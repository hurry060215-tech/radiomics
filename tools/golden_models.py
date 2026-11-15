"""Freeze original three-task data and explicit solver objectives independently.

Reference fits use public sklearn 1.0.2. Native solvers must use the documented
sample-mean objectives and unpenalized intercepts; C/alpha mappings are explicit.
This file writes only fresh candidates and never imports native model code.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def synthetic_data():
    rng = np.random.RandomState(20261114)
    datasets = {}
    for task in ("binary", "multiclass", "regression"):
        records = []
        for patient in range(66):
            latent = rng.normal(size=4)
            label = patient % (3 if task == "multiclass" else 2)
            target = (1.5 * latent[0] - .7 * latent[1] + .4 * latent[2] + rng.normal(scale=.35)) if task == "regression" else label
            for lesion in range(2):
                values = latent + rng.normal(scale=.12, size=4)
                if task != "regression":
                    values[0] += label * .85
                    values[1] += (label == 1) * .55
                    values[2] += (label == 2) * .6
                records.append({"sample_id": f"{task}-P{patient:03d}-L{lesion}",
                                "patient_id": f"{task}-P{patient:03d}", "lesion_id": str(lesion),
                                "cohort": "train" if patient < 48 else ("test" if patient < 60 else "external_A"),
                                "center": "A" if patient < 48 else ("B" if patient < 60 else "C"),
                                "values": values.tolist(), "target": target if patient < 60 else None})
        datasets[task] = {"feature_names": ["signal_a", "signal_b", "noise_a", "noise_b"],
                          "classes": ["negative", "positive"] if task == "binary" else (["benign", "intermediate", "malignant"] if task == "multiclass" else []),
                          "records": records}
    return {"schema": "python_golden_v1.models.inputs.v1", "license": "CC0-1.0",
            "provenance": "Original synthetic data generated with NumPy1.23 RandomState seed20261114, 66 patients/task, two lesions/patient.",
            "datasets": datasets}


def regression_state(model, x, y, penalty, lasso=False):
    w, b = model.coef_, float(model.intercept_)
    residual = x @ w + b - y
    gradient = x.T @ residual / len(y)
    intercept_gradient = float(residual.mean())
    if lasso:
        kkt = np.where(np.abs(w) > 1e-10, np.abs(gradient + penalty * np.sign(w)), np.maximum(np.abs(gradient) - penalty, 0))
        objective = float(np.mean(residual ** 2) / 2 + penalty * np.abs(w).sum())
    else:
        kkt = np.abs(gradient + penalty * w)
        objective = float(np.mean(residual ** 2) / 2 + penalty * np.sum(w ** 2) / 2)
    return {"coefficient": w.tolist(), "intercept": b, "objective": objective,
            "smooth_gradient": gradient.tolist(), "intercept_gradient": intercept_gradient,
            "kkt_inf": float(max(kkt.max(), abs(intercept_gradient))),
            "support": np.flatnonzero(np.abs(w) > 1e-10).tolist()}


def binary_logistic_state(model, x, y, penalty, l1=False):
    w, b = model.coef_[0], float(model.intercept_[0])
    linear = x @ w + b
    probability = expit(linear)
    gradient = x.T @ (probability - y) / len(y)
    intercept_gradient = float((probability - y).mean())
    if l1:
        kkt = np.where(np.abs(w) > 1e-10, np.abs(gradient + penalty * np.sign(w)), np.maximum(np.abs(gradient) - penalty, 0))
        penalty_value = penalty * np.abs(w).sum()
    else:
        kkt = np.abs(gradient + penalty * w)
        penalty_value = penalty * np.sum(w ** 2) / 2
    return {"coefficient": w.tolist(), "intercept": b,
            "objective": float(np.mean(np.logaddexp(0, linear) - y * linear) + penalty_value),
            "smooth_gradient": gradient.tolist(), "intercept_gradient": intercept_gradient,
            "kkt_inf": float(max(kkt.max(), abs(intercept_gradient))),
            "iterations": int(model.n_iter_[0]), "support": np.flatnonzero(np.abs(w) > 1e-10).tolist()}


def softmax_state(model, x, y, penalty):
    w, b = model.coef_, model.intercept_
    logits = x @ w.T + b
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    onehot = np.eye(len(model.classes_))[y]
    residual = np.exp(logp) - onehot
    gradient = residual.T @ x / len(y) + penalty * w
    intercept_gradient = residual.mean(axis=0)
    return {"coefficient": w.tolist(), "intercept": b.tolist(),
            "centered_coefficient": (w - w.mean(axis=0)).tolist(), "centered_intercept": (b - b.mean()).tolist(),
            "objective": float(-logp[np.arange(len(y)), y].mean() + penalty * np.sum(w ** 2) / 2),
            "penalized_gradient": gradient.tolist(), "intercept_gradient": intercept_gradient.tolist(),
            "kkt_inf": float(max(np.abs(gradient).max(), np.abs(intercept_gradient).max())),
            "iterations": int(model.n_iter_[0])}


def fit_reference(data):
    output = {}
    for task, dataset in data["datasets"].items():
        records = dataset["records"]
        train = [r for r in records if r["cohort"] == "train"]
        # Held-out and external feature rows are prediction-only. Training probes
        # additionally exercise KNN distance-zero semantics, never fit on test labels.
        evaluate = train[:3] + [r for r in records if r["cohort"] != "train"]
        raw = np.asarray([r["values"] for r in train])
        mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=0)
        x = (raw - mean) / scale
        test_x = (np.asarray([r["values"] for r in evaluate]) - mean) / scale
        y = np.asarray([r["target"] for r in train], dtype=float if task == "regression" else int)
        n = len(y)
        model_results = {}
        if task == "regression":
            for name, model, penalty, lasso in (("Linear", LinearRegression(fit_intercept=True), 0., False),
                                                ("Ridge", Ridge(alpha=n * .1, fit_intercept=True, solver="svd"), .1, False),
                                                ("LASSO", Lasso(alpha=.05, fit_intercept=True, max_iter=50000, tol=1e-12, selection="cyclic"), .05, True)):
                model.fit(x, y)
                state = regression_state(model, x, y, penalty, lasso)
                model_results[name] = {"parameters": model.get_params(), "native_penalty": penalty, "state": state,
                                       "predictions": model.predict(test_x).tolist(), "converged": True}
            for weights in ("uniform", "distance"):
                model = KNeighborsRegressor(n_neighbors=5, weights=weights, algorithm="brute", metric="euclidean", n_jobs=1).fit(x, y)
                model_results["KNN_" + weights] = {"parameters": model.get_params(), "predictions": model.predict(test_x).tolist()}
            collinear = np.c_[x[:, :2], x[:, 0]]
            model = LinearRegression().fit(collinear, y)
            failures = {"rank_deficient_OLS": {"input_column_indices": [0, 1, 0], "reference_rank": int(model.rank_), "design_columns_including_intercept": 4,
                                                "native_rule": "error_rank_deficient; do not silently return the reference minimum-norm solution"}}
        else:
            penalty = .1
            model = LogisticRegression(C=1 / (n * penalty), solver="newton-cg", penalty="l2", fit_intercept=True, multi_class="multinomial" if task == "multiclass" else "ovr", max_iter=10000, tol=1e-7, random_state=0, n_jobs=1).fit(x, y)
            state = softmax_state(model, x, y, penalty) if task == "multiclass" else binary_logistic_state(model, x, y, penalty)
            model_results["LR"] = {"parameters": model.get_params(), "native_penalty": penalty, "classes": model.classes_.tolist(),
                                   "state": state, "probabilities": model.predict_proba(test_x).tolist(), "converged": True}
            for weights in ("uniform", "distance"):
                model = KNeighborsClassifier(n_neighbors=5, weights=weights, algorithm="brute", metric="euclidean", n_jobs=1).fit(x, y)
                model_results["KNN_" + weights] = {"parameters": model.get_params(), "probabilities": model.predict_proba(test_x).tolist(), "classes": model.classes_.tolist()}
            model = GaussianNB(priors=None, var_smoothing=1e-9).fit(x, y)
            model_results["GaussianNB"] = {"parameters": model.get_params(), "classes": model.classes_.tolist(), "class_prior": model.class_prior_.tolist(),
                                           "theta": model.theta_.tolist(), "variance": model.var_.tolist(), "epsilon": float(model.epsilon_),
                                           "probabilities": model.predict_proba(test_x).tolist()}
            if task == "binary":
                model = Lasso(alpha=.03, fit_intercept=True, max_iter=50000, tol=1e-12, selection="cyclic").fit(x, y)
                model_results["LASSO"] = {"parameters": model.get_params(), "native_penalty": .03, "state": regression_state(model, x, y, .03, True),
                                           "linear_scores_not_probabilities": model.predict(test_x).tolist(), "converged": True}
            else:
                results = []
                support = set()
                for label in range(3):
                    binary = (y == label).astype(int)
                    model = LogisticRegression(C=1 / (n * .03), penalty="l1", solver="saga", fit_intercept=True, max_iter=50000, tol=1e-12, random_state=0, n_jobs=1).fit(x, binary)
                    state = binary_logistic_state(model, x, binary, .03, l1=True)
                    support.update(state["support"])
                    results.append({"positive_class": label, "parameters": model.get_params(), "state": state})
                model_results["OvR_L1_Logistic"] = {"native_penalty": .03, "per_class": results, "support_union": sorted(support), "converged": True}
            failures = {}
        output[task] = {"train_sample_ids": [r["sample_id"] for r in train], "evaluation_sample_ids": [r["sample_id"] for r in evaluate],
                        "standard_scaling": {"mean": mean.tolist(), "scale": scale.tolist(), "ddof": 0, "fit_cohort": "train"},
                        "training_design_rank_including_intercept": int(np.linalg.matrix_rank(np.c_[np.ones(n), x])),
                        "models": model_results, "failures": failures}
    return output


def main():
    required = {"numpy": "1.23.0", "scipy": "1.13.1", "scikit-learn": "1.0.2"}
    for package, version in required.items():
        if metadata.version(package) != version:
            raise SystemExit("Expected " + package + "==" + version)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; produce a fresh review candidate.")
    data = synthetic_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        reference = fit_reference(data)
    result = {"schema": "python_golden_v1.models.reference.v1", "packages": required,
              "reference_source_revisions": {"scikit-learn_1.0.2": "3363ab79a71c8c0b034bd3fc787ea0be24627fed", "scipy_v1.13.1": "34a5ce2e0e51fc2005d3c2720bcd8296cd33af94", "numpy_v1.23.0": "499ac172538d32218da9d1c63b4faa71f864bc71"},
              "input_sha256": hashlib.sha256(json_bytes(data)).hexdigest(),
              "objectives": {"LR_binary": "mean(logaddexp(0,z)-y*z) + lambda/2 * sum(w^2)",
                             "LR_multiclass": "mean(-log_softmax(z)[true_class]) + lambda/2 * sum(W^2)",
                             "Ridge_OLS": "SSE/(2*n) + lambda/2 * sum(w^2), OLS lambda=0 requires full design rank",
                             "LASSO": "SSE/(2*n) + alpha*sum(abs(w))",
                             "OvR_L1_Logistic": "mean(logaddexp(0,z)-y_binary*z) + alpha*sum(abs(w)); union support across classes",
                             "intercept": "Never penalized. Softmax common intercept shift is nonidentifiable; compare centered intercepts or probabilities/objective/gradient.",
                             "reference_mapping": "LR C=1/(n*lambda); L1 Logistic C=1/(n*alpha); Ridge sklearn alpha=n*lambda; LASSO sklearn alpha=alpha"},
              "acceptance": {"prediction_atol": 1e-6, "prediction_rtol": 1e-5, "objective_atol": 1e-10, "objective_rtol": 1e-8,
                             "gradient_or_KKT_inf_max": 1e-7, "convergence_required": True,
                             "class_mapping": "Dataset classes ordered explicitly; numeric targets are indices, never continuous values for multiclass sparse selection.",
                             "scope": "Solver-level references on train-standardized buffers, no validation preprocessing or labels fitted. Full nested workflow acceptance remains separate."},
              "datasets": reference}
    args.output.mkdir(parents=True)
    (args.output / "model_inputs.json").write_bytes(json_bytes(data))
    (args.output / "model_reference.json").write_bytes(json_bytes(result))
    print(args.output)


if __name__ == "__main__":
    main()
