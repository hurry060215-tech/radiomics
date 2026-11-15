"""Freeze forest behavioral references on the three accepted fixed datasets."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, f1_score, mean_squared_error, roc_auc_score

from golden_generate_reference import ROOT


def main():
    required = {"numpy": "1.23.0", "scikit-learn": "1.0.2"}
    if any(metadata.version(name) != version for name, version in required.items()):
        raise SystemExit("Forest reference requires NumPy1.23.0 / sklearn1.0.2.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; generate a fresh review candidate.")
    directory = ROOT / "golden/python_golden_v1"
    input_path = directory / "model_inputs.json"
    data = json.loads(input_path.read_text())
    model_ref = json.loads((directory / "model_reference.json").read_text())
    seeds = [11, 23, 37, 53, 71]
    datasets = {}
    for task, dataset in data["datasets"].items():
        records = dataset["records"]
        training = [r for r in records if r["cohort"] == "train"]
        testing = [r for r in records if r["cohort"] == "test"]
        mean, scale = (np.asarray(model_ref["datasets"][task]["standard_scaling"][name]) for name in ("mean", "scale"))
        x = (np.asarray([r["values"] for r in training]) - mean) / scale
        test_x = (np.asarray([r["values"] for r in testing]) - mean) / scale
        y = np.asarray([r["target"] for r in training], dtype=float if task == "regression" else int)
        test_y = np.asarray([r["target"] for r in testing], dtype=float if task == "regression" else int)
        patients = list(dict.fromkeys(r["patient_id"] for r in testing))
        indices = [[i for i, r in enumerate(testing) if r["patient_id"] == patient] for patient in patients]
        patient_y = np.asarray([test_y[index[0]] for index in indices])
        assert all(np.all(test_y[index] == test_y[index[0]]) for index in indices)
        per_seed = []
        parameters = dict(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                          min_weight_fraction_leaf=0., max_leaf_nodes=None, min_impurity_decrease=0., bootstrap=True,
                          oob_score=False, n_jobs=1, warm_start=False, ccp_alpha=0., max_samples=None)
        parameters.update(criterion="squared_error" if task == "regression" else "gini", max_features=1. if task == "regression" else "sqrt")
        for seed in seeds:
            klass = RandomForestRegressor if task == "regression" else RandomForestClassifier
            model = klass(random_state=seed, **parameters).fit(x, y)
            if task == "regression":
                predictions = model.predict(test_x)
                patient_predictions = np.asarray([predictions[index].mean() for index in indices])
                per_seed.append({"seed": seed, "predictions": predictions.tolist(), "patient_predictions": patient_predictions.tolist(),
                                 "sample_metrics": {"rmse": float(mean_squared_error(test_y, predictions, squared=False))},
                                 "metrics": {"rmse": float(mean_squared_error(patient_y, patient_predictions, squared=False))}})
            else:
                probabilities = model.predict_proba(test_x)
                patient_probabilities = np.asarray([probabilities[index].mean(axis=0) for index in indices])
                def metrics(labels, probability):
                    return ({"auc": float(roc_auc_score(labels, probability[:, 1])), "brier": float(brier_score_loss(labels, probability[:, 1]))}
                            if task == "binary" else {"macro_f1": float(f1_score(labels, np.argmax(probability, axis=1), labels=[0, 1, 2], average="macro", zero_division=0))})
                per_seed.append({"seed": seed, "probabilities": probabilities.tolist(), "patient_probabilities": patient_probabilities.tolist(),
                                 "sample_metrics": metrics(test_y, probabilities), "metrics": metrics(patient_y, patient_probabilities)})
        means = {key: float(np.mean([row["metrics"][key] for row in per_seed])) for key in per_seed[0]["metrics"]}
        datasets[task] = {"parameters": parameters, "train_sample_ids": [r["sample_id"] for r in training],
                          "test_sample_ids": [r["sample_id"] for r in testing], "per_seed": per_seed, "mean_metrics": means,
                          "test_patient_ids": patients, "test_patient_targets": patient_y.tolist(),
                          "test_target_std_ddof0": float(patient_y.std(ddof=0)) if task == "regression" else None}
    output = {"schema": "python_golden_v1.forest.reference.v1", "packages": required,
              "reference_source_revision": "scikit-learn 1.0.2 / 3363ab79a71c8c0b034bd3fc787ea0be24627fed",
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(), "model_seeds": seeds,
              "seed_semantics": "Reference sklearn uses its own MT19937-derived tree seeds. Native records a versioned RNG scheme and derives each tree seed independently; identical sklearn trees are not required.",
              "acceptance": {"dataset_independent": True, "n_estimators": 200, "seeds_per_task": 5,
                             "evaluation_unit": "patient", "sample_to_patient": "Equal mean of the two lesion probabilities or regression scores per test patient, before metrics. Sample metrics retained separately.",
                             "binary": {"maximum_mean_auc_decrement": .03, "maximum_mean_brier_increment": .03},
                             "multiclass": {"maximum_mean_macro_f1_decrement": .04},
                             "regression": "native mean RMSE - reference mean RMSE <= max(reference mean RMSE*0.10, test target population std*0.02)",
                             "reproducibility": "Within native implementation, same seed must reproduce across repeated calls, thread counts and save/load; no seed/data substitution after results."},
              "datasets": datasets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(output, indent=2, allow_nan=False) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
