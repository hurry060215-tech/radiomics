"""Freeze distribution tails/quantiles, hypothesis tests and complete-case Logit."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from scipy import stats
import statsmodels.api as sm

from golden_generate_reference import ROOT, clean, observed


def distributions():
    probabilities = [0., 1e-8, .0001, .025, .5, .975, .9999, 1-1e-8, 1.]
    output = []
    definitions = [("normal", {}, stats.norm, [-8., -4., -1.96, 0., 1.96, 4., 8.])]
    definitions.extend(("student_t", {"df": df}, stats.t(df), [-10., -2., 0., 1., 6.]) for df in [.5, 1., 2., 5., 30., 100.])
    definitions.extend(("chi_square", {"df": df}, stats.chi2(df), [0., .1, 1., 10., 100.]) for df in [.5, 1., 2., 5., 30.])
    definitions.extend(("f", {"dfn": dfn, "dfd": dfd}, stats.f(dfn, dfd), [0., .01, .5, 1., 10., 100.]) for dfn, dfd in [(1., 1.), (2., 10.), (5., 12.), (7., 21.)])
    for name, parameters, distribution, values in definitions:
        output.append({"distribution": name, "parameters": parameters,
                       "evaluations": [{"x": x, "cdf": float(distribution.cdf(x)), "sf": float(distribution.sf(x))} for x in values],
                       "quantiles": [{"p": p, "value": float(distribution.ppf(p))} for p in probabilities]})
    return output


def tests():
    a, b = [1., 2., 3., 5.], [2., 4., 7., 8., 10.]
    independent = {}
    for method, equal in (("student_t", True), ("welch", False)):
        result = stats.ttest_ind(a, b, equal_var=equal, nan_policy="raise", alternative="two-sided")
        independent[method] = {"a": a, "b": b, "statistic": float(result.statistic), "df": float(result.df), "p_value": float(result.pvalue)}
    for name, table in (("2x2", [[12, 5], [4, 9]]), ("2x3", [[12, 7, 3], [5, 8, 11]])):
        statistic, p, df, expected = stats.chi2_contingency(table, correction=False)
        independent["chi_square_" + name] = {"table": table, "correction": False, "statistic": float(statistic), "p_value": float(p), "df": int(df), "expected_counts": expected.tolist()}
    independent["fisher_2x2"] = []
    for table in ([[1, 9], [11, 3]], [[0, 4], [5, 2]], [[3, 0], [0, 3]], [[0, 0], [1, 3]]):
        result = stats.fisher_exact(table, alternative="two-sided")
        independent["fisher_2x2"].append({"table": table, "alternative": "two-sided", "odds_ratio": float(result.statistic), "p_value": float(result.pvalue)})
    independent["invalid_expected_cell"] = observed(lambda: stats.chi2_contingency([[0, 0], [0, 1]], correction=False))
    independent["constant_equal_t"] = observed(lambda: list(stats.ttest_ind([1, 1, 1], [1, 1, 1], equal_var=True)))
    independent["constant_different_welch"] = observed(lambda: list(stats.ttest_ind([1, 1, 1], [2, 2, 2], equal_var=False)))
    return independent


def clinical_reference():
    path = ROOT / "golden/python_golden_v1/model_inputs.json"
    source = json.loads(path.read_text())["datasets"]["binary"]["records"]
    rows = []
    for record in source:
        if record["cohort"] != "train":
            continue
        patient_index = int(record["patient_id"].split("-P")[1])
        rows.append({"sample_id": record["sample_id"], "patient_id": record["patient_id"],
                     "a": record["values"][0], "b": record["values"][1], "sex": "F" if patient_index % 3 == 0 else "M", "target": record["target"]})
    rows[0]["a"] = None
    rows[1]["b"] = None
    rows[2]["target"] = None
    rows[3]["sex"] = None
    fits = {}
    for name, features in (("a", ["a"]), ("b", ["b"]), ("sex", ["sex"]), ("multivariable", ["a", "b", "sex"])):
        complete = [row for row in rows if all(row[key] is not None for key in features + ["target"])]
        x = np.asarray([[1.] + [float(row[key] == "M") if key == "sex" else row[key] for key in features] for row in complete])
        y = np.asarray([row["target"] for row in complete])
        fit = sm.Logit(y, x).fit(method="newton", maxiter=200, tol=1e-12, disp=False)
        if not fit.mle_retvals["converged"]:
            raise RuntimeError("Reference clinical Logit did not converge.")
        covariance = np.asarray(fit.cov_params())
        information = -fit.model.hessian(fit.params)
        score = fit.model.score(fit.params)
        ci = fit.conf_int(alpha=.05)
        fits[name] = {"features": features, "terms": ["const"] + ["sex_M" if key == "sex" else key for key in features],
                      "reference_categories": {"sex": "F"} if "sex" in features else {},
                      "used_sample_ids": [row["sample_id"] for row in complete], "n_used": len(complete), "n_dropped": len(rows)-len(complete),
                      "coefficient": fit.params.tolist(), "covariance": covariance.tolist(), "information": information.tolist(), "score": score.tolist(),
                      "standard_error": fit.bse.tolist(), "z": fit.tvalues.tolist(), "p_values": fit.pvalues.tolist(),
                      "coefficient_ci95": ci.tolist(), "odds_ratio": np.exp(fit.params).tolist(), "odds_ratio_ci95": np.exp(ci).tolist(),
                      "apparent_probabilities": fit.predict(x).tolist(), "log_likelihood": float(fit.llf),
                      "design_rank": int(np.linalg.matrix_rank(x)), "converged": True}
    separation = observed(lambda: sm.Logit([0, 0, 1, 1], sm.add_constant([-2., -1., 1., 2.])).fit(disp=False, maxiter=200))
    return {"input_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": rows,
            "policy": "Unpenalized binary inference on complete cases per selected model, no imputation; sex reference F, intercept included; observed Fisher information inverse and normal Wald CI95. Apparent predictions are not validation performance.",
            "reference_parameters": {"solver": "statsmodels Logit/newton", "maxiter": 200, "tol": 1e-12, "confidence": .95, "penalty": None},
            "fits": fits, "complete_separation": separation,
            "native_error_rules": ["Rank-deficient augmented design or <=p complete cases", "Complete or quasi separation; numerical convergence alone does not prove finite MLE", "Nonconvergence or singular information matrix"]}


def main():
    required = {"numpy": "1.23.0", "scipy": "1.13.1", "statsmodels": "0.13.2"}
    if any(metadata.version(name) != version for name, version in required.items()):
        raise SystemExit("Statistics reference requires the pinned reference environment.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; generate a fresh review candidate.")
    result = {"schema": "python_golden_v1.statistics.reference.v1", "packages": required,
              "reference_source_revisions": {"scipy_v1.13.1": "34a5ce2e0e51fc2005d3c2720bcd8296cd33af94", "statsmodels_v0.13.2": "c87540849e8ac616cf768353cc156d3478d0fc93"},
              "distributions": distributions(), "hypothesis_tests": tests(), "clinical_logistic": clinical_reference(),
              "tolerances": {"scalar_atol": 1e-10, "scalar_rtol": 1e-8, "logistic_prediction_atol": 1e-6, "logistic_prediction_rtol": 1e-5, "nonfinite": "identical location/state"},
              "remaining": ["ICC21/ICC31 with independently checked CI", "Complete native multiclass/regression metrics, ROC/PR/calibration", "Patient-cluster bootstrap using frozen legacy sampling indices"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(clean(result), indent=2, allow_nan=False) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
