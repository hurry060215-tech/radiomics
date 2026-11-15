"""Freeze train-only table preprocessing, selection and basic test references."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from statsmodels.stats.multitest import multipletests

from golden_generate_reference import observed


def main():
    versions = {"numpy": "1.23.0", "scipy": "1.13.1", "scikit-learn": "1.0.2", "statsmodels": "0.13.2"}
    if any(metadata.version(k) != v for k, v in versions.items()):
        raise SystemExit("Table reference dependency versions do not match the frozen environment.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; produce a fresh reference candidate.")
    data = {"numeric_names": ["a", "b", "duplicate_a", "negative_a", "constant", "tiny", "all_missing"],
            "numeric_train": [[1, 3, 2, -1, 3, .01, None], [2, 0, 4, -2, 3, .02, None], [None, 4, None, None, 3, None, None],
                              [4, 1, 8, -4, 3, .04, None], [5, 5, 10, -5, 3, .05, None], [6, 2, 12, -6, 3, .06, None]],
            "numeric_evaluation": [[7, None, 14, -7, 3, .07, None], [None, 3, None, None, 3, None, None]],
            "categorical_names": ["site", "sex"], "categorical_train": [["B", "M"], [None, "F"], ["A", "F"], ["B", None], ["A", "M"], [None, "F"]],
            "categorical_evaluation": [["A", "F"], [None, None]], "unknown_category_evaluation": [["Z", "F"]],
            "targets": {"binary": [0, 0, 0, 1, 1, 1], "multiclass": [0, 0, 1, 1, 2, 2], "regression": [1., 2., 1., 4., 3., 6.]},
            "policy": {"all_missing_default": "error", "all_missing_explicit_drop": True, "variance_threshold": .001,
                       "correlation_threshold": .9, "p_threshold": .2, "bh_enabled": True,
                       "category_tie": "lexicographically smallest observed string", "unknown_category": "error", "one_hot_drop": None}}
    raw = np.asarray(data["numeric_train"], dtype=float)
    usable = ~np.isnan(raw).all(axis=0)
    names = [name for name, keep in zip(data["numeric_names"], usable) if keep]
    imputer = SimpleImputer(strategy="median").fit(raw[:, usable])
    x = imputer.transform(raw[:, usable])
    evaluate = imputer.transform(np.asarray(data["numeric_evaluation"], dtype=float)[:, usable])
    numeric = {"retained_names_after_explicit_all_missing_drop": names, "medians": imputer.statistics_.tolist(),
               "imputed_training": x.tolist(), "imputed_evaluation": evaluate.tolist(), "scalers": {}}
    for name, scaler in (("standard", StandardScaler()), ("minmax", MinMaxScaler(feature_range=(0, 1), clip=False)),
                         ("robust", RobustScaler(quantile_range=(25, 75), unit_variance=False))):
        scaler.fit(x)
        attributes = {key: getattr(scaler, key).tolist() for key in ("mean_", "var_", "scale_", "data_min_", "data_max_", "data_range_", "min_", "center_") if hasattr(scaler, key)}
        numeric["scalers"][name] = {"parameters": scaler.get_params(), "state": attributes,
                                    "training": scaler.transform(x).tolist(), "evaluation": scaler.transform(evaluate).tolist()}
    cats = np.asarray(data["categorical_train"], dtype=object)
    cat_imputer = SimpleImputer(missing_values=None, strategy="most_frequent").fit(cats)
    imputed = cat_imputer.transform(cats)
    encoder = OneHotEncoder(handle_unknown="error", sparse=False, drop=None).fit(imputed)
    cat_eval = cat_imputer.transform(np.asarray(data["categorical_evaluation"], dtype=object))
    categorical = {"modes": cat_imputer.statistics_.tolist(), "categories": [values.tolist() for values in encoder.categories_],
                   "imputed_training": imputed.tolist(), "imputed_evaluation": cat_eval.tolist(),
                   "one_hot_training": encoder.transform(imputed).tolist(), "one_hot_evaluation": encoder.transform(cat_eval).tolist(),
                   "unknown_category": observed(lambda: encoder.transform(np.asarray(data["unknown_category_evaluation"], dtype=object)))}
    variance = x.var(axis=0, ddof=0)
    after_variance = np.flatnonzero(variance > data["policy"]["variance_threshold"])
    candidate_names = [names[i] for i in after_variance]
    matrix = x[:, after_variance]
    correlation = np.corrcoef(matrix, rowvar=False)
    selected = []
    for j in range(len(candidate_names)):
        if not any(abs(correlation[j, k]) > data["policy"]["correlation_threshold"] for k in selected):
            selected.append(j)
    selection = {"population_variances": variance.tolist(), "after_variance": candidate_names,
                 "correlation_matrix": correlation.tolist(), "stable_correlation_only_names": [candidate_names[i] for i in selected], "univariate": {}}
    for task, target in data["targets"].items():
        y = np.asarray(target)
        statistics, pvalues = [], []
        for j in range(matrix.shape[1]):
            column = matrix[:, j]
            if task == "binary":
                statistic, p = stats.mannwhitneyu(column[y == 0], column[y == 1], alternative="two-sided", use_continuity=True, method="auto")
            elif task == "multiclass":
                statistic, p = stats.kruskal(*(column[y == label] for label in sorted(set(y))))
            else:
                statistic, p = stats.pearsonr(column, y)
            statistics.append(float(statistic))
            pvalues.append(float(p))
        adjusted = multipletests(pvalues, method="fdr_bh")[1]
        kept = [j for j, p in enumerate(adjusted) if p < data["policy"]["p_threshold"]]
        final = []
        for j in kept:
            if not any(abs(correlation[j, k]) > data["policy"]["correlation_threshold"] for k in final):
                final.append(j)
        selection["univariate"][task] = {"statistic": statistics, "p_values": pvalues, "bh_p_values": adjusted.tolist(),
                                            "passed_names": [candidate_names[i] for i in kept], "after_stable_correlation": [candidate_names[i] for i in final]}
    raw_p = [.01, .04, .03, .002, .9, .04]
    tests = {"mannwhitney_no_ties_exact": dict(zip(["statistic", "p_value"], stats.mannwhitneyu([1, 2, 3], [4, 5, 6], alternative="two-sided", use_continuity=True, method="auto"))),
             "mannwhitney_ties_asymptotic": dict(zip(["statistic", "p_value"], stats.mannwhitneyu([1, 2, 4], [4, 5, 6], alternative="two-sided", use_continuity=True, method="auto"))),
             "kruskal_all_identical": observed(lambda: stats.kruskal([1, 1], [1, 1], [1, 1])),
             "bh": {"raw_p": raw_p, "adjusted_p": multipletests(raw_p, method="fdr_bh")[1].tolist()}}
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    result = {"schema": "python_golden_v1.tables.reference.v1", "packages": versions, "source_revisions": {"sklearn": "3363ab79a71c8c0b034bd3fc787ea0be24627fed", "scipy": "34a5ce2e0e51fc2005d3c2720bcd8296cd33af94"},
              "input_sha256": hashlib.sha256(payload).hexdigest(),
              "conventions": {"fit_scope": "Only the six training rows; evaluation rows and targets never fit preprocessing.",
                              "missing": "JSON null denotes explicit missing. Default all-missing numeric/category column errors; deletion requires explicit drop policy.",
                              "standard": "Population variance ddof=0; constant scale=1.", "minmax": "Training min/max, range0..1, constant scale denominator1; no clipping prediction values.",
                              "robust": "Median center and q75-q25 using linear quantiles on imputed training data; zero IQR scale1, no Gaussian unit-variance correction.",
                              "selection": "Drop variance<=threshold, univariate test+optional BH, then absolute Pearson correlation>threshold with original input order retention. Empty feature set is an error.",
                              "mannwhitney": "Scipy1.13.1 auto: exact for smaller group<=8 with no ties; asymptotic with tie correction and continuity otherwise; two-sided.",
                              "kruskal": "Tie-corrected H with chi-square df=k-1; all-identical data errors.",
                              "pearson": "Two-sided Pearson target correlation; p-value from frozen SciPy distribution implementation.",
                              "BH": "fdr_bh, independent Benjamini-Hochberg, stable original output order."},
              "numeric": numeric, "categorical": categorical, "selection": selection, "small_tests": tests,
              "native_error_rules": ["All missing column unless explicit drop", "Unknown category at transform", "Zero features after an explicit selection threshold; never relax it automatically", "All-identical Kruskal-Wallis groups when called directly"]}
    args.output.mkdir(parents=True)
    (args.output / "table_inputs.json").write_bytes(payload)
    (args.output / "table_reference.json").write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
