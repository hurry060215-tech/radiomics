"""Freeze actual public Pingouin 0.5.5 ICC(2,1)/ICC(3,1), without display rounding."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from golden_generate_reference import clean, observed


def dependency_lock():
    pending = ["pingouin"]
    versions = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        versions[name] = metadata.version(name)
        for specification in metadata.requires(name) or []:
            requirement = Requirement(specification)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            version = metadata.version(requirement.name)
            if not requirement.specifier.contains(version, prereleases=True):
                raise RuntimeError("Reference environment violates declared dependency: " + specification)
            pending.append(requirement.name)
    return dict(sorted(versions.items()))


def execute(matrix):
    frame = pd.DataFrame([(subject, reader, value) for subject, row in enumerate(matrix) for reader, value in enumerate(row)], columns=["subject", "reader", "value"])
    previous = pg.options.copy()
    try:
        # Disable output formatting only in this reference call; restore all options.
        pg.options["round"] = None
        pg.options["round.column.CI95%"] = None
        result = pg.intraclass_corr(data=frame, targets="subject", raters="reader", ratings="value", nan_policy="raise")
    finally:
        pg.options.clear()
        pg.options.update(previous)
    output = {}
    for kind, label in (("ICC(2,1)", "ICC2"), ("ICC(3,1)", "ICC3")):
        row = result.loc[result.Type == label].iloc[0]
        output[kind] = {"type": kind, "estimate": float(row.ICC), "confidence": .95,
                        "ci95": row["CI95%"].tolist(), "f_statistic": float(row.F),
                        "df1": int(row.df1), "df2": int(row.df2), "p_value": float(row.pval)}
    return output


def anova(matrix):
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    grand = x.mean()
    subject = x.mean(axis=1)
    reader = x.mean(axis=0)
    ms_subject = k * np.sum((subject-grand)**2) / (n-1)
    ms_reader = n * np.sum((reader-grand)**2) / (k-1)
    ms_error = np.sum((x-subject[:, None]-reader[None, :]+grand)**2) / ((n-1)*(k-1))
    return {"n_subjects": n, "n_readers": k, "grand_mean": float(grand), "ms_subject": float(ms_subject),
            "ms_reader": float(ms_reader), "ms_error": float(ms_error)}


def main():
    versions = {"numpy": "1.23.0", "scipy": "1.13.1", "pandas": "1.5.3", "pingouin": "0.5.5"}
    if any(metadata.version(name) != version for name, version in versions.items()):
        raise SystemExit("ICC requires its isolated pinned Pingouin reference environment.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; produce a fresh ICC review candidate.")
    if args.lock_output and args.lock_output.exists():
        raise SystemExit("Refusing an existing dependency lock.")
    lock = dependency_lock()
    base = [[1.,2.,1.],[2.,2.,3.],[3.,4.,2.],[5.,5.,6.],[4.,3.,4.],[7.,6.,8.]]
    cases = {"balanced": base,
             "reader_bias": [[value + shift for value, shift in zip(row,[0.,5.,-3.])] for row in base],
             "low_agreement": [[1.,3.,2.],[3.,1.,2.],[2.,2.,3.],[3.,2.,1.],[1.,2.,3.],[2.,3.,1.]],
             "perfect_agreement": [[1.,1.,1.],[2.,2.,2.],[3.,3.,3.],[4.,4.,4.]],
             "constant": [[1.,1.,1.]]*4,
             "missing": [[1.,2.,1.],[2.,None,3.],[3.,4.,2.],[5.,5.,6.]]}
    output = {}
    for name, matrix in cases.items():
        case = {"subject_ids": ["S"+str(i) for i in range(len(matrix))], "reader_ids": ["R0","R1","R2"],
                "values_subject_by_reader": matrix, "reference": observed(lambda: execute(matrix))}
        if name != "missing":
            case["anova"] = anova(matrix)
        output[name] = case
    source = Path(pg.__file__).parent / "reliability.py"
    result = {"schema": "python_golden_v1.icc.reference.v1", "packages": versions, "dependency_lock": lock,
              "reference_source": {"repository": "https://github.com/raphaelvallat/pingouin", "tag": "v0.5.5", "revision": "b1488dd4e6e872d016c3d55af5b2ed255c40ea63", "reliability_py_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
              "effective_parameters": {"nan_policy": "raise", "confidence": .95, "reference_round": None, "reference_round_column_CI95": None},
              "specification": "Complete balanced subject-by-reader matrix only; ICC(2,1)=two-way random absolute agreement single rating; ICC(3,1)=two-way fixed consistency single rating. No mixed models, imputation, average-measure ICC or omitted type.",
              "confidence_definition": "Public Pingouin0.5.5 Shrout-Fleiss ANOVA/F interval, with output presentation rounding disabled temporarily and restored. Report unrounded CI. Preserve negative estimates and bounds; do not clip to zero.",
              "license": "Original synthetic input matrices CC0-1.0. Pingouin source retains its BSD-3-Clause license; this artifact contains numerical outputs and provenance.",
              "cases": output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(clean(result), indent=2, allow_nan=False) + "\n").encode())
    if args.lock_output:
        args.lock_output.parent.mkdir(parents=True, exist_ok=True)
        args.lock_output.write_bytes(("# CPython3.9.21; complete Pingouin runtime dependency closure, no extras\n" + "\n".join(k + "==" + v for k, v in lock.items()) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
