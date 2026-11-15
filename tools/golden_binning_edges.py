"""Independent actual-arange/digitize boundary reference, before native repairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import radiomics
from radiomics import imageoperations

from golden_generate_reference import clean, observed


def main():
    if radiomics.__version__ != "v3.0.1" or np.__version__ != "1.23.0":
        raise SystemExit("Requires public PyRadiomics3.0.1 and NumPy1.23.0.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing an existing output path.")
    centers = [-.6, -.4, -.2, 0., .2, .6]
    adjacent = [float(x) for center in centers for x in (np.nextafter(center, -np.inf), center, np.nextafter(center, np.inf))]
    cases = []
    for name, values in (("decimal", centers), ("nextafter", adjacent), ("large_constant", [1e16] * 6)):
        shape = [3, len(values) // 3]
        for kwargs in ({"binWidth": .1}, {"binWidth": .2}, {"binCount": 3}, {"binCount": 7}, {"binCount": 10}):
            array = np.asarray(values).reshape(tuple(reversed(shape)))
            result = observed(lambda: imageoperations.binImage(array, np.where(np.ones_like(array, dtype=bool)), **kwargs)[0].ravel())
            cases.append({"id": name + "_" + "_".join(str(x) for x in next(iter(kwargs.items()))),
                          "shape": shape, "values": values, "mask": [1] * len(values), **kwargs,
                          "native_rule": "error_invalid_or_unsupported" if name == "large_constant" else "exact_discrete_labels",
                          "bin_edges": observed(lambda: imageoperations.getBinEdges(np.asarray(values), **kwargs)),
                          "discretized": result})
    output = {"schema": "python_golden_v1.binning_edges.v1", "reference": {"pyradiomics": "3.0.1", "numpy": "1.23.0"},
              "provenance": "Independent actual execution before native floating-edge repair; all original synthetic data CC0-1.0.",
              "scope": "Actual NumPy arange/linspace/digitize behavior, not idealized floor formula.", "comparison": "Exact gray labels and finite/error state; bin edges exact f64 values.",
              "native_rejection_rule": "A reference configuration that errors or cannot produce positive ROI gray labels is explicitly rejected as InvalidInput/Unsupported by the native positive-label contract. Reference outputs are preserved; no numeric tolerance is relaxed.",
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(clean(output), indent=2, allow_nan=False) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
