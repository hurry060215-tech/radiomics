"""Record installed PyRadiomics 3.1.0 differences without altering 3.0.1 Golden."""
import argparse
import json
from pathlib import Path

from golden_generate_reference import ROOT, clean, feature_reference


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        output = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                output.append({"path": path + "/" + key, "reference": a.get(key), "installed": b.get(key)})
            else:
                output.extend(differences(a[key], b[key], path + "/" + key))
        return output
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [item for i, (x, y) in enumerate(zip(a, b)) for item in differences(x, y, path + "/" + str(i))]
    if a != b:
        return [{"path": path, "reference": a, "installed": b}]
    return []


def main():
    import radiomics
    if radiomics.__version__ != "v3.1.0":
        raise SystemExit("This separate drift report requires local PyRadiomics v3.1.0.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a report.")
    reference = json.loads((ROOT / "golden/python_golden_v1/reference.json").read_text())
    cases = json.loads((ROOT / "golden/python_golden_v1/cases.json").read_text())["cases"]
    actual = clean(feature_reference(cases))
    diff = differences(reference["features"], actual)
    report = {"reference": "PyRadiomics v3.0.1", "compared": radiomics.__version__,
              "scope": "The 10 committed synthetic fixtures only; no claim of general version equivalence.",
              "comparison": "Exact JSON values, statuses, and warnings; numeric tolerance not used to hide drift.",
              "different_entries": len(diff), "differences": diff}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("different entries:", len(diff))


if __name__ == "__main__":
    main()
