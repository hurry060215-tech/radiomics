"""Stdlib algorithm demonstration on an already frozen scalar audit; not a backend/profile."""
import argparse
import json
import math
from pathlib import Path

import audit_scalar_accuracy as audit


def freeze(base_manifest, base_sha256, output):
    if audit.sha256(base_manifest) != base_sha256:
        raise ValueError("Original frozen manifest differs.")
    output.mkdir(parents=True, exist_ok=False)
    audit.write_json(output / "manifest.json", dict(
        schema="radiomics.scalar-stable-prototype-inputs.v1", base_manifest_sha256=base_sha256,
        base_manifest=str(base_manifest.resolve()), prototype_source_sha256=audit.sha256(__file__),
        oracle_tool_sha256=audit.sha256(audit.__file__), role="Algorithm demonstration only; not a new supported backend or profile",
        mean="math.fsum(original f64 values)/n",
        variance="anchor=first f64; shifted=f64(x-anchor); center=fsum(shifted)/n; fsum(f64(shifted-center)**2)/n",
        energy="math.fsum(f64(x*x)); product rounding remains",
        limitations="Shift/subtraction/residual/product arithmetic still f64. No universal exactness guarantee; oracle remains original Fraction.from_float inputs."))


def run(manifest_path, manifest_sha256):
    if audit.sha256(manifest_path) != manifest_sha256:
        raise ValueError("Prototype manifest differs.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit.sha256(__file__) != manifest["prototype_source_sha256"] or audit.sha256(audit.__file__) != manifest["oracle_tool_sha256"]:
        raise ValueError("Prototype/oracle source changed after freeze.")
    base_path = Path(manifest["base_manifest"])
    if audit.sha256(base_path) != manifest["base_manifest_sha256"]:
        raise ValueError("Original frozen input changed.")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    before = audit.sha256(base_path.parent / "report.json")
    rows = []
    for case in base["cases"]:
        values = [float.fromhex(x) for x in case["input_f64_hex"]]
        shifted = [x - values[0] for x in values]
        center = math.fsum(shifted) / len(values)
        actual = dict(Mean=math.fsum(values) / len(values),
                      Variance=math.fsum((x - center) ** 2 for x in shifted) / len(values),
                      Energy=math.fsum(x * x for x in values))
        for name in audit.FEATURES:
            reference = audit.fraction(case["reference"][name])
            rows.append(dict(case=case["id"], feature=name, errors=audit.errors(actual[name], reference)))
    after = audit.sha256(base_path.parent / "report.json")
    if after != before:
        raise RuntimeError("Original report changed.")
    audit.write_json(manifest_path.parent / "report.json", dict(
        schema="radiomics.scalar-stable-prototype-report.v1", role=manifest["role"],
        prototype_manifest_sha256=manifest_sha256, base_manifest_sha256=manifest["base_manifest_sha256"],
        original_report_sha256_before=before, original_report_sha256_after=after,
        prototype_source_sha256=manifest["prototype_source_sha256"], rows=rows))
    print("Prototype supplement complete; original report unchanged:", before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("freeze")
    prepared.add_argument("--base-manifest", type=Path, required=True)
    prepared.add_argument("--base-sha256", required=True)
    prepared.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.base_manifest, args.base_sha256, args.output)
        print(audit.sha256(args.output / "manifest.json"))
    else:
        run(args.manifest, args.manifest_sha256)


if __name__ == "__main__":
    main()
