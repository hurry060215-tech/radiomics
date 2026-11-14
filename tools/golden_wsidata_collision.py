"""Independently expose and freeze the WSI float-tag collision rejection."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from golden_generate_reference import ROOT, observed


def load(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing collision reference candidate.")
    directory = ROOT / "golden/python_golden_v1"
    base = load(directory / "wsidata_contract_spec.py", "original_wsidata")
    guard = load(directory / "wsidata_contract_guard.py", "guard_wsidata")
    original = json.loads((directory / "wsidata_contract_reference.json").read_text())
    saved_input = copy.deepcopy(original["snapshot_payload"])
    current_input = copy.deepcopy(saved_input)
    saved_input["tile_spec"]["custom"] = 1.0
    current_input["tile_spec"]["custom"] = {"float64_hex": "0x1.0000000000000p+0"}
    saved = base.snapshot(**saved_input)
    current = base.snapshot(**current_input)
    arguments = [current, saved, saved_input["tile_key"], saved_input["tile_key"], ["T0", "T1"], ["T1", "T0"], ["barcode_B", "barcode_A"]]
    assert saved["sha256"] == current["sha256"]
    assert not guard.typed_equal(saved["snapshot"], current["snapshot"])
    legacy = observed(lambda: base.validate_embedding_link(*arguments))
    protected = observed(lambda: guard.validate_embedding_link(*arguments))
    assert legacy["status"] == "ok" and protected["status"] == "error"
    output = {
        "schema": "python_golden_v1.wsidata_collision.reference.v1",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ["wsidata_contract_spec.py", "wsidata_contract_guard.py", "wsidata_contract_reference.json"]},
        "saved_generation_input": saved_input, "current_input": current_input,
        "saved_snapshot": saved, "current_snapshot": current,
        "legacy_digest_only_validation": legacy, "required_guard_validation": protected,
        "policy": "Digest validity AND recursively identical normalized metadata structure/type/value are required. Existing schema, normal snapshots and digests remain unchanged. A float64_hex object is not evidence of the original metadata type.",
        "approval": "Main agent approved the independent protocol guard before production WSI implementation; no native output was used."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(output, indent=2, allow_nan=False) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
