"""Independent geometry/identity/typed-canonical checks for strict WSI metadata."""

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def test_wsidata_uses_rectangle_bounds_and_actual_row_bijection():
    data = read("wsidata_contract_inputs.json")
    reference = read("wsidata_contract_reference.json")
    assert (
        reference["input_sha256"]
        == hashlib.sha256((DIRECTORY / "wsidata_contract_inputs.json").read_bytes()).hexdigest()
    )
    assert (
        reference["specification_sha256"]
        == hashlib.sha256((DIRECTORY / "wsidata_contract_spec.py").read_bytes()).hexdigest()
    )
    payload = reference["current_snapshot"]["snapshot"]
    assert payload["tiles"][0]["bounds0"] == [0, 0, 4, 4]
    assert data["shape_rows"][0]["x"] == 2 and data["shape_rows"][0]["y"] == 2
    assert reference["expected_row_order"] == [1, 0]
    assert reference["aligned_feature_values"] == data["expected_aligned_values"]
    assert reference["source_row_ids_in_shape_order"] == ["barcode_A", "barcode_B"]
    assert (
        reference["variants"]["reversed_shape_order"]["snapshot"]["sha256"]
        == reference["current_snapshot"]["sha256"]
    )
    assert reference["variants"]["reversed_shape_order"]["row_order"] == [0, 1]
    edge = reference["variants"]["explicit_valid_edge"]["snapshot"]["tiles"][0]
    assert edge["bounds0"] == [18, 0, 22, 4]
    assert edge["valid_pixels"] == [0, 0, 1, 2]


def test_wsidata_nominal_roundoff_changes_no_coordinate_or_cache_rule():
    reference = read("wsidata_contract_reference.json")
    translated = reference["variants"]["translation_point1"]["snapshot"]
    assert translated["tiles"][0]["bounds0"] == [0.1, 0.1, 4.1, 4.1]
    assert translated["tiles"][0]["bounds0"][2] - translated["tiles"][0]["bounds0"][0] != 4
    assert translated["nominal_span_validation"]["operations"] == 6
    assert translated["nominal_span_validation"]["max_relative_roundoff"] == 1e-10
    base = reference["current_snapshot"]["snapshot"]
    for key in ["base_downsample", "ops_downsample", "mpp"]:
        assert type(base["tile_spec"][key]) is float
    for key in ["width", "height", "base_level", "ops_level"]:
        assert type(base["tile_spec"][key]) is int
    assert all(type(value) is float for row in base["shape_to_level0"] for value in row)
    assert all(value["status"] == "error" for value in reference["failures"].values())
    assert (
        "unreliable" in reference["failures"]["roundoff_bound_too_large"]["message"]
        or "reliably" in reference["failures"]["roundoff_bound_too_large"]["message"]
    )


def test_wsidata_canonical_float_hex_includes_signed_zero_and_subnormal():
    reference = read("wsidata_contract_reference.json")
    for case in reference["canonical_float_cases"]:
        value = case["value"]
        text = case["python_float_hex"]
        assert value.hex() == text
        assert struct.pack(">d", float.fromhex(text)) == struct.pack(">d", value)
        assert case["canonical"] == {"float64_hex": text}
    assert reference["canonical_float_cases"][0]["python_float_hex"] == "0x0.0p+0"
    assert reference["canonical_float_cases"][1]["python_float_hex"] == "-0x0.0p+0"


def test_wsidata_float_tag_collision_requires_typed_payload_guard():
    reference = read("wsidata_collision_reference.json")
    for name, digest in reference["source_sha256"].items():
        assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == digest
    loader = importlib.util.spec_from_file_location(
        "wsidata_guard_test", DIRECTORY / "wsidata_contract_guard.py"
    )
    guard = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(guard)
    saved = reference["saved_snapshot"]
    current = reference["current_snapshot"]
    assert saved["sha256"] == current["sha256"]
    assert type(saved["snapshot"]["tile_spec"]["custom"]) is float
    assert type(current["snapshot"]["tile_spec"]["custom"]) is dict
    assert reference["legacy_digest_only_validation"]["status"] == "ok"
    assert reference["required_guard_validation"]["status"] == "error"
    assert not guard.typed_equal(1, 1.0) and not guard.typed_equal(True, 1)
    assert not guard.typed_equal(saved["snapshot"], current["snapshot"])
    import pytest

    with pytest.raises(ValueError, match="structure, type or value"):
        guard.validate_embedding_link(
            current, saved, "tiles", "tiles", ["T0", "T1"], ["T1", "T0"], ["b", "a"]
        )
