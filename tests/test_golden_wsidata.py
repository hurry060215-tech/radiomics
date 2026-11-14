"""Independent geometry/identity/typed-canonical checks for strict WSI metadata."""
import hashlib
import json
from pathlib import Path
import struct


DIRECTORY=Path(__file__).resolve().parents[1]/"golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY/name).read_text())


def test_wsidata_uses_rectangle_bounds_and_actual_row_bijection():
    data=read("wsidata_contract_inputs.json");reference=read("wsidata_contract_reference.json")
    assert reference["input_sha256"]==hashlib.sha256((DIRECTORY/"wsidata_contract_inputs.json").read_bytes()).hexdigest()
    assert reference["specification_sha256"]==hashlib.sha256((DIRECTORY/"wsidata_contract_spec.py").read_bytes()).hexdigest()
    payload=reference["current_snapshot"]["snapshot"]
    assert payload["tiles"][0]["bounds0"]==[0,0,4,4]
    assert data["shape_rows"][0]["x"]==2 and data["shape_rows"][0]["y"]==2
    assert reference["expected_row_order"]==[1,0]
    assert reference["aligned_feature_values"]==data["expected_aligned_values"]
    assert reference["source_row_ids_in_shape_order"]==["barcode_A","barcode_B"]
    assert reference["variants"]["reversed_shape_order"]["snapshot"]["sha256"]==reference["current_snapshot"]["sha256"]
    assert reference["variants"]["reversed_shape_order"]["row_order"]==[0,1]
    edge=reference["variants"]["explicit_valid_edge"]["snapshot"]["tiles"][0]
    assert edge["bounds0"]==[18,0,22,4]
    assert edge["valid_pixels"]==[0,0,1,2]


def test_wsidata_nominal_roundoff_changes_no_coordinate_or_cache_rule():
    reference=read("wsidata_contract_reference.json")
    translated=reference["variants"]["translation_point1"]["snapshot"]
    assert translated["tiles"][0]["bounds0"]==[.1,.1,4.1,4.1]
    assert translated["tiles"][0]["bounds0"][2]-translated["tiles"][0]["bounds0"][0]!=4
    assert translated["nominal_span_validation"]["operations"]==6
    assert translated["nominal_span_validation"]["max_relative_roundoff"]==1e-10
    base=reference["current_snapshot"]["snapshot"]
    for key in ["base_downsample","ops_downsample","mpp"]:
        assert type(base["tile_spec"][key]) is float
    for key in ["width","height","base_level","ops_level"]:
        assert type(base["tile_spec"][key]) is int
    assert all(type(value) is float for row in base["shape_to_level0"] for value in row)
    assert all(value["status"]=="error" for value in reference["failures"].values())
    assert "unreliable" in reference["failures"]["roundoff_bound_too_large"]["message"] or "reliably" in reference["failures"]["roundoff_bound_too_large"]["message"]


def test_wsidata_canonical_float_hex_includes_signed_zero_and_subnormal():
    reference=read("wsidata_contract_reference.json")
    for case in reference["canonical_float_cases"]:
        value=case["value"]
        text=case["python_float_hex"]
        assert value.hex()==text
        assert struct.pack(">d",float.fromhex(text))==struct.pack(">d",value)
        assert case["canonical"]=={"float64_hex":text}
    assert reference["canonical_float_cases"][0]["python_float_hex"]=="0x0.0p+0"
    assert reference["canonical_float_cases"][1]["python_float_hex"]=="-0x0.0p+0"
