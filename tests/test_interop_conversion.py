"""Conversion/delegation tests. The independent Rust suite tests actual numerical alignment."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def adapter(monkeypatch):
    # Isolate conversion from the separately integrated public wrapper/PyO3 build.
    # The spy never implements sorting or identity-matching algorithms.
    class Table:
        def __init__(self, table, provenance=None):
            self.payload = copy.deepcopy(table)
            self.provenance = copy.deepcopy(provenance)

        def to_dict(self):
            return copy.deepcopy(self.payload)

        @classmethod
        def _from_native(cls, table, provenance=None):
            return cls(table, provenance=provenance)

    types_module = ModuleType("ov_radiomics._types")
    types_module.TileTable = Table
    monkeypatch.setitem(sys.modules, "ov_radiomics._types", types_module)
    path = Path(__file__).resolve().parents[1] / "src/ov_radiomics/interop.py"
    spec = importlib.util.spec_from_file_location("ov_radiomics._interop_conversion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def align(table_json, ids, buffer_json, names):
        calls.append((json.loads(table_json), list(ids), json.loads(buffer_json), list(names)))
        result = json.loads(table_json)
        result["features"] = json.loads(buffer_json)
        result["feature_names"] = list(names)
        return json.dumps(result)

    monkeypatch.setattr(
        module, "resolve_backend", lambda backend: SimpleNamespace(align_tile_embeddings=align)
    )
    return module, calls


def test_arrays_preserve_input_order_missing_and_delegate_identity_once(adapter):
    module, calls = adapter
    table = dict(tiles=[dict(tile_id="A"), dict(tile_id="B")], features=None)
    output = module.from_arrays(
        np.array([[4.0, np.nan], [1.0, 2.0]]),
        tile_ids=["B", "A"],
        tiles=table,
        feature_names=["f1", "f2"],
        source=Path("provided.npy"),
    )
    assert len(calls) == 1
    sent, ids, buffer, names = calls[0]
    assert ids == ["B", "A"]
    assert names == ["f1", "f2"]
    assert buffer == dict(
        rows=2, cols=2, values=[4.0, 0.0, 1.0, 2.0], missing=[False, True, False, False]
    )
    assert table["features"] is None
    assert sent["tiles"][0]["tile_id"] == "A"
    assert output.provenance["association"] == "external/provided"
    assert output.provenance["cache_verified"] is False
    assert output.provenance["source"] == "provided.npy"


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1 + 0j]]),
        np.array([[1 + 2j]], dtype=object),
        np.array([[np.complex128(1 + 0j)]], dtype=object),
        np.array([[np.inf]]),
        np.array([["1.2"]]),
        np.array([1.0, 2.0]),
    ],
)
def test_lossy_or_non_numeric_arrays_fail_before_native_conversion(adapter, values):
    module, calls = adapter
    with pytest.raises(ValueError):
        module.from_arrays(values, tile_ids=["A"], tiles={}, feature_names=["x"])
    assert not calls


def test_anndata_uses_explicit_authoritative_rows_without_importing_optional_packages(adapter):
    module, calls = adapter
    before = set(sys.modules)
    adata = SimpleNamespace(
        X=np.array([[2.0], [1.0]]),
        obs_names=["row-b", "row-a"],
        var_names=["fm"],
        obs={"tile_id": ["B", "A"]},
    )
    output = module.from_anndata(adata, tiles={}, tile_id_column="tile_id")
    assert calls[0][1] == ["B", "A"]
    assert calls[0][2]["values"] == [2.0, 1.0]
    assert output.provenance["original_obs_names"] == ["row-b", "row-a"]
    assert output.provenance["tile_id_column"] == "tile_id"
    assert not ({"anndata", "wsidata", "spatialdata"} & (set(sys.modules) - before))


def test_native_identity_errors_surface_without_python_fallback(adapter, monkeypatch):
    module, calls = adapter

    def fail(*args):
        raise ValueError("Alignment: embedding and tile ID sets differ.")

    monkeypatch.setattr(
        module, "resolve_backend", lambda backend: SimpleNamespace(align_tile_embeddings=fail)
    )
    with pytest.raises(ValueError, match="ID sets differ"):
        module.from_arrays([[1.0]], tile_ids=["wrong"], tiles={})
    assert not calls


def test_id_conversion_preserves_leading_zero_strings_and_rejects_float_guessing(adapter):
    module, _ = adapter
    assert module._identity_strings(["001", 1, np.int64(2)], "IDs") == ["001", "1", "2"]
    for ids in [[1.0], [None], [True]]:
        with pytest.raises(ValueError, match="identities"):
            module._identity_strings(ids, "IDs")


class Polygon:
    geom_type = "Polygon"
    is_valid = True
    is_empty = False
    has_z = False
    interiors = ()

    def __init__(self, corners):
        self.exterior = SimpleNamespace(coords=corners)


class Shapes:
    def __init__(self, rows):
        self.columns = {
            "tile_id": [row["tile_id"] for row in rows],
            "x": [row.get("x") for row in rows],
            "y": [row.get("y") for row in rows],
        }
        self.geometry = [Polygon(row["geometry_corners"]) for row in rows]
        if any("valid_pixels" in row for row in rows):
            self.columns["valid_pixels"] = [row.get("valid_pixels") for row in rows]

    def __contains__(self, key):
        return key in self.columns

    def __getitem__(self, key):
        return self.columns[key]


def fake_wsi():
    path = (
        Path(__file__).resolve().parents[1] / "golden/python_golden_v1/wsidata_contract_inputs.json"
    )
    case = json.loads(path.read_text())
    table = SimpleNamespace(
        X=np.asarray(case["embedding_values"], dtype=float),
        obs_names=case["obs_row_ids"],
        var_names=["e0", "e1"],
        obs={"tile_id": case["obs_tile_ids"], "library_id": ["tiles", "tiles"]},
        uns={
            "spatialdata_attrs": {
                "instance_key": "tile_id",
                "region_key": "library_id",
                "region": ["tiles"],
            }
        },
    )
    return (
        SimpleNamespace(
            shapes={"tiles": Shapes(case["shape_rows"])},
            attrs={"tile_spec": {"tiles": case["tile_spec"]}},
            tables={"embeddings": table},
            properties=SimpleNamespace(shape=[20, 20], mpp=0.5),
        ),
        case,
    )


def test_wsidata_reads_real_geometry_not_center_columns_and_does_not_mutate_cache(
    adapter, monkeypatch
):
    module, _ = adapter
    wsi, case = fake_wsi()
    calls = []

    def snapshot(payload):
        calls.append(json.loads(payload))
        return json.dumps({"snapshot": {"schema": "spy"}, "sha256": "spy"})

    monkeypatch.setattr(
        module, "resolve_backend", lambda backend: SimpleNamespace(wsidata_snapshot=snapshot)
    )
    before = copy.deepcopy(wsi.tables["embeddings"].uns)
    output = module.snapshot_wsidata(
        wsi, source=case["source"], shape_to_level0=case["shape_to_level0"]
    )
    assert output["sha256"] == "spy"
    assert (
        calls[0]["shape_rows"][0]["geometry_corners"] == case["shape_rows"][0]["geometry_corners"]
    )
    assert "x" not in calls[0]["shape_rows"][0]
    assert "y" not in calls[0]["shape_rows"][0]
    assert calls[0]["level0_mpp_xy"] == [0.5, 0.5]
    assert wsi.tables["embeddings"].uns == before


def test_wsidata_missing_historical_proof_never_stamps_or_falls_back(adapter, monkeypatch):
    module, _ = adapter
    wsi, case = fake_wsi()
    calls = []

    def reject(payload, buffer):
        calls.append((json.loads(payload), json.loads(buffer)))
        raise ValueError("Embedding lacks a generation-time manifest.")

    monkeypatch.setattr(
        module, "resolve_backend", lambda backend: SimpleNamespace(from_wsidata=reject)
    )
    before = copy.deepcopy(wsi.tables["embeddings"].uns)
    with pytest.raises(ValueError, match="generation-time"):
        module.from_wsidata(
            wsi,
            patient_id="P",
            slide_id="S",
            feature_key="embeddings",
            source=case["source"],
            shape_to_level0=case["shape_to_level0"],
        )
    assert len(calls) == 1
    payload, buffer = calls[0]
    assert payload["saved_at_generation"] is None
    assert payload["table_region"] == ["tiles"]
    assert payload["table_instance_key"] == "tile_id"
    assert payload["obs_tile_ids"] == case["obs_tile_ids"]
    assert buffer["values"] == list(np.asarray(case["embedding_values"], dtype=float).flat)
    assert wsi.tables["embeddings"].uns == before


def test_wsidata_manifest_string_is_only_decoded_not_regenerated(adapter, monkeypatch):
    module, _ = adapter
    wsi, case = fake_wsi()
    saved = {"snapshot": {"historical": "original"}, "sha256": "unchanged"}
    wsi.tables["embeddings"].uns["radiomics_tile_manifest"] = json.dumps(saved)
    sent = []

    def adapt(payload, buffer):
        sent.append(json.loads(payload))
        return json.dumps(
            {"table": {"features": json.loads(buffer)}, "provenance": {"cache_verified": True}}
        )

    monkeypatch.setattr(
        module, "resolve_backend", lambda backend: SimpleNamespace(from_wsidata=adapt)
    )
    result = module.from_wsidata(
        wsi,
        patient_id="P",
        slide_id="S",
        feature_key="embeddings",
        source=case["source"],
        shape_to_level0=case["shape_to_level0"],
    )
    assert sent[0]["saved_at_generation"] == saved
    assert result.provenance["cache_verified"] is True


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("geom_type", "MultiPolygon"),
        ("interiors", (object(),)),
        ("is_valid", False),
        ("has_z", True),
    ],
)
def test_wsidata_structural_geometry_errors_precede_native_builder(adapter, attribute, value):
    module, calls = adapter
    wsi, case = fake_wsi()
    setattr(wsi.shapes["tiles"].geometry[0], attribute, value)
    with pytest.raises(ValueError, match="Polygon"):
        module.snapshot_wsidata(wsi, source=case["source"], shape_to_level0=case["shape_to_level0"])
    assert not calls
