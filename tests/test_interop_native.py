"""Real extension integration, required whenever a compiled native backend is present."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_external_arrays_and_anndata_rows_reach_the_same_rust_alignment():
    from ov_radiomics._backend import try_native

    native = try_native()
    if native is None:
        pytest.skip("This source-only environment has no compiled extension.")
    assert hasattr(native, "align_tile_embeddings"), (
        "Rebuild the extension with the native interop entry point."
    )
    from ov_radiomics import interop

    directory = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"
    inputs = json.loads((directory / "pathology_aggregate_inputs.json").read_text())
    expected = json.loads((directory / "pathology_aggregate_reference.json").read_text())
    table = dict(
        tiles=inputs["tiles"],
        features=None,
        feature_names=[],
        slide_status={slide: "provided" for slide in inputs["slide_patients"]},
        slide_patients=inputs["slide_patients"],
    )
    values = np.asarray(inputs["embedding_values"], dtype=float)
    arrays = interop.from_arrays(
        values, tile_ids=inputs["embedding_ids"], tiles=table, feature_names=inputs["feature_names"]
    )
    adata = SimpleNamespace(
        X=values,
        obs_names=[f"row{i}" for i in range(len(values))],
        var_names=inputs["feature_names"],
        obs={"tile_id": inputs["embedding_ids"]},
    )
    adapted = interop.from_anndata(adata, tile_id_column="tile_id", tiles=table)
    assert arrays.to_dict() == adapted.to_dict()
    buffer = arrays.to_dict()["features"]
    rows = np.asarray(expected["alignment"]["aligned_values"], dtype=float)
    np.testing.assert_array_equal(np.asarray(buffer["missing"]).reshape(rows.shape), np.isnan(rows))
    np.testing.assert_array_equal(
        np.asarray(buffer["values"]).reshape(rows.shape), np.nan_to_num(rows)
    )
    assert arrays.provenance["association"] == "external/provided"
    assert arrays.provenance["cache_verified"] is False
    assert table["features"] is None


def test_wsidata_real_rust_snapshot_history_namespace_and_stale_geometry():
    from ov_radiomics._backend import try_native

    native = try_native()
    if native is None:
        pytest.skip("This source-only environment has no compiled extension.")
    assert hasattr(native, "wsidata_snapshot") and hasattr(native, "from_wsidata"), (
        "Rebuild the native WSIData contract entry points."
    )
    from ov_radiomics import interop

    directory = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"
    data = json.loads((directory / "wsidata_contract_inputs.json").read_text())
    reference = json.loads((directory / "wsidata_contract_reference.json").read_text())

    class Shapes(dict):
        def __init__(self):
            super().__init__(
                tile_id=[row["tile_id"] for row in data["shape_rows"]],
                x=[row["x"] for row in data["shape_rows"]],
                y=[row["y"] for row in data["shape_rows"]],
            )
            self.geometry = [
                SimpleNamespace(
                    geom_type="Polygon",
                    is_valid=True,
                    is_empty=False,
                    has_z=False,
                    interiors=[],
                    exterior=SimpleNamespace(coords=row["geometry_corners"]),
                )
                for row in data["shape_rows"]
            ]

    table = SimpleNamespace(
        X=np.asarray(data["embedding_values"], dtype=float),
        obs_names=data["obs_row_ids"],
        var_names=["e0", "e1"],
        obs={"tile_id": data["obs_tile_ids"], "library_id": ["tiles"] * len(data["obs_tile_ids"])},
        uns={
            "spatialdata_attrs": {
                "region": ["tiles"],
                "region_key": "library_id",
                "instance_key": "tile_id",
            }
        },
    )
    wsi = SimpleNamespace(
        shapes={"tiles": Shapes()},
        attrs={"tile_spec": {"tiles": data["tile_spec"]}},
        tables={"embedding_tiles": table},
        properties=SimpleNamespace(shape=[20, 20], mpp=0.5),
    )
    arguments = dict(source=data["source"], shape_to_level0=data["shape_to_level0"])
    saved = interop.snapshot_wsidata(wsi, **arguments)
    assert saved == reference["current_snapshot"]
    assert "radiomics_tile_manifest" not in table.uns
    with pytest.raises(ValueError, match="generation-time"):
        interop.from_wsidata(
            wsi, patient_id="P", slide_id="Slideα", feature_key="embedding_tiles", **arguments
        )
    table.uns["radiomics_tile_manifest"] = json.dumps(saved, ensure_ascii=False)
    adapted = interop.from_wsidata(
        wsi, patient_id="P", slide_id="Slideα", feature_key="embedding_tiles", **arguments
    )
    assert [tile["tile_id"] for tile in adapted.tiles] == [
        json.dumps(["Slideα", row["tile_id"]], ensure_ascii=False, separators=(",", ":"))
        for row in data["shape_rows"]
    ]
    np.testing.assert_array_equal(
        adapted.features, np.asarray(reference["aligned_feature_values"], dtype=float)
    )
    assert (
        adapted.provenance["source_row_ids_in_shape_order"]
        == reference["source_row_ids_in_shape_order"]
    )
    for geometry in wsi.shapes["tiles"].geometry:
        geometry.exterior.coords = [[x + 4, y] for x, y in geometry.exterior.coords]
    with pytest.raises(ValueError, match="snapshot differs"):
        interop.from_wsidata(
            wsi, patient_id="P", slide_id="Slideα", feature_key="embedding_tiles", **arguments
        )
