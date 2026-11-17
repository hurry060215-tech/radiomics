"""Validate optional installed containers, without WSI decoding or deep models.

Run only in a dedicated optional-test environment. The base mode separately
proves that array alignment/aggregation works without optional container packages.
This script never installs dependencies, mutates external library source, or
changes native data contracts. All synthetic images are already decoded.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WSIDATA_REVISION = "0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286"
OPTIONAL = {"anndata", "wsidata", "spatialdata", "geopandas", "shapely"}
DEEP_OR_VENDOR = {
    "torch",
    "torchvision",
    "cucim",
    "fastslide",
    "tiffslide",
    "openslide",
    "scyjava",
    "pyisyntax",
    "pylibCZIrw",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        return plain(value.item())
    raise TypeError("Unsupported evidence metadata type " + str(type(value)))


def metadata_fingerprint(value):
    return sha(
        json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


class RejectImports(importlib.abc.MetaPathFinder):
    def __init__(self, names, missing=False):
        self.names = names
        self.missing = missing
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            self.attempts.append(fullname)
            if self.missing:
                raise ModuleNotFoundError(
                    "Deliberately absent optional backend " + fullname, name=fullname
                )
            raise AssertionError("This metadata/container validation must not import " + fullname)


def expected_error(name, call, errors):
    try:
        call()
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        errors.append(dict(case=name, type=type(error).__name__, message=str(error)))
        return
    raise AssertionError("Expected an explicit error: " + name)


def baseline(rad):
    import numpy as np

    tile = dict(
        patient_id="P",
        slide_id="S",
        tile_id="T",
        x0=0.0,
        y0=0.0,
        width0=4.0,
        height0=4.0,
        patch_width=4,
        patch_height=4,
        valid_pixels=[0, 0, 4, 4],
        mpp_x_um=None,
        mpp_y_um=None,
        source="synthetic_predecoded",
        qc={},
        exclusion_reasons=[],
    )
    metadata = dict(
        tiles=[tile],
        features=None,
        feature_names=[],
        slide_status={"S": "provided"},
        slide_patients={"S": "P"},
    )
    table = rad.interop.from_arrays(
        np.asarray([[1.0, 2.0]]), tile_ids=["T"], tiles=metadata, feature_names=["e0", "e1"]
    )
    report = rad.pathology.aggregate(table, patient_stats=["mean"])
    if report.patient_features.features["patient_mean(slide_tile_mean(e0))"].iloc[0] != 1.0:
        raise AssertionError("Native patient aggregate differs.")
    return dict(
        tiles=len(table),
        patients=len(report.patients),
        e0_patient_mean=1.0,
        physical_scale="unknown/pixels only",
    )


def actual_anndata(rad, ad, pd, np, output):
    inputs = json.loads(
        (ROOT / "golden/python_golden_v1/pathology_aggregate_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    reference = json.loads(
        (ROOT / "golden/python_golden_v1/pathology_aggregate_reference.json").read_text(
            encoding="utf-8"
        )
    )
    tiles = dict(
        tiles=inputs["tiles"],
        features=None,
        feature_names=[],
        slide_status={s: "provided" for s in inputs["slide_patients"]},
        slide_patients=inputs["slide_patients"],
    )
    values = np.asarray(inputs["embedding_values"], dtype=float)
    table = ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {"tile_id": inputs["embedding_ids"]}, index=["row" + str(i) for i in range(len(values))]
        ),
        var=pd.DataFrame(index=inputs["feature_names"]),
    )
    table.uns["test_provenance"] = {
        "kind": "synthetic",
        "original_row_order": "explicit tile_id column",
    }
    before = metadata_fingerprint(table.uns)
    adapted = rad.interop.from_anndata(table, tiles=tiles, tile_id_column="tile_id")
    expected = np.asarray(reference["alignment"]["aligned_values"], dtype=float)
    np.testing.assert_array_equal(adapted.features, expected)
    if metadata_fingerprint(table.uns) != before:
        raise AssertionError("AnnData uns was mutated by conversion.")
    path = output / "external_embeddings.h5ad"
    table.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    restored = rad.interop.from_anndata(loaded, tiles=tiles, tile_id_column="tile_id")
    np.testing.assert_array_equal(adapted.features, restored.features)
    actual = rad.pathology.aggregate(restored)
    frozen = reference["aggregation"]["default_equal_slides"]

    def compare(a, b):
        if isinstance(b, dict):
            for key, value in b.items():
                compare(a[key], value)
        elif isinstance(b, list):
            if len(a) != len(b):
                raise AssertionError("Hierarchy length differs.")
            for av, bv in zip(a, b):
                compare(av, bv)
        elif isinstance(b, float):
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-8)
        else:
            if a != b:
                raise AssertionError("Hierarchy identity/status differs.")

    compare(actual.report, frozen)
    errors = []
    duplicate = table.copy()
    duplicate.obs.iloc[1, duplicate.obs.columns.get_loc("tile_id")] = duplicate.obs.iloc[0][
        "tile_id"
    ]
    expected_error(
        "anndata_duplicate_tile_id",
        lambda: rad.interop.from_anndata(duplicate, tiles=tiles, tile_id_column="tile_id"),
        errors,
    )
    expected_error(
        "anndata_missing_tile_id",
        lambda: rad.interop.from_anndata(table[:-1].copy(), tiles=tiles, tile_id_column="tile_id"),
        errors,
    )
    from scipy import sparse

    sparse_table = table.copy()
    sparse_table.X = sparse.csr_matrix(np.nan_to_num(values))
    expected_error(
        "anndata_sparse_requires_explicit_conversion",
        lambda: rad.interop.from_anndata(sparse_table, tiles=tiles, tile_id_column="tile_id"),
        errors,
    )
    return dict(
        class_name=type(table).__module__ + "." + type(table).__qualname__,
        shape=list(table.shape),
        shuffled_ids_preserved=True,
        h5ad_roundtrip=True,
        uns_unchanged=True,
        native_hierarchy_matches_frozen_reference=True,
        errors=errors,
    )


def actual_wsidata(rad, ad, pd, np, output):
    import geopandas as gpd
    from shapely.geometry import box
    from spatialdata.models import Image2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity
    from wsidata import TileSpec, WSIData
    from wsidata.reader import SpatialDataImage2DReader

    direct = importlib.metadata.distribution("wsidata").read_text("direct_url.json")
    provenance = {} if direct is None else json.loads(direct)
    if provenance.get("vcs_info", {}).get("commit_id") != WSIDATA_REVISION:
        raise AssertionError(
            "Installed WSIData must be the reviewed exact Git revision, not another release/local"
            " stub."
        )
    image = np.arange(3 * 20 * 24, dtype=np.uint16).astype(np.uint8).reshape(3, 20, 24)
    source = "sha256:" + sha(image.tobytes(order="C"))
    image_model = Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=["R", "G", "B"])
    spec = TileSpec(
        height=4,
        width=4,
        stride_height=4,
        stride_width=4,
        mpp=1.0,
        ops_level=0,
        ops_downsample=2.0,
        base_level=0,
        base_downsample=2.0,
    )

    def container(mpp=None):
        reader = SpatialDataImage2DReader(image_model, key="synthetic_predecoded_rgb")
        if reader.properties.mpp is not None:
            raise AssertionError("Unknown ordinary-image MPP is a guessed physical calibration.")
        # Magnification is explicitly present but cannot supply physical MPP.
        reader.properties.magnification = 20.0
        shapes = gpd.GeoDataFrame(
            {"tile_id": [0, 1, 2], "x": [4.0, 12.0, 4.0], "y": [4.0, 4.0, 12.0]},
            geometry=[box(0, 0, 8, 8), box(8, 0, 16, 8), box(0, 8, 8, 16)],
        )
        shapes = ShapesModel.parse(shapes, transformations={"global": Identity()})
        attrs = {"tile_spec": {"tiles": spec.to_dict()}}
        if mpp is not None:
            attrs["slide_properties"] = {"mpp": mpp}
        return WSIData(shapes={"tiles": shapes}, attrs=attrs, reader=reader)

    def embedding(wsi, snapshot=None):
        rows = [2, 0, 1]
        table = ad.AnnData(
            X=np.asarray([[30.0, 300.0], [10.0, 100.0], [20.0, 200.0]]),
            obs=pd.DataFrame(
                {"tile_id": rows, "library_id": pd.Categorical(["tiles"] * 3)},
                index=["row2", "row0", "row1"],
            ),
            var=pd.DataFrame(index=["e0", "e1"]),
        )
        if snapshot is not None:
            table.uns["radiomics_tile_manifest"] = json.dumps(snapshot, ensure_ascii=False)
        wsi.tables["external_features"] = TableModel.parse(
            table, region="tiles", region_key="library_id", instance_key="tile_id"
        )

    errors = []
    readerless_properties = dict(
        shape=[20, 24],
        n_level=1,
        level_shape=[[20, 24]],
        level_downsample=[1.0],
        mpp=None,
        bounds=[0, 0, 24, 20],
    )
    expected_error(
        "upstream_reader_none_despite_cached_properties",
        lambda: WSIData(attrs={"slide_properties": readerless_properties}, reader=None),
        errors,
    )
    wsi = container()
    if wsi.properties.mpp is not None:
        raise AssertionError("WSIData inferred MPP from magnification/requested tile MPP.")
    before = metadata_fingerprint(wsi.attrs)
    snapshot = rad.interop.snapshot_wsidata(wsi, source=source)
    if metadata_fingerprint(wsi.attrs) != before:
        raise AssertionError("Snapshot helper modified original WSI metadata.")
    embedding(wsi, snapshot)
    before_uns = metadata_fingerprint(wsi.tables["external_features"].uns)
    table = rad.interop.from_wsidata(
        wsi, patient_id="P", slide_id="Slideα", feature_key="external_features", source=source
    )
    if table.provenance["geometry_manifest"]["snapshot"]["slide_shape_wh"] != [24, 20]:
        raise AssertionError("Real non-square reader shape was swapped or guessed.")
    np.testing.assert_array_equal(table.features, [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    if [tile["x0"] for tile in table.tiles] != [0.0, 8.0, 0.0] or [
        tile["y0"] for tile in table.tiles
    ] != [0.0, 0.0, 8.0]:
        raise AssertionError("Adapter treated center-valued x/y columns as top-left coordinates.")
    if any(tile["mpp_x_um"] is not None or tile["mpp_y_um"] is not None for tile in table.tiles):
        raise AssertionError("Unknown physical scale was converted to a presumed MPP.")
    if metadata_fingerprint(wsi.tables["external_features"].uns) != before_uns:
        raise AssertionError("Adapter mutated historical embedding metadata.")
    native = rad.pathology.aggregate(table, patient_stats=["mean"])
    if native.patient_features.features["patient_mean(slide_tile_mean(e0))"].iloc[0] != 20.0:
        raise AssertionError("Real WSIData/native aggregation differs from hand mean.")
    expected_error(
        "unknown_mpp_physical_area_rejected",
        lambda: rad.pathology.aggregate(table, tile_weight="physical_area"),
        errors,
    )
    # Center fields are explicitly nonauthoritative; changing only them cannot move tiles.
    wsi.shapes["tiles"]["x"] = 999.0
    wsi.shapes["tiles"]["y"] = 888.0
    unchanged = rad.interop.from_wsidata(
        wsi, patient_id="P", slide_id="Slideα", feature_key="external_features", source=source
    )
    if unchanged.to_dict() != table.to_dict():
        raise AssertionError("Nonauthoritative center fields affected native geometry.")
    wsi.tables["external_features"] = wsi.tables["external_features"][[2, 0, 1]].copy()
    reordered = rad.interop.from_wsidata(
        wsi, patient_id="P", slide_id="Slideα", feature_key="external_features", source=source
    )
    np.testing.assert_array_equal(reordered.features, table.features)
    # H5AD retains the actual historical manifest string and explicit obs tile IDs.
    wsi.tables["external_features"].write_h5ad(output / "generated_wsi_embedding.h5ad")
    wsi.tables["external_features"] = ad.read_h5ad(output / "generated_wsi_embedding.h5ad")
    np.testing.assert_array_equal(
        rad.interop.from_wsidata(
            wsi, patient_id="P", slide_id="Slideα", feature_key="external_features", source=source
        ).features,
        table.features,
    )
    missing = container()
    embedding(missing)
    prior = metadata_fingerprint(missing.tables["external_features"].uns)
    expected_error(
        "missing_generation_history",
        lambda: rad.interop.from_wsidata(
            missing, patient_id="P", slide_id="M", feature_key="external_features", source=source
        ),
        errors,
    )
    if metadata_fingerprint(missing.tables["external_features"].uns) != prior:
        raise AssertionError("Missing history was retroactively stamped.")
    wrong = container()
    embedding(wrong, rad.interop.snapshot_wsidata(wrong, source=source))
    wrong.tables["external_features"].obs.loc["row2", "tile_id"] = 9
    expected_error(
        "same_count_wrong_embedding_ids",
        lambda: rad.interop.from_wsidata(
            wrong, patient_id="P", slide_id="W", feature_key="external_features", source=source
        ),
        errors,
    )
    stale = container()
    embedding(stale, rad.interop.snapshot_wsidata(stale, source=source))
    stale.shapes["tiles"].geometry = stale.shapes["tiles"].geometry.translate(xoff=1.0)
    expected_error(
        "same_count_ids_stale_geometry_history",
        lambda: rad.interop.from_wsidata(
            stale, patient_id="P", slide_id="Stale", feature_key="external_features", source=source
        ),
        errors,
    )
    calibrated = container(mpp=0.5)
    embedding(calibrated, rad.interop.snapshot_wsidata(calibrated, source=source))
    known = rad.interop.from_wsidata(
        calibrated, patient_id="P", slide_id="Known", feature_key="external_features", source=source
    )
    if any(t["mpp_x_um"] != 0.5 or t["mpp_y_um"] != 0.5 for t in known.tiles):
        raise AssertionError("Explicit recorded level-0 MPP was lost.")
    rad.pathology.aggregate(known, tile_weight="physical_area", patient_weight="physical_area")
    return dict(
        wsidata_class=type(wsi).__module__ + "." + type(wsi).__qualname__,
        geometry_class=type(wsi.shapes["tiles"]).__module__
        + "."
        + type(wsi.shapes["tiles"]).__qualname__,
        embedding_class=type(wsi.tables["external_features"]).__module__
        + "."
        + type(wsi.tables["external_features"]).__qualname__,
        reader_class=type(wsi.reader).__module__ + "." + type(wsi.reader).__qualname__,
        source_revision=WSIDATA_REVISION,
        source_install=provenance,
        reader_default_mpp=None,
        requested_tile_mpp=1.0,
        magnification_present=20.0,
        native_mpp_unknown_preserved=True,
        explicit_recorded_mpp_preserved=True,
        shape_yx=[20, 24],
        native_level0_xy=[24, 20],
        snapshot_before_embedding_generation=True,
        adapter_cache_mutation=False,
        reordered_rows_id_aligned=True,
        center_columns_not_origins=True,
        h5ad_roundtrip=True,
        native_patient_e0_mean=20.0,
        errors=errors,
        wsidata_without_reader=(
            "upstream constructor explicitly fails; actual predecoded-image reader used"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "optional"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-manifest",
        type=Path,
        help="Optional immutable installed-wheel provenance for this environment.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh optional-interop evidence output.")
    args.output.mkdir(parents=True)
    report = dict(
        schema="radiomics.optional_interop.acceptance.v1",
        mode=args.mode,
        status="running",
        python=platform.python_version(),
        environment=sys.prefix,
        wsidata_revision=WSIDATA_REVISION,
        pixel_decoding=False,
        model_training=False,
        base_dependencies_changed=False,
    )
    try:
        if args.mode == "base":
            installed = {name: importlib.util.find_spec(name) is not None for name in OPTIONAL}
            if any(installed.values()):
                raise AssertionError(
                    "Base smoke requires these optional packages to be absent: " + str(installed)
                )
            guard = RejectImports(OPTIONAL | DEEP_OR_VENDOR)
        else:
            # Upstream declares torch, but metadata/container imports are lazy at
            # the reviewed revision. Do not silently exercise deep/vendor code.
            present = {name: importlib.util.find_spec(name) is not None for name in DEEP_OR_VENDOR}
            if any(present.values()):
                raise AssertionError(
                    "This dedicated smoke requires deep/vendor backends absent: " + str(present)
                )
            guard = RejectImports(DEEP_OR_VENDOR, missing=True)
        sys.meta_path.insert(0, guard)
        import ov_radiomics as rad

        if args.build_manifest:
            build = json.loads(args.build_manifest.read_text(encoding="utf-8"))
            package = Path(rad.__file__).parent.parent
            verified = 0
            for name, expected in build["wheel_files_sha256"].items():
                if name.startswith("ov_radiomics/"):
                    if sha((package / name).read_bytes()) != expected:
                        raise AssertionError("Installed native-wheel file differs: " + name)
                    verified += 1
            report.update(
                core_implementation_sha256=build["core_implementation_sha256"],
                installed_snapshot_files_verified=verified,
                build_manifest_sha256=sha(args.build_manifest.read_bytes()),
            )
        report["baseline"] = baseline(rad)
        if args.mode == "optional":
            import anndata as ad
            import numpy as np
            import pandas as pd

            report["anndata"] = actual_anndata(rad, ad, pd, np, args.output)
            calls = []
            previous = sys.getprofile()

            def observe(frame, event, arg):
                filename = frame.f_code.co_filename.replace("\\", "/").lower()
                if (
                    event == "call"
                    and "/wsidata/reader/" in filename
                    and frame.f_code.co_name
                    in {"get_region", "get_thumbnail", "create_reader", "read_region"}
                ):
                    calls.append(dict(method=frame.f_code.co_name, source=filename))

            sys.setprofile(observe)
            try:
                report["wsidata"] = actual_wsidata(rad, ad, pd, np, args.output)
            finally:
                sys.setprofile(previous)
            report["reader_pixel_method_calls"] = calls
            if calls:
                raise AssertionError("Container/metadata adapter invoked a pixel-reader method.")
            report["packages"] = {
                name: importlib.metadata.version(name)
                for name in (
                    "ov-radiomics",
                    "numpy",
                    "pandas",
                    "scipy",
                    "anndata",
                    "spatialdata",
                    "geopandas",
                    "shapely",
                    "opencv-python-headless",
                    "wsidata",
                )
            }
        report["deep_or_vendor_modules_loaded"] = sorted(
            {name.split(".")[0] for name in sys.modules} & DEEP_OR_VENDOR
        )
        report["absent_backend_import_probes"] = sorted(set(guard.attempts))
        if report["deep_or_vendor_modules_loaded"]:
            raise AssertionError("Deep model or vendor reader was imported.")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    (args.output / "acceptance.json").write_bytes(
        (json.dumps(plain(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    print(
        json.dumps(
            {key: report.get(key) for key in ("mode", "status", "error", "environment")}, indent=2
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
