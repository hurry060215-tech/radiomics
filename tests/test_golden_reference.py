"""Consume frozen, independently generated references; never regenerate them."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def load(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def assert_scalar(actual, expected, atol=1e-10, rtol=1e-8):
    if isinstance(expected, str) and expected in ("NaN", "Inf", "-Inf"):
        assert np.isnan(actual) if expected == "NaN" else actual == float(expected)
    else:
        assert np.isfinite(actual)
        assert abs(actual - expected) <= atol + rtol * abs(expected)


def test_frozen_fixture_integrity():
    manifest = load("manifest.json")
    for file, expected in manifest["files_sha256"].items():
        actual = hashlib.sha256((GOLDEN / file).read_bytes()).hexdigest()
        assert actual == expected, f"Frozen reference changed without manifest review: {file}"
    ref = load("reference.json")
    assert ref["environment"]["packages"]["pyradiomics"] == "3.0.1"
    assert ref["baseline_metrics"]["source_revision"].startswith("2512e16")


def test_hand_calculations_crosscheck_reference_301():
    hand = load("hand_calculations.json")
    ref = load("reference.json")["features"][hand["case_id"]]
    for name, value in hand["firstorder"].items():
        assert_scalar(ref["features"]["firstorder"]["value"][name], value)
    for family, expected in (("glcm", "glcm_x_symmetric"), ("glrlm", "glrlm_x")):
        raw = ref["raw_matrices"][family]["value"]
        angle = raw["angles_xyz"].index([1, 0])
        counts = np.asarray(raw["counts"])[:, :, angle]
        target = np.asarray(hand[expected])
        np.testing.assert_array_equal(counts[:, : target.shape[1]], target)
        assert not counts[:, target.shape[1] :].any()
    for family in ("glszm", "gldm"):
        counts = np.asarray(ref["raw_matrices"][family]["value"])
        target = np.asarray(hand[family])
        np.testing.assert_array_equal(counts[:, : target.shape[1]], target)
        assert not counts[:, target.shape[1] :].any()
    np.testing.assert_allclose(
        np.asarray(ref["raw_matrices"]["ngtdm"]["value"])[:, :2],
        hand["ngtdm"],
        atol=1e-10,
        rtol=1e-8,
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_current_metrics_match_frozen_2512e16(backend):
    import ov_radiomics as rad

    if backend == "rust" and rad.backend_info()["metrics_backend"] != "rust":
        pytest.skip("Native extension unavailable; Python baseline remains checked.")
    ref = load("reference.json")["baseline_metrics"]
    y, a, b = (ref["inputs"][k] for k in ("y", "a", "b"))
    for method, expected in (("delong", "binary_delong"), ("bootstrap", "binary_bootstrap")):
        actual = rad.binary_metrics(y, a, ci_method=method, backend=backend, **ref["parameters"])
        assert list(actual.index) == list(ref[expected])
        for name, value in ref[expected].items():
            assert_scalar(actual[name], value)
    actual = rad.delong_test(y, a, b, backend=backend)
    for name, value in ref["paired_delong"].items():
        assert_scalar(actual[name], value)
    dca = rad.decision_curve(y, {"a": a, "b": b}, thresholds=[0.5, 0.2, 0.5, 0.8], backend=backend)
    pd.testing.assert_frame_equal(
        dca, pd.DataFrame(ref["dca"]), check_exact=False, atol=1e-10, rtol=1e-8
    )
    hl = rad.hosmer_lemeshow(y, a, n_bins=4, backend=backend)
    for name in ("statistic", "df", "p_value"):
        assert_scalar(hl[name], ref["hosmer_lemeshow"][name])
    np.testing.assert_array_equal(
        hl["table"][["n", "observed"]],
        [[row["n"], row["observed"]] for row in ref["hosmer_lemeshow"]["table"]],
    )
    np.testing.assert_allclose(
        hl["table"].expected,
        [row["expected"] for row in ref["hosmer_lemeshow"]["table"]],
        atol=1e-10,
        rtol=1e-8,
    )


def test_bootstrap_index_sequence_frozen():
    ref = load("reference.json")["baseline_metrics"]
    y = np.asarray(ref["inputs"]["y"])
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    rng = np.random.RandomState(ref["parameters"]["random_state"])
    for expected in ref["bootstrap_indices"]:
        np.testing.assert_array_equal(
            np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))], expected
        )


def test_degenerate_states_are_explicit():
    cases = load("reference.json")["features"]
    assert all(result["status"] == "error" for result in cases["empty_roi"]["features"].values())
    assert all(v == "NaN" for v in cases["single_voxel"]["features"]["glcm"]["value"].values())
    assert all(v == "NaN" for v in cases["single_voxel"]["features"]["glrlm"]["value"].values())
    assert cases["single_voxel"]["features"]["ngtdm"]["value"]["Coarseness"] == 1e6
    for key in ("constant_count_even", "constant_count_odd"):
        np.testing.assert_array_equal(cases[key]["discretized"]["value"], [[3, 3, 3], [3, 3, 3]])
