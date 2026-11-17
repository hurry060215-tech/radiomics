"""Independent verification of hand and HiGHS separation certificates."""

import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def test_separation_certificates_and_scale_invariance():
    reference = json.loads((DIRECTORY / "separation_reference.json").read_text())
    cases = {case["id"]: case for case in reference["cases"]}
    for case in cases.values():
        x = np.c_[np.ones(len(case["x"])), np.asarray(case["x"])]
        scale = np.max(np.abs(x), axis=0)
        scale[scale == 0] = 1
        scaled = x / scale
        np.testing.assert_array_equal(scaled, case["scaled_augmented_design"])
        singular = np.linalg.svd(scaled, compute_uv=False)
        rank = np.sum(singular > singular.max() * 1e-12)
        assert rank == case["scaled_augmented_rank"]
        if case["expected_diagnostic"] == "rank_deficient":
            assert rank < x.shape[1]
            assert case["reference_lp"] is None
            continue
        signed = scaled * (2 * np.asarray(case["y"]) - 1)[:, None]
        for coefficient in (
            case["hand_certificate_scaled_beta"],
            case["reference_lp"]["scaled_beta"],
        ):
            beta = np.asarray(coefficient)
            assert np.max(np.abs(beta)) <= 1
            margins = signed @ beta
            assert np.min(margins) >= 0
            assert margins.sum() == case["hand_optimal_margin_sum"]
        assert (case["hand_optimal_margin_sum"] > 0) == (case["expected_diagnostic"] == "separated")
    np.testing.assert_array_equal(
        cases["complete"]["scaled_augmented_design"],
        cases["complete_large_scale"]["scaled_augmented_design"],
    )
    np.testing.assert_array_equal(
        cases["complete"]["scaled_augmented_design"],
        cases["complete_small_scale"]["scaled_augmented_design"],
    )
    assert 0 in cases["complete"]["reference_lp"]["signed_margins"]
    assert min(cases["complete"]["hand_signed_margins"]) > 0
    assert cases["quasi"]["hand_signed_margins"] == [1, 0, 0, 1]
