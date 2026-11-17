"""Independent hand and linear-algebra checks of frozen statistical references."""

import json
import math
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def reference():
    return json.loads((DIRECTORY / "statistics_reference.json").read_text())


def test_distribution_reference_endpoints_and_normal_identities():
    for item in reference()["distributions"]:
        q = item["quantiles"]
        assert q[0]["value"] == ("-Inf" if item["distribution"] in ("normal", "student_t") else 0)
        assert q[-1]["value"] == "Inf"
        for row in item["evaluations"]:
            np.testing.assert_allclose(row["cdf"] + row["sf"], 1, atol=1e-12, rtol=1e-12)
            if item["distribution"] == "normal":
                np.testing.assert_allclose(
                    row["sf"], math.erfc(row["x"] / math.sqrt(2)) / 2, atol=1e-12, rtol=1e-12
                )


def test_hypothesis_statistics_hand_formulas_and_degenerate_states():
    ref = reference()["hypothesis_tests"]
    for name in ("student_t", "welch"):
        case = ref[name]
        a, b = np.asarray(case["a"]), np.asarray(case["b"])
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        if name == "student_t":
            df = len(a) + len(b) - 2
            pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / df
            se = np.sqrt(pooled * (1 / len(a) + 1 / len(b)))
        else:
            sa, sb = va / len(a), vb / len(b)
            df = (sa + sb) ** 2 / (sa**2 / (len(a) - 1) + sb**2 / (len(b) - 1))
            se = np.sqrt(sa + sb)
        np.testing.assert_allclose(
            case["statistic"], (a.mean() - b.mean()) / se, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(case["df"], df, atol=1e-12, rtol=1e-12)
    chi = ref["chi_square_2x3"]
    np.testing.assert_allclose(
        chi["p_value"], math.exp(-chi["statistic"] / 2), atol=1e-12, rtol=1e-12
    )
    assert ref["constant_equal_t"]["value"] == ["NaN", "NaN"]
    assert ref["constant_different_welch"]["value"] == ["-Inf", 0]
    assert ref["fisher_2x2"][2]["odds_ratio"] == "Inf"
    np.testing.assert_allclose(
        ref["fisher_2x2"][2]["p_value"], 2 / math.comb(6, 3), atol=1e-12, rtol=1e-12
    )
    assert ref["fisher_2x2"][3]["odds_ratio"] == "NaN"
    assert ref["fisher_2x2"][3]["p_value"] == 1


def test_complete_case_logistic_information_covariance_and_wald_ci():
    ref = reference()["clinical_logistic"]
    lookup = {row["sample_id"]: row for row in ref["rows"]}
    expected_counts = {"a": (94, 2), "b": (94, 2), "sex": (94, 2), "multivariable": (92, 4)}
    z95 = 1.959963984540054
    for name, fit in ref["fits"].items():
        assert (fit["n_used"], fit["n_dropped"]) == expected_counts[name]
        used = [lookup[key] for key in fit["used_sample_ids"]]
        x = np.asarray(
            [
                [1.0]
                + [float(row[key] == "M") if key == "sex" else row[key] for key in fit["features"]]
                for row in used
            ]
        )
        y = np.asarray([row["target"] for row in used])
        beta = np.asarray(fit["coefficient"])
        p = np.exp(-np.logaddexp(0, -(x @ beta)))
        np.testing.assert_allclose(p, fit["apparent_probabilities"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(x.T @ (y - p), fit["score"], atol=1e-12, rtol=1e-12)
        assert np.max(np.abs(x.T @ (y - p))) <= 1e-9
        information = (x.T * (p * (1 - p))) @ x
        np.testing.assert_allclose(information, fit["information"], atol=1e-12, rtol=1e-12)
        covariance = np.asarray(fit["covariance"])
        np.testing.assert_allclose(
            information @ covariance, np.eye(len(beta)), atol=1e-12, rtol=1e-12
        )
        se = np.sqrt(np.diag(covariance))
        np.testing.assert_allclose(se, fit["standard_error"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            np.exp(np.c_[beta - z95 * se, beta + z95 * se]),
            fit["odds_ratio_ci95"],
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            [math.erfc(abs(z) / math.sqrt(2)) for z in beta / se],
            fit["p_values"],
            atol=1e-12,
            rtol=1e-12,
        )
    assert ref["complete_separation"]["error_type"] == "PerfectSeparationError"
