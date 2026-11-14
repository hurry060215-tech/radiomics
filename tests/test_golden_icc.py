"""Independent ANOVA/hand checks for the frozen public ICC reference."""
import json
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def reference():
    return json.loads((DIRECTORY / "icc_reference.json").read_text())


def test_icc_hand_mean_squares_point_estimates_and_reader_bias():
    cases = reference()["cases"]
    base = cases["balanced"]
    np.testing.assert_allclose([base["anova"][key] for key in ("ms_subject", "ms_reader", "ms_error")], [116/9,2/9,28/45], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(base["reference"]["value"]["ICC(2,1)"]["estimate"], 184/209, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(base["reference"]["value"]["ICC(3,1)"]["estimate"], 46/53, atol=1e-12, rtol=1e-12)
    for name in ("balanced", "reader_bias", "low_agreement"):
        case = cases[name]
        x = np.asarray(case["values_subject_by_reader"])
        n,k = x.shape
        grand = np.mean(x)
        subject,reader = x.mean(axis=1),x.mean(axis=0)
        msb = k*np.sum((subject-grand)**2)/(n-1)
        msj = n*np.sum((reader-grand)**2)/(k-1)
        mse = np.sum((x-subject[:,None]-reader[None,:]+grand)**2)/((n-1)*(k-1))
        values = case["reference"]["value"]
        np.testing.assert_allclose(values["ICC(2,1)"]["estimate"], (msb-mse)/(msb+(k-1)*mse+k*(msj-mse)/n), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(values["ICC(3,1)"]["estimate"], (msb-mse)/(msb+(k-1)*mse), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cases["reader_bias"]["reference"]["value"]["ICC(3,1)"]["estimate"], base["reference"]["value"]["ICC(3,1)"]["estimate"], atol=1e-12, rtol=1e-12)
    assert cases["reader_bias"]["reference"]["value"]["ICC(2,1)"]["estimate"] < base["reference"]["value"]["ICC(2,1)"]["estimate"]
    assert cases["low_agreement"]["reference"]["value"]["ICC(2,1)"]["estimate"] < 0


def test_icc_types_missing_and_degenerate_states_are_preserved():
    ref = reference()
    assert ref["effective_parameters"]["reference_round_column_CI95"] is None
    assert ref["dependency_lock"]["scikit-learn"] == "1.2.2"
    for kind in ("ICC(2,1)", "ICC(3,1)"):
        perfect = ref["cases"]["perfect_agreement"]["reference"]["value"][kind]
        assert perfect["type"] == kind
        assert perfect["estimate"] == 1
        assert perfect["ci95"] == ["NaN", "NaN"]
        assert perfect["f_statistic"] == "Inf"
        constant = ref["cases"]["constant"]["reference"]["value"][kind]
        assert constant["estimate"] == "NaN"
        assert constant["ci95"] == ["NaN", "NaN"]
    assert ref["cases"]["missing"]["reference"]["status"] == "error"
