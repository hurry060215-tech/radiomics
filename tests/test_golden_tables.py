"""Independent hand checks for frozen table and selection references."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def test_table_identity_and_numeric_training_statistics():
    ref, hand = read("table_reference.json"), read("table_hand.json")
    assert ref["input_sha256"] == hashlib.sha256((DIRECTORY / "table_inputs.json").read_bytes()).hexdigest()
    numeric = ref["numeric"]
    assert numeric["medians"][0] == hand["a_observed_median"]
    assert np.asarray(numeric["imputed_training"])[:, 0].tolist() == hand["a_imputed"]
    standard = numeric["scalers"]["standard"]
    np.testing.assert_allclose(standard["state"]["mean_"][0], hand["a_population_mean"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(standard["state"]["var_"][0], hand["a_population_variance"], atol=1e-12, rtol=1e-12)
    assert numeric["scalers"]["robust"]["state"]["center_"][0] == hand["a_q50"]
    assert numeric["scalers"]["robust"]["state"]["scale_"][0] == hand["a_iqr"]
    assert numeric["scalers"]["robust"]["evaluation"][0][0] == hand["a_evaluation_robust_first"]
    np.testing.assert_allclose(numeric["scalers"]["minmax"]["evaluation"][0][0], hand["a_evaluation_minmax_first"], atol=1e-12, rtol=1e-12)
    assert numeric["scalers"]["minmax"]["evaluation"][0][0] > 1
    for scaler in numeric["scalers"].values():
        assert np.asarray(scaler["training"])[:, 4].tolist() == [0] * 6


def test_table_mode_encoding_and_univariate_hand_results():
    ref, hand = read("table_reference.json"), read("table_hand.json")
    assert ref["categorical"]["modes"] == [hand["categorical_tie_site_mode"], hand["categorical_sex_mode"]]
    assert ref["categorical"]["one_hot_training"][0] == hand["one_hot_first_row"]
    assert ref["categorical"]["unknown_category"]["status"] == "error"
    assert ref["selection"]["stable_correlation_only_names"] == hand["stable_correlation_only_names"]
    for result in ref["selection"]["univariate"].values():
        assert result["after_stable_correlation"] == hand["all_three_task_selected_names"]
        assert all(p >= .05 for p in result["bh_p_values"])
    np.testing.assert_allclose(ref["small_tests"]["bh"]["adjusted_p"], hand["bh_adjusted_p"], atol=1e-12, rtol=1e-12)
    assert ref["small_tests"]["mannwhitney_no_ties_exact"]["p_value"] == hand["mw_disjoint_two_sided_p"]
    kw = ref["selection"]["univariate"]["multiclass"]
    np.testing.assert_allclose(kw["statistic"][0], 80 / 17, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(kw["p_values"][0], math.exp(-40 / 17), atol=1e-12, rtol=1e-12)
    assert ref["small_tests"]["kruskal_all_identical"]["status"] == "error"
