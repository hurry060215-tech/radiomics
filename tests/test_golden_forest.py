"""Check frozen forest behavioral metrics and hand-tree arithmetic independently."""

import hashlib
import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def auc(y, scores):
    positive, negative = scores[y == 1, None], scores[None, y == 0]
    return np.mean((positive > negative) + 0.5 * (positive == negative))


def macro_f1(y, predictions):
    scores = []
    for label in range(3):
        tp = np.sum((y == label) & (predictions == label))
        fp = np.sum((y != label) & (predictions == label))
        fn = np.sum((y == label) & (predictions != label))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return np.mean(scores)


def test_forest_uses_frozen_data_patient_units_and_five_fixed_seeds():
    reference = read("forest_reference.json")
    inputs = read("model_inputs.json")
    assert (
        hashlib.sha256((DIRECTORY / "model_inputs.json").read_bytes()).hexdigest()
        == reference["input_sha256"]
    )
    assert reference["model_seeds"] == [11, 23, 37, 53, 71]
    assert reference["acceptance"]["evaluation_unit"] == "patient"
    for task, result in reference["datasets"].items():
        lookup = {row["sample_id"]: row for row in inputs["datasets"][task]["records"]}
        records = [lookup[key] for key in result["test_sample_ids"]]
        assert len(records) == 24
        assert len(result["test_patient_ids"]) == 12
        assert result["parameters"]["n_estimators"] == 200
        assert result["parameters"]["bootstrap"]
        y = np.asarray(result["test_patient_targets"])
        groups = [
            [i for i, row in enumerate(records) if row["patient_id"] == patient]
            for patient in result["test_patient_ids"]
        ]
        assert all(len(group) == 2 for group in groups)
        assert [row["seed"] for row in result["per_seed"]] == reference["model_seeds"]
        for seed in result["per_seed"]:
            key = "predictions" if task == "regression" else "probabilities"
            values = np.asarray(seed[key])
            patient_values = np.asarray([values[index].mean(axis=0) for index in groups])
            np.testing.assert_array_equal(patient_values, seed["patient_" + key])
            if task == "binary":
                metrics = {
                    "auc": auc(y, patient_values[:, 1]),
                    "brier": np.mean((patient_values[:, 1] - y) ** 2),
                }
            elif task == "multiclass":
                metrics = {"macro_f1": macro_f1(y, np.argmax(patient_values, axis=1))}
            else:
                metrics = {"rmse": np.sqrt(np.mean((patient_values - y) ** 2))}
                assert np.std(y, ddof=0) == result["test_target_std_ddof0"]
            for name, value in metrics.items():
                np.testing.assert_allclose(seed["metrics"][name], value, atol=1e-12, rtol=1e-12)
        for name, value in result["mean_metrics"].items():
            assert value == np.mean([seed["metrics"][name] for seed in result["per_seed"]])


def test_hand_tree_root_impurity_and_thresholds():
    cases = read("forest_hand.json")["cases"]
    for case in cases:
        x, y = np.asarray(case["x"]), np.asarray(case["y"])
        if case.get("root_is_leaf"):
            assert np.var(y) == 0
            assert all(value == np.mean(y) for value in case["predictions"])
            continue
        feature = case["root_feature"]
        threshold = case["root_threshold"]
        left = x[:, feature] <= threshold
        if case["task"] == "regression":

            def impurity(values):
                return np.var(values, ddof=0)
        else:

            def impurity(values):
                return 1 - sum(
                    (np.sum(values == label) / len(values)) ** 2 for label in case["classes"]
                )

        root = impurity(y)
        decrease = root - (left.mean() * impurity(y[left]) + (~left).mean() * impurity(y[~left]))
        np.testing.assert_allclose(root, case["root_impurity"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(decrease, case["root_impurity_decrease"], atol=1e-12, rtol=1e-12)
