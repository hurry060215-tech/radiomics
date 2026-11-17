"""Independent ID/partition/probability audits of frozen patient-level fusion."""

import hashlib
import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def test_fusion_components_share_actual_partitions_and_align_patient_ids():
    reference = read("fusion_reference.json")
    plan = read("workflow_plans.json")["binary"]
    assert (
        reference["workflow_plan_file_sha256"]
        == hashlib.sha256((DIRECTORY / "workflow_plans.json").read_bytes()).hexdigest()
    )
    inputs = read("model_inputs.json")["datasets"]["binary"]
    source = {r["sample_id"]: r for r in inputs["records"]}
    for name, component in reference["components"].items():
        expected_features = ["signal_a", "signal_b"] if name == "signal" else ["noise_a", "noise_b"]
        assert component["features"] == expected_features
        for outer, nested in zip(component["outer_fits"], plan["outer_folds"]):
            part = nested["outer"]
            assert outer["training_patients"] == part["training_patients"]
            assert outer["validation_patients"] == part["validation_patients"]
            assert outer["fit_state"]["fitted_patient_ids"] == part["training_patients"]
            assert not set(part["training_patients"]) & set(part["validation_patients"])
        oof = {p["patient_id"]: p for p in component["oof_patient_predictions"]}
        assert set(oof) == set(plan["cohorts"]["train"])
        assert len(component["oof_sample_predictions"]) == 96
        for patient in oof.values():
            assert patient["prediction_type"] == "oof"
            assert (
                patient["patient_id"]
                in plan["outer_folds"][patient["fold"]]["outer"]["validation_patients"]
            )
            assert all(
                source[sample]["patient_id"] == patient["patient_id"]
                for sample in patient["sample_ids"]
            )
        assert component["final_fit_state"]["fitted_patient_ids"] == plan["cohorts"]["train"]


def test_fusion_meta_uses_patient_oof_and_final_scores_are_not_broadcast():
    reference = read("fusion_reference.json")
    meta = reference["meta_fit"]
    assert reference["prediction_unit"] == "patient"
    assert reference["configuration"]["meta_training_rows"] == 48
    columns = meta["feature_order"]
    lookup = {
        name: {r["patient_id"]: r for r in reference["components"][name]["oof_patient_predictions"]}
        for name in columns
    }
    x = np.asarray(
        [
            [lookup[name][patient]["probabilities"][1] for name in columns]
            for patient in meta["training_patient_ids"]
        ]
    )
    np.testing.assert_array_equal(x, meta["x"])
    y = np.asarray(meta["y"])
    state = meta["numerical_state"]
    beta = np.asarray(state["coefficient"])
    z = x @ beta + state["intercept"]
    probability = np.exp(-np.logaddexp(0, -z))
    gradient = x.T @ (probability - y) / 48 + 0.1 * beta
    assert np.max(np.abs(gradient)) <= 1e-7
    assert abs(np.mean(probability - y)) <= 1e-7
    objective = np.mean(np.logaddexp(0, z) - y * z) + 0.1 * np.sum(beta**2) / 2
    np.testing.assert_allclose(objective, state["objective"], atol=1e-12, rtol=1e-12)
    assert all(
        p["prediction_type"] == "apparent_meta_training" and p["base_input_source"] == "base_oof"
        for p in reference["meta_training_predictions"]
    )
    rows = reference["final_patient_predictions"]
    assert len(rows) == len({r["patient_id"] for r in rows}) == 66
    for patient in rows:
        scores = patient["component_probabilities"]
        weighted = np.mean([scores[name] for name in columns], axis=0)
        np.testing.assert_array_equal(weighted, patient["weighted_probabilities"])
        linear = np.asarray([scores[name][1] for name in columns]) @ beta + state["intercept"]
        expected = np.exp(-np.logaddexp(0, -linear))
        np.testing.assert_allclose(
            patient["logistic_probabilities"], [1 - expected, expected], atol=1e-12, rtol=1e-12
        )
    external = [p for p in rows if p["cohort"] == "external_A"]
    assert len(external) == 6 and all(
        p["y"] is None and p["prediction_type"] == "external" for p in external
    )
