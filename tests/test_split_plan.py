"""Native patient provenance, nested-CV isolation and serialization contracts."""

import json

import pandas as pd
import pytest

import ov_radiomics as rad
from ov_radiomics._backend import try_native


@pytest.fixture
def cohort():
    if try_native() is None:
        pytest.skip("Native SplitPlan requires a built extension.")
    return pd.DataFrame(
        [
            dict(
                ID=f"P{i}-L{j}",
                patient=f"P{i}",
                label=i % 3,
                group="train" if i < 30 else "external",
                center="A" if i < 30 else "B",
            )
            for i in range(36)
            for j in range(2)
        ]
    )


def make_plan(frame):
    return rad.SplitPlan.from_data(
        frame,
        "label",
        task="multiclass",
        patient_id="patient",
        outer_cv=3,
        inner_cv=2,
        center="center",
        seed=42,
    )


def test_repeated_patients_nested_sets_and_immutable_roundtrip(cohort):
    plan = make_plan(cohort)
    plan.validate_data(cohort, "label", patient_id="patient", center="center")
    restored = rad.SplitPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    plan.assert_shared(restored)
    assert plan.fingerprint == restored.fingerprint
    data = plan.to_dict()
    data["seed"] += 1
    assert data["seed"] != plan.to_dict()["seed"]
    for fold in range(3):
        train, valid = plan.fold_indices(cohort, fold, patient_id="patient")
        a, b = set(cohort.iloc[train].patient), set(cohort.iloc[valid].patient)
        assert not a & b
        assert a | b == set(cohort.query('group == "train"').patient)
        for inner in range(2):
            it, iv = plan.fold_indices(cohort, fold, inner=inner, patient_id="patient")
            assert set(cohort.iloc[it].patient) <= a
            assert set(cohort.iloc[iv].patient) <= a


def test_row_order_and_external_labels_cannot_change_splits(cohort):
    first = make_plan(cohort)
    changed = cohort.iloc[::-1].copy()
    changed.loc[changed.group != "train", "label"] = None
    second = make_plan(changed)
    assert first.fingerprint == second.fingerprint
    first.assert_shared(second)


def test_unknown_schema_overlap_and_different_actual_folds_fail(cohort):
    plan = make_plan(cohort)
    value = plan.to_dict()
    value["schema_version"] = 999
    with pytest.raises(ValueError, match="schema"):
        rad.SplitPlan.from_dict(value)
    value = plan.to_dict()
    value["outer_folds"][0], value["outer_folds"][1] = (
        value["outer_folds"][1],
        value["outer_folds"][0],
    )
    other = rad.SplitPlan.from_dict(value)
    with pytest.raises(ValueError, match="actual training"):
        plan.assert_shared(other)
    bad = cohort.copy()
    bad.loc[bad.index[-1], "patient"] = "P0"
    with pytest.raises(ValueError, match="multiple cohorts"):
        make_plan(bad)


def test_changed_training_sample_identity_or_class_fails(cohort):
    plan = make_plan(cohort)
    changed = cohort.copy()
    changed.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="class differs"):
        plan.validate_data(changed, "label", patient_id="patient")
    changed = cohort.copy()
    changed.loc[0, "ID"] = "wrong-sample"
    with pytest.raises(ValueError, match="membership differs"):
        plan.validate_data(changed, "label", patient_id="patient")


def test_regression_has_no_class_mapping(cohort):
    plan = rad.SplitPlan.from_data(
        cohort, task="regression", patient_id="patient", outer_cv=3, inner_cv=2
    )
    assert plan.to_dict()["classes"] == []
    assert plan.to_dict()["sample_classes"] == {}
    assert plan.to_dict()["patient_classes"] == {}
