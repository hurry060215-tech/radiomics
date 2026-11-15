"""Audit explicit patient partitions and training provenance of frozen nested workflows."""
import hashlib
import json
from pathlib import Path

import numpy as np


DIRECTORY=Path(__file__).resolve().parents[1]/"golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY/name).read_text())


def check_partition(part,population):
    a,b=set(part["training_patients"]),set(part["validation_patients"])
    assert a and b and not a&b and a|b==set(population)


def check_fit(state,patients,records):
    assert set(state["fitted_patient_ids"])==set(patients)
    used=[r for r in records if r["sample_id"] in set(state["fitted_sample_ids"])]
    assert {r["patient_id"] for r in used}==set(patients)
    assert all(r["cohort"]=="train" for r in used)
    assert len(used)==state["training_rows"]
    raw=np.asarray([r["values"] for r in used])
    np.testing.assert_allclose(np.median(raw,axis=0),state["medians"],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(np.mean(raw,axis=0),state["means"],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(np.std(raw,axis=0,ddof=0),state["scales"],atol=1e-12,rtol=1e-12)
    assert state["numerical_state"]["kkt_inf"]<=1e-7


def check_tuning(tuning,parts,records,task):
    assert [c["lambda"] for c in tuning["candidates"]]==[.05,.1]
    for candidate in tuning["candidates"]:
        for fitted,part in zip(candidate["inner_folds"],parts):
            assert fitted["training_patients"]==part["training_patients"]
            assert fitted["validation_patients"]==part["validation_patients"]
            check_fit(fitted["fit_state"],part["training_patients"],records)
            assert {p["patient_id"] for p in fitted["validation_patient_predictions"]}==set(part["validation_patients"])
            assert fitted["fit_state"]["lambda"]==candidate["lambda"]
        assert candidate["mean_score"]==np.mean([fold["score"] for fold in candidate["inner_folds"]])
    best=0
    for index,candidate in enumerate(tuning["candidates"][1:],start=1):
        delta=candidate["mean_score"]-tuning["candidates"][best]["mean_score"]
        if abs(delta)>1e-12 and (delta<0 if task=="regression" else delta>0):
            best=index
    assert tuning["selected_candidate_index"]==best
    assert tuning["selected_lambda"]==tuning["candidates"][best]["lambda"]


def test_explicit_patient_plans_and_every_nested_training_state():
    inputs=read("model_inputs.json")
    plans=read("workflow_plans.json")
    reference=read("workflow_reference.json")
    assert reference["input_sha256"]==hashlib.sha256((DIRECTORY/"model_inputs.json").read_bytes()).hexdigest()
    assert reference["plan_sha256"]==hashlib.sha256((DIRECTORY/"workflow_plans.json").read_bytes()).hexdigest()
    for task,plan in plans.items():
        records=inputs["datasets"][task]["records"]
        result=reference["datasets"][task]
        assert plan["rng_scheme"]=="explicit-patient-partitions-v1"
        assert len(plan["cohorts"]["train"])==48
        assert len(plan["cohorts"]["test"])==12
        assert len(plan["cohorts"]["external_A"])==6
        assert len(plan["sample_patients"])==132
        assert plan["sample_centers"]=={r["sample_id"]:r["center"] for r in records}
        seen=[]
        for nested,fitted in zip(plan["outer_folds"],result["outer_folds"]):
            outer=nested["outer"]
            check_partition(outer,plan["cohorts"]["train"])
            seen.extend(outer["validation_patients"])
            for inner in nested["inner"]:
                check_partition(inner,outer["training_patients"])
            check_tuning(fitted["tuning"],nested["inner"],records,task)
            check_fit(fitted["fit_state"],outer["training_patients"],records)
            assert {p["patient_id"] for p in fitted["patient_predictions"]}==set(outer["validation_patients"])
        assert len(seen)==len(set(seen))==48
        assert set(seen)==set(plan["cohorts"]["train"])
        for inner in plan["final_inner_folds"]:
            check_partition(inner,plan["cohorts"]["train"])
        check_tuning(result["final_tuning"],plan["final_inner_folds"],records,task)
        check_fit(result["final_fit_state"],plan["cohorts"]["train"],records)
        assert len(result["oof_sample_predictions"])==96
        assert len(result["oof_patient_predictions"])==48
        assert len(result["final_sample_predictions"])==132
        assert len(result["final_patient_predictions"])==66
        external=[p for p in result["final_patient_predictions"] if p["cohort"]=="external_A"]
        assert len(external)==6 and all(p["y"] is None and p["prediction_type"]=="external" for p in external)


def test_workflow_selected_grid_values_and_aggregation_are_explicit():
    reference=read("workflow_reference.json")
    expected={"binary":([.1,.05,.1],.1),"multiclass":([.05,.1,.05],.05),"regression":([.05,.05,.05],.05)}
    assert reference["configuration"]["selection_tie_atol"]==1e-12
    for task,result in reference["datasets"].items():
        assert [f["tuning"]["selected_lambda"] for f in result["outer_folds"]]==expected[task][0]
        assert result["final_tuning"]["selected_lambda"]==expected[task][1]
        assert result["mean_outer_patient_score"]==np.mean([f["score"] for f in result["outer_folds"]])
        sample={p["sample_id"]:p for p in result["final_sample_predictions"]}
        for patient in result["final_patient_predictions"]:
            rows=[sample[key] for key in patient["sample_ids"]]
            assert all(row["patient_id"]==patient["patient_id"] for row in rows)
            assert len(rows)==2
            if task=="regression":
                assert patient["value"]==np.mean([row["value"] for row in rows])
            else:
                np.testing.assert_array_equal(patient["probabilities"],np.mean([row["probabilities"] for row in rows],axis=0))
