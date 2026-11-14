"""Hand arithmetic and key/fit-scope checks independent of native algorithms."""
import hashlib
import json
from pathlib import Path

import numpy as np


DIRECTORY=Path(__file__).resolve().parents[1]/"golden/python_golden_v1"


def test_research_concat_and_delta_hand_pairs():
    inputs=json.loads((DIRECTORY/"research_inputs.json").read_text())
    reference=json.loads((DIRECTORY/"research_reference.json").read_text())
    assert hashlib.sha256((DIRECTORY/"research_inputs.json").read_bytes()).hexdigest()==reference["input_sha256"]
    assert hashlib.sha256((DIRECTORY/"research_spec.py").read_bytes()).hexdigest()==reference["specification_sha256"]
    merged=reference["concat"]
    assert [row["metadata"]["sample_id"] for row in merged["rows"]]==["CT-P2","CT-P1","CT-P1b"]
    assert merged["rows"][0]["values"]==[4,20,40,1600,2,8]
    assert merged["rows"][2]["values"]==[3,None,30,900,1.5,4.5]
    assert all(row["metadata"]["modality"] is None and row["metadata"]["region"] is None for row in merged["rows"])
    paired=reference["delta_missing_policy"]["rows"]
    assert paired[0]["key"]==["P1","L1","CT","tumor"]
    assert paired[0]["values"]==[1,1.5,.5,-4,.5,-.5]
    assert paired[1]["values"]==[1,.5,-.5,2,None,None]
    assert paired[2]["values"]==[5,None,None,None,None,None]
    assert reference["delta_missing_precedes_zero"]["rows"][0]["values"]==[None,None,None]
    assert all(value["status"]=="error" for value in reference["failures"].values())
    assert len(inputs["concat"]["sources"][0]["rows"])==3  # Two lesions belong to P1.


def test_habitat_training_scaler_centers_inertia_and_partial_result():
    reference=json.loads((DIRECTORY/"research_reference.json").read_text());case=reference["habitat"]
    assert case["scaler"]["mean"]==[4.5,4.5]
    assert case["scaler"]["variance"]==[16.25,16.25]
    np.testing.assert_allclose(case["centers_raw"],[[.5,.5],[8.5,8.5]],atol=1e-10,rtol=1e-8)
    np.testing.assert_allclose(case["inertia"],4/16.25,atol=1e-10,rtol=1e-8)
    assert case["labels"]==[0,0,0,0,1,1,1,1]
    assert case["evaluation_labels"][:3]==[0,1,1]  # Third point is deliberately away from the midpoint.
    assert reference["habitat_old_midpoint_diagnostic"]["evaluation_values"]==[[4.5,4.5]]
    assert reference["habitat_exact_tie"]["squared_distances"]==[[1.,1.]]
    assert reference["habitat_exact_tie"]["labels"]==[0]
    x=np.asarray(case["training_scaled"]);centers=np.asarray(case["centers_scaled"])
    np.testing.assert_allclose(sum(np.dot(row-centers[label],row-centers[label]) for row,label in zip(x,case["labels"])),case["inertia"],atol=1e-10,rtol=1e-8)
    for cluster in [0,1]:
        np.testing.assert_allclose(x[np.asarray(case["labels"])==cluster].mean(axis=0),centers[cluster],atol=1e-10,rtol=1e-8)
    assert reference["habitat_max_iter_reference"]["n_iter"]==1
    assert reference["habitat_max_iter_reference"]["center_fixed_point_max_abs_residual"]>1e-6
