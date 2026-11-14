"""Freeze independent keyed research transformations and explicit-init Lloyd."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from golden_generate_reference import ROOT, observed


def habitat(values, initial, evaluation, max_iter, tol):
    x=np.asarray(values,dtype=float); initial=np.asarray(initial,dtype=float)
    scaler=StandardScaler().fit(x)
    z=scaler.transform(x)
    centers=scaler.transform(initial)
    # sklearn 1.0.2 calls its Lloyd algorithm 'full'; the later spelling is not accepted.
    model=KMeans(n_clusters=len(initial),init=centers,n_init=1,algorithm="full",max_iter=max_iter,tol=tol,random_state=0).fit(z)
    predicted=model.predict(scaler.transform(np.asarray(evaluation,dtype=float)))
    recomputed_centers=[]
    for cluster in range(len(initial)):
        members=z[model.labels_==cluster]
        recomputed_centers.append(None if not len(members) else members.mean(axis=0).tolist())
    center_residual=max(np.max(np.abs(np.asarray(center)-model.cluster_centers_[i])) for i,center in enumerate(recomputed_centers) if center is not None)
    return dict(scaler=dict(mean=scaler.mean_.tolist(),variance=scaler.var_.tolist(),scale=scaler.scale_.tolist()),
                training_scaled=z.tolist(),initial_centers_scaled=centers.tolist(),labels=model.labels_.tolist(),centers_scaled=model.cluster_centers_.tolist(),centers_raw=scaler.inverse_transform(model.cluster_centers_).tolist(),
                inertia=float(model.inertia_),n_iter=int(model.n_iter_),evaluation_labels=predicted.tolist(),evaluation_distances=model.transform(scaler.transform(evaluation)).tolist(),
                assigned_member_means=recomputed_centers,center_fixed_point_max_abs_residual=float(center_residual))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    if args.output.exists(): raise SystemExit("Refusing existing research candidate")
    if sklearn.__version__!="1.0.2": raise RuntimeError("Frozen sklearn 1.0.2 required")
    directory=ROOT/"golden/python_golden_v1";source=directory/"research_spec.py"
    loader=importlib.util.spec_from_file_location("independent_research",source);spec=importlib.util.module_from_spec(loader);loader.loader.exec_module(spec)
    inputs=json.loads((directory/"research_inputs.json").read_text())
    config=inputs["concat"];delta=inputs["delta"];h=inputs["habitat"]
    merged=spec.concat(**config)
    missing=copy.deepcopy(config);missing["sources"][1]["rows"].pop()
    duplicate=copy.deepcopy(config);duplicate["sources"][1]["rows"].append(dict(duplicate["sources"][1]["rows"][0],sample_id="different-sample-same-key"))
    conflict=copy.deepcopy(config);conflict["sources"][1]["rows"][0]["center"]="B"
    prefix=copy.deepcopy(config);prefix["sources"][1]["prefix"]=prefix["sources"][0]["prefix"]
    missing_key=copy.deepcopy(config);missing_key["sources"][1]["rows"][0]["lesion_id"]=None
    delta_unpaired=copy.deepcopy(delta);delta_unpaired["rows"].pop()
    delta_duplicate=copy.deepcopy(delta);delta_duplicate["rows"].append(dict(delta["rows"][0],sample_id="other-followup"))
    delta_extra=copy.deepcopy(delta);delta_extra["rows"].append(dict(delta["rows"][0],sample_id="T2",timepoint="T2"))
    delta_overflow=dict(rows=[dict(delta["rows"][0],sample_id="overflow-before",timepoint="T0",values={"Mean":-1e308}),dict(delta["rows"][0],sample_id="overflow-after",timepoint="T1",values={"Mean":1e308})],feature_names=["Mean"],baseline="T0",followup="T1")
    failures=dict(concat_missing_key_set=observed(lambda:spec.concat(**missing)),concat_duplicate_key=observed(lambda:spec.concat(**duplicate)),concat_metadata_conflict=observed(lambda:spec.concat(**conflict)),concat_duplicate_prefix=observed(lambda:spec.concat(**prefix)),concat_absent_key_field=observed(lambda:spec.concat(**missing_key)),concat_registration_undeclared=observed(lambda:spec.concat(**dict(config,already_registered=False))),delta_zero_default=observed(lambda:spec.delta(**delta)),delta_unpaired=observed(lambda:spec.delta(**delta_unpaired)),delta_duplicate_pair=observed(lambda:spec.delta(**delta_duplicate)),delta_extra_timepoint=observed(lambda:spec.delta(**delta_extra)),delta_overflow=observed(lambda:spec.delta(**delta_overflow)))
    assert all(case["status"]=="error" for case in failures.values())
    missing_before_zero=dict(rows=[dict(delta["rows"][0],sample_id="zero",timepoint="T0",values={"Mean":0}),dict(delta["rows"][0],sample_id="missing",timepoint="T1",values={"Mean":None})],feature_names=["Mean"],baseline="T0",followup="T1")
    normal=habitat(h["training_values"],h["initial_centers_raw"],h["evaluation_values"],h["max_iter"],h["tol"])
    diagnostic=habitat(h["training_values"],h["initial_centers_raw"],h["old_midpoint_diagnostic"]["evaluation_values"],h["max_iter"],h["tol"])
    diagnostic=dict(evaluation_values=h["old_midpoint_diagnostic"]["evaluation_values"],evaluation_labels=diagnostic["evaluation_labels"],evaluation_distances=diagnostic["evaluation_distances"],applicability=h["old_midpoint_diagnostic"]["applicability"])
    exact=h["exact_tie"]
    squared=[[sum((value-center)**2 for value,center in zip(row,centroid)) for centroid in exact["centers_scaled"]] for row in exact["evaluation_scaled"]]
    exact_reference=dict(centers_scaled=exact["centers_scaled"],evaluation_scaled=exact["evaluation_scaled"],squared_distances=squared,labels=[min(range(len(distances)),key=distances.__getitem__) for distances in squared],applicability=exact["applicability"])
    early=h["nonconvergence"];early_reference=habitat(early["training_values"],early["initial_centers_raw"],early["training_values"],early["max_iter"],h["tol"])
    assert early_reference["center_fixed_point_max_abs_residual"]>1e-6
    output=dict(schema="python_golden_v1.research.reference.v1",specification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((directory/"research_inputs.json").read_bytes()).hexdigest(),reference=dict(sklearn=sklearn.__version__,numpy=np.__version__,algorithm="Lloyd (sklearn1.0.2 spelling full)",n_init=1,tol=h["tol"],seed="unused for explicit initial centers"),concat=merged,delta_missing_policy=spec.delta(**dict(delta,zero_denominator="missing")),delta_missing_precedes_zero=spec.delta(**missing_before_zero),failures=failures,habitat=normal,habitat_max_iter_reference=early_reference,
                conventions=dict(concat="Exact explicit keys; first-source row order, source order then source feature order. Only declared modality/region differences allowed, output mixed fields missing and per-source values retained. Caller must declare required registration already completed; no image registration performed.",delta="Sorted(patient,lesion,modality,region) pair order; feature order explicit; difference=followup-baseline,ratio=followup/baseline,relative=(followup-baseline)/baseline without percent. Missing precedes zero check; difference remains observed under explicit zero-as-missing; unexpected timepoint rejected before pairing.",habitat="Train-only Standard(ddof0; zero variance scale1); raw explicit centers transformed with training state. Euclidean squared nearest-center assignment; exact ties lower cluster index. Cluster numbers have no cross-patient tissue meaning. Evaluation IDs/features never fit scaler or centers.",native_failure_rules={"max_iter":"Non-fixed-point output at iteration limit is explicit NonConvergence, never saved as a successful habitat model; sklearn itself returns this partial reference.","empty_cluster":"An empty assigned cluster is an explicit native error; no hidden sklearn relocation. Duplicate explicit centers create this failure on the fixed training matrix.","default_plus_plus":"New native kmeans++ uses the project recorded ChaCha8 component RNG scheme; same seed need not yield sklearn initial centers. Require native repeat/thread/save-load identity and compare simple-data inertia to the explicit fixed reference, not equal RNG sequences."},tolerances=dict(labels="exact for these fixed explicit centers",scaler_centers_inertia=dict(atol=1e-10,rtol=1e-8),prediction_state="evaluation feature/label changes cannot alter fit")))
    output["conventions"]["initial_center_space"]="Explicit initial_centers_raw are original feature units, converted using only the fitted training Standard state; they are not already scaled."
    output["habitat_old_midpoint_diagnostic"]=diagnostic
    output["habitat_exact_tie"]=exact_reference
    output["conventions"]["reference_convergence"]="sklearn1.0.2 uses tol*mean(var(training_scaled,axis=0,ddof=0)) as an absolute threshold for sum of squared center shifts, or strict unchanged assignments. This is not a per-element scalar tolerance. The normal fixture has unit variance in both columns, hence the threshold is 1e-12."
    output["conventions"]["native_convergence"]="A successful native Lloyd fit requires stable nearest-center assignments (ties lowest cluster index), no empty cluster and centers equal to the assigned member means within declared scalar tolerance; report iterations and convergence. At max_iter, a non-fixed-point result errors, including the frozen partial reference. Initial repeated centers are rejected as empty-cluster input/fit, not silently relocated."
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode());print(args.output)


if __name__=="__main__":main()
