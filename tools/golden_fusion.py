"""Freeze shared-plan, patient-level probability averaging and OOF Logistic fusion."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from golden_generate_reference import ROOT
from golden_models import binary_logistic_state
from golden_workflow import fit_pipeline,predict,aggregate,score


def base_component(dataset,plan,features):
    outer=[]
    samples=[]
    patients=[]
    for fold,nested in enumerate(plan["outer_folds"]):
        part=nested["outer"]
        fit=fit_pipeline(dataset,"binary",part["training_patients"],.1,features)
        validation=[r for r in dataset["records"] if r["patient_id"] in set(part["validation_patients"])]
        raw=predict(fit,validation)
        grouped=aggregate(raw)
        for row in raw+grouped:
            row["fold"]=fold
            row["prediction_type"]="oof"
        outer.append({"fold":fold,"training_patients":part["training_patients"],"validation_patients":part["validation_patients"],"fit_state":fit["state"]})
        samples.extend(raw)
        patients.extend(grouped)
    fit=fit_pipeline(dataset,"binary",plan["cohorts"]["train"],.1,features)
    final_samples=predict(fit,dataset["records"])
    final_patients=aggregate(final_samples)
    for row in final_samples+final_patients:
        row["prediction_type"]="apparent" if row["cohort"]=="train" else "holdout" if row["cohort"]=="test" else "external"
    return {"features":features,"base_lambda":.1,"outer_fits":outer,"oof_sample_predictions":sorted(samples,key=lambda r:r["sample_id"]),
            "oof_patient_predictions":sorted(patients,key=lambda r:r["patient_id"]),"final_fit_state":fit["state"],
            "final_sample_predictions":final_samples,"final_patient_predictions":final_patients}


def main():
    if metadata.version("scikit-learn")!="1.0.2" or np.__version__!="1.23.0":
        raise SystemExit("Fusion reference requires sklearn1.0.2 / NumPy1.23.0.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; generate a fresh fusion review candidate.")
    directory=ROOT/"golden/python_golden_v1"
    dataset=json.loads((directory/"model_inputs.json").read_text())["datasets"]["binary"]
    plan_path=directory/"workflow_plans.json"
    plan=json.loads(plan_path.read_text())["binary"]
    components={"signal":base_component(dataset,plan,["signal_a","signal_b"]),
                "noise":base_component(dataset,plan,["noise_a","noise_b"])}
    names=list(components)
    lookup={name:{r["patient_id"]:r for r in component["oof_patient_predictions"]} for name,component in components.items()}
    patient_ids=plan["cohorts"]["train"]
    assert all(set(table)==set(patient_ids) for table in lookup.values())
    x=np.asarray([[lookup[name][patient]["probabilities"][1] for name in names] for patient in patient_ids])
    y=np.asarray([lookup[names[0]][patient]["y"] for patient in patient_ids])
    assert all(lookup[names[0]][patient]["y"]==lookup[names[1]][patient]["y"] for patient in patient_ids)
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model=LogisticRegression(C=1/(len(y)*.1),penalty="l2",solver="newton-cg",fit_intercept=True,multi_class="ovr",tol=1e-8,max_iter=10000,random_state=0,n_jobs=1).fit(x,y)
    numerical=binary_logistic_state(model,x,y,.1)
    assert numerical["kkt_inf"]<=1e-7
    apparent=model.predict_proba(x)
    meta_training=[{"patient_id":patient,"y":int(target),"probabilities":probability.tolist(),"prediction_type":"apparent_meta_training","base_input_source":"base_oof"} for patient,target,probability in zip(patient_ids,y,apparent)]
    final_lookup={name:{r["patient_id"]:r for r in component["final_patient_predictions"]} for name,component in components.items()}
    final_ids=sorted(final_lookup[names[0]])
    assert set(final_ids)==set(final_lookup[names[1]])
    final_x=np.asarray([[final_lookup[name][patient]["probabilities"][1] for name in names] for patient in final_ids])
    final_meta=model.predict_proba(final_x)
    final=[]
    for i,patient in enumerate(final_ids):
        source=[final_lookup[name][patient] for name in names]
        assert source[0]["y"]==source[1]["y"] and source[0]["cohort"]==source[1]["cohort"]
        weighted=np.mean([row["probabilities"] for row in source],axis=0)
        final.append({"patient_id":patient,"cohort":source[0]["cohort"],"y":source[0]["y"],
                      "component_probabilities":{name:final_lookup[name][patient]["probabilities"] for name in names},
                      "weighted_probabilities":weighted.tolist(),"logistic_probabilities":final_meta[i].tolist(),
                      "prediction_type":"apparent" if source[0]["cohort"]=="train" else "holdout" if source[0]["cohort"]=="test" else "external"})
    cohort_scores={}
    for method in ["weighted","logistic"]:
        cohort_scores[method]={cohort:score("binary",[{**r,"probabilities":r[method+"_probabilities"]} for r in final if r["cohort"]==cohort]) for cohort in ["train","test"]}
    output={"schema":"python_golden_v1.fusion.reference.v1","prediction_unit":"patient",
            "packages":{"numpy":"1.23.0","scikit-learn":"1.0.2"},
            "reference_source_revision":"scikit-learn1.0.2 / 3363ab79a71c8c0b034bd3fc787ea0be24627fed",
            "input_sha256":hashlib.sha256((directory/"model_inputs.json").read_bytes()).hexdigest(),
            "workflow_plan_file_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),"shared_plan_key":"binary",
            "configuration":{"components":names,"base_lambda":.1,"meta_lambda":.1,"meta_training_rows":48,
                             "meta_features":"Positive-class OOF probabilities after equal-mean lesion-to-patient aggregation; no extra scaling/imputation.",
                             "weights":{"signal":.5,"noise":.5},"threshold":.5,"class_order":dataset["classes"]},
            "components":components,
            "meta_fit":{"training_patient_ids":patient_ids,"feature_order":names,"x":x.tolist(),"y":y.tolist(),"parameters":model.get_params(),"numerical_state":numerical,
                        "training_source":"Patient-aligned base OOF; same real outer training/validation patient partitions from shared SplitPlan, not merely equal fold integers."},
            "meta_training_predictions":meta_training,"final_patient_predictions":final,"final_cohort_auc":cohort_scores,
            "limitations":["Meta model scores on its training base OOF matrix are apparent, not unbiased validation of the meta model.",
                           "Full-training base predictions differ from base OOF inputs; final training predictions are apparent.",
                           "Test/external labels never fit meta coefficients or choose weights/threshold; external y is absent.",
                           "Conflicting target values inside one patient are an error; no majority vote or mean label.",
                           "Patient probabilities are not local tile scores and must not be broadcast into pathology prediction maps."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
