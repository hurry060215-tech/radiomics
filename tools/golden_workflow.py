"""Independent explicit-patient nested workflow reference, without native calls."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from golden_generate_reference import ROOT
from golden_models import binary_logistic_state, regression_state, softmax_state


TIE_ATOL=1e-12
GRID=[.05,.1]


def patient_index(patient):
    return int(patient.split("-P")[1])


def partition(population,validation):
    validation=set(validation)
    return {"training_patients":sorted(set(population)-validation),"validation_patients":sorted(validation)}


def explicit_plan(task,dataset):
    rows=dataset["records"]
    training=sorted({r["patient_id"] for r in rows if r["cohort"]=="train"})
    cohorts={name:sorted({r["patient_id"] for r in rows if r["cohort"]==name}) for name in sorted({r["cohort"] for r in rows})}
    outer=[]
    for fold in range(3):
        part=partition(training,[p for p in training if (patient_index(p)//3)%3==fold])
        inner=[partition(part["training_patients"],[p for p in part["training_patients"] if (patient_index(p)//9)%2==index]) for index in range(2)]
        outer.append({"outer":part,"inner":inner})
    final=[partition(training,[p for p in training if (patient_index(p)//3)%2==index]) for index in range(2)]
    labels={r["sample_id"]:dataset["classes"][r["target"]] for r in rows if task!="regression" and r["cohort"]=="train"}
    patient_labels={p:sorted({dataset["classes"][r["target"]] for r in rows if r["patient_id"]==p}) for p in training} if task!="regression" else {}
    return {"schema_version":1,"task":task,"seed":42,"sample_patients":{r["sample_id"]:r["patient_id"] for r in rows},
            "sample_centers":{r["sample_id"]:r.get("center") for r in rows},
            "sample_classes":labels,"cohorts":cohorts,"classes":dataset["classes"],"patient_classes":patient_labels,
            "outer_folds":outer,"final_inner_folds":final,"rng_scheme":"explicit-patient-partitions-v1"}


def fit_pipeline(dataset,task,patients,penalty,feature_names=None):
    names=dataset["feature_names"] if feature_names is None else feature_names
    columns=[dataset["feature_names"].index(name) for name in names]
    rows=[r for r in dataset["records"] if r["patient_id"] in set(patients)]
    assert {r["patient_id"] for r in rows}==set(patients)
    raw=np.asarray([[r["values"][j] for j in columns] for r in rows],dtype=float)
    y=np.asarray([r["target"] for r in rows],dtype=float if task=="regression" else int)
    if task!="regression" and set(y)!=set(range(len(dataset["classes"]))):
        raise ValueError("A reference training fold is missing a class.")
    if np.isnan(raw).all(axis=0).any():
        raise ValueError("An all-missing training feature cannot be silently dropped.")
    imputer=SimpleImputer(strategy="median").fit(raw)
    imputed=imputer.transform(raw)
    scaler=StandardScaler().fit(imputed)
    z=scaler.transform(imputed)
    variance=imputed.var(axis=0,ddof=0)
    candidates=np.flatnonzero(variance>0)
    correlation=np.corrcoef(z[:,candidates],rowvar=False)
    if len(candidates)==1:
        correlation=np.ones((1,1))
    retained=[]
    for index,j in enumerate(candidates):
        if not any(abs(correlation[index,previous])>.9 for previous,_ in retained):
            retained.append((index,j))
    selected=[int(j) for _,j in retained]
    if not selected:
        raise ValueError("Reference selection retained zero features.")
    design=z[:,selected]
    if task=="regression":
        model=Ridge(alpha=len(y)*penalty,fit_intercept=True,solver="svd").fit(design,y)
        numerical=regression_state(model,design,y,penalty)
    else:
        model=LogisticRegression(C=1/(len(y)*penalty),penalty="l2",solver="newton-cg",fit_intercept=True,multi_class="multinomial" if task=="multiclass" else "ovr",tol=1e-7,max_iter=10000,random_state=0,n_jobs=1).fit(design,y)
        numerical=softmax_state(model,design,y,penalty) if task=="multiclass" else binary_logistic_state(model,design,y,penalty)
    if numerical["kkt_inf"]>1e-7:
        raise RuntimeError("Reference workflow model failed stationary-point tolerance.")
    state={"fitted_patient_ids":sorted(patients),"fitted_sample_ids":[r["sample_id"] for r in rows],"training_rows":len(rows),
           "feature_names":list(names),"medians":imputer.statistics_.tolist(),"means":scaler.mean_.tolist(),"scales":scaler.scale_.tolist(),
           "raw_population_variances":variance.tolist(),"selected_indices":selected,"selected_names":[names[j] for j in selected],
           "lambda":penalty,"model_parameters":model.get_params(),"numerical_state":numerical}
    return {"imputer":imputer,"scaler":scaler,"model":model,"columns":columns,"selected":selected,"state":state,"task":task}


def predict(pipeline,rows):
    raw=np.asarray([[r["values"][j] for j in pipeline["columns"]] for r in rows],dtype=float)
    z=pipeline["scaler"].transform(pipeline["imputer"].transform(raw))[:,pipeline["selected"]]
    values=pipeline["model"].predict(z) if pipeline["task"]=="regression" else pipeline["model"].predict_proba(z)
    sample=[]
    for row,value in zip(rows,values):
        sample.append({"sample_id":row["sample_id"],"patient_id":row["patient_id"],"cohort":row["cohort"],"y":row["target"],
                       "value":float(value) if pipeline["task"]=="regression" else None,
                       "probabilities":[] if pipeline["task"]=="regression" else value.tolist()})
    return sample


def aggregate(sample):
    patient=[]
    for identifier in sorted({r["patient_id"] for r in sample}):
        rows=[r for r in sample if r["patient_id"]==identifier]
        assert len({r["y"] for r in rows})==1
        assert len({r["cohort"] for r in rows})==1
        if rows[0]["probabilities"]:
            probability=np.mean([r["probabilities"] for r in rows],axis=0).tolist()
            value=None
        else:
            probability=[]
            value=float(np.mean([r["value"] for r in rows]))
        patient.append({"patient_id":identifier,"sample_ids":[r["sample_id"] for r in rows],"cohort":rows[0]["cohort"],"y":rows[0]["y"],"probabilities":probability,"value":value})
    return patient


def score(task,patient):
    y=np.asarray([r["y"] for r in patient],dtype=float if task=="regression" else int)
    if task=="regression":
        return float(mean_squared_error(y,[r["value"] for r in patient],squared=False))
    probability=np.asarray([r["probabilities"] for r in patient])
    return float(roc_auc_score(y,probability[:,1])) if task=="binary" else float(roc_auc_score(y,probability,multi_class="ovr",average="macro",labels=list(range(probability.shape[1]))))


def tune(dataset,task,folds):
    candidates=[]
    for penalty in GRID:
        per_fold=[]
        for index,part in enumerate(folds):
            fit=fit_pipeline(dataset,task,part["training_patients"],penalty)
            validation=[r for r in dataset["records"] if r["patient_id"] in set(part["validation_patients"])]
            predicted=aggregate(predict(fit,validation))
            per_fold.append({"fold":index,"training_patients":part["training_patients"],"validation_patients":part["validation_patients"],
                             "fit_state":fit["state"],"validation_patient_predictions":predicted,"score":score(task,predicted)})
        candidates.append({"lambda":penalty,"inner_folds":per_fold,"mean_score":float(np.mean([r["score"] for r in per_fold]))})
    best=0
    for index,candidate in enumerate(candidates[1:],start=1):
        delta=candidate["mean_score"]-candidates[best]["mean_score"]
        if abs(delta)>TIE_ATOL and ((delta<0) if task=="regression" else (delta>0)):
            best=index
    return {"candidates":candidates,"selected_candidate_index":best,"selected_lambda":candidates[best]["lambda"],"selection_tie_atol":TIE_ATOL}


def workflow(dataset,task,plan):
    outer=[]
    oof_sample=[]
    for index,nested in enumerate(plan["outer_folds"]):
        tuning=tune(dataset,task,nested["inner"])
        fit=fit_pipeline(dataset,task,nested["outer"]["training_patients"],tuning["selected_lambda"])
        validation=[r for r in dataset["records"] if r["patient_id"] in set(nested["outer"]["validation_patients"])]
        samples=predict(fit,validation)
        patient=aggregate(samples)
        for row in samples:
            row["fold"]=index
            row["prediction_type"]="oof"
        for row in patient:
            row["fold"]=index
            row["prediction_type"]="oof"
        outer.append({"fold":index,"tuning":tuning,"fit_state":fit["state"],"patient_predictions":patient,"score":score(task,patient)})
        oof_sample.extend(samples)
    final_tuning=tune(dataset,task,plan["final_inner_folds"])
    final=fit_pipeline(dataset,task,plan["cohorts"]["train"],final_tuning["selected_lambda"])
    sample=predict(final,dataset["records"])
    for row in sample:
        row["prediction_type"]="apparent" if row["cohort"]=="train" else "external" if row["y"] is None else "holdout"
    patient=aggregate(sample)
    for row in patient:
        row["prediction_type"]="apparent" if row["cohort"]=="train" else "external" if row["y"] is None else "holdout"
    return {"outer_folds":outer,"mean_outer_patient_score":float(np.mean([r["score"] for r in outer])),
            "oof_sample_predictions":sorted(oof_sample,key=lambda r:r["sample_id"]),"oof_patient_predictions":sorted([r for fold in outer for r in fold["patient_predictions"]],key=lambda r:r["patient_id"]),
            "final_tuning":final_tuning,"final_fit_state":final["state"],"final_sample_predictions":sample,"final_patient_predictions":patient,
            "final_cohort_scores":{cohort:score(task,[r for r in patient if r["cohort"]==cohort]) for cohort in ["train","test"]}}


def main():
    if metadata.version("scikit-learn")!="1.0.2" or np.__version__!="1.23.0":
        raise SystemExit("Workflow reference requires sklearn1.0.2 / NumPy1.23.0.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; produce a fresh workflow candidate.")
    source=ROOT/"golden/python_golden_v1/model_inputs.json"
    inputs=json.loads(source.read_text())
    plans={task:explicit_plan(task,data) for task,data in inputs["datasets"].items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        results={task:workflow(data,task,plans[task]) for task,data in inputs["datasets"].items()}
    plan_bytes=(json.dumps(plans,indent=2,allow_nan=False)+"\n").encode()
    output={"schema":"python_golden_v1.workflow.reference.v1","packages":{"numpy":"1.23.0","scikit-learn":"1.0.2"},
            "input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"plan_sha256":hashlib.sha256(plan_bytes).hexdigest(),
            "configuration":{"scaling":"standard_population","imputation":"median_train_only","univariate":None,"variance_threshold":0.,"absolute_correlation_threshold":.9,
                             "candidate_lambda_order":GRID,"selection_tie_atol":TIE_ATOL,"outer_folds":3,"inner_folds":2,"binary_threshold":.5,
                             "CV_score_unit":"patient","CV_fold_aggregation":"Equal mean of fold-level patient metrics, not pooled OOF or patient-count weighted.",
                             "scoring":{"binary":"maximize ROC AUC","multiclass":"maximize macro OvR ROC AUC","regression":"minimize RMSE"}},
            "split_formula":{"outer":"(patient_index//3)%3","inner_inside_outer":"(patient_index//9)%2","final_inner":"(patient_index//3)%2","scope":"Original48 training patients only; test/external identities and outcomes never choose candidate or fit state."},
            "objectives":"Identical mean-loss objectives/unpenalized intercepts as model_reference.json; LR C=1/(n_rows*lambda), Ridge sklearnalpha=n_rows*lambda.",
            "prediction_provenance":"OOF fit excludes that patient's entire outer fold; train final predictions apparent; test holdout; unlabeled external prediction-only. Patient scores equal-mean lesion predictions.",
            "datasets":results}
    args.output.mkdir(parents=True)
    (args.output/"workflow_plans.json").write_bytes(plan_bytes)
    (args.output/"workflow_reference.json").write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
