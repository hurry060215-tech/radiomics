"""Freeze complete patient-level evaluation from public sklearn 1.0.2 references."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, auc, average_precision_score, brier_score_loss,
    confusion_matrix, log_loss, mean_absolute_error, mean_squared_error, precision_recall_curve,
    precision_recall_fscore_support, r2_score, roc_auc_score, roc_curve)

from golden_generate_reference import ROOT, clean, observed


def calibration(y, probability, strategy):
    true, predicted = calibration_curve(y, probability, n_bins=5, strategy=strategy)
    if strategy == "uniform":
        edges=np.linspace(0.,1.+1e-8,6)
    else:
        edges=np.percentile(probability,np.linspace(0.,1.,6)*100)
        edges[-1]+=1e-8
    membership = np.digitize(probability,edges,right=False)-1
    return {"strategy": strategy, "n_bins": 5, "edges": edges.tolist(), "bin_membership": membership.tolist(),
            "bin_counts": np.bincount(membership, minlength=5).tolist(), "nonempty_fraction_positive": true.tolist(), "nonempty_mean_predicted": predicted.tolist(),
            "boundary_rule": "sklearn1.0.2: uniform linspace(0,1+1e-8); quantile percentile(linspace(0,1)*100), last edge+1e-8; numpy.digitize(right=False)-1, left-closed bins. Empty bins omitted from returned means, retained in counts."}


def binary(y, probability):
    fpr, tpr, roc_thresholds = roc_curve(y, probability, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(y, probability)
    predicted = (probability >= .5).astype(int)
    p,r,f,support = precision_recall_fscore_support(y,predicted,labels=[0,1],zero_division=0)
    return {"roc_auc": float(roc_auc_score(y, probability)), "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
            "pr": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()},
            "average_precision": float(average_precision_score(y, probability)), "pr_auc_trapezoidal": float(auc(recall, precision)),
            "brier": float(brier_score_loss(y, probability)), "log_loss_eps1e15": float(log_loss(y, np.c_[1-probability,probability], labels=[0,1], eps=1e-15)),
            "threshold": .5, "confusion": confusion_matrix(y,predicted,labels=[0,1]).tolist(), "accuracy": float(accuracy_score(y,predicted)),
            "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": support.tolist()},
            "calibration": {strategy: calibration(y,probability,strategy) for strategy in ["uniform","quantile"]}}


def multiclass(y, probabilities):
    classes = [0,1,2]
    predicted = np.argmax(probabilities,axis=1)
    onehot = np.eye(3)[y]
    p,r,f,support = precision_recall_fscore_support(y,predicted,labels=classes,zero_division=0)
    per_class = [binary((y==label).astype(int), probabilities[:,label]) for label in classes]
    summary = {}
    for average in ["macro","micro","weighted"]:
        precision,recall,f1,_ = precision_recall_fscore_support(y,predicted,labels=classes,average=average,zero_division=0)
        summary[average] = {"precision":float(precision),"recall":float(recall),"f1":float(f1)}
    return {"accuracy":float(accuracy_score(y,predicted)),"confusion":confusion_matrix(y,predicted,labels=classes).tolist(),
            "per_class": {"precision":p.tolist(),"recall":r.tolist(),"f1":f.tolist(),"support":support.tolist()},
            "summary":summary, "ovr":per_class, "macro_ovr_auc":float(roc_auc_score(y,probabilities,multi_class="ovr",average="macro",labels=classes)),
            "weighted_ovr_auc":float(roc_auc_score(y,probabilities,multi_class="ovr",average="weighted",labels=classes)),
            "micro_ovr_auc":float(roc_auc_score(onehot.ravel(),probabilities.ravel())),
            "macro_average_precision":float(average_precision_score(onehot,probabilities,average="macro")),
            "weighted_average_precision":float(average_precision_score(onehot,probabilities,average="weighted")),
            "micro_average_precision":float(average_precision_score(onehot,probabilities,average="micro")),
            "log_loss_eps1e15":float(log_loss(y,probabilities,labels=classes,eps=1e-15)),
            "multiclass_brier_sum":float(np.mean(np.sum((probabilities-onehot)**2,axis=1)))}


def main():
    if metadata.version("scikit-learn") != "1.0.2" or np.__version__ != "1.23.0":
        raise SystemExit("Evaluation reference requires sklearn1.0.2 / NumPy1.23.0.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; generate a fresh evaluation candidate.")
    directory=ROOT/"golden/python_golden_v1"
    data=json.loads((directory/"model_inputs.json").read_text())
    reference=json.loads((directory/"model_reference.json").read_text())
    cases={}
    for task,dataset in data["datasets"].items():
        records=[r for r in dataset["records"] if r["cohort"]=="test"]
        patient_ids=list(dict.fromkeys(r["patient_id"] for r in records))
        model=reference["datasets"][task]
        result=model["models"]["Ridge" if task=="regression" else "LR"]
        ids=model["evaluation_sample_ids"]
        source=np.asarray(result["predictions"] if task=="regression" else result["probabilities"])
        predictions=np.asarray([source[ids.index(r["sample_id"])] for r in records])
        groups=[[i for i,r in enumerate(records) if r["patient_id"]==p] for p in patient_ids]
        patient=np.asarray([predictions[group].mean(axis=0) for group in groups])
        y=np.asarray([records[group[0]]["target"] for group in groups],dtype=float if task=="regression" else int)
        if task=="binary":
            metrics=binary(y,patient[:,1])
        elif task=="multiclass":
            metrics=multiclass(y,patient)
        else:
            metrics={"mae":float(mean_absolute_error(y,patient)),"rmse":float(mean_squared_error(y,patient,squared=False)),"r2":float(r2_score(y,patient))}
        cases[task]={"patient_ids":patient_ids,"y":y.tolist(),"predictions":patient.tolist(),"metrics":metrics,
                     "aggregation":"Equal mean of two lesion predictions per patient before any metric; source test IDs matched explicitly."}
    y=np.asarray([0,1,0,1,1,0,1,0,0,1])
    score=np.asarray([0.,.2,.2,.4,.6,.8,1.,1.,.5,.5])
    cases["binary_boundary_ties"]={"y":y.tolist(),"score":score.tolist(),"metrics":binary(y,score)}
    patient_y=np.asarray(cases["binary"]["y"])
    patient_score=np.asarray(cases["binary"]["predictions"])[:,1]
    positive,negative=np.flatnonzero(patient_y),np.flatnonzero(patient_y==0)
    rng=np.random.RandomState(42)
    indices=[np.r_[rng.choice(positive,len(positive)),rng.choice(negative,len(negative))] for _ in range(64)]
    values=np.asarray([roc_auc_score(patient_y[index],patient_score[index]) for index in indices])
    bootstrap={"unit":"patient","method":"stratified binary legacy NumPy RandomState","seed":42,"replicates":64,
               "indices":[index.tolist() for index in indices],"auc_replicates":values.tolist(),"confidence":.95,
               "ci":np.quantile(values,[.025,.975]).tolist(),"quantile_method":"linear"}
    degenerate={"r2_constant_perfect":observed(lambda:r2_score([2.,2.,2.],[2.,2.,2.])),
                "r2_constant_imperfect":observed(lambda:r2_score([2.,2.,2.],[1.,2.,3.])),
                "r2_single_observation":observed(lambda:r2_score([2.],[2.]))}
    output={"schema":"python_golden_v1.evaluation.reference.v1","packages":{"numpy":"1.23.0","scikit-learn":"1.0.2"},
            "input_sha256":hashlib.sha256((directory/"model_inputs.json").read_bytes()).hexdigest(),
            "source_model_reference_sha256":hashlib.sha256((directory/"model_reference.json").read_bytes()).hexdigest(),
            "prediction_unit":"patient","class_order":"Explicit source dataset class order; multiclass hard prediction argmax ties choose lowest class index.",
            "roc_convention":"sklearn1.0.2 roc_curve(drop_intermediate=False): initial threshold max(score)+1, not the later sklearn Inf sentinel.",
            "pr_convention":"sklearn1.0.2 precision_recall_curve reverses unique thresholds only back to the first full-recall point, then appends precision1/recall0; this can omit lower all-negative thresholds. AP step integral and trapezoidal PR AUC are separately named.",
            "binary_only":"DeLong, DCA and H-L remain binary semantics; not implicitly applied to multiclass overall.",
            "cases":cases,"patient_binary_bootstrap":bootstrap,"degenerate_regression":degenerate}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(clean(output),indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
