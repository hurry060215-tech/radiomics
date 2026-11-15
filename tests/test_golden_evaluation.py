"""Audit legacy curve conventions, bin membership and patient bootstrap artifacts."""
import bisect
import json
from pathlib import Path

import numpy as np


DIRECTORY=Path(__file__).resolve().parents[1]/"golden/python_golden_v1"


def read():
    return json.loads((DIRECTORY/"evaluation_reference.json").read_text())


def check_binary(y,score,metrics):
    y=np.asarray(y);score=np.asarray(score)
    assert metrics["roc"]["thresholds"][0]==max(score)+1
    for threshold,fpr,tpr in zip(metrics["roc"]["thresholds"],metrics["roc"]["fpr"],metrics["roc"]["tpr"]):
        predicted=score>=threshold
        np.testing.assert_allclose(fpr,np.sum(predicted&(y==0))/np.sum(y==0),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(tpr,np.sum(predicted&(y==1))/np.sum(y==1),atol=1e-12,rtol=1e-12)
    pr=metrics["pr"]
    assert len(pr["thresholds"])+1==len(pr["precision"])==len(pr["recall"])
    for threshold,precision,recall in zip(pr["thresholds"],pr["precision"],pr["recall"]):
        predicted=score>=threshold
        np.testing.assert_allclose(precision,np.sum(predicted&(y==1))/np.sum(predicted),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(recall,np.sum(predicted&(y==1))/np.sum(y==1),atol=1e-12,rtol=1e-12)
    assert pr["precision"][-1]==1 and pr["recall"][-1]==0
    for calibration in metrics["calibration"].values():
        membership=np.asarray([bisect.bisect_right(calibration["edges"],p)-1 for p in score])
        np.testing.assert_array_equal(membership,calibration["bin_membership"])
        counts=np.bincount(membership,minlength=5)
        np.testing.assert_array_equal(counts,calibration["bin_counts"])
        nonempty=np.flatnonzero(counts)
        expected_true=[y[membership==j].mean() for j in nonempty]
        expected_pred=[score[membership==j].mean() for j in nonempty]
        np.testing.assert_allclose(expected_true,calibration["nonempty_fraction_positive"],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(expected_pred,calibration["nonempty_mean_predicted"],atol=1e-12,rtol=1e-12)


def test_all_calibration_memberships_reproduce_actual_reference_curve_means():
    reference=read()
    case=reference["cases"]["binary"]
    check_binary(case["y"],np.asarray(case["predictions"])[:,1],case["metrics"])
    case=reference["cases"]["binary_boundary_ties"]
    check_binary(case["y"],case["score"],case["metrics"])
    multi=reference["cases"]["multiclass"]
    y=np.asarray(multi["y"]);predictions=np.asarray(multi["predictions"])
    for label,metrics in enumerate(multi["metrics"]["ovr"]):
        check_binary((y==label).astype(int),predictions[:,label],metrics)


def test_multiclass_metrics_regression_and_frozen_patient_bootstrap():
    reference=read()
    multi=reference["cases"]["multiclass"]
    y=np.asarray(multi["y"]);probability=np.asarray(multi["predictions"])
    predicted=np.argmax(probability,axis=1)
    confusion=np.asarray([[np.sum((y==i)&(predicted==j)) for j in range(3)] for i in range(3)])
    np.testing.assert_array_equal(confusion,multi["metrics"]["confusion"])
    assert multi["metrics"]["summary"]["micro"]["f1"]==multi["metrics"]["accuracy"]
    case=reference["cases"]["regression"]
    y=np.asarray(case["y"]);p=np.asarray(case["predictions"])
    np.testing.assert_allclose(case["metrics"]["rmse"],np.sqrt(np.mean((y-p)**2)),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(case["metrics"]["r2"],1-np.sum((y-p)**2)/np.sum((y-y.mean())**2),atol=1e-12,rtol=1e-12)
    case=reference["cases"]["binary"]
    y=np.asarray(case["y"]);p=np.asarray(case["predictions"])[:,1]
    bootstrap=reference["patient_binary_bootstrap"]
    rng=np.random.RandomState(42)
    values=[]
    for indices,expected in zip(bootstrap["indices"],bootstrap["auc_replicates"]):
        native_order=np.r_[rng.choice(np.flatnonzero(y),np.sum(y==1)),rng.choice(np.flatnonzero(y==0),np.sum(y==0))]
        np.testing.assert_array_equal(indices,native_order)
        yy,pp=y[indices],p[indices]
        positive,negative=pp[yy==1,None],pp[None,yy==0]
        value=np.mean((positive>negative)+.5*(positive==negative))
        np.testing.assert_allclose(value,expected,atol=1e-12,rtol=1e-12)
        values.append(value)
    np.testing.assert_allclose(np.quantile(values,[.025,.975]),bootstrap["ci"],atol=1e-12,rtol=1e-12)
    assert reference["degenerate_regression"]["r2_single_observation"]["value"]=="NaN"
