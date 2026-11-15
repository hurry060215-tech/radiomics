"""Independent mathematical checks on solver-level reference artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def logistic(linear):
    return np.exp(-np.logaddexp(0, -linear))


def test_model_input_identity_and_patient_groups():
    data = read("model_inputs.json")
    ref = read("model_reference.json")
    assert hashlib.sha256((DIRECTORY / "model_inputs.json").read_bytes()).hexdigest() == ref["input_sha256"]
    for task, dataset in data["datasets"].items():
        assert len(dataset["records"]) == 132
        cohorts = {}
        for record in dataset["records"]:
            assert cohorts.setdefault(record["patient_id"], record["cohort"]) == record["cohort"]
            assert (record["target"] is None) == (record["cohort"] == "external_A")
        assert len(cohorts) == 66
        assert len(ref["datasets"][task]["train_sample_ids"]) == 96


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_frozen_solver_stationarity_and_probability_formulas(task):
    dataset = read("model_inputs.json")["datasets"][task]
    ref = read("model_reference.json")["datasets"][task]
    lookup = {record["sample_id"]: record for record in dataset["records"]}
    train = [lookup[key] for key in ref["train_sample_ids"]]
    mean, scale = (np.asarray(ref["standard_scaling"][key]) for key in ("mean", "scale"))
    raw = np.asarray([row["values"] for row in train])
    np.testing.assert_allclose(mean, raw.mean(axis=0), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scale, raw.std(axis=0, ddof=0), atol=1e-12, rtol=1e-12)
    x = (raw - mean) / scale
    y = np.asarray([row["target"] for row in train])
    evaluate = (np.asarray([lookup[key]["values"] for key in ref["evaluation_sample_ids"]]) - mean) / scale
    assert np.linalg.matrix_rank(np.c_[np.ones(len(x)), x]) == 5
    for name, result in ref["models"].items():
        if "state" not in result:
            continue
        state = result["state"]
        w, b = np.asarray(state["coefficient"]), np.asarray(state["intercept"])
        penalty = result["native_penalty"]
        if name == "LR":
            logits = x @ w.T + b
            if task == "binary":
                residual = logistic(logits) - y
                gradient = x.T @ residual / len(y) + penalty * w
                objective = np.mean(np.logaddexp(0, logits) - y * logits) + penalty * np.sum(w ** 2) / 2
                p = logistic(evaluate @ w + b)
                expected_probabilities = np.c_[1 - p, p]
            else:
                # Stable softmax implemented directly, without importing sklearn.
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities = exp / exp.sum(axis=1, keepdims=True)
                residual = probabilities - np.eye(3)[y.astype(int)]
                gradient = residual.T @ x / len(y) + penalty * w
                objective = -np.log(probabilities[np.arange(len(y)), y.astype(int)]).mean() + penalty * np.sum(w ** 2) / 2
                eval_logits = evaluate @ w.T + b
                exp = np.exp(eval_logits - eval_logits.max(axis=1, keepdims=True))
                expected_probabilities = exp / exp.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(expected_probabilities, result["probabilities"], atol=1e-12, rtol=1e-12)
        else:
            residual = x @ w + b - y
            smooth = x.T @ residual / len(y)
            if name == "LASSO":
                gradient = np.where(np.abs(w) > 1e-10, smooth + penalty * np.sign(w), np.maximum(np.abs(smooth) - penalty, 0))
                objective = np.mean(residual ** 2) / 2 + penalty * np.abs(w).sum()
            else:
                gradient = smooth + penalty * w
                objective = np.mean(residual ** 2) / 2 + penalty * np.sum(w ** 2) / 2
            key = "linear_scores_not_probabilities" if task == "binary" else "predictions"
            np.testing.assert_allclose(evaluate @ w + b, result[key], atol=1e-12, rtol=1e-12)
        assert np.max(np.abs(gradient)) <= 1e-7
        assert np.max(np.abs(np.mean(residual, axis=0))) <= 1e-7
        assert abs(objective - state["objective"]) <= 1e-10 + 1e-8 * abs(state["objective"])
        assert state["kkt_inf"] <= 1e-7
        assert result["converged"]
    if task == "multiclass":
        sparse = ref["models"]["OvR_L1_Logistic"]
        union = set()
        for one in sparse["per_class"]:
            state = one["state"]
            w = np.asarray(state["coefficient"])
            labels = (y == one["positive_class"]).astype(int)
            residual = logistic(x @ w + state["intercept"]) - labels
            smooth = x.T @ residual / len(y)
            alpha = sparse["native_penalty"]
            kkt = np.where(np.abs(w) > 1e-10, smooth + alpha * np.sign(w), np.maximum(np.abs(smooth) - alpha, 0))
            assert np.max(np.abs(kkt)) <= 1e-7
            assert abs(residual.mean()) <= 1e-7
            union.update(state["support"])
        assert sorted(union) == sparse["support_union"]
