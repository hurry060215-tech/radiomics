use super::*;
use serde_json::Value;
fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "actual {actual:.16e}, expected {expected:.16e}"
    );
}
fn fixture() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/model_inputs.json"
        )))
        .unwrap(),
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/model_reference.json"
        )))
        .unwrap(),
    )
}
fn data(task: &str) -> (NumericBuffer, ModelTarget, NumericBuffer, Value) {
    let (inputs, reference) = fixture();
    let dataset = &inputs["datasets"][task];
    let reference = reference["datasets"][task].clone();
    let records = dataset["records"].as_array().unwrap();
    let means: Vec<_> = reference["standard_scaling"]["mean"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let scales: Vec<_> = reference["standard_scaling"]["scale"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let training: Vec<_> = records.iter().filter(|r| r["cohort"] == "train").collect();
    let transform = |rows: &[&Value]| {
        buffer(
            rows.len(),
            means.len(),
            rows.iter()
                .flat_map(|row| {
                    row["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(j, v)| (v.as_f64().unwrap() - means[j]) / scales[j])
                })
                .collect(),
        )
    };
    let eval: Vec<_> = reference["evaluation_sample_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| records.iter().find(|r| r["sample_id"] == *id).unwrap())
        .collect();
    let target = if task == "regression" {
        ModelTarget::Regression {
            values: training
                .iter()
                .map(|r| r["target"].as_f64().unwrap())
                .collect(),
        }
    } else {
        ModelTarget::Classification {
            classes: dataset["classes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
            labels: training
                .iter()
                .map(|r| r["target"].as_u64().unwrap() as usize)
                .collect(),
        }
    };
    (transform(&training), target, transform(&eval), reference)
}
#[test]
fn five_model_families_match_frozen_predictions_objectives_and_gradients() {
    for task in ["binary", "multiclass", "regression"] {
        let (x, y, test, reference) = data(task);
        let mut specifications = vec![
            (
                "KNN_uniform",
                ModelSpec::KNN {
                    k: 5,
                    weights: KnnWeights::Uniform,
                },
            ),
            (
                "KNN_distance",
                ModelSpec::KNN {
                    k: 5,
                    weights: KnnWeights::Distance,
                },
            ),
        ];
        if task == "regression" {
            specifications.extend([
                (
                    "Linear",
                    ModelSpec::Linear {
                        rank_tolerance: 1e-12,
                    },
                ),
                (
                    "Ridge",
                    ModelSpec::Ridge {
                        lambda: 0.1,
                        rank_tolerance: 1e-12,
                    },
                ),
            ]);
        } else {
            specifications.extend([
                (
                    "LR",
                    ModelSpec::LR {
                        lambda: 0.1,
                        tolerance: 1e-9,
                        max_iterations: 5000,
                    },
                ),
                (
                    "GaussianNB",
                    ModelSpec::GaussianNB {
                        priors: None,
                        var_smoothing: 1e-9,
                    },
                ),
            ]);
        }
        for (name, spec) in specifications {
            let model = fit(&spec, &x, &y).unwrap_or_else(|e| panic!("{task}/{name}: {e}"));
            let prediction = model.predict(&test).unwrap();
            let r = &reference["models"][name];
            let (actual, expected): (Vec<f64>, Vec<f64>) = match prediction {
                ModelPrediction::Classification { probabilities, .. } => (
                    probabilities.values,
                    r["probabilities"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                        .collect(),
                ),
                ModelPrediction::Regression { values } => (
                    values,
                    r["predictions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect(),
                ),
            };
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                close(*a, b, 1e-6, 1e-5);
            }
            if let Some(objective) = model.report.objective {
                close(
                    objective,
                    r["state"]["objective"].as_f64().unwrap(),
                    1e-10,
                    1e-8,
                );
                assert!(model.report.gradient_or_kkt_inf.unwrap() <= 1e-7);
            }
            let restored: FittedModel =
                serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
            restored.validate().unwrap();
            assert_eq!(
                model.predict(&test).unwrap(),
                restored.predict(&test).unwrap()
            );
        }
    }
}
#[test]
fn sparse_solvers_match_frozen_objectives_kkt_and_support() {
    for task in ["binary", "multiclass", "regression"] {
        let (x, y, test, reference) = data(task);
        let method = if task == "multiclass" {
            SparseMethod::OvrL1Logistic
        } else {
            SparseMethod::SquaredLasso
        };
        let alpha = if task == "regression" { 0.05 } else { 0.03 };
        let fitted = fit_sparse(
            &SparseSpec {
                method,
                alpha,
                tolerance: 1e-9,
                max_iterations: 10000,
            },
            &x,
            &y,
        )
        .unwrap_or_else(|e| panic!("{task}: {e}"));
        let r = &reference["models"][if task == "multiclass" {
            "OvR_L1_Logistic"
        } else {
            "LASSO"
        }];
        let states: Vec<_> = if task == "multiclass" {
            r["per_class"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| &r["state"])
                .collect()
        } else {
            vec![&r["state"]]
        };
        for (report, state) in fitted.per_class_reports.iter().zip(states) {
            close(
                report.objective.unwrap(),
                state["objective"].as_f64().unwrap(),
                1e-10,
                1e-8,
            );
            assert!(report.gradient_or_kkt_inf.unwrap() <= 1e-7);
        }
        if task == "multiclass" {
            let expected: Vec<_> = r["support_union"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(fitted.support, expected);
        } else {
            let expected: Vec<_> = r["state"]["support"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(fitted.support, expected);
            let scores = fitted.linear_scores(&test).unwrap();
            let expected = r[if task == "binary" {
                "linear_scores_not_probabilities"
            } else {
                "predictions"
            }]
            .as_array()
            .unwrap();
            for (a, b) in scores.values.iter().zip(expected) {
                close(*a, b.as_f64().unwrap(), 1e-6, 1e-5);
            }
        }
        let restored: SparseFit =
            serde_json::from_slice(&serde_json::to_vec(&fitted).unwrap()).unwrap();
        assert_eq!(
            fitted.linear_scores(&test).unwrap(),
            restored.linear_scores(&test).unwrap()
        );
    }
}
#[test]
fn invalid_input_rank_nonconvergence_and_state_errors_are_explicit() {
    let (mut x, y, _, _) = data("regression");
    for row in x.values.chunks_mut(x.cols) {
        row[3] = row[0];
    }
    assert!(fit(
        &ModelSpec::Linear {
            rank_tolerance: 1e-12
        },
        &x,
        &y
    )
    .unwrap_err()
    .message
    .contains("rank deficient"));
    assert!(fit(
        &ModelSpec::Ridge {
            lambda: 0.1,
            rank_tolerance: 1e-12
        },
        &x,
        &y
    )
    .is_ok());
    let (x, y, _, _) = data("multiclass");
    assert!(fit_sparse(
        &SparseSpec {
            method: SparseMethod::SquaredLasso,
            alpha: 0.03,
            tolerance: 1e-9,
            max_iterations: 100
        },
        &x,
        &y
    )
    .is_err());
    assert_eq!(
        fit(
            &ModelSpec::LR {
                lambda: 0.1,
                tolerance: 1e-15,
                max_iterations: 1
            },
            &x,
            &y
        )
        .unwrap_err()
        .code,
        ErrorCode::Convergence
    );
    let mut missing = x.clone();
    missing.missing[0] = true;
    missing.values[0] = 0.0;
    assert!(fit(
        &ModelSpec::KNN {
            k: 2,
            weights: KnnWeights::Uniform
        },
        &missing,
        &y
    )
    .is_err());
    let ModelTarget::Classification {
        classes,
        mut labels,
    } = y
    else {
        unreachable!()
    };
    labels.iter_mut().for_each(|c| {
        if *c == 2 {
            *c = 1;
        }
    });
    assert!(fit(
        &ModelSpec::KNN {
            k: 2,
            weights: KnnWeights::Uniform
        },
        &x,
        &ModelTarget::Classification { classes, labels }
    )
    .is_err());
}
#[test]
fn softmax_and_ovr_support_ignore_class_names_and_index_permutation() {
    let (x, y, test, _) = data("multiclass");
    let ModelTarget::Classification { classes: _, labels } = y.clone() else {
        unreachable!()
    };
    let changed = ModelTarget::Classification {
        classes: vec!["z".into(), "x".into(), "y".into()],
        labels: labels.iter().map(|l| (l + 1) % 3).collect(),
    };
    let spec = ModelSpec::LR {
        lambda: 0.1,
        tolerance: 1e-9,
        max_iterations: 5000,
    };
    let original = fit(&spec, &x, &y).unwrap().predict(&test).unwrap();
    let permuted = fit(&spec, &x, &changed).unwrap().predict(&test).unwrap();
    let (
        ModelPrediction::Classification {
            probabilities: a, ..
        },
        ModelPrediction::Classification {
            probabilities: b, ..
        },
    ) = (original, permuted)
    else {
        unreachable!()
    };
    for row in 0..test.rows {
        for c in 0..3 {
            close(
                a.values[row * 3 + c],
                b.values[row * 3 + (c + 1) % 3],
                1e-10,
                1e-8,
            );
        }
    }
    let sparse = SparseSpec {
        method: SparseMethod::OvrL1Logistic,
        alpha: 0.03,
        tolerance: 1e-9,
        max_iterations: 10000,
    };
    assert_eq!(
        fit_sparse(&sparse, &x, &y).unwrap().support,
        fit_sparse(&sparse, &x, &changed).unwrap().support
    );
}
