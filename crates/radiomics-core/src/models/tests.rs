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

#[test]
fn predictive_lr_requires_regularization_but_clinical_solver_is_shared_and_not_an_inference_result()
{
    let x = buffer(4, 1, vec![-2., -1., 1., 2.]);
    let y = ModelTarget::Classification {
        classes: vec!["negative".into(), "positive".into()],
        labels: vec![0, 1, 0, 1],
    };
    let error = fit(
        &ModelSpec::LR {
            lambda: 0.0,
            tolerance: 1e-9,
            max_iterations: 1000,
        },
        &x,
        &y,
    )
    .unwrap_err();
    assert!(error.message.contains("strictly positive L2"));
    let (state, report) = solve_logistic(&x, &y, 0.0, 1e-9, 1000).unwrap();
    report.validate(true).unwrap();
    let not_predictive = FittedModel {
        spec: ModelSpec::LR {
            lambda: 0.0,
            tolerance: 1e-9,
            max_iterations: 1000,
        },
        feature_count: 1,
        state,
        report,
    };
    assert!(not_predictive.validate().is_err());
    let spec = ModelSpec::LR {
        lambda: 0.1,
        tolerance: 1e-9,
        max_iterations: 1000,
    };
    let model = fit(&spec, &x, &y).unwrap();
    let (state, report) = solve_logistic(&x, &y, 0.1, 1e-9, 1000).unwrap();
    assert_eq!(model.state, state);
    assert_eq!(model.report, report);
    let mut broken = model;
    broken.report.objective = None;
    assert!(broken.validate().is_err());
    broken.report.objective = Some(0.0);
    broken.report.converged = false;
    assert_eq!(broken.validate().unwrap_err().code, ErrorCode::Convergence);
}

fn forest_spec(seed: u64, regression: bool, trees: usize) -> ModelSpec {
    ModelSpec::RandomForest {
        trees,
        max_depth: None,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_features: if regression {
            MaxFeatures::All
        } else {
            MaxFeatures::Sqrt
        },
        bootstrap: true,
        seed,
    }
}
#[test]
fn cart_hand_splits_ties_leaves_and_thresholds_are_exact() {
    let hand: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../golden/python_golden_v1/forest_hand.json"
    )))
    .unwrap();
    for case in hand["cases"].as_array().unwrap() {
        let rows = case["x"].as_array().unwrap();
        let cols = rows[0].as_array().unwrap().len();
        let x = buffer(
            rows.len(),
            cols,
            rows.iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .collect(),
        );
        let y = if case["task"] == "regression" {
            ModelTarget::Regression {
                values: case["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            }
        } else {
            ModelTarget::Classification {
                classes: case["classes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_string())
                    .collect(),
                labels: case["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
            }
        };
        let model = fit(
            &ModelSpec::RandomForest {
                trees: 1,
                max_depth: Some(case["max_depth"].as_u64().unwrap() as usize),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::All,
                bootstrap: false,
                seed: 1,
            },
            &x,
            &y,
        )
        .unwrap();
        let ModelState::Forest { trees, .. } = &model.state else {
            unreachable!()
        };
        match &trees[0].nodes[0] {
            TreeNode::Leaf { impurity, .. } => {
                assert_eq!(case["root_is_leaf"], true);
                close(
                    *impurity,
                    case["root_impurity"].as_f64().unwrap(),
                    1e-15,
                    1e-15,
                );
            }
            TreeNode::Split {
                feature,
                threshold,
                impurity,
                impurity_decrease,
                ..
            } => {
                assert_eq!(*feature, case["root_feature"].as_u64().unwrap() as usize);
                assert_eq!(*threshold, case["root_threshold"].as_f64().unwrap());
                close(
                    *impurity,
                    case["root_impurity"].as_f64().unwrap(),
                    1e-15,
                    1e-15,
                );
                close(
                    *impurity_decrease,
                    case["root_impurity_decrease"].as_f64().unwrap(),
                    1e-15,
                    1e-15,
                );
            }
        }
        if let Some(evaluation) = case["evaluation_x"].as_array() {
            let x = buffer(
                evaluation.len(),
                cols,
                evaluation
                    .iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect(),
            );
            match model.predict(&x).unwrap() {
                ModelPrediction::Classification { probabilities, .. } => {
                    let expected: Vec<_> = case["probabilities"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                        .collect();
                    assert_eq!(probabilities.values, expected);
                }
                ModelPrediction::Regression { values } => {
                    let expected: Vec<_> = case["predictions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect();
                    assert_eq!(values, expected);
                }
            }
        }
    }
}
#[test]
fn forest_fixed_datasets_pass_patient_level_behavior_gates() {
    let (inputs, model_reference) = fixture();
    let reference: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../golden/python_golden_v1/forest_reference.json"
    )))
    .unwrap();
    for task in ["binary", "multiclass", "regression"] {
        let (x, y, _, _) = data(task);
        let expected = &reference["datasets"][task];
        let records = inputs["datasets"][task]["records"].as_array().unwrap();
        let sample_ids = expected["test_sample_ids"].as_array().unwrap();
        let rows: Vec<_> = sample_ids
            .iter()
            .map(|id| records.iter().find(|r| r["sample_id"] == *id).unwrap())
            .collect();
        let means = model_reference["datasets"][task]["standard_scaling"]["mean"]
            .as_array()
            .unwrap();
        let scales = model_reference["datasets"][task]["standard_scaling"]["scale"]
            .as_array()
            .unwrap();
        let test = buffer(
            rows.len(),
            x.cols,
            rows.iter()
                .flat_map(|row| {
                    row["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(j, v)| {
                            (v.as_f64().unwrap() - means[j].as_f64().unwrap())
                                / scales[j].as_f64().unwrap()
                        })
                })
                .collect(),
        );
        let patient_ids = expected["test_patient_ids"].as_array().unwrap();
        let targets: Vec<_> = expected["test_patient_targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mut metrics: Vec<Vec<f64>> = Vec::new();
        for seed in [11, 23, 37, 53, 71] {
            let model = fit(&forest_spec(seed, task == "regression", 200), &x, &y).unwrap();
            let prediction = model.predict(&test).unwrap();
            let (values, columns) = match prediction {
                ModelPrediction::Classification { probabilities, .. } => {
                    (probabilities.values, probabilities.cols)
                }
                ModelPrediction::Regression { values } => (values, 1),
            };
            let mut patient_values = vec![0.0; patient_ids.len() * columns];
            for (patient_index, patient) in patient_ids.iter().enumerate() {
                let indices: Vec<_> = rows
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (r["patient_id"] == *patient).then_some(i))
                    .collect();
                assert_eq!(indices.len(), 2);
                for j in 0..columns {
                    patient_values[patient_index * columns + j] = indices
                        .iter()
                        .map(|&i| values[i * columns + j] / indices.len() as f64)
                        .sum();
                }
            }
            let row = match task {
                "binary" => {
                    let scores: Vec<_> = patient_values.chunks(2).map(|r| r[1]).collect();
                    let labels: Vec<_> = targets.iter().map(|&y| y as u8).collect();
                    vec![
                        crate::auc(&labels, &scores).unwrap(),
                        scores
                            .iter()
                            .zip(&targets)
                            .map(|(p, y)| (p - y).powi(2))
                            .sum::<f64>()
                            / targets.len() as f64,
                    ]
                }
                "multiclass" => {
                    let predicted: Vec<_> = patient_values
                        .chunks(3)
                        .map(|r| {
                            r.iter()
                                .enumerate()
                                .fold((0, f64::NEG_INFINITY), |(best, value), (j, &p)| {
                                    if p > value {
                                        (j, p)
                                    } else {
                                        (best, value)
                                    }
                                })
                                .0
                        })
                        .collect();
                    let f1 = (0..3)
                        .map(|c| {
                            let tp = targets
                                .iter()
                                .zip(&predicted)
                                .filter(|(y, p)| **y == c as f64 && **p == c)
                                .count();
                            let fp = targets
                                .iter()
                                .zip(&predicted)
                                .filter(|(y, p)| **y != c as f64 && **p == c)
                                .count();
                            let fn_count = targets
                                .iter()
                                .zip(&predicted)
                                .filter(|(y, p)| **y == c as f64 && **p != c)
                                .count();
                            let denominator = 2 * tp + fp + fn_count;
                            if denominator == 0 {
                                0.0
                            } else {
                                2.0 * tp as f64 / denominator as f64
                            }
                        })
                        .sum::<f64>()
                        / 3.0;
                    vec![f1]
                }
                _ => vec![(patient_values
                    .iter()
                    .zip(&targets)
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f64>()
                    / targets.len() as f64)
                    .sqrt()],
            };
            println!("forest/{task}/seed{seed}: {row:?}");
            metrics.push(row);
        }
        let mean = |j| metrics.iter().map(|row| row[j] / 5.0).sum::<f64>();
        match task {
            "binary" => {
                assert!(mean(0) >= expected["mean_metrics"]["auc"].as_f64().unwrap() - 0.03);
                assert!(mean(1) <= expected["mean_metrics"]["brier"].as_f64().unwrap() + 0.03);
            }
            "multiclass" => {
                assert!(mean(0) >= expected["mean_metrics"]["macro_f1"].as_f64().unwrap() - 0.04)
            }
            _ => {
                let baseline = expected["mean_metrics"]["rmse"].as_f64().unwrap();
                let limit = (baseline * 0.1)
                    .max(expected["test_target_std_ddof0"].as_f64().unwrap() * 0.02);
                assert!(mean(0) <= baseline + limit);
            }
        }
    }
}
#[test]
fn forest_is_reproducible_across_outer_threads_and_json_and_rejects_corrupt_topology() {
    let (x, y, test, _) = data("binary");
    let spec = forest_spec(11, false, 20);
    let first = fit(&spec, &x, &y).unwrap();
    let repeated = fit(&spec, &x, &y).unwrap();
    assert_eq!(first, repeated);
    for n in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        assert_eq!(first, pool.install(|| fit(&spec, &x, &y).unwrap()));
    }
    let restored: FittedModel =
        serde_json::from_slice(&serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(
        first.predict(&test).unwrap(),
        restored.predict(&test).unwrap()
    );
    let mut invalid = first.clone();
    let ModelState::Forest { trees, .. } = &mut invalid.state else {
        unreachable!()
    };
    let TreeNode::Split { left, .. } = &mut trees[0].nodes[0] else {
        panic!("fixed training root must split")
    };
    *left = 0;
    assert!(invalid.validate().is_err());
}
