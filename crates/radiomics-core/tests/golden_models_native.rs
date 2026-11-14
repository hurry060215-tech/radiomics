//! Golden A's independent black-box solver acceptance and stationary-point audit.
use ov_radiomics_core::contracts::NumericBuffer;
use ov_radiomics_core::models::TreeNode;
use ov_radiomics_core::models::{
    self, KnnWeights, MaxFeatures, ModelPrediction, ModelSpec, ModelState, ModelTarget,
    SparseMethod, SparseSpec,
};
use serde_json::Value;

fn read(name: &str) -> Value {
    let path = std::path::Path::new(env!("RAD_GOLDEN_DIR"))
        .join("python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= atol + rtol * expected.abs(),
        "actual={actual:.16e}, expected={expected:.16e}"
    );
}
fn prepare(task: &str) -> (NumericBuffer, ModelTarget, NumericBuffer, Value) {
    let input = read("model_inputs.json");
    let reference = read("model_reference.json")["datasets"][task].clone();
    let data = &input["datasets"][task];
    let rows = data["records"].as_array().unwrap();
    let training: Vec<_> = rows.iter().filter(|r| r["cohort"] == "train").collect();
    let evaluate: Vec<_> = reference["evaluation_sample_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| rows.iter().find(|r| r["sample_id"] == *id).unwrap())
        .collect();
    let transform = |rows: &[&Value]| {
        let values: Vec<_> = rows
            .iter()
            .flat_map(|r| {
                r["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(j, v)| {
                        (v.as_f64().unwrap()
                            - reference["standard_scaling"]["mean"][j].as_f64().unwrap())
                            / reference["standard_scaling"]["scale"][j].as_f64().unwrap()
                    })
            })
            .collect();
        NumericBuffer {
            rows: rows.len(),
            cols: 4,
            missing: vec![false; values.len()],
            values,
        }
    };
    let target = if task == "regression" {
        ModelTarget::Regression {
            values: training
                .iter()
                .map(|r| r["target"].as_f64().unwrap())
                .collect(),
        }
    } else {
        ModelTarget::Classification {
            classes: data["classes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().into())
                .collect(),
            labels: training
                .iter()
                .map(|r| r["target"].as_u64().unwrap() as usize)
                .collect(),
        }
    };
    (
        transform(&training),
        target,
        transform(&evaluate),
        reference,
    )
}

fn objective_gradient(
    x: &NumericBuffer,
    y: &ModelTarget,
    state: &ModelState,
    lambda: f64,
) -> (f64, f64) {
    let (coefficients, intercepts, outputs) = match state {
        ModelState::Logistic {
            coefficients,
            intercepts,
            ..
        } => (&coefficients.values, intercepts.clone(), coefficients.rows),
        ModelState::Linear {
            coefficients,
            intercept,
            ..
        } => (coefficients, vec![*intercept], 1),
        _ => panic!("Expected optimized state"),
    };
    let mut gradient = vec![0.; outputs * (x.cols + 1)];
    let mut loss = 0.;
    for i in 0..x.rows {
        let row = &x.values[i * x.cols..(i + 1) * x.cols];
        let z: Vec<_> = (0..outputs)
            .map(|k| {
                intercepts[k]
                    + (0..x.cols)
                        .map(|j| row[j] * coefficients[k * x.cols + j])
                        .sum::<f64>()
            })
            .collect();
        let residual = match y {
            ModelTarget::Regression { values } => {
                loss += (z[0] - values[i]).powi(2) / 2.;
                vec![z[0] - values[i]]
            }
            ModelTarget::Classification { labels, .. } if outputs == 1 => {
                let softplus = z[0].max(0.) + (-z[0].abs()).exp().ln_1p();
                loss += softplus - labels[i] as f64 * z[0];
                vec![
                    if z[0] >= 0. {
                        1. / (1. + (-z[0]).exp())
                    } else {
                        z[0].exp() / (1. + z[0].exp())
                    } - labels[i] as f64,
                ]
            }
            ModelTarget::Classification { labels, .. } => {
                let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<_> = z.iter().map(|v| (v - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                loss += max + sum.ln() - z[labels[i]];
                exp.iter()
                    .enumerate()
                    .map(|(k, e)| e / sum - f64::from(k == labels[i]))
                    .collect()
            }
        };
        for k in 0..outputs {
            for j in 0..x.cols {
                gradient[k * (x.cols + 1) + j] += residual[k] * row[j] / x.rows as f64;
            }
            gradient[k * (x.cols + 1) + x.cols] += residual[k] / x.rows as f64;
        }
    }
    for k in 0..outputs {
        for j in 0..x.cols {
            gradient[k * (x.cols + 1) + j] += lambda * coefficients[k * x.cols + j];
        }
    }
    (
        loss / x.rows as f64 + lambda * coefficients.iter().map(|w| w * w).sum::<f64>() / 2.,
        gradient.into_iter().map(f64::abs).fold(0., f64::max),
    )
}

#[test]
fn native_model_predictions_and_independently_recomputed_stationarity() {
    for task in ["binary", "multiclass", "regression"] {
        let (x, y, evaluate, reference) = prepare(task);
        let mut models = vec![
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
            models.extend([
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
            models.extend([
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
        for (name, spec) in models {
            let fit = models::fit(&spec, &x, &y).unwrap();
            let expected = &reference["models"][name];
            let prediction = fit.predict(&evaluate).unwrap();
            let (actual, expected_values) = match prediction {
                ModelPrediction::Regression { values } => (
                    values,
                    expected["predictions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect::<Vec<_>>(),
                ),
                ModelPrediction::Classification {
                    classes,
                    probabilities,
                } => {
                    let ModelTarget::Classification {
                        classes: ordered, ..
                    } = &y
                    else {
                        panic!("wrong task")
                    };
                    assert_eq!(&classes, ordered);
                    (
                        probabilities.values,
                        expected["probabilities"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                            .collect::<Vec<_>>(),
                    )
                }
            };
            assert_eq!(actual.len(), expected_values.len());
            for (a, b) in actual.iter().zip(expected_values) {
                close(*a, b, 1e-6, 1e-5);
            }
            if let Some(reported_objective) = fit.report.objective {
                let lambda = if name == "Linear" { 0. } else { 0.1 };
                let (objective, residual) = objective_gradient(&x, &y, &fit.state, lambda);
                assert!(
                    residual <= 1e-7,
                    "{task}/{name} actual stationarity {residual}"
                );
                close(
                    objective,
                    expected["state"]["objective"].as_f64().unwrap(),
                    1e-10,
                    1e-8,
                );
                close(reported_objective, objective, 1e-12, 1e-12);
            }
        }
    }
}

#[test]
fn sparse_native_states_satisfy_independent_kkt_and_frozen_support() {
    for task in ["binary", "multiclass", "regression"] {
        let (x, y, _, reference) = prepare(task);
        let method = if task == "multiclass" {
            SparseMethod::OvrL1Logistic
        } else {
            SparseMethod::SquaredLasso
        };
        let alpha = if task == "regression" { 0.05 } else { 0.03 };
        let fit = models::fit_sparse(
            &SparseSpec {
                method,
                alpha,
                tolerance: 1e-9,
                max_iterations: 10000,
            },
            &x,
            &y,
        )
        .unwrap();
        let mut union = vec![false; x.cols];
        for k in 0..fit.coefficients.rows {
            let coef = &fit.coefficients.values[k * x.cols..(k + 1) * x.cols];
            let b = fit.intercepts[k];
            let mut smooth = vec![0.; x.cols + 1];
            for i in 0..x.rows {
                let row = &x.values[i * x.cols..(i + 1) * x.cols];
                let z = b + row.iter().zip(coef).map(|(a, w)| a * w).sum::<f64>();
                let residual = match &y {
                    ModelTarget::Regression { values } => z - values[i],
                    ModelTarget::Classification { labels, .. } if task == "binary" => {
                        z - labels[i] as f64
                    }
                    ModelTarget::Classification { labels, .. } => {
                        1. / (1. + (-z).exp()) - f64::from(labels[i] == k)
                    }
                };
                for j in 0..x.cols {
                    smooth[j] += residual * row[j] / x.rows as f64;
                }
                smooth[x.cols] += residual / x.rows as f64;
            }
            assert!(smooth[x.cols].abs() <= 1e-7);
            for j in 0..x.cols {
                let residual = if coef[j].abs() > 1e-10 {
                    (smooth[j] + alpha * coef[j].signum()).abs()
                } else {
                    (smooth[j].abs() - alpha).max(0.)
                };
                assert!(residual <= 1e-7, "{task}/{k}/{j} actual KKT {residual}");
                union[j] |= coef[j].abs() > 1e-10;
            }
        }
        let support: Vec<_> = union
            .iter()
            .enumerate()
            .filter_map(|(j, &yes)| yes.then_some(j))
            .collect();
        let expected = if task == "multiclass" {
            &reference["models"]["OvR_L1_Logistic"]["support_union"]
        } else {
            &reference["models"]["LASSO"]["state"]["support"]
        };
        assert_eq!(
            support,
            expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
        assert_eq!(fit.support, support);
    }
}

#[test]
fn predictive_zero_penalty_is_rejected_and_shared_solver_matches_positive_fit() {
    let (x, y, _, _) = prepare("binary");
    assert!(models::fit(
        &ModelSpec::LR {
            lambda: 0.,
            tolerance: 1e-9,
            max_iterations: 5000
        },
        &x,
        &y
    )
    .is_err());
    let fit = models::fit(
        &ModelSpec::LR {
            lambda: 0.1,
            tolerance: 1e-9,
            max_iterations: 5000,
        },
        &x,
        &y,
    )
    .unwrap();
    let (state, report) = models::solve_logistic(&x, &y, 0.1, 1e-9, 5000).unwrap();
    assert_eq!(state, fit.state);
    assert_eq!(report, fit.report);
}

#[test]
fn forest_five_seed_patient_behavior_gates_are_independent_per_dataset() {
    let forest = read("forest_reference.json");
    let inputs = read("model_inputs.json");
    for task in ["binary", "multiclass", "regression"] {
        let (x, y, evaluate, model_reference) = prepare(task);
        let reference = &forest["datasets"][task];
        let ids = reference["test_sample_ids"].as_array().unwrap();
        let eval_ids = model_reference["evaluation_sample_ids"].as_array().unwrap();
        let values: Vec<_> = ids
            .iter()
            .flat_map(|id| {
                let index = eval_ids.iter().position(|v| v == id).unwrap();
                evaluate.values[index * evaluate.cols..(index + 1) * evaluate.cols]
                    .iter()
                    .copied()
            })
            .collect();
        let testing = NumericBuffer {
            rows: ids.len(),
            cols: evaluate.cols,
            missing: vec![false; values.len()],
            values,
        };
        let source = inputs["datasets"][task]["records"].as_array().unwrap();
        let groups: Vec<Vec<usize>> = reference["test_patient_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|patient| {
                ids.iter()
                    .enumerate()
                    .filter_map(|(i, id)| {
                        (source.iter().find(|r| r["sample_id"] == *id).unwrap()["patient_id"]
                            == *patient)
                            .then_some(i)
                    })
                    .collect()
            })
            .collect();
        assert!(groups.iter().all(|g| g.len() == 2));
        let actual_y: Vec<_> = reference["test_patient_targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mut metrics = vec![];
        for (seed_index, seed) in forest["model_seeds"].as_array().unwrap().iter().enumerate() {
            let spec = ModelSpec::RandomForest {
                trees: 200,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: if task == "regression" {
                    MaxFeatures::All
                } else {
                    MaxFeatures::Sqrt
                },
                bootstrap: true,
                seed: seed.as_u64().unwrap(),
            };
            let fitted = models::fit(&spec, &x, &y).unwrap();
            let prediction = fitted.predict(&testing).unwrap();
            let restored: models::FittedModel =
                serde_json::from_slice(&serde_json::to_vec(&fitted).unwrap()).unwrap();
            assert_eq!(prediction, restored.predict(&testing).unwrap());
            if seed_index == 0 {
                let workers: Vec<_> = (0..3)
                    .map(|_| {
                        let x = x.clone();
                        let y = y.clone();
                        let spec = spec.clone();
                        std::thread::spawn(move || models::fit(&spec, &x, &y).unwrap())
                    })
                    .collect();
                for worker in workers {
                    assert_eq!(worker.join().unwrap(), fitted);
                }
            }
            match prediction {
                ModelPrediction::Regression { values } => {
                    let patient: Vec<_> = groups
                        .iter()
                        .map(|g| g.iter().map(|&i| values[i]).sum::<f64>() / g.len() as f64)
                        .collect();
                    let rmse = (patient
                        .iter()
                        .zip(&actual_y)
                        .map(|(p, y)| (p - y).powi(2))
                        .sum::<f64>()
                        / actual_y.len() as f64)
                        .sqrt();
                    metrics.push(vec![rmse]);
                }
                ModelPrediction::Classification {
                    classes,
                    probabilities,
                } => {
                    let count = classes.len();
                    let patient: Vec<Vec<f64>> = groups
                        .iter()
                        .map(|g| {
                            (0..count)
                                .map(|c| {
                                    g.iter()
                                        .map(|&i| probabilities.values[i * count + c])
                                        .sum::<f64>()
                                        / g.len() as f64
                                })
                                .collect()
                        })
                        .collect();
                    if task == "binary" {
                        let positives: Vec<_> = actual_y
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &y)| (y == 1.).then_some(patient[i][1]))
                            .collect();
                        let negatives: Vec<_> = actual_y
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &y)| (y == 0.).then_some(patient[i][1]))
                            .collect();
                        let auc = positives
                            .iter()
                            .flat_map(|p| {
                                negatives.iter().map(move |n| {
                                    if p > n {
                                        1.
                                    } else if p == n {
                                        0.5
                                    } else {
                                        0.
                                    }
                                })
                            })
                            .sum::<f64>()
                            / (positives.len() * negatives.len()) as f64;
                        let brier = patient
                            .iter()
                            .zip(&actual_y)
                            .map(|(p, y)| (p[1] - y).powi(2))
                            .sum::<f64>()
                            / actual_y.len() as f64;
                        metrics.push(vec![auc, brier]);
                    } else {
                        let predicted: Vec<_> = patient
                            .iter()
                            .map(|p| {
                                (0..count)
                                    .min_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)))
                                    .unwrap()
                            })
                            .collect();
                        let mut total = 0.;
                        for c in 0..count {
                            let tp = actual_y
                                .iter()
                                .zip(&predicted)
                                .filter(|(y, p)| **y == c as f64 && **p == c)
                                .count();
                            let fp = actual_y
                                .iter()
                                .zip(&predicted)
                                .filter(|(y, p)| **y != c as f64 && **p == c)
                                .count();
                            let fn_count = actual_y
                                .iter()
                                .zip(&predicted)
                                .filter(|(y, p)| **y == c as f64 && **p != c)
                                .count();
                            let denom = 2 * tp + fp + fn_count;
                            total += if denom == 0 {
                                0.
                            } else {
                                2. * tp as f64 / denom as f64
                            };
                        }
                        metrics.push(vec![total / count as f64]);
                    }
                }
            }
        }
        let mean = |j: usize| metrics.iter().map(|row| row[j]).sum::<f64>() / metrics.len() as f64;
        let reference_metrics = &reference["mean_metrics"];
        if task == "binary" {
            assert!(
                mean(0) >= reference_metrics["auc"].as_f64().unwrap() - 0.03,
                "binary patient AUC {}",
                mean(0)
            );
            assert!(
                mean(1) <= reference_metrics["brier"].as_f64().unwrap() + 0.03,
                "binary patient Brier {}",
                mean(1)
            );
        } else if task == "multiclass" {
            assert!(
                mean(0) >= reference_metrics["macro_f1"].as_f64().unwrap() - 0.04,
                "multiclass patient macro-F1 {}",
                mean(0)
            );
        } else {
            let expected = reference_metrics["rmse"].as_f64().unwrap();
            let allowed =
                (expected * 0.1).max(reference["test_target_std_ddof0"].as_f64().unwrap() * 0.02);
            assert!(
                mean(0) - expected <= allowed,
                "regression patient RMSE {} > {}+{}",
                mean(0),
                expected,
                allowed
            );
        }
        println!("{task} native five-seed patient mean={metrics:?}");
    }
}

#[test]
fn hand_cart_ties_and_leaf_predictions_match_frozen_math() {
    let hand = read("forest_hand.json");
    for case in hand["cases"].as_array().unwrap() {
        let rows = case["x"].as_array().unwrap();
        let cols = rows[0].as_array().unwrap().len();
        let values: Vec<_> = rows
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let x = NumericBuffer {
            rows: rows.len(),
            cols,
            missing: vec![false; values.len()],
            values,
        };
        let target = if case["task"] == "regression" {
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
        let spec = ModelSpec::RandomForest {
            trees: 1,
            max_depth: Some(case["max_depth"].as_u64().unwrap() as usize),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            bootstrap: false,
            seed: 11,
        };
        let fit = models::fit(&spec, &x, &target).unwrap();
        let ModelState::Forest { trees, .. } = &fit.state else {
            panic!("Wrong state");
        };
        let root = &trees[0].nodes[0];
        match root {
            TreeNode::Leaf { impurity, .. } => {
                assert_eq!(case["root_is_leaf"], true);
                close(
                    *impurity,
                    case["root_impurity"].as_f64().unwrap(),
                    1e-10,
                    1e-8,
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
                    1e-10,
                    1e-8,
                );
                close(
                    *impurity_decrease,
                    case["root_impurity_decrease"].as_f64().unwrap(),
                    1e-10,
                    1e-8,
                );
            }
        }
        if let Some(rows) = case["evaluation_x"].as_array() {
            let values: Vec<_> = rows
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .collect();
            let evaluate = NumericBuffer {
                rows: rows.len(),
                cols,
                missing: vec![false; values.len()],
                values,
            };
            match fit.predict(&evaluate).unwrap() {
                ModelPrediction::Regression { values } => assert_eq!(
                    values,
                    case["predictions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect::<Vec<_>>()
                ),
                ModelPrediction::Classification { probabilities, .. } => assert_eq!(
                    probabilities.values,
                    case["probabilities"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                        .collect::<Vec<_>>()
                ),
            }
        }
    }
}
