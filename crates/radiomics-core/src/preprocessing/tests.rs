use super::*;
use serde_json::Value;
fn fixtures() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/table_inputs.json"
        )))
        .unwrap(),
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/table_reference.json"
        )))
        .unwrap(),
    )
}
fn table(input: &Value, cohort: &str) -> TabularData {
    let numeric_names: Vec<_> = input["numeric_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let categorical_names: Vec<_> = input["categorical_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let rows = input[format!("numeric_{cohort}")].as_array().unwrap();
    let mut values = Vec::new();
    let mut missing = Vec::new();
    for row in rows {
        for v in row.as_array().unwrap() {
            missing.push(v.is_null());
            values.push(v.as_f64().unwrap_or(0.0));
        }
    }
    TabularData {
        rows: rows.len(),
        numeric: NumericBuffer {
            rows: rows.len(),
            cols: numeric_names.len(),
            values,
            missing,
        },
        numeric_names,
        categorical_names,
        categorical_values: input[format!("categorical_{cohort}")]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().map(str::to_string))
            })
            .collect(),
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-10 + 1e-8 * b.abs(), "{a} != {b}");
}
fn compare(output: &NumericBuffer, numeric: &Value, categorical: &Value) {
    for i in 0..output.rows {
        let expected: Vec<_> = numeric[i]
            .as_array()
            .unwrap()
            .iter()
            .chain(categorical[i].as_array().unwrap().iter())
            .map(|v| v.as_f64().unwrap())
            .collect();
        for (j, v) in expected.into_iter().enumerate() {
            close(output.values[i * output.cols + j], v);
        }
    }
}
#[test]
fn imputation_onehot_and_three_scalers_match_frozen_train_and_evaluation() {
    let (input, reference) = fixtures();
    let training = table(&input, "train");
    let evaluation = table(&input, "evaluation");
    assert!(fit_preprocessor(&training, &PreprocessConfig::default())
        .unwrap_err()
        .message
        .contains("entirely missing"));
    for (name, scaling) in [
        ("none", Scaling::None),
        ("standard", Scaling::Standard),
        ("minmax", Scaling::MinMax),
        ("robust", Scaling::Robust),
    ] {
        let fitted = fit_preprocessor(
            &training,
            &PreprocessConfig {
                scaling,
                all_missing: AllMissingPolicy::Drop,
            },
        )
        .unwrap();
        assert_eq!(fitted.dropped_numeric, vec!["all_missing"]);
        assert_eq!(
            fitted
                .categorical
                .iter()
                .map(|c| c.mode.as_str())
                .collect::<Vec<_>>(),
            vec!["A", "F"]
        );
        let r = if scaling == Scaling::None {
            &reference["numeric"]
        } else {
            &reference["numeric"]["scalers"][name]
        };
        compare(
            &fitted.transform(&training).unwrap(),
            &r[if scaling == Scaling::None {
                "imputed_training"
            } else {
                "training"
            }],
            &reference["categorical"]["one_hot_training"],
        );
        compare(
            &fitted.transform(&evaluation).unwrap(),
            &r[if scaling == Scaling::None {
                "imputed_evaluation"
            } else {
                "evaluation"
            }],
            &reference["categorical"]["one_hot_evaluation"],
        );
        let restored: FittedPreprocessor =
            serde_json::from_slice(&serde_json::to_vec(&fitted).unwrap()).unwrap();
        assert_eq!(
            fitted.transform(&evaluation).unwrap(),
            restored.transform(&evaluation).unwrap()
        );
        let variances = fitted.unscaled_variances().unwrap();
        for (a, b) in variances.iter().zip(
            reference["selection"]["population_variances"]
                .as_array()
                .unwrap(),
        ) {
            close(*a, b.as_f64().unwrap());
        }
    }
}
#[test]
fn reordered_columns_and_changed_evaluation_do_not_change_fitted_state() {
    let (input, _) = fixtures();
    let training = table(&input, "train");
    let evaluation = table(&input, "evaluation");
    let fitted = fit_preprocessor(
        &training,
        &PreprocessConfig {
            scaling: Scaling::Standard,
            all_missing: AllMissingPolicy::Drop,
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&fitted).unwrap();
    let expected = fitted.transform(&evaluation).unwrap();
    let mut changed = evaluation.clone();
    changed.numeric_names.reverse();
    for row in changed.numeric.values.chunks_mut(changed.numeric.cols) {
        row.reverse();
    }
    for row in changed.numeric.missing.chunks_mut(changed.numeric.cols) {
        row.reverse();
    }
    changed.categorical_names.reverse();
    for row in changed
        .categorical_values
        .chunks_mut(changed.categorical_names.len())
    {
        row.reverse();
    }
    assert_eq!(expected, fitted.transform(&changed).unwrap());
    changed.numeric.values.iter_mut().for_each(|v| *v += 100.0);
    for (v, m) in changed
        .numeric
        .values
        .iter_mut()
        .zip(&changed.numeric.missing)
    {
        if *m {
            *v = 0.0;
        }
    }
    let _ = fitted.transform(&changed).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&fitted).unwrap());
}
#[test]
fn unknown_category_all_missing_and_name_collisions_are_errors() {
    let (input, _) = fixtures();
    let training = table(&input, "train");
    let mut evaluation = table(&input, "evaluation");
    let config = PreprocessConfig {
        scaling: Scaling::None,
        all_missing: AllMissingPolicy::Drop,
    };
    let fitted = fit_preprocessor(&training, &config).unwrap();
    evaluation.categorical_values[0] = Some("Z".into());
    assert!(fitted
        .transform(&evaluation)
        .unwrap_err()
        .message
        .contains("Unknown category"));
    let mut missing = training.clone();
    missing.numeric.missing.fill(true);
    missing.numeric.values.fill(0.0);
    missing.categorical_values.fill(None);
    assert_eq!(
        fit_preprocessor(&missing, &config).unwrap_err().code,
        ErrorCode::EmptyResult
    );
    let mut collision = training.clone();
    collision.numeric_names[0] = "site=A".into();
    assert!(fit_preprocessor(&collision, &config)
        .unwrap_err()
        .message
        .contains("unique"));
    let mut wrong_type = training.clone();
    wrong_type.numeric_names[0] = "unrelated".into();
    assert_eq!(
        fitted.transform(&wrong_type).unwrap_err().code,
        ErrorCode::Alignment
    );
}

fn numeric_table(input: &Value) -> TabularData {
    let mut data = table(input, "train");
    data.categorical_names.clear();
    data.categorical_values.clear();
    data
}
fn target(input: &Value, task: &str) -> crate::models::ModelTarget {
    let values = input["targets"][task].as_array().unwrap();
    if task == "regression" {
        crate::models::ModelTarget::Regression {
            values: values.iter().map(|v| v.as_f64().unwrap()).collect(),
        }
    } else {
        crate::models::ModelTarget::Classification {
            classes: (0..if task == "binary" { 2 } else { 3 })
                .map(|v| v.to_string())
                .collect(),
            labels: values
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
        }
    }
}
#[test]
fn three_task_univariate_bh_and_stable_absolute_correlation_match_golden() {
    let (input, reference) = fixtures();
    let data = numeric_table(&input);
    let preprocess = fit_preprocessor(
        &data,
        &PreprocessConfig {
            scaling: Scaling::None,
            all_missing: AllMissingPolicy::Drop,
        },
    )
    .unwrap();
    let x = preprocess.transform(&data).unwrap();
    for task in ["binary", "multiclass", "regression"] {
        let config = SelectorConfig {
            variance_threshold: 0.001,
            corr_threshold: Some(0.9),
            univariate: UnivariateMethod::Auto,
            p_threshold: 0.2,
            adjust_bh: true,
            sparse: None,
            coefficient_threshold: 1e-6,
        };
        let selector = fit_selector(
            &x,
            &preprocess.output_names,
            &target(&input, task),
            &config,
            Some(&preprocess.unscaled_variances().unwrap()),
        )
        .unwrap();
        let r = &reference["selection"]["univariate"][task];
        assert_eq!(selector.selected_names().unwrap(), vec!["a"]);
        for (j, result) in selector.univariate_results.iter().enumerate() {
            close(result.statistic, r["statistic"][j].as_f64().unwrap());
            close(result.p_value, r["p_values"][j].as_f64().unwrap());
            close(
                result.adjusted_p_value,
                r["bh_p_values"][j].as_f64().unwrap(),
            );
        }
        let restored: FittedSelector =
            serde_json::from_slice(&serde_json::to_vec(&selector).unwrap()).unwrap();
        assert_eq!(
            selector.transform(&x, &preprocess.output_names).unwrap(),
            restored.transform(&x, &preprocess.output_names).unwrap()
        );
    }
    let config = SelectorConfig {
        variance_threshold: 0.001,
        univariate: UnivariateMethod::None,
        ..SelectorConfig::default()
    };
    let selector = fit_selector(
        &x,
        &preprocess.output_names,
        &target(&input, "binary"),
        &config,
        None,
    )
    .unwrap();
    assert_eq!(selector.selected_names().unwrap(), vec!["a", "b"]);
    assert!(selector
        .correlation_drops
        .iter()
        .any(|r| r.correlation < -0.9));
}
#[test]
fn unscaled_variance_reference_protects_low_variance_screen_after_standardization() {
    let (input, _) = fixtures();
    let data = numeric_table(&input);
    let preprocess = fit_preprocessor(
        &data,
        &PreprocessConfig {
            scaling: Scaling::Standard,
            all_missing: AllMissingPolicy::Drop,
        },
    )
    .unwrap();
    let x = preprocess.transform(&data).unwrap();
    let config = SelectorConfig {
        variance_threshold: 0.001,
        corr_threshold: None,
        univariate: UnivariateMethod::None,
        ..SelectorConfig::default()
    };
    let variances = preprocess.unscaled_variances().unwrap();
    let explicit = fit_selector(
        &x,
        &preprocess.output_names,
        &target(&input, "binary"),
        &config,
        Some(&variances),
    )
    .unwrap();
    let implicit = fit_selector(
        &x,
        &preprocess.output_names,
        &target(&input, "binary"),
        &config,
        None,
    )
    .unwrap();
    assert!(!explicit.selected_names().unwrap().contains(&"tiny".into()));
    assert!(implicit.selected_names().unwrap().contains(&"tiny".into()));
    assert_eq!(explicit.variance_source, VarianceSource::ExplicitUnscaled);
    assert!(fit_selector(
        &x,
        &preprocess.output_names,
        &target(&input, "binary"),
        &config,
        Some(&[1.0])
    )
    .is_err());
}
#[test]
fn selector_reorders_names_and_never_relaxes_empty_thresholds_or_multiclass_sparse_method() {
    let (input, _) = fixtures();
    let data = numeric_table(&input);
    let preprocess = fit_preprocessor(
        &data,
        &PreprocessConfig {
            scaling: Scaling::Standard,
            all_missing: AllMissingPolicy::Drop,
        },
    )
    .unwrap();
    let x = preprocess.transform(&data).unwrap();
    let config = SelectorConfig {
        univariate: UnivariateMethod::None,
        ..SelectorConfig::default()
    };
    let selector = fit_selector(
        &x,
        &preprocess.output_names,
        &target(&input, "binary"),
        &config,
        Some(&preprocess.unscaled_variances().unwrap()),
    )
    .unwrap();
    let expected = selector.transform(&x, &preprocess.output_names).unwrap();
    let mut changed = x.clone();
    let mut names = preprocess.output_names.clone();
    names.reverse();
    for row in changed.values.chunks_mut(changed.cols) {
        row.reverse();
    }
    assert_eq!(expected, selector.transform(&changed, &names).unwrap());
    names[changed.cols - 1] = "missing_a".into();
    assert_eq!(
        selector.transform(&changed, &names).unwrap_err().code,
        ErrorCode::Alignment
    );
    let mut config = config;
    config.variance_threshold = 1e9;
    assert_eq!(
        fit_selector(
            &x,
            &preprocess.output_names,
            &target(&input, "binary"),
            &config,
            None
        )
        .unwrap_err()
        .code,
        ErrorCode::EmptyResult
    );
    config.variance_threshold = 0.;
    config.univariate = UnivariateMethod::Auto;
    config.p_threshold = 1e-99;
    assert_eq!(
        fit_selector(
            &x,
            &preprocess.output_names,
            &target(&input, "binary"),
            &config,
            None
        )
        .unwrap_err()
        .code,
        ErrorCode::EmptyResult
    );
    config.univariate = UnivariateMethod::None;
    config.p_threshold = 0.05;
    config.sparse = Some(crate::models::SparseSpec {
        method: crate::models::SparseMethod::SquaredLasso,
        alpha: 1e9,
        tolerance: 1e-9,
        max_iterations: 1000,
    });
    assert_eq!(
        fit_selector(
            &x,
            &preprocess.output_names,
            &target(&input, "binary"),
            &config,
            None
        )
        .unwrap_err()
        .code,
        ErrorCode::EmptyResult
    );
    assert!(fit_selector(
        &x,
        &preprocess.output_names,
        &target(&input, "multiclass"),
        &config,
        None
    )
    .unwrap_err()
    .message
    .contains("multiclass"));
}
#[test]
fn sparse_selection_uses_shared_solver_and_explicit_per_class_union_threshold() {
    let (input, _) = fixtures();
    let data = numeric_table(&input);
    let preprocess = fit_preprocessor(
        &data,
        &PreprocessConfig {
            scaling: Scaling::Standard,
            all_missing: AllMissingPolicy::Drop,
        },
    )
    .unwrap();
    let x = preprocess.transform(&data).unwrap();
    for task in ["binary", "multiclass", "regression"] {
        let sparse = crate::models::SparseSpec {
            method: if task == "multiclass" {
                crate::models::SparseMethod::OvrL1Logistic
            } else {
                crate::models::SparseMethod::SquaredLasso
            },
            alpha: 0.03,
            tolerance: 1e-9,
            max_iterations: 10000,
        };
        let config = SelectorConfig {
            univariate: UnivariateMethod::None,
            sparse: Some(sparse),
            ..SelectorConfig::default()
        };
        let fitted = fit_selector(
            &x,
            &preprocess.output_names,
            &target(&input, task),
            &config,
            Some(&preprocess.unscaled_variances().unwrap()),
        )
        .unwrap();
        assert!(fitted.selected_names().unwrap().contains(&"a".into()));
        assert!(fitted
            .sparse
            .as_ref()
            .unwrap()
            .per_class_reports
            .iter()
            .all(|r| r.converged));
        let mut impossible = config;
        impossible.coefficient_threshold = 1e9;
        assert_eq!(
            fit_selector(
                &x,
                &preprocess.output_names,
                &target(&input, task),
                &impossible,
                None
            )
            .unwrap_err()
            .code,
            ErrorCode::EmptyResult
        );
    }
}
