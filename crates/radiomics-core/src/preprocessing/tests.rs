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
