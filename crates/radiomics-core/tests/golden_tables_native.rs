//! Independent numerical and failure-contract acceptance of D's train-fitted tables.
use ov_radiomics_core::contracts::{NumericBuffer, TabularData};
use ov_radiomics_core::models::ModelTarget;
use ov_radiomics_core::preprocessing::{
    self, AllMissingPolicy, PreprocessConfig, Scaling, SelectorConfig, UnivariateMethod,
};
use serde_json::Value;

fn read(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn names(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().into())
        .collect()
}
fn table(input: &Value, cohort: &str) -> TabularData {
    let rows = input[format!("numeric_{cohort}")].as_array().unwrap();
    let cells: Vec<_> = rows.iter().flat_map(|r| r.as_array().unwrap()).collect();
    TabularData {
        rows: rows.len(),
        numeric_names: names(&input["numeric_names"]),
        numeric: NumericBuffer {
            rows: rows.len(),
            cols: input["numeric_names"].as_array().unwrap().len(),
            values: cells.iter().map(|v| v.as_f64().unwrap_or(0.)).collect(),
            missing: cells.iter().map(|v| v.is_null()).collect(),
        },
        categorical_names: names(&input["categorical_names"]),
        categorical_values: input[format!("categorical_{cohort}")]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().map(String::from))
            })
            .collect(),
    }
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * b.abs(),
        "{a} != {b}"
    )
}
fn reorder(data: &TabularData) -> TabularData {
    let mut out = data.clone();
    out.numeric_names.reverse();
    out.categorical_names.reverse();
    for row in out.numeric.values.chunks_mut(out.numeric.cols) {
        row.reverse();
    }
    for row in out.numeric.missing.chunks_mut(out.numeric.cols) {
        row.reverse();
    }
    for row in out
        .categorical_values
        .chunks_mut(out.categorical_names.len())
    {
        row.reverse();
    }
    out
}

#[test]
fn preprocessing_matches_frozen_training_statistics_and_prediction_order() {
    let input = read("table_inputs.json");
    let expected = read("table_reference.json");
    let train = table(&input, "train");
    let evaluate = table(&input, "evaluation");
    assert!(preprocessing::fit_preprocessor(&train, &PreprocessConfig::default()).is_err());
    for (name, scaling) in [
        ("standard", Scaling::Standard),
        ("minmax", Scaling::MinMax),
        ("robust", Scaling::Robust),
    ] {
        let fit = preprocessing::fit_preprocessor(
            &train,
            &PreprocessConfig {
                scaling,
                all_missing: AllMissingPolicy::Drop,
            },
        )
        .unwrap();
        assert_eq!(fit.dropped_numeric, vec!["all_missing"]);
        assert_eq!(
            fit.numeric
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
            names(&expected["numeric"]["retained_names_after_explicit_all_missing_drop"])
        );
        for (j, state) in fit.numeric.iter().enumerate() {
            close(
                state.median,
                expected["numeric"]["medians"][j].as_f64().unwrap(),
            );
        }
        assert_eq!(
            fit.categorical
                .iter()
                .map(|s| s.mode.clone())
                .collect::<Vec<_>>(),
            names(&expected["categorical"]["modes"])
        );
        for (cohort, data) in [("training", &train), ("evaluation", &evaluate)] {
            let found = fit.transform(data).unwrap();
            let numeric = expected["numeric"]["scalers"][name][cohort]
                .as_array()
                .unwrap();
            let encoded = expected["categorical"][format!("one_hot_{cohort}")]
                .as_array()
                .unwrap();
            assert_eq!(found.rows, numeric.len());
            for i in 0..found.rows {
                let row: Vec<_> = numeric[i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .chain(encoded[i].as_array().unwrap())
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                assert_eq!(found.cols, row.len());
                for (j, expected) in row.into_iter().enumerate() {
                    close(found.values[i * found.cols + j], expected);
                }
            }
            assert_eq!(found, fit.transform(&reorder(data)).unwrap());
        }
        let mut unknown = evaluate.clone();
        unknown.categorical_values[0] = Some("Z".into());
        assert!(fit.transform(&unknown).is_err());
        let restored: preprocessing::FittedPreprocessor =
            serde_json::from_slice(&serde_json::to_vec(&fit).unwrap()).unwrap();
        assert_eq!(
            fit.transform(&evaluate).unwrap(),
            restored.transform(&evaluate).unwrap()
        );
    }
}

#[test]
fn task_aware_screening_uses_raw_variance_bh_and_stable_input_order() {
    let input = read("table_inputs.json");
    let expected = read("table_reference.json");
    let mut data = table(&input, "train");
    data.categorical_names.clear();
    data.categorical_values.clear();
    let fit = preprocessing::fit_preprocessor(
        &data,
        &PreprocessConfig {
            scaling: Scaling::Standard,
            all_missing: AllMissingPolicy::Drop,
        },
    )
    .unwrap();
    let x = fit.transform(&data).unwrap();
    let variance = fit.unscaled_variances().unwrap();
    for task in ["binary", "multiclass", "regression"] {
        let target = if task == "regression" {
            ModelTarget::Regression {
                values: input["targets"][task]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            }
        } else {
            ModelTarget::Classification {
                classes: if task == "binary" {
                    vec!["negative".into(), "positive".into()]
                } else {
                    vec!["A".into(), "B".into(), "C".into()]
                },
                labels: input["targets"][task]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
            }
        };
        let mut config = SelectorConfig {
            variance_threshold: 0.001,
            corr_threshold: Some(0.9),
            univariate: UnivariateMethod::Auto,
            p_threshold: 0.2,
            adjust_bh: true,
            sparse: None,
            coefficient_threshold: 1e-6,
        };
        let selected =
            preprocessing::fit_selector(&x, &fit.output_names, &target, &config, Some(&variance))
                .unwrap();
        assert_eq!(selected.selected_names().unwrap(), vec!["a"]);
        assert_eq!(
            selected
                .after_variance
                .iter()
                .map(|&j| fit.output_names[j].clone())
                .collect::<Vec<_>>(),
            names(&expected["selection"]["after_variance"])
        );
        for (j, test) in selected.univariate_results.iter().enumerate() {
            close(
                test.statistic,
                expected["selection"]["univariate"][task]["statistic"][j]
                    .as_f64()
                    .unwrap(),
            );
            close(
                test.p_value,
                expected["selection"]["univariate"][task]["p_values"][j]
                    .as_f64()
                    .unwrap(),
            );
            close(
                test.adjusted_p_value,
                expected["selection"]["univariate"][task]["bh_p_values"][j]
                    .as_f64()
                    .unwrap(),
            );
        }
        config.p_threshold = 0.05;
        assert!(preprocessing::fit_selector(
            &x,
            &fit.output_names,
            &target,
            &config,
            Some(&variance)
        )
        .is_err());
        config.univariate = UnivariateMethod::None;
        let correlation_only =
            preprocessing::fit_selector(&x, &fit.output_names, &target, &config, Some(&variance))
                .unwrap();
        assert_eq!(correlation_only.selected_names().unwrap(), vec!["a", "b"]);
    }
}
