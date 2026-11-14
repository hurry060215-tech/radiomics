//! Golden A's independent actual-state checks of clinical LP/Logit/ICC results.
use ov_radiomics_core::clinical_native::{self, ClinicalConfig, SeparationStatus};
use ov_radiomics_core::contracts::{NumericBuffer, TabularData};
use ov_radiomics_core::reliability::{self, IccType};
use serde_json::Value;
use std::collections::BTreeMap;

fn read(name: &str) -> Value {
    let path = std::path::Path::new(env!("RAD_GOLDEN_DIR"))
        .join("python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn buffer(rows: &Value) -> NumericBuffer {
    let values = rows.as_array().unwrap();
    let cols = values[0].as_array().unwrap().len();
    let cells: Vec<_> = values.iter().flat_map(|r| r.as_array().unwrap()).collect();
    NumericBuffer {
        rows: values.len(),
        cols,
        values: cells.iter().map(|v| v.as_f64().unwrap_or(0.)).collect(),
        missing: cells.iter().map(|v| v.is_null()).collect(),
    }
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * b.abs(),
        "{a:.17e} != {b:.17e}"
    )
}
fn state(a: f64, b: &Value) {
    match b.as_str() {
        Some("NaN") => assert!(a.is_nan()),
        Some("Inf") => assert_eq!(a, f64::INFINITY),
        Some("-Inf") => assert_eq!(a, f64::NEG_INFINITY),
        _ => close(a, b.as_f64().unwrap()),
    }
}

#[test]
fn lp_diagnostics_use_verified_original_problem_certificates() {
    let reference = read("separation_reference.json");
    for case in reference["cases"].as_array().unwrap() {
        let x = buffer(&case["x"]);
        let y: Vec<_> = case["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let diagnostic = clinical_native::separation_diagnostic(&x, &y, 1e-12).unwrap();
        let expected = case["expected_diagnostic"].as_str().unwrap();
        if expected == "rank_deficient" {
            assert_eq!(diagnostic.status, SeparationStatus::RankDeficient);
            continue;
        }
        assert_eq!(
            diagnostic.status,
            if expected == "separated" {
                SeparationStatus::Separated
            } else {
                SeparationStatus::NoSeparationDetected
            }
        );
        assert_eq!(diagnostic.scaled_beta.len(), x.cols + 1);
        let mut sum = 0.;
        for (i, &label) in y.iter().enumerate() {
            let sign = if label == 1 { 1. } else { -1. };
            let margin = sign
                * (diagnostic.scaled_beta[0]
                    + (0..x.cols)
                        .map(|j| {
                            x.values[i * x.cols + j] / diagnostic.column_scales[j + 1]
                                * diagnostic.scaled_beta[j + 1]
                        })
                        .sum::<f64>());
            assert!(margin >= 0., "Unverified negative margin accepted");
            close(margin, diagnostic.signed_margins[i]);
            sum += margin;
        }
        assert!(diagnostic.scaled_beta.iter().all(|v| v.abs() <= 1. + 1e-9));
        close(sum, case["hand_optimal_margin_sum"].as_f64().unwrap());
        close(sum, diagnostic.objective.unwrap());
    }
}

#[test]
fn clinical_actual_coefficients_information_and_inference_match_complete_cases() {
    let reference = read("statistics_reference.json");
    let clinical = &reference["clinical_logistic"];
    let rows = clinical["rows"].as_array().unwrap();
    let numeric = Value::Array(
        rows.iter()
            .map(|row| Value::Array(vec![row["a"].clone(), row["b"].clone()]))
            .collect(),
    );
    let data = TabularData {
        rows: rows.len(),
        numeric_names: vec!["a".into(), "b".into()],
        numeric: buffer(&numeric),
        categorical_names: vec!["sex".into()],
        categorical_values: rows
            .iter()
            .map(|r| r["sex"].as_str().map(String::from))
            .collect(),
    };
    let y: Vec<_> = rows
        .iter()
        .map(|r| r["target"].as_u64().map(|v| v as u8))
        .collect();
    for (_name, expected) in clinical["fits"].as_object().unwrap() {
        let features: Vec<_> = expected["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let reference_categories = if features.iter().any(|v| v == "sex") {
            BTreeMap::from([("sex".into(), "F".into())])
        } else {
            BTreeMap::new()
        };
        let fit = clinical_native::logistic_inference(
            &data,
            &y,
            &features,
            &reference_categories,
            &ClinicalConfig::default(),
        )
        .unwrap();
        assert_eq!(fit.n_used, expected["n_used"].as_u64().unwrap() as usize);
        assert_eq!(
            fit.n_dropped,
            expected["n_dropped"].as_u64().unwrap() as usize
        );
        let ids: Vec<_> = fit
            .used_indices
            .iter()
            .map(|&i| rows[i]["sample_id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            expected["used_sample_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>()
        );
        for (actual, key) in [
            (&fit.coefficient, "coefficient"),
            (&fit.covariance.values, "covariance"),
            (&fit.information.values, "information"),
            (&fit.score, "score"),
            (&fit.standard_error, "standard_error"),
            (&fit.p_values, "p_values"),
            (&fit.odds_ratio, "odds_ratio"),
            (&fit.apparent_probabilities, "apparent_probabilities"),
        ] {
            let expected_flat: Vec<_> = if key == "covariance" || key == "information" {
                expected[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect()
            } else {
                expected[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            };
            assert_eq!(actual.len(), expected_flat.len());
            for (&a, b) in actual.iter().zip(expected_flat) {
                close(a, b);
            }
        }
        let d = fit.coefficient.len();
        let mut information = vec![0.; d * d];
        let mut score = vec![0.; d];
        for (position, &index) in fit.used_indices.iter().enumerate() {
            let row = &rows[index];
            let mut design = vec![1.];
            design.extend(features.iter().map(|name| {
                if name == "sex" {
                    f64::from(row["sex"] == "M")
                } else {
                    row[name].as_f64().unwrap()
                }
            }));
            let linear: f64 = design
                .iter()
                .zip(&fit.coefficient)
                .map(|(x, b)| x * b)
                .sum();
            let p = if linear >= 0. {
                1. / (1. + (-linear).exp())
            } else {
                linear.exp() / (1. + linear.exp())
            };
            close(p, fit.apparent_probabilities[position]);
            for j in 0..d {
                score[j] += design[j] * (y[index].unwrap() as f64 - p);
                for k in 0..d {
                    information[j * d + k] += design[j] * design[k] * p * (1. - p);
                }
            }
        }
        for j in 0..d {
            close(score[j], fit.score[j]);
            assert!(score[j].abs() <= 1e-10);
            for k in 0..d {
                close(information[j * d + k], fit.information.values[j * d + k]);
                let product: f64 = (0..d)
                    .map(|l| information[j * d + l] * fit.covariance.values[l * d + k])
                    .sum();
                close(product, f64::from(j == k));
            }
        }
        let serialized = serde_json::to_vec(&fit).unwrap();
        let restored: clinical_native::LogisticInference =
            serde_json::from_slice(&serialized).unwrap();
        restored.validate().unwrap();
    }
}

#[test]
fn icc_keeps_type_nonfinite_locations_unrounded_intervals_and_missing_errors() {
    let reference = read("icc_reference.json");
    for (_name, case) in reference["cases"].as_object().unwrap() {
        let data = buffer(&case["values_subject_by_reader"]);
        for (kind, label) in [(IccType::Icc21, "ICC(2,1)"), (IccType::Icc31, "ICC(3,1)")] {
            let result = reliability::icc(&data, kind, 0.95);
            if case["reference"]["status"] == "error" {
                assert!(result.is_err());
                continue;
            }
            let result = result.unwrap();
            let expected = &case["reference"]["value"][label];
            assert_eq!(result.icc_type, kind);
            state(result.estimate, &expected["estimate"]);
            state(result.f_statistic, &expected["f_statistic"]);
            state(result.p_value, &expected["p_value"]);
            for (actual, expected) in result
                .confidence_interval
                .iter()
                .zip(expected["ci95"].as_array().unwrap())
            {
                state(*actual, expected);
            }
            if !result.estimate.is_finite()
                || result.confidence_interval.iter().any(|v| !v.is_finite())
            {
                assert!(!result.diagnostics.is_empty());
            }
            let encoded = serde_json::to_value(&result).unwrap();
            let restored: reliability::IccResult = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(serde_json::to_value(restored).unwrap(), encoded);
        }
    }
}
