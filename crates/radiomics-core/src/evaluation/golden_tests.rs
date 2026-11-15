use super::*;
use crate::contracts::NumericBuffer;
use serde_json::Value;
#[test]
fn legacy_seeded_bootstrap_stream_matches_frozen16_and64_without_evaluator_rng() {
    let old: Value = serde_json::from_str(include_str!(concat!(
        env!("RAD_GOLDEN_DIR"),
        "/python_golden_v1/reference.json"
    )))
    .unwrap();
    let old = &old["baseline_metrics"];
    let y: Vec<_> = old["inputs"]["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let p = numbers(&old["inputs"]["a"]);
    let draws: Vec<Vec<_>> = old["bootstrap_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    let seeded = bootstrap_binary_seeded(&y, &p, 16, 42, 0.95).unwrap();
    let explicit = bootstrap_auc_explicit(&y, &p, &draws, 0.95).unwrap();
    assert_eq!(seeded.replicate_scores, explicit.replicate_scores);
    assert_eq!(seeded.low, explicit.low);
    assert_eq!(seeded.high, explicit.high);
    assert_eq!(seeded.seed, Some(42));
    assert_eq!(
        seeded.rng_scheme.as_deref(),
        Some(crate::random::LEGACY_BOOTSTRAP_SCHEME)
    );
    assert_eq!(seeded.rng_namespace, None);
    let fixture = fixture();
    let case = &fixture["cases"]["binary"];
    let y: Vec<_> = case["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let p: Vec<_> = case["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row[1].as_f64().unwrap())
        .collect();
    let reference = &fixture["patient_binary_bootstrap"];
    let result = bootstrap_binary_seeded(&y, &p, 64, 42, 0.95).unwrap();
    vec_close(&result.replicate_scores, &reference["auc_replicates"]);
    close(result.low, reference["ci"][0].as_f64().unwrap());
    close(result.high, reference["ci"][1].as_f64().unwrap());
    for n in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        assert_eq!(
            result,
            pool.install(|| bootstrap_binary_seeded(&y, &p, 64, 42, 0.95).unwrap())
        );
    }
    assert!(bootstrap_binary_seeded(&y, &p, 1, 42, 0.95).is_err());
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("RAD_GOLDEN_DIR"),
        "/python_golden_v1/evaluation_reference.json"
    )))
    .unwrap()
}
fn close(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else if expected.is_infinite() {
        assert_eq!(actual, expected);
    } else {
        assert!(
            (actual - expected).abs() <= 1e-10 + 1e-8 * expected.abs(),
            "{actual:.16e} != {expected:.16e}"
        );
    }
}
fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn vec_close(actual: &[f64], reference: &Value) {
    let reference = numbers(reference);
    assert_eq!(actual.len(), reference.len());
    for (a, b) in actual.iter().zip(reference) {
        close(*a, b);
    }
}
fn per_class(actual: &[ClassMetrics], reference: &Value) {
    for (j, c) in actual.iter().enumerate() {
        close(c.precision, reference["precision"][j].as_f64().unwrap());
        close(c.recall, reference["recall"][j].as_f64().unwrap());
        close(c.f1, reference["f1"][j].as_f64().unwrap());
        assert_eq!(c.support, reference["support"][j].as_u64().unwrap());
    }
}
fn binary_case(y: &[u8], scores: &[f64], reference: &Value) {
    let metrics = binary_metrics(y, scores, 0.5).unwrap();
    for (a, key) in [
        (metrics.auc, "roc_auc"),
        (metrics.brier, "brier"),
        (metrics.accuracy, "accuracy"),
        (metrics.log_loss_eps1e15, "log_loss_eps1e15"),
        (metrics.average_precision, "average_precision"),
        (metrics.pr_auc_trapezoidal, "pr_auc_trapezoidal"),
    ] {
        close(a, reference[key].as_f64().unwrap());
    }
    let counts: Vec<_> = reference["confusion"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()))
        .collect();
    assert_eq!(metrics.confusion.counts, counts);
    per_class(&metrics.per_class, &reference["per_class"]);
    let roc = roc_curve(y, scores).unwrap();
    vec_close(&roc.thresholds, &reference["roc"]["thresholds"]);
    vec_close(&roc.false_positive_rate, &reference["roc"]["fpr"]);
    vec_close(&roc.true_positive_rate, &reference["roc"]["tpr"]);
    let pr = precision_recall_curve(y, scores).unwrap();
    vec_close(&pr.thresholds, &reference["pr"]["thresholds"]);
    vec_close(&pr.precision, &reference["pr"]["precision"]);
    vec_close(&pr.recall, &reference["pr"]["recall"]);
    for (name, strategy) in [
        ("uniform", CalibrationStrategy::Uniform),
        ("quantile", CalibrationStrategy::Quantile),
    ] {
        let curve = calibration_curve(y, scores, 5, strategy).unwrap();
        let r = &reference["calibration"][name];
        vec_close(&curve.edges, &r["edges"]);
        assert_eq!(
            curve.bin_membership,
            r["bin_membership"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            curve.bin_counts,
            r["bin_counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect::<Vec<_>>()
        );
        vec_close(&curve.fraction_positive, &r["nonempty_fraction_positive"]);
        vec_close(&curve.mean_predicted, &r["nonempty_mean_predicted"]);
    }
}
#[test]
fn binary_full_curves_metrics_and_calibration_membership_match_frozen_102() {
    let fixture = fixture();
    for name in ["binary", "binary_boundary_ties"] {
        let case = &fixture["cases"][name];
        let y: Vec<_> = case["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let scores = if let Some(p) = case.get("score") {
            numbers(p)
        } else {
            case["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row[1].as_f64().unwrap())
                .collect()
        };
        binary_case(&y, &scores, &case["metrics"]);
    }
}
#[test]
fn multiclass_per_class_and_macro_micro_weighted_metrics_match_frozen_102() {
    let fixture = fixture();
    let case = &fixture["cases"]["multiclass"];
    let reference = &case["metrics"];
    let labels: Vec<_> = case["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let values: Vec<_> = case["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect();
    let classes = vec!["A".into(), "B".into(), "C".into()];
    let target = ModelTarget::Classification {
        classes: classes.clone(),
        labels: labels.clone(),
    };
    let prediction = ModelPrediction::Classification {
        classes,
        probabilities: NumericBuffer {
            rows: labels.len(),
            cols: 3,
            missing: vec![false; values.len()],
            values: values.clone(),
        },
    };
    let metrics = classification_metrics(Task::Multiclass, &target, &prediction, 0.5).unwrap();
    per_class(&metrics.per_class, &reference["per_class"]);
    let counts: Vec<_> = reference["confusion"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()))
        .collect();
    assert_eq!(metrics.confusion.counts, counts);
    for (value, name) in [
        (&metrics.macro_average, "macro"),
        (&metrics.micro_average, "micro"),
        (&metrics.weighted_average, "weighted"),
    ] {
        for (actual, key) in [
            (value.precision, "precision"),
            (value.recall, "recall"),
            (value.f1, "f1"),
        ] {
            close(actual, reference["summary"][name][key].as_f64().unwrap());
        }
    }
    for (actual, key) in [
        (metrics.accuracy, "accuracy"),
        (metrics.macro_ovr_auc, "macro_ovr_auc"),
        (metrics.weighted_ovr_auc, "weighted_ovr_auc"),
        (metrics.micro_ovr_auc, "micro_ovr_auc"),
        (metrics.macro_average_precision, "macro_average_precision"),
        (
            metrics.weighted_average_precision,
            "weighted_average_precision",
        ),
        (metrics.micro_average_precision, "micro_average_precision"),
        (metrics.log_loss_eps1e15, "log_loss_eps1e15"),
        (metrics.multiclass_brier_sum, "multiclass_brier_sum"),
    ] {
        close(actual, reference[key].as_f64().unwrap());
    }
    for c in 0..3 {
        let y: Vec<_> = labels.iter().map(|&y| u8::from(y == c)).collect();
        let p: Vec<_> = values.chunks(3).map(|row| row[c]).collect();
        binary_case(&y, &p, &reference["ovr"][c]);
    }
}
#[test]
fn regression_and_degenerate_r2_follow_frozen_reference_not_invented_zeroes() {
    let fixture = fixture();
    let case = &fixture["cases"]["regression"];
    let output = regression_metrics(&numbers(&case["y"]), &numbers(&case["predictions"])).unwrap();
    for (actual, key) in [
        (output.mae, "mae"),
        (output.rmse, "rmse"),
        (output.r2, "r2"),
    ] {
        close(actual, case["metrics"][key].as_f64().unwrap());
    }
    assert_eq!(
        regression_metrics(&[2., 2., 2.], &[2., 2., 2.]).unwrap().r2,
        1.0
    );
    assert_eq!(
        regression_metrics(&[2., 2., 2.], &[1., 2., 3.]).unwrap().r2,
        0.0
    );
    let single = regression_metrics(&[2.], &[2.]).unwrap();
    assert!(single.r2.is_nan());
    let reloaded: RegressionMetrics =
        serde_json::from_slice(&serde_json::to_vec(&single).unwrap()).unwrap();
    assert!(reloaded.r2.is_nan());
    let no_positive = binary_metrics(&[0, 1], &[0., 0.], 0.5).unwrap();
    assert!(no_positive.ppv.is_nan());
    assert_eq!(no_positive.precision, 0.0);
    assert!(serde_json::to_string(&no_positive)
        .unwrap()
        .contains("\"NaN\""));
}

#[test]
fn native_delong_dca_and_hl_bins_tails_match_the_original_python_baseline() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("RAD_GOLDEN_DIR"),
        "/python_golden_v1/reference.json"
    )))
    .unwrap();
    let reference = &fixture["baseline_metrics"];
    let y: Vec<_> = reference["inputs"]["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let a = numbers(&reference["inputs"]["a"]);
    let b = numbers(&reference["inputs"]["b"]);
    let interval = delong_interval(&y, &a, 0.95).unwrap();
    close(
        interval.auc,
        reference["binary_delong"]["auc"].as_f64().unwrap(),
    );
    close(
        interval.low,
        reference["binary_delong"]["ci_low"].as_f64().unwrap(),
    );
    close(
        interval.high,
        reference["binary_delong"]["ci_high"].as_f64().unwrap(),
    );
    let paired = paired_delong(&y, &a, &b).unwrap();
    for (value, key) in [
        (paired.auc_a, "auc_a"),
        (paired.auc_b, "auc_b"),
        (paired.difference, "difference"),
        (paired.z, "z"),
        (paired.p_value, "p_value"),
    ] {
        close(value, reference["paired_delong"][key].as_f64().unwrap());
    }
    for (name, scores) in [("a", &a), ("b", &b)] {
        let dca = decision_curve(&y, scores, &[0.5, 0.2, 0.5, 0.8]).unwrap();
        let expected: Vec<_> = reference["dca"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["model"] == name)
            .collect();
        for (i, row) in expected.iter().enumerate() {
            close(
                dca.model_net_benefit[i],
                row["net_benefit"].as_f64().unwrap(),
            );
        }
        let all: Vec<_> = reference["dca"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["model"] == "Treat all")
            .collect();
        for (a, b) in dca.treat_all.iter().zip(all) {
            close(*a, b["net_benefit"].as_f64().unwrap());
        }
    }
    let hl = hosmer_lemeshow(&y, &a, 4).unwrap();
    let expected = &reference["hosmer_lemeshow"];
    close(hl.statistic, expected["statistic"].as_f64().unwrap());
    close(hl.p_value, expected["p_value"].as_f64().unwrap());
    assert_eq!(hl.df, expected["df"].as_u64().unwrap() as usize);
    for (bin, r) in hl.table.iter().zip(expected["table"].as_array().unwrap()) {
        assert_eq!(bin.n, r["n"].as_f64().unwrap() as u64);
        close(bin.observed, r["observed"].as_f64().unwrap());
        close(bin.expected, r["expected"].as_f64().unwrap());
    }
}
#[test]
fn explicit_bootstrap_indices_reproduce_old16_and_patient64_auc_intervals() {
    let old: Value = serde_json::from_str(include_str!(concat!(
        env!("RAD_GOLDEN_DIR"),
        "/python_golden_v1/reference.json"
    )))
    .unwrap();
    let old = &old["baseline_metrics"];
    let y: Vec<_> = old["inputs"]["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let p = numbers(&old["inputs"]["a"]);
    let draws: Vec<Vec<_>> = old["bootstrap_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    let result = bootstrap_auc_explicit(&y, &p, &draws, 0.95).unwrap();
    close(
        result.low,
        old["binary_bootstrap"]["ci_low"].as_f64().unwrap(),
    );
    close(
        result.high,
        old["binary_bootstrap"]["ci_high"].as_f64().unwrap(),
    );
    let fixture = fixture();
    let case = &fixture["cases"]["binary"];
    let y: Vec<_> = case["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let p: Vec<_> = case["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row[1].as_f64().unwrap())
        .collect();
    let reference = &fixture["patient_binary_bootstrap"];
    let draws: Vec<Vec<_>> = reference["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    let output = bootstrap_auc_explicit(&y, &p, &draws, 0.95).unwrap();
    vec_close(&output.replicate_scores, &reference["auc_replicates"]);
    close(output.low, reference["ci"][0].as_f64().unwrap());
    close(output.high, reference["ci"][1].as_f64().unwrap());
    let classes = vec!["negative".into(), "positive".into()];
    let target = ModelTarget::Classification {
        classes: classes.clone(),
        labels: y.iter().map(|&v| v as usize).collect(),
    };
    let values: Vec<_> = p.iter().flat_map(|&p| [1.0 - p, p]).collect();
    let prediction = ModelPrediction::Classification {
        classes,
        probabilities: NumericBuffer {
            rows: y.len(),
            cols: 2,
            missing: vec![false; values.len()],
            values,
        },
    };
    assert_eq!(
        output,
        bootstrap_model_score_explicit(Task::Binary, &target, &prediction, &draws, 0.95).unwrap()
    );
}
#[test]
fn binary_inference_undefined_and_invalid_requests_are_explicit() {
    let y = [0, 0, 1, 1];
    let a = [0., 0., 1., 1.];
    let b = [1., 1., 0., 0.];
    let paired = paired_delong(&y, &a, &b).unwrap();
    assert!(paired.z.is_infinite());
    assert_eq!(paired.p_value, 0.0);
    assert!(serde_json::to_string(&paired).unwrap().contains("\"Inf\""));
    assert_eq!(paired_delong(&y, &a, &a).unwrap().p_value, 1.0);
    assert!(hosmer_lemeshow(&y, &[0.5; 4], 10).is_err());
    assert!(hosmer_lemeshow(&y, &a, 2).is_err());
    assert!(decision_curve(&y, &a, &[1.0]).is_err());
    assert!(delong_interval(&[0, 1], &[0., 1.], 0.95).is_err());
    assert!(bootstrap_auc_explicit(&y, &a, &[vec![0, 1, 2, 4], vec![0, 1, 2, 3]], 0.95).is_err());
    assert!(bootstrap_auc_explicit(&y, &a, &[vec![0; 4], vec![1; 4]], 0.95).is_err());
}

#[test]
fn new_multiclass_and_regression_bootstrap_record_component_rng_and_reproduce_without_inner_pools()
{
    let fixture = fixture();
    for (name, task) in [
        ("multiclass", Task::Multiclass),
        ("regression", Task::Regression),
    ] {
        let case = &fixture["cases"][name];
        let (target, prediction) = if task == Task::Regression {
            (
                ModelTarget::Regression {
                    values: numbers(&case["y"]),
                },
                ModelPrediction::Regression {
                    values: numbers(&case["predictions"]),
                },
            )
        } else {
            let classes = vec!["A".into(), "B".into(), "C".into()];
            let labels: Vec<_> = case["y"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let values: Vec<_> = case["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .collect();
            (
                ModelTarget::Classification {
                    classes: classes.clone(),
                    labels: labels.clone(),
                },
                ModelPrediction::Classification {
                    classes,
                    probabilities: NumericBuffer {
                        rows: labels.len(),
                        cols: 3,
                        missing: vec![false; values.len()],
                        values,
                    },
                },
            )
        };
        let run = |replicates| {
            if task == Task::Regression {
                bootstrap_regression_seeded(&target, &prediction, replicates, 42, 0.95)
            } else {
                bootstrap_multiclass_seeded(&target, &prediction, replicates, 42, 0.95)
            }
        };
        let report = run(32).unwrap();
        assert_eq!(report, run(32).unwrap());
        assert_eq!(
            report.rng_scheme.as_deref(),
            Some(crate::random::RNG_SCHEME)
        );
        assert_eq!(
            report.rng_namespace.as_deref(),
            Some("evaluation-bootstrap-v1")
        );
        assert_eq!(report.seed, Some(42));
        assert_eq!(
            report.replicate_scores.as_slice(),
            &run(64).unwrap().replicate_scores[..32]
        );
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert_eq!(report, pool.install(|| run(32).unwrap()));
        }
        let reloaded: BootstrapInterval =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(report, reloaded);
        assert!(run(1).is_err());
    }
}
