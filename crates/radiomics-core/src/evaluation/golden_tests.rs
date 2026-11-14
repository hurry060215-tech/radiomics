use super::*;
use crate::contracts::NumericBuffer;
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../golden/python_golden_v1/evaluation_reference.json"
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
