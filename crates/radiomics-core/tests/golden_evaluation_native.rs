//! Independent native evaluation acceptance, consuming immutable public references.
use ov_radiomics_core::contracts::{NumericBuffer, Task};
use ov_radiomics_core::evaluation::{self, CalibrationStrategy};
use ov_radiomics_core::models::{ModelPrediction, ModelTarget};
use serde_json::Value;

fn read(name: &str) -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * b.abs(),
        "{a:.17e} != {b:.17e}"
    )
}
fn vector(a: &[f64], b: &Value) {
    let b = numbers(b);
    assert_eq!(a.len(), b.len());
    for (&a, b) in a.iter().zip(b) {
        close(a, b);
    }
}
fn check_binary(y: &[u8], p: &[f64], expected: &Value) {
    let metrics = evaluation::binary_metrics(y, p, 0.5).unwrap();
    close(metrics.auc, expected["roc_auc"].as_f64().unwrap());
    close(metrics.brier, expected["brier"].as_f64().unwrap());
    close(
        metrics.log_loss_eps1e15,
        expected["log_loss_eps1e15"].as_f64().unwrap(),
    );
    let roc = evaluation::roc_curve(y, p).unwrap();
    vector(&roc.false_positive_rate, &expected["roc"]["fpr"]);
    vector(&roc.true_positive_rate, &expected["roc"]["tpr"]);
    vector(&roc.thresholds, &expected["roc"]["thresholds"]);
    let pr = evaluation::precision_recall_curve(y, p).unwrap();
    vector(&pr.precision, &expected["pr"]["precision"]);
    vector(&pr.recall, &expected["pr"]["recall"]);
    vector(&pr.thresholds, &expected["pr"]["thresholds"]);
    close(
        pr.average_precision,
        expected["average_precision"].as_f64().unwrap(),
    );
    close(
        pr.trapezoidal_auc,
        expected["pr_auc_trapezoidal"].as_f64().unwrap(),
    );
    for (strategy, name) in [
        (CalibrationStrategy::Uniform, "uniform"),
        (CalibrationStrategy::Quantile, "quantile"),
    ] {
        let curve = evaluation::calibration_curve(y, p, 5, strategy).unwrap();
        let reference = &expected["calibration"][name];
        assert_eq!(
            curve.bin_membership,
            reference["bin_membership"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            curve.bin_counts,
            reference["bin_counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect::<Vec<_>>()
        );
        vector(&curve.edges, &reference["edges"]);
        vector(
            &curve.fraction_positive,
            &reference["nonempty_fraction_positive"],
        );
        vector(&curve.mean_predicted, &reference["nonempty_mean_predicted"]);
    }
}

#[test]
fn all_binary_and_ovr_curves_calibration_counts_and_regression_metrics_match() {
    let reference = read("evaluation_reference.json");
    for name in ["binary", "binary_boundary_ties"] {
        let case = &reference["cases"][name];
        let y: Vec<_> = case["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let p = if name == "binary" {
            case["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r[1].as_f64().unwrap())
                .collect::<Vec<_>>()
        } else {
            numbers(&case["score"])
        };
        check_binary(&y, &p, &case["metrics"]);
    }
    let case = &reference["cases"]["multiclass"];
    let labels: Vec<_> = case["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let probabilities: Vec<_> = case["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect();
    let classes = vec!["benign".into(), "intermediate".into(), "malignant".into()];
    let target = ModelTarget::Classification {
        classes: classes.clone(),
        labels: labels.clone(),
    };
    let prediction = ModelPrediction::Classification {
        classes,
        probabilities: NumericBuffer {
            rows: labels.len(),
            cols: 3,
            missing: vec![false; probabilities.len()],
            values: probabilities.clone(),
        },
    };
    let report =
        evaluation::classification_metrics(Task::Multiclass, &target, &prediction, 0.5).unwrap();
    let expected = &case["metrics"];
    assert_eq!(
        report.confusion.counts,
        expected["confusion"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()))
            .collect::<Vec<_>>()
    );
    close(report.accuracy, expected["accuracy"].as_f64().unwrap());
    let encoded = serde_json::to_value(&report).unwrap();
    for name in [
        "macro_ovr_auc",
        "weighted_ovr_auc",
        "micro_ovr_auc",
        "macro_average_precision",
        "weighted_average_precision",
        "micro_average_precision",
        "log_loss_eps1e15",
        "multiclass_brier_sum",
    ] {
        close(
            encoded[name].as_f64().unwrap(),
            expected[name].as_f64().unwrap(),
        );
    }
    for (actual, name) in [
        (&report.macro_average, "macro"),
        (&report.micro_average, "micro"),
        (&report.weighted_average, "weighted"),
    ] {
        let e = &expected["summary"][name];
        close(actual.precision, e["precision"].as_f64().unwrap());
        close(actual.recall, e["recall"].as_f64().unwrap());
        close(actual.f1, e["f1"].as_f64().unwrap());
    }
    for c in 0..3 {
        let y: Vec<_> = labels.iter().map(|&l| u8::from(l == c)).collect();
        let p: Vec<_> = probabilities.chunks(3).map(|r| r[c]).collect();
        check_binary(&y, &p, &expected["ovr"][c]);
    }
    let case = &reference["cases"]["regression"];
    let report =
        evaluation::regression_metrics(&numbers(&case["y"]), &numbers(&case["predictions"]))
            .unwrap();
    close(report.mae, case["metrics"]["mae"].as_f64().unwrap());
    close(report.rmse, case["metrics"]["rmse"].as_f64().unwrap());
    close(report.r2, case["metrics"]["r2"].as_f64().unwrap());
    assert!(evaluation::regression_metrics(&[2.], &[2.])
        .unwrap()
        .r2
        .is_nan());
}

#[test]
fn native_explicit_patient_and_legacy_bootstrap_and_binary_inference_match() {
    let reference = read("evaluation_reference.json");
    let case = &reference["cases"]["binary"];
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
        .map(|r| r[1].as_f64().unwrap())
        .collect();
    let bootstrap = &reference["patient_binary_bootstrap"];
    let draws: Vec<Vec<usize>> = serde_json::from_value(bootstrap["indices"].clone()).unwrap();
    let result = evaluation::bootstrap_auc_explicit(&y, &p, &draws, 0.95).unwrap();
    vector(&result.replicate_scores, &bootstrap["auc_replicates"]);
    close(result.low, bootstrap["ci"][0].as_f64().unwrap());
    close(result.high, bootstrap["ci"][1].as_f64().unwrap());
    let baseline = read("reference.json");
    let legacy = &baseline["baseline_metrics"];
    let y: Vec<_> = legacy["inputs"]["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let a = numbers(&legacy["inputs"]["a"]);
    let b = numbers(&legacy["inputs"]["b"]);
    let legacy_draws: Vec<Vec<usize>> =
        serde_json::from_value(legacy["bootstrap_indices"].clone()).unwrap();
    let bootstrap = evaluation::bootstrap_auc_explicit(&y, &a, &legacy_draws, 0.95).unwrap();
    close(
        bootstrap.low,
        legacy["binary_bootstrap"]["ci_low"].as_f64().unwrap(),
    );
    close(
        bootstrap.high,
        legacy["binary_bootstrap"]["ci_high"].as_f64().unwrap(),
    );
    for (name, values) in [("a", &a), ("b", &b)] {
        let curve = evaluation::decision_curve(&y, values, &[0.5, 0.2, 0.5, 0.8]).unwrap();
        let expected: Vec<_> = legacy["dca"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["model"] == name)
            .collect();
        for (i, row) in expected.iter().enumerate() {
            close(curve.thresholds[i], row["threshold"].as_f64().unwrap());
            close(
                curve.model_net_benefit[i],
                row["net_benefit"].as_f64().unwrap(),
            );
        }
        for (label, actual) in [
            ("Treat all", &curve.treat_all),
            ("Treat none", &curve.treat_none),
        ] {
            let expected: Vec<_> = legacy["dca"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["model"] == label)
                .collect();
            for (i, row) in expected.iter().enumerate() {
                close(actual[i], row["net_benefit"].as_f64().unwrap());
            }
        }
    }
    let interval = evaluation::delong_interval(&y, &a, 0.95).unwrap();
    close(
        interval.low,
        legacy["binary_delong"]["ci_low"].as_f64().unwrap(),
    );
    close(
        interval.high,
        legacy["binary_delong"]["ci_high"].as_f64().unwrap(),
    );
    let test = evaluation::paired_delong(&y, &a, &b).unwrap();
    close(
        test.p_value,
        legacy["paired_delong"]["p_value"].as_f64().unwrap(),
    );
    let hl = evaluation::hosmer_lemeshow(&y, &a, 4).unwrap();
    close(
        hl.statistic,
        legacy["hosmer_lemeshow"]["statistic"].as_f64().unwrap(),
    );
    close(
        hl.p_value,
        legacy["hosmer_lemeshow"]["p_value"].as_f64().unwrap(),
    );
    assert_eq!(
        hl.table.iter().map(|row| row.n).collect::<Vec<_>>(),
        legacy["hosmer_lemeshow"]["table"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["n"].as_f64().unwrap() as u64)
            .collect::<Vec<_>>()
    );
}
