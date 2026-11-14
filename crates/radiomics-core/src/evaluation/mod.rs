//! Native evaluation. Metric rows represent the caller's explicit evaluation unit.
use crate::contracts::Task;
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::models::{ModelPrediction, ModelTarget};
mod bootstrap;
mod curves;
mod inference;
mod reports;
pub use bootstrap::{
    bootstrap_auc_explicit, bootstrap_model_score_explicit, bootstrap_multiclass_seeded,
    bootstrap_regression_seeded, BootstrapInterval,
};
pub use curves::{
    calibration_curve, precision_recall_curve, roc_curve, CalibrationCurve, CalibrationStrategy,
    PrecisionRecallCurve, RocCurve,
};
pub use inference::{
    decision_curve, delong_interval, hosmer_lemeshow, paired_delong, AucInterval, DecisionCurve,
    HosmerLemeshow, HosmerLemeshowBin, PairedDelong,
};
pub use reports::{
    binary_metrics, classification_metrics, regression_metrics, BinaryMetrics, ClassMetrics,
    ClassificationMetrics, ConfusionMatrix, MetricAverage, RegressionMetrics,
};
#[cfg(test)]
mod golden_tests;

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn kernel(error: crate::InputError) -> RadiomicsError {
    invalid(error.to_string())
}

/// Enforce task, ordered class names, complete targets and valid probabilities
/// before any metric is computed. Classification requires every declared class.
pub fn validate_prediction(
    task: Task,
    target: &ModelTarget,
    prediction: &ModelPrediction,
) -> RadResult<()> {
    match (task, target, prediction) {
        (
            Task::Regression,
            ModelTarget::Regression { values: y },
            ModelPrediction::Regression { values: p },
        ) => {
            target.validate(p.len())?;
            if y.is_empty() || p.iter().any(|v| !v.is_finite()) {
                return Err(invalid(
                    "Regression evaluation requires nonempty finite target and prediction vectors.",
                ));
            }
        }
        (
            Task::Binary | Task::Multiclass,
            ModelTarget::Classification { classes, labels },
            ModelPrediction::Classification {
                classes: predicted_classes,
                probabilities,
            },
        ) => {
            target.validate(probabilities.rows)?;
            probabilities.validate()?;
            if (task == Task::Binary && classes.len() != 2)
                || (task == Task::Multiclass && classes.len() < 3)
                || classes != predicted_classes
                || probabilities.cols != classes.len()
                || labels.is_empty()
                || probabilities.missing.iter().any(|&m| m)
                || probabilities
                    .values
                    .iter()
                    .any(|v| !(0.0..=1.0).contains(v))
                || probabilities
                    .values
                    .chunks(probabilities.cols)
                    .any(|row| (row.iter().sum::<f64>() - 1.0).abs() > 1e-10)
            {
                return Err(invalid("Classification prediction task, class order, dimensions or probability sums disagree with the target."));
            }
        }
        _ => {
            return Err(invalid(
                "Evaluation task and target/prediction types disagree.",
            ))
        }
    }
    Ok(())
}

/// The sole RMSE implementation, shared by model comparison and regression reports.
pub fn rmse(target: &[f64], prediction: &[f64]) -> RadResult<f64> {
    if target.is_empty()
        || target.len() != prediction.len()
        || target.iter().chain(prediction).any(|v| !v.is_finite())
    {
        return Err(invalid(
            "RMSE requires equal nonempty finite target and prediction vectors.",
        ));
    }
    let value = (target
        .iter()
        .zip(prediction)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / target.len() as f64)
        .sqrt();
    if !value.is_finite() {
        return Err(invalid(
            "RMSE is not representable for these values; check target and prediction scale.",
        ));
    }
    Ok(value)
}

/// Return the actual comparison metric, not a sign-adjusted optimization score:
/// binary ROC AUC, multiclass macro OvR ROC AUC, or regression RMSE.
/// The training workflow decides whether to maximize or minimize it.
pub fn model_score(
    task: Task,
    target: &ModelTarget,
    prediction: &ModelPrediction,
) -> RadResult<f64> {
    validate_prediction(task, target, prediction)?;
    match (task, target, prediction) {
        (
            Task::Regression,
            ModelTarget::Regression { values: y },
            ModelPrediction::Regression { values: p },
        ) => rmse(y, p),
        (
            Task::Binary | Task::Multiclass,
            ModelTarget::Classification { classes, labels },
            ModelPrediction::Classification { probabilities, .. },
        ) => {
            let positive_classes: Vec<_> = if task == Task::Binary {
                vec![1]
            } else {
                (0..classes.len()).collect()
            };
            let mut sum = 0.0;
            for &class in &positive_classes {
                let y: Vec<_> = labels
                    .iter()
                    .map(|&label| u8::from(label == class))
                    .collect();
                let p: Vec<_> = probabilities
                    .values
                    .chunks(probabilities.cols)
                    .map(|row| row[class])
                    .collect();
                sum += crate::auc(&y, &p).map_err(kernel)?;
            }
            Ok(sum / positive_classes.len() as f64)
        }
        _ => Err(invalid("Evaluation task resolution failed.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::NumericBuffer;
    fn probability(rows: usize, classes: Vec<String>, values: Vec<f64>) -> ModelPrediction {
        let cols = classes.len();
        let missing = vec![false; values.len()];
        ModelPrediction::Classification {
            classes,
            probabilities: NumericBuffer {
                rows,
                cols,
                values,
                missing,
            },
        }
    }
    #[test]
    fn comparison_scores_preserve_task_metric_direction_and_class_order() {
        let classes = vec!["negative".into(), "positive".into()];
        let y = ModelTarget::Classification {
            classes: classes.clone(),
            labels: vec![0, 0, 1, 1],
        };
        let p = probability(4, classes, vec![0.9, 0.1, 0.6, 0.4, 0.65, 0.35, 0.2, 0.8]);
        assert_eq!(model_score(Task::Binary, &y, &p).unwrap(), 0.75);
        let y = ModelTarget::Regression {
            values: vec![1., 2., 3.],
        };
        let p = ModelPrediction::Regression {
            values: vec![1., 3., 2.],
        };
        assert_eq!(
            model_score(Task::Regression, &y, &p).unwrap(),
            (2f64 / 3.).sqrt()
        );
        let classes = vec!["A".into(), "B".into(), "C".into()];
        let y = ModelTarget::Classification {
            classes: classes.clone(),
            labels: vec![0, 1, 2],
        };
        let p = probability(
            3,
            classes,
            vec![0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8],
        );
        assert_eq!(model_score(Task::Multiclass, &y, &p).unwrap(), 1.0);
    }
    #[test]
    fn score_rejects_absent_classes_reordered_names_and_invalid_probabilities() {
        let classes = vec!["A".into(), "B".into(), "C".into()];
        let y = ModelTarget::Classification {
            classes: classes.clone(),
            labels: vec![0, 1, 2],
        };
        assert!(model_score(
            Task::Multiclass,
            &y,
            &probability(
                3,
                vec!["B".into(), "A".into(), "C".into()],
                vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]
            )
        )
        .is_err());
        let missing = ModelTarget::Classification {
            classes: classes.clone(),
            labels: vec![0, 0, 1],
        };
        assert!(model_score(
            Task::Multiclass,
            &missing,
            &probability(3, classes.clone(), vec![1., 0., 0., 0., 1., 0., 0., 0., 1.])
        )
        .is_err());
        assert!(model_score(
            Task::Multiclass,
            &y,
            &probability(3, classes, vec![0.8, 0.8, 0., 0., 1., 0., 0., 0., 1.])
        )
        .is_err());
        assert!(model_score(
            Task::Regression,
            &y,
            &ModelPrediction::Regression {
                values: vec![0.; 3]
            }
        )
        .is_err());
    }
    #[test]
    fn model_comparison_scores_match_every_frozen_inner_and_outer_patient_metric() {
        use serde_json::Value;
        fn visit(value: &Value, task: Task, classes: &[String], count: &mut usize) {
            if let Some(object) = value.as_object() {
                if let Some(expected) = object.get("score").and_then(Value::as_f64) {
                    if let Some(rows) = object
                        .get("patient_predictions")
                        .or_else(|| object.get("validation_patient_predictions"))
                        .and_then(Value::as_array)
                    {
                        let (target, prediction) = if task == Task::Regression {
                            (
                                ModelTarget::Regression {
                                    values: rows.iter().map(|r| r["y"].as_f64().unwrap()).collect(),
                                },
                                ModelPrediction::Regression {
                                    values: rows
                                        .iter()
                                        .map(|r| r["value"].as_f64().unwrap())
                                        .collect(),
                                },
                            )
                        } else {
                            (
                                ModelTarget::Classification {
                                    classes: classes.to_vec(),
                                    labels: rows
                                        .iter()
                                        .map(|r| r["y"].as_u64().unwrap() as usize)
                                        .collect(),
                                },
                                probability(
                                    rows.len(),
                                    classes.to_vec(),
                                    rows.iter()
                                        .flat_map(|r| {
                                            r["probabilities"]
                                                .as_array()
                                                .unwrap()
                                                .iter()
                                                .map(|v| v.as_f64().unwrap())
                                        })
                                        .collect(),
                                ),
                            )
                        };
                        let actual = model_score(task, &target, &prediction).unwrap();
                        assert!((actual - expected).abs() <= 1e-10 + 1e-8 * expected.abs());
                        *count += 1;
                    }
                }
                for child in object.values() {
                    visit(child, task, classes, count);
                }
            } else if let Some(values) = value.as_array() {
                for child in values {
                    visit(child, task, classes, count);
                }
            }
        }
        let reference: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/workflow_reference.json"
        )))
        .unwrap();
        let input: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/model_inputs.json"
        )))
        .unwrap();
        for (name, task) in [
            ("binary", Task::Binary),
            ("multiclass", Task::Multiclass),
            ("regression", Task::Regression),
        ] {
            let classes: Vec<_> = input["datasets"][name]["classes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let mut count = 0;
            visit(&reference["datasets"][name], task, &classes, &mut count);
            assert!(count >= 12);
        }
    }
}
