use super::*;
use crate::contracts::NumericBuffer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfusionMatrix {
    pub size: usize,
    pub counts: Vec<u64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassMetrics {
    pub class_index: usize,
    pub support: u64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub specificity: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricAverage {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}
fn ratio_zero(n: u64, d: u64) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}
fn ratio_nan(n: u64, d: u64) -> f64 {
    if d == 0 {
        f64::NAN
    } else {
        n as f64 / d as f64
    }
}
fn measures(
    confusion: &ConfusionMatrix,
) -> (
    Vec<ClassMetrics>,
    MetricAverage,
    MetricAverage,
    MetricAverage,
    f64,
) {
    let k = confusion.size;
    let n = confusion.counts.iter().sum::<u64>();
    let mut classes = Vec::with_capacity(k);
    let mut total_tp = 0;
    let mut total_fp = 0;
    let mut total_fn = 0;
    for c in 0..k {
        let tp = confusion.counts[c * k + c];
        let support = confusion.counts[c * k..(c + 1) * k].iter().sum::<u64>();
        let predicted = (0..k).map(|r| confusion.counts[r * k + c]).sum::<u64>();
        let fp = predicted - tp;
        let fn_count = support - tp;
        let tn = n - tp - fp - fn_count;
        total_tp += tp;
        total_fp += fp;
        total_fn += fn_count;
        classes.push(ClassMetrics {
            class_index: c,
            support,
            precision: ratio_zero(tp, predicted),
            recall: ratio_zero(tp, support),
            f1: ratio_zero(2 * tp, 2 * tp + fp + fn_count),
            specificity: ratio_zero(tn, tn + fp),
        });
    }
    let mean = MetricAverage {
        precision: classes.iter().map(|c| c.precision).sum::<f64>() / k as f64,
        recall: classes.iter().map(|c| c.recall).sum::<f64>() / k as f64,
        f1: classes.iter().map(|c| c.f1).sum::<f64>() / k as f64,
    };
    let weighted = MetricAverage {
        precision: classes
            .iter()
            .map(|c| c.precision * c.support as f64 / n as f64)
            .sum(),
        recall: classes
            .iter()
            .map(|c| c.recall * c.support as f64 / n as f64)
            .sum(),
        f1: classes
            .iter()
            .map(|c| c.f1 * c.support as f64 / n as f64)
            .sum(),
    };
    let micro = MetricAverage {
        precision: ratio_zero(total_tp, total_tp + total_fp),
        recall: ratio_zero(total_tp, total_tp + total_fn),
        f1: ratio_zero(2 * total_tp, 2 * total_tp + total_fp + total_fn),
    };
    (classes, mean, micro, weighted, total_tp as f64 / n as f64)
}
fn log_loss(labels: &[usize], probabilities: &NumericBuffer) -> f64 {
    labels
        .iter()
        .zip(probabilities.values.chunks(probabilities.cols))
        .map(|(&label, row)| {
            let sum = row
                .iter()
                .map(|&p| p.clamp(1e-15, 1.0 - 1e-15))
                .sum::<f64>();
            -(row[label].clamp(1e-15, 1.0 - 1e-15) / sum).ln()
        })
        .sum::<f64>()
        / labels.len() as f64
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinaryMetrics {
    pub n: usize,
    pub positive_n: u64,
    pub threshold: f64,
    pub accuracy: f64,
    pub auc: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    #[serde(with = "crate::statistics::float_state")]
    pub ppv: f64,
    #[serde(with = "crate::statistics::float_state")]
    pub npv: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub brier: f64,
    pub log_loss_eps1e15: f64,
    pub average_precision: f64,
    pub pr_auc_trapezoidal: f64,
    pub confusion: ConfusionMatrix,
    pub per_class: Vec<ClassMetrics>,
}
pub fn binary_metrics(y: &[u8], scores: &[f64], threshold: f64) -> RadResult<BinaryMetrics> {
    let summary = crate::binary_summary(y, scores, threshold).map_err(kernel)?;
    let confusion = ConfusionMatrix {
        size: 2,
        counts: vec![summary.tn, summary.fp, summary.fn_count, summary.tp],
    };
    let (per_class, _, _, _, accuracy) = measures(&confusion);
    let labels: Vec<_> = y.iter().map(|&v| v as usize).collect();
    let values: Vec<_> = scores.iter().flat_map(|&p| [1.0 - p, p]).collect();
    let probabilities = NumericBuffer {
        rows: y.len(),
        cols: 2,
        missing: vec![false; values.len()],
        values,
    };
    let pr = precision_recall_curve(y, scores)?;
    Ok(BinaryMetrics {
        n: y.len(),
        positive_n: summary.tp + summary.fn_count,
        threshold,
        accuracy,
        auc: summary.auc,
        sensitivity: ratio_nan(summary.tp, summary.tp + summary.fn_count),
        specificity: ratio_nan(summary.tn, summary.tn + summary.fp),
        ppv: ratio_nan(summary.tp, summary.tp + summary.fp),
        npv: ratio_nan(summary.tn, summary.tn + summary.fn_count),
        precision: per_class[1].precision,
        recall: per_class[1].recall,
        f1: per_class[1].f1,
        brier: summary.brier,
        log_loss_eps1e15: log_loss(&labels, &probabilities),
        average_precision: pr.average_precision,
        pr_auc_trapezoidal: pr.trapezoidal_auc,
        confusion,
        per_class,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationMetrics {
    pub n: usize,
    pub classes: Vec<String>,
    pub accuracy: f64,
    pub confusion: ConfusionMatrix,
    pub per_class: Vec<ClassMetrics>,
    pub macro_average: MetricAverage,
    pub micro_average: MetricAverage,
    pub weighted_average: MetricAverage,
    pub ovr: Vec<BinaryMetrics>,
    pub macro_ovr_auc: f64,
    pub weighted_ovr_auc: f64,
    pub micro_ovr_auc: f64,
    pub macro_average_precision: f64,
    pub weighted_average_precision: f64,
    pub micro_average_precision: f64,
    pub log_loss_eps1e15: f64,
    pub multiclass_brier_sum: f64,
}
pub fn classification_metrics(
    task: Task,
    target: &ModelTarget,
    prediction: &ModelPrediction,
    threshold: f64,
) -> RadResult<ClassificationMetrics> {
    validate_prediction(task, target, prediction)?;
    let (
        ModelTarget::Classification { classes, labels },
        ModelPrediction::Classification { probabilities, .. },
    ) = (target, prediction)
    else {
        return Err(invalid(
            "Classification metrics require a classification task.",
        ));
    };
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(invalid("Binary classification threshold must be in [0,1]."));
    }
    let k = classes.len();
    let mut counts = vec![0u64; k * k];
    for (&label, row) in labels.iter().zip(probabilities.values.chunks(k)) {
        let predicted = if task == Task::Binary {
            usize::from(row[1] >= threshold)
        } else {
            row.iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |(best, value), (j, &p)| {
                    if p > value {
                        (j, p)
                    } else {
                        (best, value)
                    }
                })
                .0
        };
        counts[label * k + predicted] += 1;
    }
    let confusion = ConfusionMatrix { size: k, counts };
    let (per_class, macro_average, micro_average, weighted_average, accuracy) =
        measures(&confusion);
    let mut ovr = Vec::with_capacity(k);
    let mut flat_y = Vec::with_capacity(labels.len() * k);
    let mut brier = 0.0;
    for (&label, row) in labels.iter().zip(probabilities.values.chunks(k)) {
        for (c, &p) in row.iter().enumerate() {
            let actual = u8::from(c == label);
            flat_y.push(actual);
            brier += (p - f64::from(actual)).powi(2) / labels.len() as f64;
        }
    }
    for c in 0..k {
        let y: Vec<_> = labels.iter().map(|&label| u8::from(label == c)).collect();
        let p: Vec<_> = probabilities.values.chunks(k).map(|row| row[c]).collect();
        ovr.push(binary_metrics(&y, &p, threshold)?);
    }
    let macro_ovr_auc = ovr.iter().map(|r| r.auc).sum::<f64>() / k as f64;
    let macro_average_precision = ovr.iter().map(|r| r.average_precision).sum::<f64>() / k as f64;
    let weighted_ovr_auc = ovr
        .iter()
        .zip(&per_class)
        .map(|(r, c)| r.auc * c.support as f64 / labels.len() as f64)
        .sum();
    let weighted_average_precision = ovr
        .iter()
        .zip(&per_class)
        .map(|(r, c)| r.average_precision * c.support as f64 / labels.len() as f64)
        .sum();
    let micro_ovr_auc = crate::auc(&flat_y, &probabilities.values).map_err(kernel)?;
    let micro_average_precision =
        precision_recall_curve(&flat_y, &probabilities.values)?.average_precision;
    Ok(ClassificationMetrics {
        n: labels.len(),
        classes: classes.clone(),
        accuracy,
        confusion,
        per_class,
        macro_average,
        micro_average,
        weighted_average,
        ovr,
        macro_ovr_auc,
        weighted_ovr_auc,
        micro_ovr_auc,
        macro_average_precision,
        weighted_average_precision,
        micro_average_precision,
        log_loss_eps1e15: log_loss(labels, probabilities),
        multiclass_brier_sum: brier,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionMetrics {
    pub n: usize,
    pub mae: f64,
    pub rmse: f64,
    #[serde(with = "crate::statistics::float_state")]
    pub r2: f64,
}
pub fn regression_metrics(y: &[f64], predictions: &[f64]) -> RadResult<RegressionMetrics> {
    let error = rmse(y, predictions)?;
    let n = y.len() as f64;
    let mae = y
        .iter()
        .zip(predictions)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n;
    let mean = y.iter().sum::<f64>() / n;
    let numerator = y
        .iter()
        .zip(predictions)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let denominator = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    // Actual frozen sklearn 1.0.2 behavior: constant-perfect is one,
    // constant-imperfect is zero; a single observation is undefined (NaN).
    let r2 = if y.len() < 2 {
        f64::NAN
    } else if denominator == 0.0 {
        if numerator == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - numerator / denominator
    };
    Ok(RegressionMetrics {
        n: y.len(),
        mae,
        rmse: error,
        r2,
    })
}
