use super::*;
use serde::{Deserialize, Serialize};

pub(super) struct ThresholdCounts {
    pub thresholds: Vec<f64>,
    pub tp: Vec<u64>,
    pub fp: Vec<u64>,
    pub positive: u64,
    pub negative: u64,
}
pub(super) fn threshold_counts(y: &[u8], scores: &[f64]) -> RadResult<ThresholdCounts> {
    crate::auc(y, scores).map_err(kernel)?;
    let mut order: Vec<_> = (0..y.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let (mut true_positive, mut false_positive) = (0u64, 0u64);
    let mut thresholds = Vec::new();
    let mut tp = Vec::new();
    let mut fp = Vec::new();
    let mut i = 0;
    while i < order.len() {
        let score = scores[order[i]];
        let mut end = i + 1;
        while end < order.len() && scores[order[end]] == score {
            end += 1;
        }
        for &j in &order[i..end] {
            if y[j] == 1 {
                true_positive += 1;
            } else {
                false_positive += 1;
            }
        }
        thresholds.push(score);
        tp.push(true_positive);
        fp.push(false_positive);
        i = end;
    }
    Ok(ThresholdCounts {
        thresholds,
        tp,
        fp,
        positive: true_positive,
        negative: false_positive,
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RocCurve {
    pub false_positive_rate: Vec<f64>,
    pub true_positive_rate: Vec<f64>,
    pub thresholds: Vec<f64>,
}
/// Frozen sklearn 1.0.2 full ROC: the first threshold is max(score)+1.
pub fn roc_curve(y: &[u8], scores: &[f64]) -> RadResult<RocCurve> {
    let counts = threshold_counts(y, scores)?;
    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let mut thresholds = vec![counts.thresholds[0] + 1.0];
    fpr.extend(counts.fp.iter().map(|&v| v as f64 / counts.negative as f64));
    tpr.extend(counts.tp.iter().map(|&v| v as f64 / counts.positive as f64));
    thresholds.extend(counts.thresholds);
    Ok(RocCurve {
        false_positive_rate: fpr,
        true_positive_rate: tpr,
        thresholds,
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionRecallCurve {
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub average_precision: f64,
    pub trapezoidal_auc: f64,
}
/// Frozen sklearn 1.0.2 reverses only through the first full-recall threshold,
/// then appends precision=1, recall=0. AP and trapezoidal PR area differ.
pub fn precision_recall_curve(y: &[u8], scores: &[f64]) -> RadResult<PrecisionRecallCurve> {
    let counts = threshold_counts(y, scores)?;
    let last = counts
        .tp
        .iter()
        .position(|&n| n == counts.positive)
        .unwrap();
    let mut precision = Vec::new();
    let mut recall = Vec::new();
    let mut thresholds = Vec::new();
    for j in (0..=last).rev() {
        precision.push(counts.tp[j] as f64 / (counts.tp[j] + counts.fp[j]) as f64);
        recall.push(counts.tp[j] as f64 / counts.positive as f64);
        thresholds.push(counts.thresholds[j]);
    }
    precision.push(1.0);
    recall.push(0.0);
    let average_precision = (0..precision.len() - 1)
        .map(|i| (recall[i] - recall[i + 1]) * precision[i])
        .sum();
    let trapezoidal_auc = (0..precision.len() - 1)
        .map(|i| (recall[i] - recall[i + 1]) * (precision[i] + precision[i + 1]) / 2.0)
        .sum();
    Ok(PrecisionRecallCurve {
        precision,
        recall,
        thresholds,
        average_precision,
        trapezoidal_auc,
    })
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStrategy {
    Uniform,
    Quantile,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationCurve {
    pub strategy: CalibrationStrategy,
    pub n_bins: usize,
    pub edges: Vec<f64>,
    pub bin_membership: Vec<usize>,
    pub bin_counts: Vec<u64>,
    pub nonempty_bins: Vec<usize>,
    pub fraction_positive: Vec<f64>,
    pub mean_predicted: Vec<f64>,
}
pub fn calibration_curve(
    y: &[u8],
    scores: &[f64],
    n_bins: usize,
    strategy: CalibrationStrategy,
) -> RadResult<CalibrationCurve> {
    crate::auc(y, scores).map_err(kernel)?;
    if n_bins == 0 {
        return Err(invalid(
            "Calibration requires a positive requested bin count.",
        ));
    }
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(
            n_bins
                .checked_add(1)
                .ok_or_else(|| invalid("Calibration edge count overflows."))?,
        )
        .map_err(|_| invalid("Calibration edge allocation failed."))?;
    for i in 0..=n_bins {
        let fraction = if i == n_bins {
            1.0
        } else {
            i as f64 * (1.0 / n_bins as f64)
        };
        edges.push(match strategy {
            CalibrationStrategy::Uniform => {
                if i == n_bins {
                    1.0 + 1e-8
                } else {
                    i as f64 * ((1.0 + 1e-8) / n_bins as f64)
                }
            }
            CalibrationStrategy::Quantile => {
                crate::statistics::quantile_linear(scores, (fraction * 100.0) / 100.0)?
            }
        });
    }
    if strategy == CalibrationStrategy::Quantile {
        *edges.last_mut().unwrap() += 1e-8;
    }
    let mut bin_counts = vec![0u64; n_bins];
    let mut positive = vec![0u64; n_bins];
    let mut sum = vec![0.0; n_bins];
    let mut membership = Vec::with_capacity(y.len());
    for (&label, &score) in y.iter().zip(scores) {
        let right = edges.partition_point(|&edge| edge <= score);
        if right == 0 || right > n_bins {
            return Err(invalid(
                "Calibration bin edges did not contain a probability.",
            ));
        }
        let bin = right - 1;
        membership.push(bin);
        bin_counts[bin] += 1;
        positive[bin] += u64::from(label);
        sum[bin] += score;
    }
    let nonempty_bins: Vec<_> = (0..n_bins).filter(|&j| bin_counts[j] > 0).collect();
    let fraction_positive = nonempty_bins
        .iter()
        .map(|&j| positive[j] as f64 / bin_counts[j] as f64)
        .collect();
    let mean_predicted = nonempty_bins
        .iter()
        .map(|&j| sum[j] / bin_counts[j] as f64)
        .collect();
    Ok(CalibrationCurve {
        strategy,
        n_bins,
        edges,
        bin_membership: membership,
        bin_counts,
        nonempty_bins,
        fraction_positive,
        mean_predicted,
    })
}
