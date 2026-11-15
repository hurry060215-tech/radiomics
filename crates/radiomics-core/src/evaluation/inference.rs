use super::*;
use crate::statistics::Distribution;
use serde::{Deserialize, Serialize};

pub(super) fn confidence(value: f64) -> RadResult<()> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        Err(invalid(
            "Confidence must be finite and strictly between zero and one.",
        ))
    } else {
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AucInterval {
    pub auc: f64,
    pub variance: f64,
    pub confidence: f64,
    pub low: f64,
    pub high: f64,
}
pub fn delong_interval(y: &[u8], scores: &[f64], level: f64) -> RadResult<AucInterval> {
    confidence(level)?;
    let auc = crate::auc(y, scores).map_err(kernel)?;
    let (_, variance) = crate::delong_summary(y, scores).map_err(kernel)?;
    let delta = Distribution::Normal.quantile(0.5 + level / 2.0)? * variance.max(0.0).sqrt();
    Ok(AucInterval {
        auc,
        variance,
        confidence: level,
        low: (auc - delta).max(0.0),
        high: (auc + delta).min(1.0),
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedDelong {
    pub auc_a: f64,
    pub auc_b: f64,
    pub difference: f64,
    pub variance: f64,
    #[serde(with = "crate::statistics::float_state")]
    pub z: f64,
    pub p_value: f64,
}
pub fn paired_delong(y: &[u8], a: &[f64], b: &[f64]) -> RadResult<PairedDelong> {
    let (auc_a, auc_b, variance) = crate::paired_delong_summary(y, a, b).map_err(kernel)?;
    let difference = auc_a - auc_b;
    let z = if variance > 0.0 {
        difference / variance.sqrt()
    } else if difference == 0.0 {
        0.0
    } else {
        difference.signum() * f64::INFINITY
    };
    let p_value = 2.0 * Distribution::Normal.sf(z.abs())?;
    Ok(PairedDelong {
        auc_a,
        auc_b,
        difference,
        variance,
        z,
        p_value,
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionCurve {
    pub thresholds: Vec<f64>,
    pub model_net_benefit: Vec<f64>,
    pub treat_all: Vec<f64>,
    pub treat_none: Vec<f64>,
}
pub fn decision_curve(y: &[u8], scores: &[f64], thresholds: &[f64]) -> RadResult<DecisionCurve> {
    let model_net_benefit = crate::net_benefit(y, scores, thresholds).map_err(kernel)?;
    let prevalence = y.iter().map(|&v| f64::from(v)).sum::<f64>() / y.len() as f64;
    let treat_all = thresholds
        .iter()
        .map(|&t| prevalence - (1.0 - prevalence) * t / (1.0 - t))
        .collect();
    Ok(DecisionCurve {
        thresholds: thresholds.to_vec(),
        model_net_benefit,
        treat_all,
        treat_none: vec![0.0; thresholds.len()],
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HosmerLemeshowBin {
    pub lower: f64,
    pub upper: f64,
    pub include_lowest: bool,
    pub n: u64,
    pub observed: f64,
    pub expected: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HosmerLemeshow {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    pub requested_bins: usize,
    pub quantile_edges: Vec<f64>,
    pub bin_membership: Vec<usize>,
    pub table: Vec<HosmerLemeshowBin>,
}
/// pandas qcut-compatible numerical grouping: unique quantile edges, right
/// closed intervals, minimum included. Table bounds are raw numerical bounds;
/// presentation rounding of interval labels belongs to the language wrapper.
pub fn hosmer_lemeshow(y: &[u8], scores: &[f64], n_bins: usize) -> RadResult<HosmerLemeshow> {
    crate::auc(y, scores).map_err(kernel)?;
    if n_bins < 3 {
        return Err(invalid("H-L requires at least three requested bins."));
    }
    let count = n_bins
        .checked_add(1)
        .ok_or_else(|| invalid("H-L requested bin count overflows."))?;
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(count)
        .map_err(|_| invalid("H-L edge allocation failed."))?;
    for i in 0..=n_bins {
        let p = if i == n_bins {
            1.0
        } else {
            i as f64 * (1.0 / n_bins as f64)
        };
        edges.push(crate::statistics::quantile_linear(scores, p)?);
    }
    edges.dedup_by(|a, b| *a == *b);
    if edges.len() < 4 {
        return Err(invalid(
            "Fewer than three distinct probability bins; H-L is undefined.",
        ));
    }
    let mut counts = vec![0u64; edges.len() - 1];
    let mut observed = vec![0.0; counts.len()];
    let mut expected = vec![0.0; counts.len()];
    let mut membership = Vec::with_capacity(y.len());
    for (&label, &score) in y.iter().zip(scores) {
        let position = edges.partition_point(|&edge| edge < score);
        let position = if position == 0 { 1 } else { position };
        let bin = position - 1;
        if bin >= counts.len() {
            return Err(invalid("H-L quantile edges did not contain a score."));
        }
        membership.push(bin);
        counts[bin] += 1;
        observed[bin] += f64::from(label);
        expected[bin] += score;
    }
    let table: Vec<_> = (0..counts.len())
        .filter(|&j| counts[j] > 0)
        .map(|j| HosmerLemeshowBin {
            lower: edges[j],
            upper: edges[j + 1],
            include_lowest: j == 0,
            n: counts[j],
            observed: observed[j],
            expected: expected[j],
        })
        .collect();
    if table.len() < 3 {
        return Err(invalid(
            "Fewer than three nonempty probability bins; H-L is undefined.",
        ));
    }
    let n: Vec<_> = table.iter().map(|b| b.n).collect();
    let o: Vec<_> = table.iter().map(|b| b.observed).collect();
    let e: Vec<_> = table.iter().map(|b| b.expected).collect();
    let statistic = crate::hl_statistic(&n, &o, &e).map_err(kernel)?;
    let df = table.len() - 2;
    let p_value = Distribution::ChiSquare { df: df as f64 }.sf(statistic)?;
    Ok(HosmerLemeshow {
        statistic,
        df,
        p_value,
        requested_bins: n_bins,
        quantile_edges: edges,
        bin_membership: membership,
        table,
    })
}
