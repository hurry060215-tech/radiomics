//! Shared native tests and distribution operations; reference definitions are frozen.
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, ChiSquared, ContinuousCDF, Normal};

fn invalid(message: &str) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn finite(values: &[f64], minimum: usize) -> RadResult<()> {
    if values.len() < minimum || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "Insufficient or nonfinite observations for the requested statistic.",
        ));
    }
    Ok(())
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
}

pub fn population_mean_variance(values: &[f64]) -> RadResult<(f64, f64)> {
    finite(values, 1)?;
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    if !mean.is_finite() || !variance.is_finite() {
        return Err(invalid(
            "Mean or variance overflows; rescale input explicitly.",
        ));
    }
    Ok((mean, variance))
}
pub fn quantile_linear(values: &[f64], probability: f64) -> RadResult<f64> {
    finite(values, 1)?;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(invalid("Quantile probability must lie in [0,1]."));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    Ok(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64))
}
fn ranks(values: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.; values.len()];
    let mut ties = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.;
        for &i in &order[start..end] {
            ranks[i] = rank;
        }
        ties.push(end - start);
        start = end;
    }
    (ranks, ties)
}

/// SciPy 1.13.1's two-sided auto rule: exact for min(n)<=8 without ties;
/// otherwise normal approximation with tie and continuity corrections.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> RadResult<TestResult> {
    finite(a, 1)?;
    finite(b, 1)?;
    let mut values = a.to_vec();
    values.extend_from_slice(b);
    let (r, ties) = ranks(&values);
    let (n1, n2) = (a.len(), b.len());
    let n = n1 + n2;
    let pairs = n1
        .checked_mul(n2)
        .ok_or_else(|| invalid("Mann–Whitney dimensions overflow."))?;
    let u1 = r[..n1].iter().sum::<f64>() - (n1 as f64) * (n1 + 1) as f64 / 2.;
    let u2 = pairs as f64 - u1;
    let p = if n1.min(n2) <= 8 && ties.iter().all(|&t| t == 1) {
        // Gaussian-binomial generating polynomial. Division by (1-x^i)
        // uses a forward recurrence; multiplication by its numerator goes backward.
        let m = n1.min(n2);
        let larger = n1.max(n2);
        let upto = u1.min(u2).round() as usize;
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(upto + 1)
            .map_err(|_| invalid("Exact Mann–Whitney allocation exceeds available memory."))?;
        counts.resize(upto + 1, 0.);
        counts[0] = 1.;
        for i in 1..=m {
            for u in i..=upto {
                counts[u] += counts[u - i];
            }
            let exponent = larger + i;
            if exponent <= upto {
                for u in (exponent..=upto).rev() {
                    counts[u] -= counts[u - exponent];
                }
            }
        }
        let mut combinations = 1.;
        for i in 1..=m {
            combinations *= ((larger + i) as f64) / (i as f64);
        }
        (2. * counts.iter().sum::<f64>() / combinations).min(1.)
    } else {
        let n_float = n as f64;
        let correction = ties
            .iter()
            .map(|&t| (t as f64).powi(3) - t as f64)
            .sum::<f64>();
        let variance = (n1 as f64 * n2 as f64 / 12.)
            * (n_float + 1. - correction / (n_float * (n_float - 1.)));
        if variance == 0. {
            1.
        } else {
            let z = (u1.max(u2) - pairs as f64 / 2. - 0.5) / variance.sqrt();
            (2. * Normal::new(0., 1.).unwrap().sf(z)).min(1.)
        }
    };
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(invalid(
            "Mann–Whitney probability is numerically undefined.",
        ));
    }
    Ok(TestResult {
        statistic: u1,
        p_value: p,
    })
}

/// Kruskal–Wallis H with average tied ranks and chi-square survival probability.
pub fn kruskal_wallis(groups: &[Vec<f64>]) -> RadResult<TestResult> {
    if groups.len() < 2 {
        return Err(invalid("Kruskal–Wallis requires at least two groups."));
    }
    for group in groups {
        finite(group, 1)?;
    }
    let all: Vec<_> = groups.iter().flatten().copied().collect();
    let (ranks, ties) = ranks(&all);
    let n = all.len() as f64;
    let tie_correction = 1.
        - ties
            .iter()
            .map(|&t| (t as f64).powi(3) - t as f64)
            .sum::<f64>()
            / (n.powi(3) - n);
    if tie_correction <= 0. {
        return Err(invalid(
            "Kruskal–Wallis is undefined when every observation is identical.",
        ));
    }
    let mut position = 0;
    let mut sum = 0.;
    for group in groups {
        let rank_sum = ranks[position..position + group.len()].iter().sum::<f64>();
        sum += rank_sum.powi(2) / group.len() as f64;
        position += group.len();
    }
    let statistic = ((12. * sum / (n * (n + 1.)) - 3. * (n + 1.)) / tie_correction).max(0.);
    let distribution = ChiSquared::new((groups.len() - 1) as f64)
        .map_err(|_| invalid("Invalid chi-square degrees of freedom."))?;
    Ok(TestResult {
        statistic,
        p_value: distribution.sf(statistic),
    })
}

/// Two-sided Pearson correlation test; Beta distribution on [-1,1], n=2 special case.
pub fn pearson(x: &[f64], y: &[f64]) -> RadResult<TestResult> {
    finite(x, 2)?;
    finite(y, 2)?;
    if x.len() != y.len() {
        return Err(invalid("Correlation inputs have different lengths."));
    }
    let (mx, vx) = population_mean_variance(x)?;
    let (my, vy) = population_mean_variance(y)?;
    if vx == 0. || vy == 0. {
        return Err(invalid("Correlation is undefined for a constant vector."));
    }
    let covariance = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / x.len() as f64;
    let statistic = (covariance / (vx.sqrt() * vy.sqrt())).clamp(-1., 1.);
    let p_value = if x.len() == 2 {
        1.
    } else {
        let shape = x.len() as f64 / 2. - 1.;
        2. * Beta::new(shape, shape)
            .map_err(|_| invalid("Invalid Pearson null distribution."))?
            .cdf((1. - statistic.abs()) / 2.)
    };
    if !statistic.is_finite() || !p_value.is_finite() {
        return Err(invalid("Correlation computation overflowed."));
    }
    Ok(TestResult {
        statistic,
        p_value: p_value.clamp(0., 1.),
    })
}

/// Benjamini–Hochberg adjusted p-values returned in original feature order.
pub fn benjamini_hochberg(pvalues: &[f64]) -> RadResult<Vec<f64>> {
    if pvalues
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err(invalid("BH requires finite probabilities in [0,1]."));
    }
    let mut order: Vec<_> = (0..pvalues.len()).collect();
    order.sort_by(|&a, &b| pvalues[a].total_cmp(&pvalues[b]).then(a.cmp(&b)));
    let mut adjusted = vec![0.; pvalues.len()];
    let mut minimum = 1f64;
    for rank in (0..order.len()).rev() {
        let i = order[rank];
        minimum = minimum.min(pvalues[i] * pvalues.len() as f64 / (rank + 1) as f64);
        adjusted[i] = minimum;
    }
    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-10 + 1e-8 * b.abs(), "{a} != {b}");
    }
    #[test]
    fn independent_frozen_hand_statistics() {
        let mw = mann_whitney_u(&[1., 2., 3.], &[4., 5., 6.]).unwrap();
        assert_eq!(mw.statistic, 0.);
        close(mw.p_value, 0.1);
        let kw = kruskal_wallis(&[vec![1., 2.], vec![4., 4.], vec![5., 6.]]).unwrap();
        close(kw.statistic, 80. / 17.);
        close(kw.p_value, (-40f64 / 17.).exp());
        let actual = benjamini_hochberg(&[0.01, 0.04, 0.03, 0.002, 0.9, 0.04]).unwrap();
        for (a, b) in actual
            .into_iter()
            .zip([0.03, 0.048, 0.048, 0.012, 0.9, 0.048])
        {
            close(a, b);
        }
        close(
            quantile_linear(&[1., 2., 4., 4., 5., 6.], 0.25).unwrap(),
            2.5,
        );
    }
    #[test]
    fn matches_frozen_three_task_univariate_references() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../golden/python_golden_v1/table_reference.json"
        ))
        .unwrap();
        let input: serde_json::Value = serde_json::from_str(include_str!(
            "../../../golden/python_golden_v1/table_inputs.json"
        ))
        .unwrap();
        let rows = reference["numeric"]["imputed_training"].as_array().unwrap();
        for task in ["binary", "multiclass", "regression"] {
            let y: Vec<_> = input["targets"][task]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let expected = &reference["selection"]["univariate"][task];
            for col in 0..4 {
                let x: Vec<_> = rows.iter().map(|r| r[col].as_f64().unwrap()).collect();
                let test = match task {
                    "binary" => mann_whitney_u(
                        &x.iter()
                            .zip(&y)
                            .filter(|(_, y)| **y == 0.)
                            .map(|(x, _)| *x)
                            .collect::<Vec<_>>(),
                        &x.iter()
                            .zip(&y)
                            .filter(|(_, y)| **y == 1.)
                            .map(|(x, _)| *x)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                    "multiclass" => kruskal_wallis(
                        &(0..3)
                            .map(|k| {
                                x.iter()
                                    .zip(&y)
                                    .filter(|(_, y)| **y == k as f64)
                                    .map(|(x, _)| *x)
                                    .collect()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                    _ => pearson(&x, &y).unwrap(),
                };
                close(test.statistic, expected["statistic"][col].as_f64().unwrap());
                close(test.p_value, expected["p_values"][col].as_f64().unwrap());
            }
        }
    }
}
