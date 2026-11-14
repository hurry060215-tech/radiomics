//! Typed single-measure ICC for complete balanced subject-by-reader matrices.
use crate::contracts::{NumericBuffer, QualityRecord};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::statistics::Distribution;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IccType {
    #[serde(rename = "ICC(2,1)")]
    Icc21,
    #[serde(rename = "ICC(3,1)")]
    Icc31,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IccResult {
    #[serde(rename = "type")]
    pub icc_type: IccType,
    pub n_subjects: usize,
    pub n_readers: usize,
    pub confidence: f64,
    #[serde(with = "crate::statistics::float_state")]
    pub estimate: f64,
    #[serde(with = "crate::statistics::float_vec_state")]
    pub confidence_interval: Vec<f64>,
    #[serde(with = "crate::statistics::float_state")]
    pub f_statistic: f64,
    pub df1: usize,
    pub df2: usize,
    #[serde(with = "crate::statistics::float_state")]
    pub p_value: f64,
    pub ms_subject: f64,
    pub ms_reader: f64,
    pub ms_error: f64,
    pub diagnostics: Vec<QualityRecord>,
}
fn invalid(message: &str) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn warning(code: &str, message: &str) -> QualityRecord {
    QualityRecord {
        code: code.into(),
        message: message.into(),
        severity: "warning".into(),
        sample_id: None,
    }
}
/// ICC(2,1): two-way random absolute agreement, single rating.
/// ICC(3,1): two-way fixed consistency, single rating.
/// No missing-value deletion, average-measure ICC or threshold-only interface.
pub fn icc(data: &NumericBuffer, kind: IccType, confidence: f64) -> RadResult<IccResult> {
    data.validate()?;
    if data.rows < 2 || data.cols < 2 || data.missing.iter().any(|&m| m) {
        return Err(invalid("ICC requires a complete balanced subject-by-reader matrix with at least two subjects and two readers; missing values are not imputed or omitted."));
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(invalid(
            "ICC confidence must be finite and strictly between zero and one.",
        ));
    }
    let (n, k) = (data.rows, data.cols);
    let (nf, kf) = (n as f64, k as f64);
    let df1 = n - 1;
    let df2 = df1
        .checked_mul(k - 1)
        .ok_or_else(|| invalid("ICC degrees of freedom overflow usize."))?;
    if data.values.iter().all(|&v| v == data.values[0]) {
        return Ok(IccResult {
            icc_type: kind,
            n_subjects: n,
            n_readers: k,
            confidence,
            estimate: f64::NAN,
            confidence_interval: vec![f64::NAN; 2],
            f_statistic: f64::NAN,
            df1,
            df2,
            p_value: f64::NAN,
            ms_subject: 0.0,
            ms_reader: 0.0,
            ms_error: 0.0,
            diagnostics: vec![warning(
                "constant_ratings",
                "All ratings are constant; ICC, F, p-value and confidence limits are undefined.",
            )],
        });
    }
    let grand = data.values.iter().sum::<f64>() / (nf * kf);
    let subject: Vec<_> = data
        .values
        .chunks(k)
        .map(|row| row.iter().sum::<f64>() / kf)
        .collect();
    let reader: Vec<_> = (0..k)
        .map(|j| (0..n).map(|i| data.values[i * k + j]).sum::<f64>() / nf)
        .collect();
    let ms_subject = kf * subject.iter().map(|v| (v - grand).powi(2)).sum::<f64>() / df1 as f64;
    let ms_reader = nf * reader.iter().map(|v| (v - grand).powi(2)).sum::<f64>() / (k - 1) as f64;
    let perfect = data
        .values
        .chunks(k)
        .all(|row| row.iter().all(|&v| v == row[0]));
    let ms_error = if perfect {
        0.0
    } else {
        (0..n)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .map(|(i, j)| (data.values[i * k + j] - subject[i] - reader[j] + grand).powi(2))
            .sum::<f64>()
            / df2 as f64
    };
    if [grand, ms_subject, ms_reader, ms_error]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(invalid(
            "ICC ANOVA arithmetic overflowed; rescale ratings explicitly.",
        ));
    }
    let estimate = match kind {
        IccType::Icc21 => {
            (ms_subject - ms_error)
                / (ms_subject + (kf - 1.0) * ms_error + kf * (ms_reader - ms_error) / nf)
        }
        IccType::Icc31 => (ms_subject - ms_error) / (ms_subject + (kf - 1.0) * ms_error),
    };
    let f_statistic = ms_subject / ms_error;
    let p_value = if f_statistic.is_nan() {
        f64::NAN
    } else {
        Distribution::F {
            dfn: df1 as f64,
            dfd: df2 as f64,
        }
        .sf(f_statistic)?
    };
    let q = 0.5 + confidence / 2.0;
    let confidence_interval = if ms_error == 0.0 {
        vec![f64::NAN; 2]
    } else {
        match kind {
            IccType::Icc31 => {
                let lower_f = f_statistic
                    / Distribution::F {
                        dfn: df1 as f64,
                        dfd: df2 as f64,
                    }
                    .quantile(q)?;
                let upper_f = f_statistic
                    * Distribution::F {
                        dfn: df2 as f64,
                        dfd: df1 as f64,
                    }
                    .quantile(q)?;
                vec![
                    (lower_f - 1.0) / (lower_f + kf - 1.0),
                    (upper_f - 1.0) / (upper_f + kf - 1.0),
                ]
            }
            IccType::Icc21 => {
                let fj = ms_reader / ms_error;
                let common = nf * (1.0 + (kf - 1.0) * estimate) - kf * estimate;
                let v = df2 as f64 * (kf * estimate * fj + common).powi(2)
                    / (df1 as f64 * kf.powi(2) * estimate.powi(2) * fj.powi(2) + common.powi(2));
                if !v.is_finite() || v <= 0.0 {
                    vec![f64::NAN; 2]
                } else {
                    let upper_f = Distribution::F {
                        dfn: df1 as f64,
                        dfd: v,
                    }
                    .quantile(q)?;
                    let lower_f = Distribution::F {
                        dfn: v,
                        dfd: df1 as f64,
                    }
                    .quantile(q)?;
                    let shared = kf * ms_reader + (kf * nf - kf - nf) * ms_error;
                    vec![
                        nf * (ms_subject - upper_f * ms_error)
                            / (upper_f * shared + nf * ms_subject),
                        nf * (lower_f * ms_subject - ms_error)
                            / (shared + nf * lower_f * ms_subject),
                    ]
                }
            }
        }
    };
    let mut diagnostics = Vec::new();
    if ms_error == 0.0 {
        diagnostics.push(warning("zero_residual_variance","Zero residual ANOVA variance; the frozen reference has an infinite/undefined F statistic and undefined confidence limits."));
    }
    if !estimate.is_finite()
        || confidence_interval.iter().any(|v| !v.is_finite())
        || !p_value.is_finite()
    {
        diagnostics.push(warning("undefined_icc_components","Undefined ICC components are retained as explicit NaN/Inf states rather than replaced by zero."));
    }
    Ok(IccResult {
        icc_type: kind,
        n_subjects: n,
        n_readers: k,
        confidence,
        estimate,
        confidence_interval,
        f_statistic,
        df1,
        df2,
        p_value,
        ms_subject,
        ms_reader,
        ms_error,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn number(value: &Value) -> f64 {
        value
            .as_f64()
            .unwrap_or_else(|| match value.as_str().unwrap() {
                "NaN" => f64::NAN,
                "Inf" => f64::INFINITY,
                "-Inf" => f64::NEG_INFINITY,
                _ => panic!("invalid reference scalar"),
            })
    }
    fn compare(actual: f64, expected: f64) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!(
                (actual - expected).abs() <= 1e-10 + 1e-8 * expected.abs(),
                "{actual} != {expected}"
            );
        }
    }
    #[test]
    fn typed_icc_and_unrounded_intervals_match_frozen_reference_including_degeneracy() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/icc_reference.json"
        )))
        .unwrap();
        for (name, case) in fixture["cases"].as_object().unwrap() {
            let rows = case["values_subject_by_reader"].as_array().unwrap();
            let cols = rows[0].as_array().unwrap().len();
            let mut values = Vec::new();
            let mut missing = Vec::new();
            for row in rows {
                for v in row.as_array().unwrap() {
                    values.push(v.as_f64().unwrap_or(0.0));
                    missing.push(v.is_null());
                }
            }
            let data = NumericBuffer {
                rows: rows.len(),
                cols,
                values,
                missing,
            };
            for (kind, label) in [(IccType::Icc21, "ICC(2,1)"), (IccType::Icc31, "ICC(3,1)")] {
                let output = icc(&data, kind, 0.95);
                if name == "missing" {
                    assert!(output.is_err());
                    continue;
                }
                let output = output.unwrap();
                let reference = &case["reference"]["value"][label];
                compare(output.estimate, number(&reference["estimate"]));
                compare(output.f_statistic, number(&reference["f_statistic"]));
                compare(output.p_value, number(&reference["p_value"]));
                for (a, b) in output
                    .confidence_interval
                    .iter()
                    .zip(reference["ci95"].as_array().unwrap())
                {
                    compare(*a, number(b));
                }
                if name == "constant" || name == "perfect_agreement" {
                    assert!(!output.diagnostics.is_empty());
                }
                let reloaded: IccResult =
                    serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
                compare(reloaded.estimate, output.estimate);
                compare(reloaded.f_statistic, output.f_statistic);
            }
        }
    }
}
