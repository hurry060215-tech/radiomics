//! Observed-data baseline tables; no imputation or cross-variable case deletion.
use crate::contracts::{unique_nonempty, NumericBuffer, QualityRecord, TabularData};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> RadiomicsError {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContinuousTest {
    #[default]
    Auto,
    MannWhitney,
    StudentT,
    Welch,
    KruskalWallis,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CategoricalTest {
    #[default]
    Auto,
    Fisher,
    ChiSquare,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct BaselineConfig {
    pub continuous_test: ContinuousTest,
    pub categorical_test: CategoricalTest,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineMethod {
    MannWhitney,
    StudentT,
    Welch,
    KruskalWallis,
    #[serde(rename = "fisher_2x2")]
    Fisher2x2,
    ChiSquareNoYates,
    InsufficientObservations,
    ConstantOrAbsentClass,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BaselineGroup {
    Continuous {
        class_index: usize,
        class_label: String,
        n: u64,
        missing: u64,
        #[serde(with = "crate::statistics::float_state")]
        mean: f64,
        #[serde(with = "crate::statistics::float_state")]
        sd_ddof1: f64,
        #[serde(with = "crate::statistics::float_state")]
        median: f64,
        #[serde(with = "crate::statistics::float_state")]
        q25: f64,
        #[serde(with = "crate::statistics::float_state")]
        q75: f64,
    },
    Categorical {
        class_index: usize,
        class_label: String,
        n: u64,
        missing: u64,
        counts: BTreeMap<String, u64>,
        proportions: BTreeMap<String, f64>,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountTable {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<u64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineRow {
    pub cohort: String,
    pub feature: String,
    pub groups: Vec<BaselineGroup>,
    pub method: BaselineMethod,
    #[serde(with = "crate::statistics::float_state")]
    pub statistic: f64,
    #[serde(with = "crate::statistics::float_state")]
    pub p_value: f64,
    pub categorical_levels: Vec<String>,
    pub contingency: Option<CountTable>,
    pub expected_counts: Option<NumericBuffer>,
    pub diagnostics: Vec<QualityRecord>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineTable {
    pub n_input: usize,
    pub classes: Vec<String>,
    pub cohorts: Vec<String>,
    pub features: Vec<String>,
    pub config: BaselineConfig,
    pub analysis_unit: String,
    pub outcome_missing_policy: String,
    pub rows: Vec<BaselineRow>,
    pub diagnostics: Vec<QualityRecord>,
}
enum Feature {
    Numeric(usize),
    Categorical(usize),
}
fn numeric_summary(
    values: &[f64],
    total: usize,
    index: usize,
    label: &str,
) -> RadResult<BaselineGroup> {
    let n = values.len();
    let (mean, sd, median, q25, q75) = if n == 0 {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let (mean, variance) = crate::statistics::population_mean_variance(values)?;
        (
            mean,
            if n < 2 {
                f64::NAN
            } else {
                (variance * n as f64 / (n - 1) as f64).sqrt()
            },
            crate::statistics::quantile_linear(values, 0.5)?,
            crate::statistics::quantile_linear(values, 0.25)?,
            crate::statistics::quantile_linear(values, 0.75)?,
        )
    };
    Ok(BaselineGroup::Continuous {
        class_index: index,
        class_label: label.into(),
        n: n as u64,
        missing: (total - n) as u64,
        mean,
        sd_ddof1: sd,
        median,
        q25,
        q75,
    })
}
fn categorical_summary(
    values: &[String],
    total: usize,
    index: usize,
    label: &str,
) -> BaselineGroup {
    let mut counts = BTreeMap::new();
    for value in values {
        *counts.entry(value.clone()).or_insert(0u64) += 1;
    }
    let proportions = counts
        .iter()
        .map(|(label, &count)| (label.clone(), count as f64 / values.len() as f64))
        .collect();
    BaselineGroup::Categorical {
        class_index: index,
        class_label: label.into(),
        n: values.len() as u64,
        missing: (total - values.len()) as u64,
        counts,
        proportions,
    }
}

/// Compare declared outcome classes within each explicit cohort. Unlike Logit
/// complete-case inference, any missing outcome is an error. Feature missingness
/// is handled separately per variable/group and remains visible in the table.
pub fn baseline_table(
    data: &TabularData,
    classes: &[String],
    labels: &[Option<usize>],
    features: &[String],
    cohorts: Option<&[String]>,
    config: &BaselineConfig,
) -> RadResult<BaselineTable> {
    data.validate()?;
    unique_nonempty(classes, "Baseline class labels")?;
    unique_nonempty(features, "Baseline features")?;
    if classes.len() < 2
        || features.is_empty()
        || labels.len() != data.rows
        || labels.iter().any(|v| v.is_none_or(|c| c >= classes.len()))
    {
        return Err(invalid("Baseline tables require at least two classes and a nonmissing declared outcome for every row; missing outcomes are not silently omitted."));
    }
    let labels: Vec<_> = labels.iter().map(|v| v.unwrap()).collect();
    if (0..classes.len()).any(|c| !labels.contains(&c)) {
        return Err(invalid(
            "Every declared outcome class must appear in the complete baseline input.",
        ));
    }
    if config.categorical_test == CategoricalTest::Fisher && classes.len() != 2 {
        return Err(invalid(
            "Explicit Fisher exact test requires a 2x2 table; it is not a global multiclass test.",
        ));
    }
    if classes.len() != 2
        && matches!(
            config.continuous_test,
            ContinuousTest::MannWhitney | ContinuousTest::StudentT | ContinuousTest::Welch
        )
    {
        return Err(invalid(
            "The requested continuous baseline test requires exactly two outcome classes.",
        ));
    }
    if cohorts.is_some_and(|c| c.len() != data.rows || c.iter().any(|v| v.trim().is_empty())) {
        return Err(invalid(
            "Cohort labels must be nonempty and align with every baseline row.",
        ));
    }
    let mut specs = Vec::with_capacity(features.len());
    for name in features {
        if let Some(j) = data.numeric_names.iter().position(|n| n == name) {
            specs.push(Feature::Numeric(j));
        } else if let Some(j) = data.categorical_names.iter().position(|n| n == name) {
            specs.push(Feature::Categorical(j));
        } else {
            return Err(RadiomicsError::new(
                ErrorCode::Alignment,
                format!("Missing baseline feature '{name}'."),
            ));
        }
    }
    let mut partitions: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..data.rows {
        partitions
            .entry(cohorts.map_or("all", |c| c[i].as_str()).into())
            .or_default()
            .push(i);
    }
    let mut rows = Vec::new();
    for (cohort, indices) in &partitions {
        let group_indices: Vec<Vec<_>> = (0..classes.len())
            .map(|c| {
                indices
                    .iter()
                    .copied()
                    .filter(|&i| labels[i] == c)
                    .collect()
            })
            .collect();
        for (name, spec) in features.iter().zip(&specs) {
            let mut row = BaselineRow {
                cohort: cohort.clone(),
                feature: name.clone(),
                groups: Vec::new(),
                method: BaselineMethod::InsufficientObservations,
                statistic: f64::NAN,
                p_value: f64::NAN,
                categorical_levels: Vec::new(),
                contingency: None,
                expected_counts: None,
                diagnostics: Vec::new(),
            };
            match spec {
                Feature::Numeric(j) => {
                    let observed: Vec<Vec<_>> = group_indices
                        .iter()
                        .map(|indices| {
                            indices
                                .iter()
                                .filter_map(|&i| {
                                    let index = i * data.numeric.cols + j;
                                    (!data.numeric.missing[index])
                                        .then_some(data.numeric.values[index])
                                })
                                .collect()
                        })
                        .collect();
                    for (c, values) in observed.iter().enumerate() {
                        row.groups.push(numeric_summary(
                            values,
                            group_indices[c].len(),
                            c,
                            &classes[c],
                        )?);
                    }
                    if observed.iter().any(Vec::is_empty) {
                        row.diagnostics.push(warning("insufficient_observations","At least one outcome group has no observed values for this variable. Counts are retained; no imputation or cross-variable case deletion is performed."));
                    } else {
                        let method = if config.continuous_test == ContinuousTest::Auto {
                            if classes.len() == 2 {
                                ContinuousTest::MannWhitney
                            } else {
                                ContinuousTest::KruskalWallis
                            }
                        } else {
                            config.continuous_test
                        };
                        let result = match method {
                            ContinuousTest::MannWhitney => {
                                row.method = BaselineMethod::MannWhitney;
                                Some(crate::statistics::mann_whitney_u(
                                    &observed[0],
                                    &observed[1],
                                )?)
                            }
                            ContinuousTest::KruskalWallis => {
                                row.method = BaselineMethod::KruskalWallis;
                                Some(crate::statistics::kruskal_wallis(&observed)?)
                            }
                            ContinuousTest::StudentT | ContinuousTest::Welch => {
                                row.method = if method == ContinuousTest::StudentT {
                                    BaselineMethod::StudentT
                                } else {
                                    BaselineMethod::Welch
                                };
                                if observed.iter().any(|v| v.len() < 2) {
                                    row.method = BaselineMethod::InsufficientObservations;
                                    row.diagnostics.push(warning("insufficient_test_observations","Student/Welch comparison needs at least two observed values per group; summaries and missing counts are retained."));
                                    None
                                } else {
                                    let result = crate::statistics::t_test(
                                        &observed[0],
                                        &observed[1],
                                        method == ContinuousTest::StudentT,
                                    )?;
                                    if let Some(message) = result.diagnostic {
                                        row.diagnostics
                                            .push(warning("degenerate_t_test", &message));
                                    }
                                    Some(crate::statistics::TestResult {
                                        statistic: result.statistic,
                                        p_value: result.p_value,
                                    })
                                }
                            }
                            ContinuousTest::Auto => {
                                return Err(invalid("Baseline continuous test resolution failed."))
                            }
                        };
                        if let Some(result) = result {
                            row.statistic = result.statistic;
                            row.p_value = result.p_value;
                        }
                    }
                }
                Feature::Categorical(j) => {
                    let observed: Vec<Vec<String>> = group_indices
                        .iter()
                        .map(|indices| {
                            indices
                                .iter()
                                .filter_map(|&i| {
                                    data.categorical_values[i * data.categorical_names.len() + j]
                                        .clone()
                                })
                                .collect()
                        })
                        .collect();
                    for (c, values) in observed.iter().enumerate() {
                        row.groups.push(categorical_summary(
                            values,
                            group_indices[c].len(),
                            c,
                            &classes[c],
                        ));
                    }
                    if observed.iter().any(Vec::is_empty) {
                        row.diagnostics.push(warning("insufficient_observations","At least one outcome group has no observed values for this variable. Counts and observed-value proportion denominators are retained."));
                    } else {
                        let levels: Vec<_> = observed
                            .iter()
                            .flatten()
                            .cloned()
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect();
                        let contingency: Vec<Vec<_>> = levels
                            .iter()
                            .map(|level| {
                                observed
                                    .iter()
                                    .map(|group| {
                                        group.iter().filter(|value| *value == level).count() as u64
                                    })
                                    .collect()
                            })
                            .collect();
                        row.categorical_levels = levels;
                        row.contingency = Some(CountTable {
                            rows: contingency.len(),
                            cols: classes.len(),
                            values: contingency.iter().flatten().copied().collect(),
                        });
                        if config.categorical_test == CategoricalTest::Fisher
                            && contingency.len() != 2
                        {
                            return Err(invalid(format!(
                                "Explicit Fisher exact test for '{name}' requires a 2x2 table."
                            )));
                        }
                        if contingency.len() < 2 {
                            row.method = BaselineMethod::ConstantOrAbsentClass;
                            row.diagnostics.push(warning("constant_or_absent_class","Categorical variable has fewer than two observed levels; counts are retained and the association statistic/p-value are undefined."));
                        } else {
                            let chi = crate::statistics::chi_square(&contingency)?;
                            let fisher = config.categorical_test == CategoricalTest::Fisher
                                || (config.categorical_test == CategoricalTest::Auto
                                    && contingency.len() == 2
                                    && classes.len() == 2
                                    && chi.expected.values.iter().any(|&e| e < 5.0));
                            row.expected_counts = Some(chi.expected);
                            if fisher {
                                let result = crate::statistics::fisher_exact([
                                    [contingency[0][0], contingency[0][1]],
                                    [contingency[1][0], contingency[1][1]],
                                ])?;
                                row.method = BaselineMethod::Fisher2x2;
                                row.statistic = result.odds_ratio;
                                row.p_value = result.p_value;
                            } else {
                                row.method = BaselineMethod::ChiSquareNoYates;
                                row.statistic = chi.statistic;
                                row.p_value = chi.p_value;
                            }
                        }
                    }
                }
            }
            rows.push(row);
        }
    }
    Ok(BaselineTable{n_input:data.rows,classes:classes.to_vec(),cohorts:partitions.keys().cloned().collect(),features:features.to_vec(),config:config.clone(),analysis_unit:"supplied_rows".into(),outcome_missing_policy:"error".into(),rows,diagnostics:vec![warning("row_level_baseline_scope","This numerical kernel summarizes supplied rows. Public analysis-unit and patient-ID safeguards must be enforced separately; repeated lesions do not establish independent patients.")]})
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/baseline_reference.json"
        )))
        .unwrap()
    }
    fn build(case: &Value) -> (TabularData, Vec<String>, Vec<Option<usize>>, Vec<String>) {
        let rows = case["rows"].as_array().unwrap();
        let numeric_names: Vec<String> = vec!["a".into(), "b".into(), "all_missing".into()];
        let categorical_names: Vec<String> = vec![
            "sex".into(),
            "rare".into(),
            "constant_category".into(),
            "missing_category".into(),
        ];
        let mut values = Vec::new();
        let mut missing = Vec::new();
        let mut categorical_values = Vec::new();
        for row in rows {
            for name in &numeric_names {
                let v = &row[name.as_str()];
                values.push(v.as_f64().unwrap_or(0.0));
                missing.push(v.is_null());
            }
            for name in &categorical_names {
                categorical_values.push(row[name.as_str()].as_str().map(str::to_string));
            }
        }
        let features = numeric_names
            .iter()
            .chain(&categorical_names)
            .cloned()
            .collect();
        (
            TabularData {
                rows: rows.len(),
                numeric: NumericBuffer {
                    rows: rows.len(),
                    cols: numeric_names.len(),
                    values,
                    missing,
                },
                numeric_names,
                categorical_names,
                categorical_values,
            },
            case["classes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
            rows.iter()
                .map(|r| r["target"].as_u64().map(|v| v as usize))
                .collect(),
            features,
        )
    }
    fn close(a: f64, b: &Value) {
        if b.as_str() == Some("NaN") {
            assert!(a.is_nan());
        } else if b.as_str() == Some("Inf") {
            assert_eq!(a, f64::INFINITY);
        } else {
            let b = b.as_f64().unwrap();
            assert!((a - b).abs() <= 1e-10 + 1e-8 * b.abs(), "{a} != {b}");
        }
    }
    #[test]
    fn observed_binary_and_multiclass_baselines_match_frozen_sample_reference() {
        let fixture = fixture();
        for case in fixture["datasets"].as_object().unwrap().values() {
            let (data, classes, labels, features) = build(case);
            let result = baseline_table(
                &data,
                &classes,
                &labels,
                &features,
                None,
                &BaselineConfig::default(),
            )
            .unwrap();
            assert_eq!(result.n_input, case["n_input"].as_u64().unwrap() as usize);
            assert_eq!(result.rows.len(), 7);
            for (actual, expected) in result.rows.iter().zip(case["table"].as_array().unwrap()) {
                assert_eq!(actual.feature, expected["feature"].as_str().unwrap());
                assert_eq!(
                    serde_json::to_value(actual.method).unwrap(),
                    expected["method"]
                );
                close(actual.statistic, &expected["statistic"]);
                close(actual.p_value, &expected["p_value"]);
                for group in &actual.groups {
                    match group {
                        BaselineGroup::Continuous {
                            class_label,
                            n,
                            missing,
                            mean,
                            sd_ddof1,
                            median,
                            q25,
                            q75,
                            ..
                        } => {
                            let r = &expected["groups"][class_label];
                            assert_eq!(*n, r["n"].as_u64().unwrap());
                            assert_eq!(*missing, r["missing"].as_u64().unwrap());
                            for (a, key) in [
                                (*mean, "mean"),
                                (*sd_ddof1, "sd_ddof1"),
                                (*median, "median"),
                                (*q25, "q25"),
                                (*q75, "q75"),
                            ] {
                                close(a, &r[key]);
                            }
                        }
                        BaselineGroup::Categorical {
                            class_label,
                            n,
                            missing,
                            counts,
                            proportions,
                            ..
                        } => {
                            let r = &expected["groups"][class_label];
                            assert_eq!(*n, r["n"].as_u64().unwrap());
                            assert_eq!(*missing, r["missing"].as_u64().unwrap());
                            assert_eq!(serde_json::to_value(counts).unwrap(), r["counts"]);
                            for (level, p) in proportions {
                                close(*p, &r["proportions"][level]);
                            }
                        }
                    }
                }
                if let Some(counts) = &actual.contingency {
                    assert_eq!(
                        counts.values,
                        expected["contingency"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .flat_map(|row| row
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_u64().unwrap()))
                            .collect::<Vec<_>>()
                    );
                } else {
                    assert!(expected["contingency"].is_null());
                }
                if let Some(counts) = &actual.expected_counts {
                    for (a, b) in counts.values.iter().zip(
                        expected["expected_counts"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .flat_map(|row| row.as_array().unwrap().iter()),
                    ) {
                        close(*a, b);
                    }
                } else {
                    assert!(expected["expected_counts"].is_null());
                }
                if !actual.p_value.is_finite() {
                    assert!(!actual.diagnostics.is_empty());
                }
            }
            let restored: BaselineTable =
                serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
            assert_eq!(
                serde_json::to_value(&result).unwrap(),
                serde_json::to_value(restored).unwrap()
            );
        }
    }
    #[test]
    fn baseline_missing_outcomes_illegal_fisher_and_named_alignment_are_explicit() {
        let fixture = fixture();
        let (data, classes, mut labels, features) = build(&fixture["datasets"]["binary"]);
        labels[0] = None;
        assert!(baseline_table(
            &data,
            &classes,
            &labels,
            &features,
            None,
            &BaselineConfig::default()
        )
        .unwrap_err()
        .message
        .contains("missing"));
        labels[0] = Some(0);
        let mut changed = data.clone();
        changed.numeric_names.reverse();
        for row in changed.numeric.values.chunks_mut(3) {
            row.reverse();
        }
        for row in changed.numeric.missing.chunks_mut(3) {
            row.reverse();
        }
        changed.categorical_names.reverse();
        for row in changed.categorical_values.chunks_mut(4) {
            row.reverse();
        }
        let a = baseline_table(
            &data,
            &classes,
            &labels,
            &features,
            None,
            &BaselineConfig::default(),
        )
        .unwrap();
        let b = baseline_table(
            &changed,
            &classes,
            &labels,
            &features,
            None,
            &BaselineConfig::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        let (data, classes, labels, features) = build(&fixture["datasets"]["multiclass"]);
        assert!(baseline_table(
            &data,
            &classes,
            &labels,
            &features,
            None,
            &BaselineConfig {
                categorical_test: CategoricalTest::Fisher,
                ..BaselineConfig::default()
            }
        )
        .unwrap_err()
        .message
        .contains("2x2"));
        let table = baseline_table(
            &data,
            &classes,
            &labels,
            &features,
            None,
            &BaselineConfig::default(),
        )
        .unwrap();
        assert_eq!(
            table
                .rows
                .iter()
                .find(|r| r.feature == "rare")
                .unwrap()
                .method,
            BaselineMethod::ChiSquareNoYates
        );
    }

    #[test]
    fn explicit_student_and_welch_use_frozen_shared_tests_and_cohorts_do_not_mix() {
        let reference: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/statistics_reference.json"
        )))
        .unwrap();
        for (name, method) in [
            ("student_t", ContinuousTest::StudentT),
            ("welch", ContinuousTest::Welch),
        ] {
            let expected = &reference["hypothesis_tests"][name];
            let a: Vec<_> = expected["a"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let b: Vec<_> = expected["b"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let n = a.len() + b.len();
            let mut values = a.clone();
            values.extend(&b);
            let data = TabularData {
                rows: n,
                numeric_names: vec!["value".into()],
                numeric: NumericBuffer {
                    rows: n,
                    cols: 1,
                    missing: vec![false; n],
                    values,
                },
                categorical_names: vec![],
                categorical_values: vec![],
            };
            let labels: Vec<_> = (0..n).map(|i| Some(usize::from(i >= a.len()))).collect();
            let output = baseline_table(
                &data,
                &["A".into(), "B".into()],
                &labels,
                &["value".into()],
                None,
                &BaselineConfig {
                    continuous_test: method,
                    categorical_test: CategoricalTest::Auto,
                },
            )
            .unwrap();
            close(output.rows[0].statistic, &expected["statistic"]);
            close(output.rows[0].p_value, &expected["p_value"]);
        }
        let data = TabularData {
            rows: 4,
            numeric_names: vec!["value".into()],
            numeric: NumericBuffer {
                rows: 4,
                cols: 1,
                values: vec![1., 3., 10., 30.],
                missing: vec![false; 4],
            },
            categorical_names: vec![],
            categorical_values: vec![],
        };
        let cohorts = vec!["A".into(), "A".into(), "B".into(), "B".into()];
        let table = baseline_table(
            &data,
            &["zero".into(), "one".into()],
            &[Some(0), Some(1), Some(0), Some(1)],
            &["value".into()],
            Some(&cohorts),
            &BaselineConfig::default(),
        )
        .unwrap();
        assert_eq!(table.rows.len(), 2);
        for (row, expected) in table.rows.iter().zip([[1., 3.], [10., 30.]]) {
            for (group, value) in row.groups.iter().zip(expected) {
                let BaselineGroup::Continuous {
                    n,
                    missing,
                    mean,
                    sd_ddof1,
                    ..
                } = group
                else {
                    unreachable!()
                };
                assert_eq!(*n, 1);
                assert_eq!(*missing, 0);
                assert_eq!(*mean, value);
                assert!(sd_ddof1.is_nan());
            }
        }
    }
}
