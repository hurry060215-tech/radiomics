use super::common::*;
use clap::{Args, ValueEnum};
use ov_radiomics_core::{
    contracts::*,
    errors::{ErrorCode, RadResult, RadiomicsError},
    evaluation as metrics,
    learning::{self, Outcome, StudyData},
    models::{ModelPrediction, ModelTarget},
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationUnit {
    Sample,
    Patient,
}
fn is_false(value: &bool) -> bool {
    !*value
}
#[derive(Debug, Args, Serialize)]
pub struct EvaluateCommand {
    #[arg(long)]
    pub predictions: PathBuf,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long, default_value = "radiomics-evaluation")]
    pub output: PathBuf,
    #[arg(long, value_enum, default_value = "patient")]
    pub prediction_unit: EvaluationUnit,
    /// Additionally evaluate each recorded center, partitioning actual source units.
    #[arg(long)]
    #[serde(skip_serializing_if = "is_false")]
    pub by_center: bool,
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f64,
    #[arg(long, default_value_t = 0.95)]
    pub confidence: f64,
    #[arg(long, default_value_t = 2000)]
    pub n_bootstrap: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long)]
    pub overwrite: bool,
}
#[derive(Debug, Clone, PartialEq)]
enum Truth {
    Class(String),
    Regression(f64),
}
fn truth_at(data: &StudyData, index: usize) -> Option<Truth> {
    match &data.outcome {
        Outcome::Unlabeled => None,
        Outcome::Classification { values, .. } => {
            values[index].as_ref().map(|v| Truth::Class(v.clone()))
        }
        Outcome::Regression { values, .. } => values[index].map(Truth::Regression),
    }
}
fn unit_key(row: &PredictionRow, unit: PredictionUnit) -> RadResult<&str> {
    match unit {
        PredictionUnit::Sample => Ok(&row.sample_id),
        PredictionUnit::Patient => Ok(&row.patient_id),
        PredictionUnit::Slide => row
            .slide_id
            .as_deref()
            .ok_or_else(|| invalid("Slide prediction has no slide identity.")),
        PredictionUnit::Tile => row
            .tile_id
            .as_deref()
            .ok_or_else(|| invalid("Tile prediction has no tile identity.")),
    }
}
fn data_groups(data: &StudyData, unit: PredictionUnit) -> BTreeMap<&str, Vec<usize>> {
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, row) in data.samples.iter().enumerate() {
        let key = match unit {
            PredictionUnit::Sample | PredictionUnit::Tile => Some(row.sample_id.as_str()),
            PredictionUnit::Patient => Some(row.patient_id.as_str()),
            PredictionUnit::Slide => row.slide_id.as_deref(),
        };
        if let Some(key) = key {
            groups.entry(key).or_default().push(i);
        }
    }
    groups
}
fn center_sources(
    source: &PredictionTable,
    data: &StudyData,
) -> RadResult<BTreeMap<String, PredictionTable>> {
    // Validate the same real unit/parent/cohort association used by ordinary
    // evaluation before deriving any center-specific population.
    align_source(source, data)?;
    let groups = data_groups(data, source.unit);
    let mut result: BTreeMap<String, PredictionTable> = BTreeMap::new();
    for row in &source.rows {
        let key = unit_key(row, source.unit)?;
        let indices = &groups[key];
        let centers: BTreeSet<_> = indices
            .iter()
            .map(|&i| data.samples[i].center.as_deref())
            .collect();
        let center=if centers.len()==1 {
            centers.iter().next().copied().flatten().filter(|value|!value.trim().is_empty())
        } else {None}.ok_or_else(||invalid(format!("Center evaluation requires one known center for actual {:?} prediction '{key}'; missing/mixed centers cannot disaggregate an already aggregated score. Supply genuine finer-unit predictions or correct the metadata.",source.unit)))?;
        result
            .entry(center.to_owned())
            .or_insert_with(|| PredictionTable {
                task: source.task,
                unit: source.unit,
                classes: source.classes.clone(),
                rows: vec![],
            })
            .rows
            .push(row.clone());
    }
    Ok(result)
}
fn align_source(source: &PredictionTable, data: &StudyData) -> RadResult<BTreeSet<usize>> {
    source.validate()?;
    data.validate()?;
    let groups = data_groups(data, source.unit);
    let mut seen = BTreeSet::new();
    let mut matched = BTreeSet::new();
    for row in &source.rows {
        let key = unit_key(row, source.unit)?;
        if !seen.insert(key) {
            return Err(invalid(
                "Prediction identities are duplicated at the declared prediction unit.",
            ));
        }
        let indices = groups.get(key).ok_or_else(|| {
            RadiomicsError::new(
                ErrorCode::Alignment,
                format!(
                    "Prediction identity '{key}' has no truth-data match at unit {:?}.",
                    source.unit
                ),
            )
        })?;
        for &i in indices {
            let actual = &data.samples[i];
            if actual.patient_id != row.patient_id
                || data.cohorts[i] != row.cohort
                || row
                    .slide_id
                    .as_ref()
                    .is_some_and(|slide| actual.slide_id.as_ref() != Some(slide))
            {
                return Err(RadiomicsError::new(
                    ErrorCode::Alignment,
                    format!(
                        "Prediction '{key}' disagrees with truth-data patient/cohort/slide parent."
                    ),
                ));
            }
            matched.insert(i);
        }
    }
    Ok(matched)
}
fn evaluation_rows(
    source: &PredictionTable,
    data: &StudyData,
    unit: EvaluationUnit,
) -> RadResult<(PredictionTable, Vec<Option<Truth>>, Value)> {
    let matched = align_source(source, data)?;
    if unit == EvaluationUnit::Sample && source.unit != PredictionUnit::Sample {
        return Err(invalid("Sample evaluation requires actual sample predictions; patient, slide or tile scores are not broadcast or renamed as samples."));
    }
    let predictions = if unit == EvaluationUnit::Patient {
        learning::aggregate_patient_predictions(source)?
    } else {
        source.clone()
    };
    let groups = data_groups(data, predictions.unit);
    let mut truths = Vec::with_capacity(predictions.rows.len());
    let mut used = BTreeSet::new();
    for row in &predictions.rows {
        let key = unit_key(row, predictions.unit)?;
        let indices = groups.get(key).ok_or_else(|| {
            RadiomicsError::new(
                ErrorCode::Alignment,
                format!("Evaluation identity '{key}' has no truth group."),
            )
        })?;
        if indices
            .iter()
            .any(|&i| data.samples[i].patient_id != row.patient_id || data.cohorts[i] != row.cohort)
        {
            return Err(invalid(format!(
                "Truth rows for evaluation identity '{key}' mix patient/cohort identities."
            )));
        }
        let first = truth_at(data, indices[0]);
        if indices.iter().any(|&i| truth_at(data, i) != first) {
            return Err(invalid(format!("Truth for patient '{key}' differs across rows or mixes missing and observed outcomes; no target is selected or silently dropped.")));
        }
        truths.push(first);
        used.extend(indices.iter().copied());
    }
    let unused: Vec<_> = data
        .samples
        .iter()
        .enumerate()
        .filter_map(|(i, row)| (!used.contains(&i)).then_some(&row.sample_id))
        .collect();
    let unmatched: Vec<_> = data
        .samples
        .iter()
        .enumerate()
        .filter_map(|(i, row)| (!matched.contains(&i)).then_some(&row.sample_id))
        .collect();
    let alignment = json!({"source_unit":source.unit,"requested_unit":unit,"effective_unit":predictions.unit,"source_rows":source.rows.len(),"effective_rows":predictions.rows.len(),"truth_input_rows":data.samples.len(),"unused_truth_sample_ids":unused,"not_predicted_source_sample_ids":unmatched,"same_patient_extra_truth_rows_checked":used.difference(&matched).map(|&i|&data.samples[i].sample_id).collect::<Vec<_>>(),"row_order":"effective_prediction_table_order"});
    Ok((predictions, truths, alignment))
}
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Calculation<T: Serialize> {
    Available { data: T },
    Unavailable { error: RadiomicsError },
}
fn calculation<T: Serialize>(result: RadResult<T>) -> Calculation<T> {
    match result {
        Ok(data) => Calculation::Available { data },
        Err(error) => Calculation::Unavailable { error },
    }
}
#[derive(Serialize)]
struct CurveSet {
    class_index: usize,
    class_label: String,
    observed_binary: Vec<u8>,
    scores: Vec<f64>,
    roc: metrics::RocCurve,
    pr: metrics::PrecisionRecallCurve,
    uniform: metrics::CalibrationCurve,
    quantile: metrics::CalibrationCurve,
}
#[derive(Serialize)]
struct CohortReport {
    format: &'static str,
    schema_version: u32,
    cohort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    center: Option<String>,
    status: &'static str,
    performance_scope: &'static str,
    n_predictions: usize,
    source_prediction_unit: PredictionUnit,
    effective_prediction_unit: PredictionUnit,
    metrics: Option<Value>,
    curves: Vec<CurveSet>,
    bootstrap: Option<metrics::BootstrapInterval>,
    delong: Option<Calculation<metrics::AucInterval>>,
    decision_curve: Option<metrics::DecisionCurve>,
    hosmer_lemeshow: Option<Calculation<metrics::HosmerLemeshow>>,
    diagnostics: Vec<String>,
}
fn model_arrays(
    table: &PredictionTable,
    truth: &[Option<Truth>],
) -> RadResult<(ModelTarget, ModelPrediction)> {
    if table.task == Task::Regression {
        let y = truth
            .iter()
            .map(|v| match v {
                Some(Truth::Regression(value)) => Ok(*value),
                _ => Err(invalid(
                    "Regression predictions need observed regression truth on each evaluated row.",
                )),
            })
            .collect::<RadResult<Vec<_>>>()?;
        let p = table
            .rows
            .iter()
            .map(|r| {
                r.value
                    .ok_or_else(|| invalid("Regression prediction has no value."))
            })
            .collect::<RadResult<Vec<_>>>()?;
        Ok((
            ModelTarget::Regression { values: y },
            ModelPrediction::Regression { values: p },
        ))
    } else {
        let labels=truth.iter().map(|v|match v {Some(Truth::Class(value))=>table.classes.iter().position(|c|c==value).ok_or_else(||invalid(format!("Unknown truth class '{value}' outside the prediction class order."))),_=>Err(invalid("Classification predictions need observed categorical truth on each evaluated row."))}).collect::<RadResult<Vec<_>>>()?;
        let values: Vec<_> = table
            .rows
            .iter()
            .flat_map(|r| r.probabilities.iter().copied())
            .collect();
        Ok((
            ModelTarget::Classification {
                classes: table.classes.clone(),
                labels,
            },
            ModelPrediction::Classification {
                classes: table.classes.clone(),
                probabilities: NumericBuffer {
                    rows: table.rows.len(),
                    cols: table.classes.len(),
                    missing: vec![false; values.len()],
                    values,
                },
            },
        ))
    }
}
fn cohort_report(
    table: &PredictionTable,
    truth: &[Option<Truth>],
    source_unit: PredictionUnit,
    args: &EvaluateCommand,
    thresholds: &[f64],
) -> RadResult<CohortReport> {
    let cohort = table.rows[0].cohort.clone();
    let scope = if table.rows.iter().all(|r| r.fold.is_some()) {
        "provided_fold_predictions"
    } else if cohort == "train" {
        "apparent"
    } else if cohort == "test" {
        "heldout_cohort"
    } else {
        "external_cohort"
    };
    let mut report = CohortReport {
        format: "radiomics.cli.cohort-evaluation",
        schema_version: 1,
        cohort,
        center: None,
        status: "prediction_only",
        performance_scope: scope,
        n_predictions: table.rows.len(),
        source_prediction_unit: source_unit,
        effective_prediction_unit: table.unit,
        metrics: None,
        curves: vec![],
        bootstrap: None,
        delong: None,
        decision_curve: None,
        hosmer_lemeshow: None,
        diagnostics: vec![],
    };
    let missing = truth.iter().filter(|v| v.is_none()).count();
    if missing == truth.len() {
        report.diagnostics.push("No outcomes are present for this cohort; predictions are retained without invented metrics.".into());
        return Ok(report);
    }
    if missing > 0 {
        return Err(invalid(format!(
            "Cohort '{}' has partially missing outcomes; no rows are silently removed.",
            report.cohort
        )));
    }
    let (target, prediction) = model_arrays(table, truth)?;
    metrics::validate_prediction(table.task, &target, &prediction)?;
    report.status = "evaluated";
    if args.prediction_unit == EvaluationUnit::Sample
        && table
            .rows
            .iter()
            .map(|r| r.patient_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != table.rows.len()
    {
        report.diagnostics.push("Repeated patients are present in explicitly requested sample-unit evaluation. Row resampling is not independent-patient bootstrap inference; use patient unit for patient intervals.".into());
    }
    match (&target, &prediction) {
        (ModelTarget::Regression { values: y }, ModelPrediction::Regression { values: p }) => {
            report.metrics = Some(serde_json::to_value(metrics::regression_metrics(y, p)?)?);
            report.bootstrap = Some(metrics::bootstrap_regression_seeded(
                &target,
                &prediction,
                args.n_bootstrap,
                args.seed,
                args.confidence,
            )?);
        }
        (
            ModelTarget::Classification { labels, .. },
            ModelPrediction::Classification { probabilities, .. },
        ) => {
            if table.task == Task::Binary {
                let y: Vec<_> = labels.iter().map(|&v| v as u8).collect();
                let p: Vec<_> = probabilities.values.chunks(2).map(|r| r[1]).collect();
                report.metrics = Some(serde_json::to_value(metrics::binary_metrics(
                    &y,
                    &p,
                    args.threshold,
                )?)?);
                report.bootstrap = Some(metrics::bootstrap_binary_seeded(
                    &y,
                    &p,
                    args.n_bootstrap,
                    u32::try_from(args.seed)
                        .map_err(|_| invalid("Binary legacy bootstrap seed must fit uint32."))?,
                    args.confidence,
                )?);
                report.delong = Some(calculation(metrics::delong_interval(
                    &y,
                    &p,
                    args.confidence,
                )));
                report.decision_curve = Some(metrics::decision_curve(&y, &p, thresholds)?);
                report.hosmer_lemeshow = Some(calculation(metrics::hosmer_lemeshow(&y, &p, 10)));
            } else {
                report.metrics = Some(serde_json::to_value(metrics::classification_metrics(
                    table.task,
                    &target,
                    &prediction,
                    args.threshold,
                )?)?);
                report.bootstrap = Some(metrics::bootstrap_multiclass_seeded(
                    &target,
                    &prediction,
                    args.n_bootstrap,
                    args.seed,
                    args.confidence,
                )?);
            }
            let classes: Vec<_> = if table.task == Task::Binary {
                vec![1]
            } else {
                (0..table.classes.len()).collect()
            };
            for c in classes {
                let y: Vec<_> = labels.iter().map(|&v| u8::from(v == c)).collect();
                let p: Vec<_> = probabilities
                    .values
                    .chunks(probabilities.cols)
                    .map(|r| r[c])
                    .collect();
                report.curves.push(CurveSet {
                    class_index: c,
                    class_label: table.classes[c].clone(),
                    roc: metrics::roc_curve(&y, &p)?,
                    pr: metrics::precision_recall_curve(&y, &p)?,
                    uniform: metrics::calibration_curve(
                        &y,
                        &p,
                        5,
                        metrics::CalibrationStrategy::Uniform,
                    )?,
                    quantile: metrics::calibration_curve(
                        &y,
                        &p,
                        5,
                        metrics::CalibrationStrategy::Quantile,
                    )?,
                    observed_binary: y,
                    scores: p,
                });
            }
        }
        _ => return Err(invalid("Evaluation target/prediction resolution failed.")),
    }
    Ok(report)
}
fn cohort_groups(predictions: &PredictionTable) -> BTreeMap<String, Vec<usize>> {
    let mut result = BTreeMap::new();
    for (i, row) in predictions.rows.iter().enumerate() {
        result
            .entry(row.cohort.clone())
            .or_insert_with(Vec::new)
            .push(i);
    }
    result
}
fn grouped_reports(
    predictions: &PredictionTable,
    truth: &[Option<Truth>],
    source_unit: PredictionUnit,
    args: &EvaluateCommand,
    thresholds: &[f64],
    center: Option<&str>,
) -> RadResult<Vec<(PredictionTable, CohortReport)>> {
    let mut reports = Vec::new();
    for (name, indices) in cohort_groups(predictions) {
        let table = PredictionTable {
            task: predictions.task,
            unit: predictions.unit,
            classes: predictions.classes.clone(),
            rows: indices
                .iter()
                .map(|&i| predictions.rows[i].clone())
                .collect(),
        };
        let labels: Vec<_> = indices.iter().map(|&i| truth[i].clone()).collect();
        let mut report = cohort_report(&table, &labels, source_unit, args, thresholds)?;
        if report.cohort != name {
            return Err(schema("Cohort identity changed during grouping."));
        }
        report.center = center.map(str::to_owned);
        reports.push((table, report));
    }
    Ok(reports)
}
fn report_files(names: &mut Vec<PathBuf>, prefix: &str, source: &PredictionTable) {
    for suffix in [".json", "-metrics.csv", "-bootstrap.csv"] {
        names.push(format!("{prefix}{suffix}").into());
    }
    for curve in curve_prefixes(prefix, source.task, source.classes.len()) {
        names.extend(curve_files(&curve));
    }
    if source.task == Task::Binary {
        names.push(format!("{prefix}-dca.csv").into());
        names.push(format!("{prefix}-hosmer-lemeshow.csv").into());
    }
}
struct CenterEvaluation {
    center: String,
    predictions: PredictionTable,
    alignment: Value,
    reports: Vec<(PredictionTable, CohortReport)>,
}
fn report_summary(report: &CohortReport, prefix: &str) -> Value {
    let mut summary = json!({"cohort":report.cohort,"status":report.status,"performance_scope":report.performance_scope,"n_predictions":report.n_predictions,"file":format!("{prefix}.json"),"effective_unit":report.effective_prediction_unit});
    if let Some(center) = &report.center {
        summary["center"] = json!(center);
    }
    summary
}
fn curve_prefixes(prefix: &str, task: Task, k: usize) -> Vec<String> {
    match task {
        Task::Binary => vec![prefix.into()],
        Task::Multiclass => (0..k).map(|c| format!("{prefix}-ovr-{c:03}")).collect(),
        Task::Regression => vec![],
    }
}
fn curve_files(prefix: &str) -> Vec<PathBuf> {
    [
        "-roc.csv",
        "-pr.csv",
        "-calibration-uniform-bins.csv",
        "-calibration-uniform-membership.csv",
        "-calibration-quantile-bins.csv",
        "-calibration-quantile-membership.csv",
    ]
    .into_iter()
    .map(|suffix| format!("{prefix}{suffix}").into())
    .collect()
}
fn flatten(value: &Value, path: &str, rows: &mut Vec<Vec<String>>) {
    match value {
        Value::Number(n) => rows.push(vec![path.into(), n.to_string()]),
        Value::String(v) if matches!(v.as_str(), "NaN" | "Inf" | "-Inf") => {
            rows.push(vec![path.into(), v.clone()])
        }
        Value::Object(map) => {
            for (key, value) in map {
                flatten(
                    value,
                    &if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    },
                    rows,
                )
            }
        }
        Value::Array(values) => {
            for (i, value) in values.iter().enumerate() {
                flatten(value, &format!("{path}[{i}]"), rows)
            }
        }
        _ => (),
    }
}
fn export_curve(
    output: &Output,
    prefix: &str,
    curve: Option<&CurveSet>,
    table: &PredictionTable,
) -> RadResult<()> {
    let Some(curve) = curve else {
        for (suffix, headers) in [
            (
                "-roc.csv",
                vec!["threshold", "false_positive_rate", "true_positive_rate"],
            ),
            ("-pr.csv", vec!["threshold", "precision", "recall"]),
            (
                "-calibration-uniform-bins.csv",
                vec![
                    "bin",
                    "lower",
                    "upper",
                    "count",
                    "mean_predicted",
                    "fraction_positive",
                ],
            ),
            (
                "-calibration-uniform-membership.csv",
                vec!["sample_id", "patient_id", "bin", "score", "observed_binary"],
            ),
            (
                "-calibration-quantile-bins.csv",
                vec![
                    "bin",
                    "lower",
                    "upper",
                    "count",
                    "mean_predicted",
                    "fraction_positive",
                ],
            ),
            (
                "-calibration-quantile-membership.csv",
                vec!["sample_id", "patient_id", "bin", "score", "observed_binary"],
            ),
        ] {
            output.csv(
                &format!("{prefix}{suffix}"),
                &headers,
                std::iter::empty::<Vec<String>>(),
            )?;
        }
        return Ok(());
    };
    output.csv(
        &format!("{prefix}-roc.csv"),
        &["threshold", "false_positive_rate", "true_positive_rate"],
        curve.roc.thresholds.iter().enumerate().map(|(i, v)| {
            vec![
                v.to_string(),
                curve.roc.false_positive_rate[i].to_string(),
                curve.roc.true_positive_rate[i].to_string(),
            ]
        }),
    )?;
    output.csv(
        &format!("{prefix}-pr.csv"),
        &["threshold", "precision", "recall"],
        curve.pr.precision.iter().enumerate().map(|(i, v)| {
            vec![
                curve
                    .pr
                    .thresholds
                    .get(i)
                    .map(ToString::to_string)
                    .unwrap_or_default(),
                v.to_string(),
                curve.pr.recall[i].to_string(),
            ]
        }),
    )?;
    for (name, bins) in [("uniform", &curve.uniform), ("quantile", &curve.quantile)] {
        output.csv(
            &format!("{prefix}-calibration-{name}-bins.csv"),
            &[
                "bin",
                "lower",
                "upper",
                "count",
                "mean_predicted",
                "fraction_positive",
            ],
            (0..bins.n_bins).map(|i| {
                let value = bins.nonempty_bins.iter().position(|&v| v == i);
                vec![
                    i.to_string(),
                    bins.edges[i].to_string(),
                    bins.edges[i + 1].to_string(),
                    bins.bin_counts[i].to_string(),
                    value
                        .map(|j| bins.mean_predicted[j].to_string())
                        .unwrap_or_else(|| "NaN".into()),
                    value
                        .map(|j| bins.fraction_positive[j].to_string())
                        .unwrap_or_else(|| "NaN".into()),
                ]
            }),
        )?;
        output.csv(
            &format!("{prefix}-calibration-{name}-membership.csv"),
            &["sample_id", "patient_id", "bin", "score", "observed_binary"],
            table.rows.iter().enumerate().map(|(i, row)| {
                vec![
                    row.sample_id.clone(),
                    row.patient_id.clone(),
                    bins.bin_membership[i].to_string(),
                    curve.scores[i].to_string(),
                    curve.observed_binary[i].to_string(),
                ]
            }),
        )?;
    }
    Ok(())
}
fn export_cohort(
    output: &Output,
    prefix: &str,
    table: &PredictionTable,
    report: &CohortReport,
) -> RadResult<()> {
    output.json(&format!("{prefix}.json"), report)?;
    let mut scalar = report.metrics.clone().unwrap_or(Value::Null);
    if let Some(interval) = &report.bootstrap {
        scalar["bootstrap_interval"] = json!({"estimate":interval.estimate,"confidence":interval.confidence,"low":interval.low,"high":interval.high});
    }
    if let Some(Calculation::Available { data }) = &report.delong {
        scalar["delong_interval"] = serde_json::to_value(data)?;
    }
    if let Some(Calculation::Available { data }) = &report.hosmer_lemeshow {
        scalar["hosmer_lemeshow"] =
            json!({"statistic":data.statistic,"df":data.df,"p_value":data.p_value});
    }
    let mut rows = Vec::new();
    flatten(&scalar, "", &mut rows);
    output.csv(&format!("{prefix}-metrics.csv"), &["metric", "value"], rows)?;
    output.csv(
        &format!("{prefix}-bootstrap.csv"),
        &["replicate", "score"],
        report.bootstrap.iter().flat_map(|b| {
            b.replicate_scores
                .iter()
                .enumerate()
                .map(|(i, v)| vec![i.to_string(), v.to_string()])
        }),
    )?;
    for (i, name) in curve_prefixes(prefix, table.task, table.classes.len())
        .iter()
        .enumerate()
    {
        export_curve(output, name, report.curves.get(i), table)?;
    }
    if table.task == Task::Binary {
        output.csv(
            &format!("{prefix}-dca.csv"),
            &["threshold", "model_net_benefit", "treat_all", "treat_none"],
            report.decision_curve.iter().flat_map(|d| {
                d.thresholds.iter().enumerate().map(|(i, t)| {
                    vec![
                        t.to_string(),
                        d.model_net_benefit[i].to_string(),
                        d.treat_all[i].to_string(),
                        d.treat_none[i].to_string(),
                    ]
                })
            }),
        )?;
        let bins = match &report.hosmer_lemeshow {
            Some(Calculation::Available { data }) => data.table.as_slice(),
            _ => &[],
        };
        output.csv(
            &format!("{prefix}-hosmer-lemeshow.csv"),
            &[
                "lower",
                "upper",
                "include_lowest",
                "n",
                "observed",
                "expected",
            ],
            bins.iter().map(|b| {
                vec![
                    b.lower.to_string(),
                    b.upper.to_string(),
                    b.include_lowest.to_string(),
                    b.n.to_string(),
                    b.observed.to_string(),
                    b.expected.to_string(),
                ]
            }),
        )?;
    }
    Ok(())
}
pub fn evaluate(args: EvaluateCommand) -> RadResult<u8> {
    if !args.threshold.is_finite()
        || !(0.0..=1.0).contains(&args.threshold)
        || !args.confidence.is_finite()
        || args.confidence <= 0.0
        || args.confidence >= 1.0
        || args.n_bootstrap < 2
    {
        return Err(invalid(
            "Evaluation threshold/confidence or bootstrap replicate count is invalid.",
        ));
    }
    let source: PredictionTable = read_json(&args.predictions)?;
    source.validate()?;
    if source.rows.is_empty() {
        return Err(invalid("Evaluation requires a nonempty prediction table."));
    }
    let (study_spec, loaded) = load_study_record(&args.data, source.task)?;
    let study_spec_directory = std::fs::canonicalize(parent(&args.data))?;
    let (predictions, truth, alignment) =
        evaluation_rows(&source, &loaded.data, args.prediction_unit)?;
    let thresholds: Vec<_> = (1..100).map(|i| i as f64 / 100.0).collect();
    let reports = grouped_reports(&predictions, &truth, source.unit, &args, &thresholds, None)?;
    let mut centers = Vec::new();
    if args.by_center {
        for (center, input) in center_sources(&source, &loaded.data)? {
            let (center_predictions, center_truth, mut center_alignment) =
                evaluation_rows(&input, &loaded.data, args.prediction_unit)?;
            center_alignment["center"] = json!(center);
            center_alignment["center_partition_unit"] = json!(source.unit);
            center_alignment["center_source_sample_ids"] = json!(input
                .rows
                .iter()
                .map(|row| &row.sample_id)
                .collect::<Vec<_>>());
            let center_reports = grouped_reports(
                &center_predictions,
                &center_truth,
                source.unit,
                &args,
                &thresholds,
                Some(&center),
            )?;
            centers.push(CenterEvaluation {
                center,
                predictions: center_predictions,
                alignment: center_alignment,
                reports: center_reports,
            });
        }
    }
    let mut names: Vec<PathBuf> = [
        "source-predictions.json",
        "predictions.json",
        "predictions.csv",
        "load_report.json",
        "alignment_report.json",
        "evaluation_config.json",
        "summary.json",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect();
    for index in 0..reports.len() {
        report_files(&mut names, &format!("cohort-{index:03}"), &source);
    }
    for (i, center) in centers.iter().enumerate() {
        for suffix in ["-predictions.json", "-predictions.csv", "-alignment.json"] {
            names.push(format!("center-{i:03}{suffix}").into());
        }
        for j in 0..center.reports.len() {
            report_files(&mut names, &format!("center-{i:03}-cohort-{j:03}"), &source);
        }
    }
    let output = Output::preflight(&args.output, args.overwrite, names)?;
    output.begin()?;
    output.json("source-predictions.json", &source)?;
    output.predictions("predictions.json", "predictions.csv", &predictions)?;
    output.json("load_report.json", &loaded.report)?;
    output.json("alignment_report.json", &alignment)?;
    output.json("evaluation_config.json",&json!({"format":"radiomics.cli.evaluation-config","schema_version":1,"arguments":args,"study_spec":study_spec,"study_spec_directory":study_spec_directory,"task":source.task,"classes":source.classes,"effective_unit":predictions.unit,"calibration_bins":5,"hosmer_lemeshow_requested_bins":10,"dca_thresholds":thresholds,"dca_default_configuration":"i/100, i=1..99; explicit CLI configuration, not a claim of legacy numpy.linspace bitwise identity","runtime":"rust"}))?;
    let mut summary = Vec::new();
    for (index, (table, report)) in reports.iter().enumerate() {
        let prefix = format!("cohort-{index:03}");
        export_cohort(&output, &prefix, table, report)?;
        summary.push(report_summary(report, &prefix));
    }
    let status = if reports.iter().all(|(_, r)| r.status == "prediction_only") {
        "prediction_only"
    } else {
        "complete"
    };
    let mut summary = json!({"format":"radiomics.cli.evaluation","schema_version":1,"status":status,"task":source.task,"classes":source.classes,"source_unit":source.unit,"effective_unit":predictions.unit,"cohorts":summary});
    if args.by_center {
        let mut center_summary = Vec::new();
        for (i, center) in centers.iter().enumerate() {
            let prefix = format!("center-{i:03}");
            output.predictions(
                &format!("{prefix}-predictions.json"),
                &format!("{prefix}-predictions.csv"),
                &center.predictions,
            )?;
            output.json(&format!("{prefix}-alignment.json"), &center.alignment)?;
            let mut cohorts = Vec::new();
            for (j, (table, report)) in center.reports.iter().enumerate() {
                let name = format!("{prefix}-cohort-{j:03}");
                export_cohort(&output, &name, table, report)?;
                cohorts.push(report_summary(report, &name));
            }
            center_summary.push(json!({"center":center.center,"source_unit":source.unit,"effective_unit":center.predictions.unit,"predictions_file":format!("{prefix}-predictions.json"),"alignment_file":format!("{prefix}-alignment.json"),"cohorts":cohorts}));
        }
        summary["by_center"] = json!(true);
        summary["centers"] = json!(center_summary);
    }
    output.json("summary.json", &summary)?;
    println!(
        "{}",
        json!({"status":status,"command":"evaluate","output":args.output,"cohorts":reports.len(),"prediction_unit":predictions.unit})
    );
    Ok(0)
}
