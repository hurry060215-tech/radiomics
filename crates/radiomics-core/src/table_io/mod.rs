//! Flat CSV/Parquet study inputs, strict keyed joins and feature manifests.
use crate::contracts::*;
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::learning::{
    ClinicalJoinKey, FeatureTableManifest, JoinPolicy, Outcome, StudyData, StudyTableSpec,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

mod csv_io;
mod feature_io;
mod parquet_io;
pub use feature_io::{
    feature_manifest_path, read_feature_table, write_feature_table, FEATURE_MANIFEST_KEY,
};

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn schema(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Schema, message)
}
fn alignment(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Alignment, message)
}
fn csv_error(error: csv::Error) -> RadiomicsError {
    invalid(format!("CSV: {error}"))
}
fn parquet_error(error: parquet::errors::ParquetError) -> RadiomicsError {
    schema(format!("Parquet: {error}"))
}
fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}
fn resolve(base: &Path, path: &Path) -> RadResult<PathBuf> {
    let base = if base.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base
    };
    Ok(fs::canonicalize(if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    })?)
}
fn sha256_file(path: &Path) -> RadResult<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut block = [0u8; 65536];
    loop {
        let n = input.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn format(path: &Path) -> RadResult<TableFormat> {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("csv") => Ok(TableFormat::Csv),
        Some("parquet") | Some("pq") => Ok(TableFormat::Parquet),
        _ => Err(RadiomicsError::new(
            ErrorCode::Unsupported,
            "Study table files must be CSV or flat Parquet.",
        )),
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum TableFormat {
    Csv,
    Parquet,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum TextRole {
    Utf8,
    Categorical,
    Id,
}
#[derive(Clone)]
struct Projection {
    numeric: Vec<String>,
    text: BTreeMap<String, TextRole>,
    missing_values: Vec<String>,
    nan_as_missing: bool,
}
impl Projection {
    fn validate(&self) -> RadResult<()> {
        unique_nonempty(&self.numeric, "Projected numeric names")?;
        if self.numeric.iter().any(|n| self.text.contains_key(n)) {
            return Err(schema("A numeric feature/regression outcome cannot simultaneously use a string metadata/category role; explicitly remap or remove that optional metadata role."));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnConversion {
    pub column: String,
    pub source_dtype: String,
    pub conversion: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputContent {
    pub role: String,
    pub path: PathBuf,
    pub sha256: String,
    pub rows: usize,
    pub columns: Vec<String>,
    pub projected_columns: Vec<String>,
    pub ignored_columns: Vec<String>,
    pub source_dtypes: BTreeMap<String, String>,
    pub conversions: Vec<ColumnConversion>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinReport {
    pub role: String,
    pub key: ClinicalJoinKey,
    pub policy: JoinPolicy,
    pub matched_base_rows: usize,
    pub missing_base_sample_ids: Vec<String>,
    pub missing_keys: Vec<String>,
    pub unused_auxiliary_keys: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadReport {
    pub base_rows: usize,
    pub inputs: Vec<InputContent>,
    pub joins: Vec<JoinReport>,
    pub missing_optional_metadata: Vec<String>,
    pub defaulted_cohort: Option<String>,
    pub base_missing_values: Vec<String>,
    pub base_missing_source: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadedStudy {
    pub data: StudyData,
    pub report: LoadReport,
    pub feature_manifest: Option<FeatureTableManifest>,
}

struct RawTable {
    headers: Vec<String>,
    numeric: NumericBuffer,
    numeric_names: Vec<String>,
    text: BTreeMap<String, Vec<Option<String>>>,
    embedded_manifest: Option<FeatureTableManifest>,
    report: InputContent,
}
impl RawTable {
    fn has(&self, name: &str) -> bool {
        self.headers.iter().any(|n| n == name)
    }
    fn text(&self, name: &str, row: usize) -> Option<&str> {
        self.text.get(name).and_then(|v| v[row].as_deref())
    }
    fn rows(&self) -> usize {
        self.numeric.rows
    }
}
fn header(path: &Path) -> RadResult<Vec<String>> {
    match format(path)? {
        TableFormat::Csv => csv_io::header(path),
        TableFormat::Parquet => parquet_io::header(path),
    }
}
fn read_raw(path: &Path, role: &str, projection: &Projection) -> RadResult<RawTable> {
    projection.validate()?;
    let initial = fs::metadata(path)?;
    let mut table = match format(path)? {
        TableFormat::Csv => csv_io::read(path, role, projection)?,
        TableFormat::Parquet => parquet_io::read(path, role, projection)?,
    };
    table.report.sha256 = sha256_file(path)?;
    let final_metadata = fs::metadata(path)?;
    if initial.len() != final_metadata.len()
        || initial.modified().ok() != final_metadata.modified().ok()
    {
        return Err(RadiomicsError::new(ErrorCode::Io,"An input changed while it was being parsed/hashed; no stable study content can be reported."));
    }
    Ok(table)
}
fn numeric_value(value: f64, nan_as_missing: bool, column: &str) -> RadResult<Option<f64>> {
    if value.is_nan() && nan_as_missing {
        Ok(None)
    } else if !value.is_finite() {
        Err(schema(format!("Nonfinite numeric value in '{column}'; NaN requires explicit nan_as_missing and infinity is not supported.")))
    } else {
        Ok(Some(value))
    }
}
fn raw_shell(
    path: &Path,
    role: &str,
    headers: Vec<String>,
    numeric_names: Vec<String>,
    projection: &Projection,
    rows: usize,
) -> RadResult<RawTable> {
    unique_nonempty(&headers, "Table column names")?;
    let size = rows
        .checked_mul(numeric_names.len())
        .ok_or_else(|| invalid("Input numeric dimensions overflow usize."))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| invalid("Input numeric buffer cannot be allocated."))?;
    values.resize(size, 0.0);
    let mut missing = Vec::new();
    missing
        .try_reserve_exact(size)
        .map_err(|_| invalid("Input missing-state buffer cannot be allocated."))?;
    missing.resize(size, true);
    let text = projection
        .text
        .keys()
        .filter(|n| headers.contains(n))
        .map(|n| (n.clone(), vec![None; rows]))
        .collect();
    let projected_columns: Vec<_> = headers
        .iter()
        .filter(|n| numeric_names.contains(n) || projection.text.contains_key(*n))
        .cloned()
        .collect();
    let ignored_columns = headers
        .iter()
        .filter(|n| !projected_columns.contains(n))
        .cloned()
        .collect();
    Ok(RawTable {
        headers: headers.clone(),
        numeric: NumericBuffer {
            rows,
            cols: numeric_names.len(),
            values,
            missing,
        },
        numeric_names,
        text,
        embedded_manifest: None,
        report: InputContent {
            role: role.into(),
            path: path.into(),
            sha256: String::new(),
            rows,
            columns: headers,
            projected_columns,
            ignored_columns,
            source_dtypes: BTreeMap::new(),
            conversions: Vec::new(),
        },
    })
}
fn required_ids(table: &RawTable, column: &str) -> RadResult<Vec<String>> {
    if !table.has(column) {
        return Err(schema(format!(
            "Missing ID column '{column}' in {}.",
            table.report.path.display()
        )));
    }
    let values: Vec<_> = (0..table.rows())
        .map(|i| {
            table
                .text(column, i)
                .filter(|v| !v.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    alignment(format!(
                        "Missing ID '{column}' on row {} in {}.",
                        i + 2,
                        table.report.path.display()
                    ))
                })
        })
        .collect::<RadResult<_>>()?;
    unique_nonempty(
        &values,
        &format!("{} ID column '{column}'", table.report.role),
    )?;
    Ok(values)
}
struct Joined {
    table: RawTable,
    positions: Vec<Option<usize>>,
    report: JoinReport,
}
fn join(
    table: RawTable,
    column: &str,
    base_keys: &[String],
    base_ids: &[String],
    key: ClinicalJoinKey,
    policy: JoinPolicy,
) -> RadResult<Joined> {
    let keys = required_ids(&table, column)?;
    let lookup: BTreeMap<_, _> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let expected: BTreeSet<_> = base_keys.iter().cloned().collect();
    let positions: Vec<_> = base_keys
        .iter()
        .map(|k| lookup.get(k.as_str()).copied())
        .collect();
    let missing_base_sample_ids = positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.is_none().then_some(base_ids[i].clone()))
        .collect::<Vec<_>>();
    let missing_keys = positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.is_none().then_some(base_keys[i].clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let unused_auxiliary_keys = keys
        .into_iter()
        .filter(|k| !expected.contains(k))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if policy == JoinPolicy::Exact
        && (!missing_keys.is_empty() || !unused_auxiliary_keys.is_empty())
    {
        return Err(alignment(format!("Exact {} join ID sets differ. Missing keys: {missing_keys:?}; unused auxiliary keys: {unused_auxiliary_keys:?}.",table.report.role)));
    }
    let report = JoinReport {
        role: table.report.role.clone(),
        key,
        policy,
        matched_base_rows: positions.iter().filter(|p| p.is_some()).count(),
        missing_base_sample_ids,
        missing_keys,
        unused_auxiliary_keys,
    };
    Ok(Joined {
        table,
        positions,
        report,
    })
}
fn agree<T: PartialEq + std::fmt::Debug>(
    current: Option<T>,
    candidate: Option<T>,
    name: &str,
    sample: &str,
) -> RadResult<Option<T>> {
    match (current, candidate) {
        (Some(a), Some(b)) if a != b => Err(alignment(format!(
            "Conflicting nonmissing field '{name}' for sample '{sample}': {a:?} versus {b:?}."
        ))),
        (Some(a), _) => Ok(Some(a)),
        (None, b) => Ok(b),
    }
}
fn merged_text<'a>(
    name: &str,
    row: usize,
    identity: (&'a str, &str),
    base: &'a RawTable,
    metadata: Option<&'a Joined>,
    clinical: Option<&'a Joined>,
    manifest: Option<&'a str>,
) -> RadResult<Option<&'a str>> {
    let (id, id_column) = identity;
    let mut value = if name == id_column {
        Some(id)
    } else {
        base.text(name, row)
    };
    for source in [metadata, clinical].into_iter().flatten() {
        let candidate = source.positions[row].and_then(|j| source.table.text(name, j));
        value = agree(value, candidate, name, id)?;
    }
    agree(value, manifest, name, id)
}
fn sample_optional<'a>(sample: Option<&'a SampleMetadata>, role: &str) -> Option<&'a str> {
    let sample = sample?;
    match role {
        "slide_id" => sample.slide_id.as_deref(),
        "lesion_id" => sample.lesion_id.as_deref(),
        "modality" => sample.modality.as_deref(),
        "region" => sample.region.as_deref(),
        "timepoint" => sample.timepoint.as_deref(),
        "center" => sample.center.as_deref(),
        _ => None,
    }
}
fn manifest_column<'a>(
    sample: Option<&'a SampleMetadata>,
    column: &str,
    spec: &StudyTableSpec,
    id: &str,
) -> RadResult<Option<&'a str>> {
    let Some(sample) = sample else {
        return Ok(None);
    };
    let mut value = None;
    if column == spec.id_column {
        value = Some(sample.sample_id.as_str());
    }
    if column == spec.patient_column {
        value = agree(value, Some(sample.patient_id.as_str()), column, id)?;
    }
    for (role, name) in &spec.optional_metadata {
        if name == column {
            value = agree(value, sample_optional(Some(sample), role), column, id)?;
        }
    }
    Ok(value)
}
fn check_manifest(
    manifest: &FeatureTableManifest,
    path: &Path,
    headers: &[String],
    ids: &[String],
) -> RadResult<()> {
    manifest.validate()?;
    if let Some(expected) = &manifest.table_sha256 {
        if &sha256_file(path)? != expected {
            return Err(schema(
                "Feature manifest table SHA256 differs from the actual file content.",
            ));
        }
    }
    let expected: BTreeSet<_> = manifest
        .samples
        .iter()
        .map(|s| s.sample_id.as_str())
        .collect();
    let actual: BTreeSet<_> = ids.iter().map(String::as_str).collect();
    if expected != actual {
        return Err(alignment("Feature manifest sample IDs differ from the actual table IDs; row counts are not an alignment proof."));
    }
    if manifest.features.iter().any(|f| !headers.contains(&f.name)) {
        return Err(alignment(
            "Feature manifest declares columns absent from the table.",
        ));
    }
    Ok(())
}

/// Read relative paths against the supplied spec directory without changing cwd.
/// Auxiliary features enter X only through explicit feature declarations.
pub fn read_study(spec: &StudyTableSpec, task: Task, base: &Path) -> RadResult<LoadedStudy> {
    spec.validate()?;
    let path = resolve(base, &spec.table)?;
    let headers = header(&path)?;
    let mut text = BTreeMap::new();
    for name in [&spec.id_column, &spec.patient_column] {
        text.insert(name.clone(), TextRole::Id);
    }
    text.insert(spec.cohort_column.clone(), TextRole::Utf8);
    for name in spec.optional_metadata.values() {
        text.entry(name.clone()).or_insert(TextRole::Utf8);
    }
    for name in &spec.categorical_features {
        if text.get(name) != Some(&TextRole::Id) {
            text.insert(name.clone(), TextRole::Categorical);
        }
    }
    let metadata_id = spec.metadata_id_column.as_ref().unwrap_or(&spec.id_column);
    let clinical_id = spec
        .clinical_id_column
        .as_ref()
        .unwrap_or(match spec.clinical_key {
            ClinicalJoinKey::Sample => &spec.id_column,
            ClinicalJoinKey::Patient => &spec.patient_column,
        });
    for name in [metadata_id, clinical_id] {
        text.insert(name.clone(), TextRole::Id);
    }
    let ids: BTreeSet<_> = [
        &spec.id_column,
        &spec.patient_column,
        metadata_id,
        clinical_id,
    ]
    .into_iter()
    .collect();
    if spec
        .numeric_features
        .iter()
        .flatten()
        .chain(&spec.categorical_features)
        .any(|n| ids.contains(n))
    {
        return Err(schema("Declared join ID columns cannot be predictors."));
    }
    let mut numeric_names = spec.numeric_features.clone().unwrap_or_else(|| {
        headers
            .iter()
            .filter(|name| !text.contains_key(*name) && spec.outcome_column.as_ref() != Some(*name))
            .cloned()
            .collect()
    });
    if let Some(outcome) = &spec.outcome_column {
        if task == Task::Regression {
            if numeric_names.contains(outcome) {
                return Err(schema("Regression outcome cannot be a feature."));
            }
        } else {
            if text.get(outcome) != Some(&TextRole::Id) {
                text.insert(outcome.clone(), TextRole::Categorical);
            }
        }
    }
    unique_nonempty(&numeric_names, "Numeric features")?;
    if numeric_names.is_empty() && spec.categorical_features.is_empty() {
        return Err(invalid(
            "No numeric or categorical features were declared or found in the base table.",
        ));
    }
    let feature_count = numeric_names.len();
    if task == Task::Regression {
        if let Some(outcome) = &spec.outcome_column {
            numeric_names.push(outcome.clone());
        }
    }
    let projection = Projection {
        numeric: numeric_names.clone(),
        text,
        missing_values: spec.missing_values.clone(),
        nan_as_missing: spec.nan_as_missing,
    };
    projection.validate()?;
    let mut manifest_content = None;
    let mut external_manifest = None;
    if let Some(manifest_path) = &spec.feature_manifest {
        let manifest_path = resolve(base, manifest_path)?;
        let external: FeatureTableManifest = serde_json::from_reader(File::open(&manifest_path)?)?;
        external.validate()?;
        manifest_content = Some(InputContent {
            role: "feature_manifest".into(),
            path: manifest_path.clone(),
            sha256: sha256_file(&manifest_path)?,
            rows: external.samples.len(),
            columns: external.features.iter().map(|f| f.name.clone()).collect(),
            projected_columns: Vec::new(),
            ignored_columns: Vec::new(),
            source_dtypes: BTreeMap::new(),
            conversions: Vec::new(),
        });
        external_manifest = Some(external);
    }
    let mut base_projection = projection.clone();
    let base_missing_source = if format(&path)? == TableFormat::Parquet {
        "parquet_definition_levels"
    } else if let Some(token) = external_manifest
        .as_ref()
        .and_then(|m| m.csv_null_value.as_ref())
    {
        base_projection.missing_values = vec![token.clone()];
        "feature_manifest.csv_null_value"
    } else {
        "study_spec.missing_values"
    };
    let base_missing_values = if format(&path)? == TableFormat::Parquet {
        Vec::new()
    } else {
        base_projection.missing_values.clone()
    };
    let mut raw = read_raw(&path, "base", &base_projection)?;
    if raw.headers != headers {
        return Err(RadiomicsError::new(
            ErrorCode::Io,
            "Base headers changed between role resolution and reading.",
        ));
    }
    let base_ids = required_ids(&raw, &spec.id_column)?;
    if raw.rows() == 0 {
        return Err(RadiomicsError::new(
            ErrorCode::EmptyResult,
            "Base study table is empty.",
        ));
    }
    let mut feature_manifest = raw.embedded_manifest.take();
    if let Some(external) = external_manifest {
        if let Some(embedded) = &feature_manifest {
            let mut expected = external.clone();
            expected.table_sha256 = None;
            let mut actual = embedded.clone();
            actual.table_sha256 = None;
            if expected != actual {
                return Err(schema("External and embedded feature manifests disagree."));
            }
        }
        feature_manifest = Some(external);
    }
    if let Some(manifest) = &feature_manifest {
        check_manifest(manifest, &path, &headers, &base_ids)?;
    }
    let manifest_samples: BTreeMap<_, _> = feature_manifest
        .as_ref()
        .map(|m| {
            m.samples
                .iter()
                .map(|s| (s.sample_id.as_str(), s))
                .collect()
        })
        .unwrap_or_default();
    let aux_projection = |path: &Path| -> RadResult<Projection> {
        let present = header(path)?;
        let mut projected = projection.clone();
        projected.numeric.retain(|n| present.contains(n));
        Ok(projected)
    };
    let metadata = if let Some(p) = &spec.metadata {
        let p = resolve(base, p)?;
        let table = read_raw(&p, "metadata", &aux_projection(&p)?)?;
        Some(join(
            table,
            metadata_id,
            &base_ids,
            &base_ids,
            ClinicalJoinKey::Sample,
            spec.join_policy,
        )?)
    } else {
        None
    };
    let initial_patients: Vec<_> = base_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let from_manifest = manifest_column(
                manifest_samples.get(id.as_str()).copied(),
                &spec.patient_column,
                spec,
                id,
            )?;
            merged_text(
                &spec.patient_column,
                i,
                (id, &spec.id_column),
                &raw,
                metadata.as_ref(),
                None,
                from_manifest,
            )
            .map(|v| v.map(str::to_string))
        })
        .collect::<RadResult<_>>()?;
    let clinical = if let Some(p) = &spec.clinical {
        let p = resolve(base, p)?;
        let table = read_raw(&p, "clinical", &aux_projection(&p)?)?;
        let keys=match spec.clinical_key {ClinicalJoinKey::Sample=>base_ids.clone(),ClinicalJoinKey::Patient=>initial_patients.iter().zip(&base_ids).map(|(p,id)|p.as_ref().filter(|p|!p.trim().is_empty()).cloned().ok_or_else(||alignment(format!("Patient-keyed clinical join requires known patient metadata for sample '{id}' before the clinical join.")))).collect::<RadResult<_>>()?};
        Some(join(
            table,
            clinical_id,
            &keys,
            &base_ids,
            spec.clinical_key,
            spec.join_policy,
        )?)
    } else {
        None
    };
    let has_physical_column = |name: &str| {
        raw.has(name)
            || metadata.as_ref().is_some_and(|s| s.table.has(name))
            || clinical.as_ref().is_some_and(|s| s.table.has(name))
    };
    let has_column = |name: &str| {
        has_physical_column(name)
            || (feature_manifest.is_some()
                && (name == spec.id_column
                    || name == spec.patient_column
                    || spec.optional_metadata.values().any(|column| column == name)))
    };
    for name in numeric_names.iter().chain(&spec.categorical_features) {
        if !has_column(name) {
            return Err(schema(format!(
                "Missing declared feature/outcome column '{name}' in all input tables."
            )));
        }
    }
    if let Some(outcome) = &spec.outcome_column {
        if !has_column(outcome) {
            return Err(schema(format!(
                "Missing declared outcome column '{outcome}'."
            )));
        }
    }
    let has_cohort = has_column(&spec.cohort_column);
    if !has_cohort && spec.default_cohort.is_none() {
        return Err(schema(
            "No cohort column exists and default_cohort was not provided.",
        ));
    }
    let missing_optional_metadata = spec
        .optional_metadata
        .iter()
        .filter(|(role, column)| {
            !has_physical_column(column)
                && feature_manifest.as_ref().is_none_or(|m| {
                    m.samples
                        .iter()
                        .all(|s| sample_optional(Some(s), role).is_none())
                })
        })
        .map(|(role, _)| role.clone())
        .collect();
    let mut samples = Vec::with_capacity(raw.rows());
    let mut cohorts = Vec::with_capacity(raw.rows());
    let mut categorical_values = Vec::with_capacity(raw.rows() * spec.categorical_features.len());
    let mut class_outcomes = Vec::with_capacity(raw.rows());
    let mut regression_outcomes = Vec::with_capacity(raw.rows());
    let auxiliary_numeric: Vec<_> = [metadata.as_ref(), clinical.as_ref()]
        .into_iter()
        .flatten()
        .map(|source| {
            (
                source,
                numeric_names
                    .iter()
                    .map(|name| source.table.numeric_names.iter().position(|n| n == name))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    for (i, id) in base_ids.iter().enumerate() {
        for (j, name) in numeric_names.iter().enumerate() {
            let index = i * raw.numeric.cols + j;
            let mut value = (!raw.numeric.missing[index]).then_some(raw.numeric.values[index]);
            for (source, columns) in &auxiliary_numeric {
                let candidate = source.positions[i].zip(columns[j]).and_then(|(row, col)| {
                    let index = row * source.table.numeric.cols + col;
                    (!source.table.numeric.missing[index])
                        .then_some(source.table.numeric.values[index])
                });
                value = agree(value, candidate, name, id)?;
            }
            raw.numeric.values[index] = value.unwrap_or(0.0);
            raw.numeric.missing[index] = value.is_none();
        }
        let manifest = manifest_samples.get(id.as_str()).copied();
        let from_manifest = |name: &str| manifest_column(manifest, name, spec, id);
        // Join-key aliases do not authorize a conflicting same-named sample_id
        // payload. Verify it even though the base ID remains the output key.
        merged_text(
            &spec.id_column,
            i,
            (id, &spec.id_column),
            &raw,
            metadata.as_ref(),
            clinical.as_ref(),
            from_manifest(&spec.id_column)?,
        )?;
        let patient = merged_text(
            &spec.patient_column,
            i,
            (id, &spec.id_column),
            &raw,
            metadata.as_ref(),
            clinical.as_ref(),
            from_manifest(&spec.patient_column)?,
        )?
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| alignment(format!("Missing required patient ID for sample '{id}'.")))?
        .to_string();
        let cohort = if has_cohort {
            merged_text(&spec.cohort_column,i,(id,&spec.id_column),&raw,metadata.as_ref(),clinical.as_ref(),from_manifest(&spec.cohort_column)?)?.filter(|s|!s.trim().is_empty()).ok_or_else(||alignment(format!("Missing cohort value for sample '{id}'; default_cohort applies only when no cohort column exists.")))?.to_string()
        } else {
            spec.default_cohort.as_ref().unwrap().clone()
        };
        cohorts.push(cohort);
        let mut optional = BTreeMap::new();
        for (role, column) in &spec.optional_metadata {
            optional.insert(
                role.as_str(),
                merged_text(
                    column,
                    i,
                    (id, &spec.id_column),
                    &raw,
                    metadata.as_ref(),
                    clinical.as_ref(),
                    from_manifest(column)?,
                )?
                .map(str::to_string),
            );
        }
        samples.push(SampleMetadata {
            sample_id: id.clone(),
            patient_id: patient,
            slide_id: optional.remove("slide_id").flatten(),
            lesion_id: optional.remove("lesion_id").flatten(),
            modality: optional.remove("modality").flatten(),
            region: optional.remove("region").flatten(),
            timepoint: optional.remove("timepoint").flatten(),
            center: optional.remove("center").flatten(),
        });
        for name in &spec.categorical_features {
            categorical_values.push(
                merged_text(
                    name,
                    i,
                    (id, &spec.id_column),
                    &raw,
                    metadata.as_ref(),
                    clinical.as_ref(),
                    from_manifest(name)?,
                )?
                .map(str::to_string),
            );
        }
        if let Some(outcome) = &spec.outcome_column {
            if task == Task::Regression {
                let j = numeric_names.len() - 1;
                let index = i * raw.numeric.cols + j;
                regression_outcomes
                    .push((!raw.numeric.missing[index]).then_some(raw.numeric.values[index]));
            } else {
                class_outcomes.push(
                    merged_text(
                        outcome,
                        i,
                        (id, &spec.id_column),
                        &raw,
                        metadata.as_ref(),
                        clinical.as_ref(),
                        from_manifest(outcome)?,
                    )?
                    .map(str::to_string),
                );
            }
        }
    }
    if raw.numeric.cols != feature_count {
        let old = raw.numeric.cols;
        for i in 0..raw.rows() {
            raw.numeric
                .values
                .copy_within(i * old..i * old + feature_count, i * feature_count);
            raw.numeric
                .missing
                .copy_within(i * old..i * old + feature_count, i * feature_count);
        }
        raw.numeric.values.truncate(raw.rows() * feature_count);
        raw.numeric.missing.truncate(raw.rows() * feature_count);
        raw.numeric.cols = feature_count;
    }
    numeric_names.truncate(feature_count);
    drop(auxiliary_numeric);
    let mut inputs = vec![raw.report];
    let mut joins = Vec::new();
    for aux in [metadata, clinical].into_iter().flatten() {
        inputs.push(aux.table.report);
        joins.push(aux.report);
    }
    if let Some(content) = manifest_content {
        inputs.push(content);
    }
    let data = StudyData {
        samples,
        cohorts,
        features: TabularData {
            rows: raw.numeric.rows,
            numeric_names,
            numeric: raw.numeric,
            categorical_names: spec.categorical_features.clone(),
            categorical_values,
        },
        outcome: match &spec.outcome_column {
            None => Outcome::Unlabeled,
            Some(name) if task == Task::Regression => Outcome::Regression {
                name: name.clone(),
                values: regression_outcomes,
            },
            Some(name) => Outcome::Classification {
                name: name.clone(),
                values: class_outcomes,
            },
        },
    };
    data.validate()?;
    Ok(LoadedStudy {
        report: LoadReport {
            base_rows: data.samples.len(),
            inputs,
            joins,
            missing_optional_metadata,
            defaulted_cohort: (!has_cohort).then(|| spec.default_cohort.clone()).flatten(),
            base_missing_values,
            base_missing_source: base_missing_source.into(),
        },
        data,
        feature_manifest,
    })
}

#[cfg(test)]
mod tests;
