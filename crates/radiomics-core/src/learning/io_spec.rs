//! Versioned column-role and strict join contract for native study table loading.
use super::invalid;
use crate::{
    contracts::{unique_nonempty, FeatureDefinition, QualityRecord, SampleMetadata},
    errors::{ErrorCode, RadResult, RadiomicsError},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ClinicalJoinKey {
    #[default]
    Sample,
    Patient,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum JoinPolicy {
    #[default]
    Exact,
    Left,
}
fn sample_column() -> String {
    "sample_id".into()
}
fn patient_column() -> String {
    "patient_id".into()
}
fn cohort_column() -> String {
    "cohort".into()
}
fn missing_values() -> Vec<String> {
    vec![String::new()]
}
fn optional_columns() -> BTreeMap<String, String> {
    [
        "slide_id",
        "lesion_id",
        "modality",
        "region",
        "timepoint",
        "center",
    ]
    .into_iter()
    .map(|n| (n.into(), n.into()))
    .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyTableSpec {
    pub schema_version: u32,
    /// CSV or flat Parquet; paths are relative to the specification file.
    pub table: PathBuf,
    #[serde(default)]
    pub metadata: Option<PathBuf>,
    #[serde(default)]
    pub clinical: Option<PathBuf>,
    #[serde(default)]
    pub clinical_key: ClinicalJoinKey,
    #[serde(default)]
    pub join_policy: JoinPolicy,
    #[serde(default = "sample_column")]
    pub id_column: String,
    #[serde(default = "patient_column")]
    pub patient_column: String,
    #[serde(default = "cohort_column")]
    pub cohort_column: String,
    #[serde(default)]
    pub metadata_id_column: Option<String>,
    #[serde(default)]
    pub clinical_id_column: Option<String>,
    #[serde(default)]
    pub outcome_column: Option<String>,
    /// Used only if no cohort column exists, e.g. an unlabeled external cohort.
    #[serde(default)]
    pub default_cohort: Option<String>,
    /// None selects base-table feature columns only, excluding declared roles.
    /// Clinical auxiliary columns are never automatically made model features.
    #[serde(default)]
    pub numeric_features: Option<Vec<String>>,
    #[serde(default)]
    pub categorical_features: Vec<String>,
    #[serde(default = "optional_columns")]
    pub optional_metadata: BTreeMap<String, String>,
    #[serde(default = "missing_values")]
    pub missing_values: Vec<String>,
    #[serde(default)]
    pub nan_as_missing: bool,
    #[serde(default)]
    pub feature_manifest: Option<PathBuf>,
}
impl StudyTableSpec {
    pub fn validate(&self) -> RadResult<()> {
        if self.schema_version != 1 {
            return Err(RadiomicsError::new(
                ErrorCode::Schema,
                "Unknown study table schema version.",
            ));
        }
        if self.table.as_os_str().is_empty()
            || [&self.id_column, &self.patient_column, &self.cohort_column]
                .iter()
                .any(|s| s.trim().is_empty())
        {
            return Err(invalid(
                "Base table path and ID/patient/cohort column roles must be nonempty.",
            ));
        }
        if [
            self.metadata_id_column.as_ref(),
            self.clinical_id_column.as_ref(),
            self.outcome_column.as_ref(),
            self.default_cohort.as_ref(),
        ]
        .iter()
        .flatten()
        .any(|s| s.trim().is_empty())
        {
            return Err(invalid(
                "Optional column roles/cohort labels must be nonempty when provided.",
            ));
        }
        for (role, column) in &self.optional_metadata {
            if !matches!(
                role.as_str(),
                "slide_id" | "lesion_id" | "modality" | "region" | "timepoint" | "center"
            ) || column.trim().is_empty()
            {
                return Err(invalid(
                    "Unknown optional sample-metadata role or empty column name.",
                ));
            }
        }
        let mut features = self.numeric_features.clone().unwrap_or_default();
        features.extend(self.categorical_features.clone());
        unique_nonempty(&features, "Explicit feature names")?;
        for feature in features {
            if [&self.id_column, &self.patient_column, &self.cohort_column].contains(&&feature)
                || self.outcome_column.as_ref() == Some(&feature)
            {
                return Err(invalid(
                    "IDs, cohort and outcome cannot be declared as model features.",
                ));
            }
        }
        if self
            .missing_values
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != self.missing_values.len()
        {
            return Err(invalid("Missing-value tokens must be unique."));
        }
        Ok(())
    }
}

/// Small sidecar/Parquet metadata retains the feature contract without duplicating values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureTableManifest {
    pub format: String,
    pub schema_version: u32,
    pub samples: Vec<SampleMetadata>,
    pub features: Vec<FeatureDefinition>,
    pub effective_parameters: BTreeMap<String, serde_json::Value>,
    pub quality: Vec<QualityRecord>,
    pub csv_null_value: Option<String>,
    /// Present on an external sidecar, absent in the same file's embedded metadata.
    pub table_sha256: Option<String>,
}
impl FeatureTableManifest {
    pub fn validate(&self) -> RadResult<()> {
        if self.format != "radiomics.feature-table" || self.schema_version != 1 {
            return Err(RadiomicsError::new(
                ErrorCode::Schema,
                "Unknown feature table manifest format/schema.",
            ));
        }
        unique_nonempty(
            &self
                .samples
                .iter()
                .map(|s| s.sample_id.clone())
                .collect::<Vec<_>>(),
            "Manifest sample IDs",
        )?;
        unique_nonempty(
            &self
                .features
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
            "Manifest feature names",
        )?;
        for sample in &self.samples {
            sample.validate()?;
        }
        if self.features.is_empty()
            || self
                .table_sha256
                .as_ref()
                .is_some_and(|h| h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err(invalid(
                "Feature manifest needs named definitions and a valid optional content SHA256.",
            ));
        }
        Ok(())
    }
}
