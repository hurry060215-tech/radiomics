use super::invalid;
use crate::{contracts::*, errors::RadResult, models::ModelTarget, split::SplitObservation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Unlabeled,
    Classification {
        name: String,
        values: Vec<Option<String>>,
    },
    Regression {
        name: String,
        values: Vec<Option<f64>>,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyData {
    pub samples: Vec<SampleMetadata>,
    pub cohorts: Vec<String>,
    pub features: TabularData,
    pub outcome: Outcome,
}
impl StudyData {
    /// Identity of this ordered training subset without retaining a second matrix.
    pub fn content_sha256(&self) -> RadResult<String> {
        use sha2::{Digest, Sha256};
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"radiomics-study-content-v1\0");
        hash.update(serde_json::to_vec(&(
            &self.samples,
            &self.cohorts,
            self.features.rows,
            &self.features.numeric_names,
            &self.features.categorical_names,
            &self.features.categorical_values,
            &self.outcome,
        ))?);
        let mut block = [0u8; 8192];
        for chunk in self.features.numeric.values.chunks(1024) {
            for (i, value) in chunk.iter().enumerate() {
                block[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
            }
            hash.update(&block[..chunk.len() * 8]);
        }
        for chunk in self.features.numeric.missing.chunks(8192) {
            for (i, &value) in chunk.iter().enumerate() {
                block[i] = u8::from(value);
            }
            hash.update(&block[..chunk.len()]);
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    pub fn validate(&self) -> RadResult<()> {
        self.features.validate()?;
        if self.samples.len() != self.features.rows
            || self.cohorts.len() != self.samples.len()
            || self.cohorts.iter().any(|s| s.trim().is_empty())
        {
            return Err(invalid("Study samples, cohorts and feature rows disagree."));
        }
        for sample in &self.samples {
            sample.validate()?;
        }
        validate_patient_cohorts(
            self.samples
                .iter()
                .zip(&self.cohorts)
                .map(|(sample, cohort)| (sample.patient_id.as_str(), cohort.as_str())),
        )?;
        unique_nonempty(
            &self
                .samples
                .iter()
                .map(|s| s.sample_id.clone())
                .collect::<Vec<_>>(),
            "Study sample IDs",
        )?;
        let name = match &self.outcome {
            Outcome::Classification { name, values } => {
                if values.len() != self.samples.len()
                    || values.iter().flatten().any(|s| s.trim().is_empty())
                {
                    return Err(invalid(
                        "Classification outcome dimensions or labels are invalid.",
                    ));
                }
                Some(name)
            }
            Outcome::Regression { name, values } => {
                if values.len() != self.samples.len()
                    || values.iter().flatten().any(|v| !v.is_finite())
                {
                    return Err(invalid("Regression outcomes require finite observed values and explicit missing markers."));
                }
                Some(name)
            }
            Outcome::Unlabeled => None,
        };
        if name.is_some_and(|name| name.trim().is_empty()) {
            return Err(invalid("Outcome name must be explicit."));
        }
        for feature in self
            .features
            .numeric_names
            .iter()
            .chain(&self.features.categorical_names)
        {
            if name.is_some_and(|name| feature == name)
                || matches!(
                    feature.as_str(),
                    "ID" | "sample_id"
                        | "patient_id"
                        | "slide_id"
                        | "tile_id"
                        | "lesion_id"
                        | "cohort"
                        | "split"
                        | "group"
                )
            {
                return Err(invalid(
                    "IDs, cohort assignments and the outcome cannot enter the feature matrix.",
                ));
            }
        }
        Ok(())
    }
    pub fn observations(&self) -> RadResult<Vec<SplitObservation>> {
        self.validate()?;
        Ok(self
            .samples
            .iter()
            .enumerate()
            .map(|(i, s)| SplitObservation {
                sample_id: s.sample_id.clone(),
                patient_id: s.patient_id.clone(),
                cohort: self.cohorts[i].clone(),
                class_label: match &self.outcome {
                    Outcome::Classification { values, .. } => values[i].clone(),
                    _ => None,
                },
                center: s.center.clone(),
            })
            .collect())
    }
    pub fn indices_for(&self, patients: &BTreeSet<String>) -> Vec<usize> {
        self.samples
            .iter()
            .enumerate()
            .filter(|(_, s)| patients.contains(&s.patient_id))
            .map(|(i, _)| i)
            .collect()
    }
    pub fn take_rows(&self, indices: &[usize]) -> RadResult<Self> {
        self.validate()?;
        if indices.is_empty() || indices.iter().any(|&i| i >= self.samples.len()) {
            return Err(invalid(
                "A study subset must contain valid nonempty row indices.",
            ));
        }
        let data = &self.features;
        let mut numeric = Vec::new();
        let mut missing = Vec::new();
        let mut categorical = Vec::new();
        for &i in indices {
            numeric.extend_from_slice(
                &data.numeric.values[i * data.numeric.cols..(i + 1) * data.numeric.cols],
            );
            missing.extend_from_slice(
                &data.numeric.missing[i * data.numeric.cols..(i + 1) * data.numeric.cols],
            );
            categorical.extend_from_slice(
                &data.categorical_values
                    [i * data.categorical_names.len()..(i + 1) * data.categorical_names.len()],
            );
        }
        let result = Self {
            samples: indices.iter().map(|&i| self.samples[i].clone()).collect(),
            cohorts: indices.iter().map(|&i| self.cohorts[i].clone()).collect(),
            features: TabularData {
                rows: indices.len(),
                numeric_names: data.numeric_names.clone(),
                numeric: NumericBuffer {
                    rows: indices.len(),
                    cols: data.numeric.cols,
                    values: numeric,
                    missing,
                },
                categorical_names: data.categorical_names.clone(),
                categorical_values: categorical,
            },
            outcome: match &self.outcome {
                Outcome::Unlabeled => Outcome::Unlabeled,
                Outcome::Classification { name, values } => Outcome::Classification {
                    name: name.clone(),
                    values: indices.iter().map(|&i| values[i].clone()).collect(),
                },
                Outcome::Regression { name, values } => Outcome::Regression {
                    name: name.clone(),
                    values: indices.iter().map(|&i| values[i]).collect(),
                },
            },
        };
        result.validate()?;
        Ok(result)
    }
    pub fn canonical(&self) -> RadResult<Self> {
        let mut indices: Vec<_> = (0..self.samples.len()).collect();
        indices.sort_by(|&a, &b| self.samples[a].sample_id.cmp(&self.samples[b].sample_id));
        self.take_rows(&indices)
    }
    pub fn target(&self, task: Task, classes: &[String]) -> RadResult<ModelTarget> {
        self.validate()?;
        match (&self.outcome,task) {
            (Outcome::Classification{values,..},Task::Binary|Task::Multiclass)=>Ok(ModelTarget::Classification{classes:classes.to_vec(),labels:values.iter().map(|v|v.as_ref().and_then(|label|classes.iter().position(|c|c==label)).ok_or_else(||invalid("Training/scoring requires a known observed class on every selected row."))).collect::<RadResult<_>>()?}),
            (Outcome::Regression{values,..},Task::Regression)=>Ok(ModelTarget::Regression{values:values.iter().map(|v|v.ok_or_else(||invalid("Training/scoring requires an observed regression outcome on every selected row."))).collect::<RadResult<_>>()?}),
            _=>Err(invalid("Study outcome type differs from the declared task.")),
        }
    }
    pub fn validate_unit(&self, unit: PredictionUnit) -> RadResult<()> {
        self.validate()?;
        let identifiers: Vec<_> = self
            .samples
            .iter()
            .map(|s| match unit {
                PredictionUnit::Patient => s.patient_id.clone(),
                PredictionUnit::Slide => s.slide_id.clone().unwrap_or_default(),
                _ => s.sample_id.clone(),
            })
            .collect();
        unique_nonempty(&identifiers, "Prediction/training unit IDs")?;
        if matches!(unit, PredictionUnit::Slide | PredictionUnit::Tile)
            && self
                .samples
                .iter()
                .any(|s| s.slide_id.as_ref().is_none_or(|v| v.trim().is_empty()))
        {
            return Err(invalid(
                "Slide/tile observations require explicit parent slide IDs.",
            ));
        }
        Ok(())
    }
}
