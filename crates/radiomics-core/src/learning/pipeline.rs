use super::{digest, invalid, StudyData};
use crate::{
    bundle::{FittedState, ModelManifest},
    contracts::*,
    errors::RadResult,
    models::{self, FittedModel, ModelPrediction},
    preprocessing::{self, FittedPreprocessor, FittedSelector, PreprocessConfig, SelectorConfig},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    pub preprocessing: PreprocessConfig,
    pub selector: Option<SelectorConfig>,
    pub model: models::ModelSpec,
}
impl PipelineSpec {
    pub fn validate(&self) -> RadResult<()> {
        self.model.validate()?;
        if let Some(s) = &self.selector {
            s.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedPipeline {
    pub spec: PipelineSpec,
    pub preprocessing: FittedPreprocessor,
    pub selector: Option<FittedSelector>,
    pub model: FittedModel,
    pub selected_features: Vec<String>,
    pub training_samples: BTreeMap<String, String>,
    pub training_data_sha256: String,
    pub fit_unit: PredictionUnit,
}
impl FittedPipeline {
    pub fn validate_state(&self) -> RadResult<()> {
        self.spec.validate()?;
        if self.training_data_sha256.len() != 64
            || !self
                .training_data_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid(
                "Pipeline training content SHA256 is missing or invalid.",
            ));
        }
        self.preprocessing.validate()?;
        self.model.validate()?;
        if self.spec.model != self.model.spec
            || self.spec.preprocessing != self.preprocessing.config
        {
            return Err(invalid(
                "Fitted pipeline state differs from its declared configuration.",
            ));
        }
        let expected = match (&self.selector, &self.spec.selector) {
            (Some(fitted), Some(spec)) => {
                fitted.validate()?;
                if &fitted.config != spec || fitted.feature_names != self.preprocessing.output_names
                {
                    return Err(invalid(
                        "Fitted selector schema/configuration differs from preprocessing.",
                    ));
                }
                fitted.selected_names()?
            }
            (None, None) => self.preprocessing.output_names.clone(),
            _ => {
                return Err(invalid(
                    "Fitted selector is missing or unexpectedly present.",
                ))
            }
        };
        if expected != self.selected_features
            || expected.len() != self.model.feature_count
            || self.training_samples.len() != self.preprocessing.training_rows
            || self.training_samples.values().any(|p| p.trim().is_empty())
        {
            return Err(invalid(
                "Pipeline feature order, model dimensions or training provenance disagree.",
            ));
        }
        unique_nonempty(
            &self.training_samples.keys().cloned().collect::<Vec<_>>(),
            "Fitted training sample IDs",
        )?;
        if self.fit_unit == PredictionUnit::Patient
            && self
                .training_samples
                .values()
                .collect::<BTreeSet<_>>()
                .len()
                != self.training_samples.len()
        {
            return Err(invalid(
                "A patient-level fitted pipeline must have one training row per patient.",
            ));
        }
        Ok(())
    }
    pub fn predict(&self, data: &TabularData) -> RadResult<ModelPrediction> {
        self.validate_state()?;
        let transformed = self.preprocessing.transform(data)?;
        let transformed = if let Some(selector) = &self.selector {
            selector.transform(&transformed, &self.preprocessing.output_names)?
        } else {
            transformed
        };
        self.model.predict(&transformed)
    }
    pub fn required_features(&self) -> Vec<String> {
        self.preprocessing
            .numeric
            .iter()
            .map(|s| s.name.clone())
            .chain(
                self.preprocessing
                    .categorical
                    .iter()
                    .map(|s| s.name.clone()),
            )
            .collect()
    }
    pub fn training_patients(&self) -> BTreeSet<String> {
        self.training_samples.values().cloned().collect()
    }
    pub fn state_sha256(&self) -> RadResult<String> {
        self.validate_state()?;
        digest(self)
    }
}
impl FittedState for FittedPipeline {
    fn validate(&self, manifest: &ModelManifest) -> RadResult<()> {
        self.validate_state()?;
        if manifest
            .training_configuration
            .get("model_kind")
            .and_then(|v| v.as_str())
            != Some("pipeline")
        {
            return Err(invalid(
                "Unknown or incompatible saved pipeline model type.",
            ));
        }
        if self.required_features() != manifest.feature_names {
            return Err(invalid(
                "Model manifest raw feature order differs from the fitted pipeline.",
            ));
        }
        match self.model.classes() {
            Some(classes)
                if classes == manifest.classes
                    && matches!(manifest.task, Task::Binary | Task::Multiclass) => {}
            None if manifest.task == Task::Regression && manifest.classes.is_empty() => (),
            _ => {
                return Err(invalid(
                    "Pipeline task/classes differ from the model manifest.",
                ))
            }
        }
        let plan: crate::split::SplitPlan = serde_json::from_value(
            manifest
                .training_configuration
                .get("split_plan")
                .cloned()
                .ok_or_else(|| {
                    invalid("Fitted pipeline manifest is missing its typed SplitPlan.")
                })?,
        )?;
        plan.validate()?;
        let training = plan
            .cohorts
            .get("train")
            .ok_or_else(|| invalid("Model SplitPlan has no training cohort."))?;
        let expected: BTreeMap<_, _> = plan
            .sample_patients
            .iter()
            .filter(|(_, p)| training.contains(*p))
            .map(|(s, p)| (s.clone(), p.clone()))
            .collect();
        if plan.fingerprint()? != manifest.split_plan_sha256
            || plan.task != manifest.task
            || plan.classes != manifest.classes
            || expected != self.training_samples
        {
            return Err(invalid(
                "Saved pipeline training patients/classes differ from its declared SplitPlan.",
            ));
        }
        Ok(())
    }
}
pub(super) fn fit(
    data: &StudyData,
    spec: &PipelineSpec,
    task: Task,
    classes: &[String],
    unit: PredictionUnit,
) -> RadResult<FittedPipeline> {
    data.validate_unit(unit)?;
    spec.validate()?;
    let target = data.target(task, classes)?;
    let preprocessing = preprocessing::fit_preprocessor(&data.features, &spec.preprocessing)?;
    let x = preprocessing.transform(&data.features)?;
    let selector = spec
        .selector
        .as_ref()
        .map(|config| {
            preprocessing::fit_selector(
                &x,
                &preprocessing.output_names,
                &target,
                config,
                Some(&preprocessing.unscaled_variances()?),
            )
        })
        .transpose()?;
    let (selected_features, x) = if let Some(selector) = &selector {
        (
            selector.selected_names()?,
            selector.transform(&x, &preprocessing.output_names)?,
        )
    } else {
        (preprocessing.output_names.clone(), x)
    };
    let model = models::fit(&spec.model, &x, &target)?;
    let fitted = FittedPipeline {
        spec: spec.clone(),
        preprocessing,
        selector,
        model,
        selected_features,
        training_samples: data
            .samples
            .iter()
            .map(|s| (s.sample_id.clone(), s.patient_id.clone()))
            .collect(),
        training_data_sha256: data.content_sha256()?,
        fit_unit: unit,
    };
    fitted.validate_state()?;
    Ok(fitted)
}
