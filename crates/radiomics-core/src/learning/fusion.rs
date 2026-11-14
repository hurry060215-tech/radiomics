//! Patient-level probability fusion with verified shared OOF training provenance.
use super::{
    digest, invalid, predict, training, FitAudit, FittedPipeline, StudyData, TrainingResult,
};
use crate::{
    bundle::{FittedState, ModelBundle, ModelManifest, MODEL_SCHEMA_VERSION},
    contracts::*,
    errors::{ErrorCode, RadResult, RadiomicsError},
    models::{self, FittedModel, ModelPrediction, ModelSpec, ModelTarget},
    split::{FoldPartition, SplitPlan},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub struct FusionInput<'a> {
    pub name: &'a str,
    pub training: &'a TrainingResult,
    pub model_name: &'a str,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum FusionMethod {
    Weighted {
        weights: Vec<f64>,
    },
    Logistic {
        lambda: f64,
        tolerance: f64,
        max_iterations: usize,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionConfig {
    pub method: FusionMethod,
    pub threshold: Option<f64>,
}
impl FusionConfig {
    fn validate(&self, task: Task, count: usize) -> RadResult<()> {
        if count < 2 || task == Task::Regression {
            return Err(invalid(
                "Probability/Logistic fusion requires at least two classification components.",
            ));
        }
        match &self.method {
            FusionMethod::Weighted { weights } => {
                let sum = weights.iter().sum::<f64>();
                if weights.len() != count
                    || weights.iter().any(|v| !v.is_finite() || *v < 0.)
                    || !sum.is_finite()
                    || sum <= 0.
                {
                    return Err(invalid("Explicit fusion weights must be finite, nonnegative, match components and have positive finite sum."));
                }
            }
            FusionMethod::Logistic {
                lambda,
                tolerance,
                max_iterations,
            } => ModelSpec::LR {
                lambda: *lambda,
                tolerance: *tolerance,
                max_iterations: *max_iterations,
            }
            .validate()?,
        }
        if task == Task::Binary {
            if self
                .threshold
                .is_none_or(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            {
                return Err(invalid(
                    "Binary fusion requires an explicit threshold in [0,1].",
                ));
            }
        } else if self.threshold.is_some() {
            return Err(invalid(
                "A binary threshold does not apply to multiclass fusion.",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionOofSource {
    pub fold: usize,
    pub partition: FoldPartition,
    pub fit: FitAudit,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedComponent {
    pub name: String,
    pub base_model_name: String,
    pub model: ModelBundle<FittedPipeline>,
    pub oof: PredictionTable,
    pub oof_sources: Vec<FusionOofSource>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedFusion {
    pub config: FusionConfig,
    pub split_plan: SplitPlan,
    pub components: Vec<FittedComponent>,
    pub meta_feature_names: Vec<String>,
    pub meta_training_sha256: String,
    pub meta_model: Option<FittedModel>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionResult {
    pub model: ModelBundle<FittedFusion>,
    pub training: PredictionTable,
    pub training_prediction_type: String,
    pub predictions: PredictionTable,
}

/// Actual sample/slide/tile predictions become patient predictions; they never
/// become local tile scores. Tile inputs average within slide then across slides.
pub fn aggregate_patient_predictions(table: &PredictionTable) -> RadResult<PredictionTable> {
    table.validate()?;
    let mut groups: BTreeMap<&str, Vec<&PredictionRow>> = BTreeMap::new();
    for row in &table.rows {
        groups.entry(&row.patient_id).or_default().push(row);
    }
    let mut rows = Vec::new();
    for (patient, group) in groups {
        let first = group[0];
        if group.iter().any(|r| {
            r.cohort != first.cohort
                || r.fold != first.fold
                || r.training_source != first.training_source
        }) {
            return Err(RadiomicsError::new(
                ErrorCode::Leakage,
                "Patient aggregation cannot mix cohorts, folds or fitted training sources.",
            ));
        }
        let subsets: Vec<Vec<&PredictionRow>> = if table.unit == PredictionUnit::Tile {
            let mut slides: BTreeMap<&str, Vec<&PredictionRow>> = BTreeMap::new();
            for row in &group {
                slides
                    .entry(
                        row.slide_id
                            .as_deref()
                            .ok_or_else(|| invalid("Tile predictions require slide identity."))?,
                    )
                    .or_default()
                    .push(*row);
            }
            slides.into_values().collect()
        } else {
            vec![group.clone()]
        };
        let (probabilities, value) = if table.task == Task::Regression {
            let means: Vec<_> = subsets
                .iter()
                .map(|subset| {
                    subset.iter().map(|r| r.value.unwrap()).sum::<f64>() / subset.len() as f64
                })
                .collect();
            (vec![], Some(means.iter().sum::<f64>() / means.len() as f64))
        } else {
            let probabilities = (0..table.classes.len())
                .map(|c| {
                    subsets
                        .iter()
                        .map(|subset| {
                            subset.iter().map(|r| r.probabilities[c]).sum::<f64>()
                                / subset.len() as f64
                        })
                        .sum::<f64>()
                        / subsets.len() as f64
                })
                .collect();
            (probabilities, None)
        };
        rows.push(PredictionRow {
            sample_id: patient.into(),
            patient_id: patient.into(),
            slide_id: None,
            tile_id: None,
            cohort: first.cohort.clone(),
            fold: first.fold,
            training_source: format!("patient_mean:{}", digest(&(table.unit, &group))?),
            probabilities,
            value,
        });
    }
    let result = PredictionTable {
        task: table.task,
        unit: PredictionUnit::Patient,
        classes: table.classes.clone(),
        rows,
    };
    result.validate()?;
    Ok(result)
}

type AlignedPatientInputs = (NumericBuffer, Vec<String>, Vec<String>, Vec<Option<usize>>);
fn aligned_matrix(tables: &[PredictionTable], names: &[String]) -> RadResult<AlignedPatientInputs> {
    if tables.is_empty() || tables.len() != names.len() {
        return Err(invalid("Supply named patient prediction tables."));
    }
    let first = &tables[0];
    first.validate()?;
    if first.unit != PredictionUnit::Patient
        || first.task == Task::Regression
        || first.rows.is_empty()
    {
        return Err(invalid(
            "Fusion input must contain patient classification probabilities.",
        ));
    }
    let ids: Vec<_> = first
        .rows
        .iter()
        .map(|r| r.patient_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let maps: Vec<BTreeMap<&str, &PredictionRow>> = tables
        .iter()
        .map(|t| t.rows.iter().map(|r| (r.patient_id.as_str(), r)).collect())
        .collect();
    for (table, map) in tables.iter().zip(&maps) {
        table.validate()?;
        if table.unit != PredictionUnit::Patient
            || table.task != first.task
            || table.classes != first.classes
            || map.keys().copied().collect::<BTreeSet<_>>()
                != ids.iter().map(String::as_str).collect()
        {
            return Err(invalid(
                "Fusion patient populations, ordered classes or prediction levels differ.",
            ));
        }
    }
    let per_component = if first.task == Task::Binary {
        1
    } else {
        first.classes.len()
    };
    let cols = names.len() * per_component;
    let mut values = Vec::new();
    let mut cohorts = Vec::new();
    let mut folds = Vec::new();
    for id in &ids {
        let source = maps[0][id.as_str()];
        cohorts.push(source.cohort.clone());
        folds.push(source.fold);
        for map in &maps {
            let row = map[id.as_str()];
            if row.cohort != source.cohort || row.fold != source.fold {
                return Err(invalid(
                    "Fusion components have different actual cohort/fold patient identities.",
                ));
            }
            if per_component == 1 {
                values.push(row.probabilities[1]);
            } else {
                values.extend_from_slice(&row.probabilities);
            }
        }
    }
    Ok((
        NumericBuffer {
            rows: ids.len(),
            cols,
            missing: vec![false; values.len()],
            values,
        },
        ids,
        cohorts,
        folds,
    ))
}
fn feature_names(components: &[FittedComponent], classes: &[String]) -> Vec<String> {
    components
        .iter()
        .flat_map(|c| {
            if classes.len() == 2 {
                vec![c.name.clone()]
            } else {
                classes.iter().map(|k| format!("{}::{k}", c.name)).collect()
            }
        })
        .collect()
}
fn meta_input(
    state: &FittedFusion,
) -> RadResult<(NumericBuffer, ModelTarget, Vec<PredictionTable>)> {
    let tables: Vec<_> = state
        .components
        .iter()
        .map(|c| aggregate_patient_predictions(&c.oof))
        .collect::<RadResult<_>>()?;
    let names: Vec<_> = state.components.iter().map(|c| c.name.clone()).collect();
    let (x, ids, cohorts, _) = aligned_matrix(&tables, &names)?;
    if ids.iter().cloned().collect::<BTreeSet<_>>() != state.split_plan.cohorts["train"]
        || cohorts.iter().any(|c| c != "train")
    {
        return Err(RadiomicsError::new(
            ErrorCode::Leakage,
            "Meta training must use the entire patient-aligned base OOF training population.",
        ));
    }
    let mut labels = Vec::new();
    for patient in &ids {
        let labels_set = &state.split_plan.patient_classes[patient];
        if labels_set.len() != 1 {
            return Err(invalid("Patient-level fusion requires one consistent outcome per patient, never majority voting."));
        }
        labels.push(
            state
                .split_plan
                .classes
                .iter()
                .position(|c| labels_set.contains(c))
                .ok_or_else(|| invalid("Unknown fusion training class."))?,
        );
    }
    let target = ModelTarget::Classification {
        classes: state.split_plan.classes.clone(),
        labels,
    };
    target.validate(x.rows)?;
    Ok((x, target, tables))
}
impl FittedFusion {
    pub fn required_features(&self) -> Vec<String> {
        self.components
            .iter()
            .flat_map(|c| {
                c.model
                    .manifest
                    .feature_names
                    .iter()
                    .map(|f| format!("{}::{f}", c.name))
            })
            .collect()
    }
    pub fn validate_state(&self) -> RadResult<()> {
        self.split_plan.validate()?;
        self.config
            .validate(self.split_plan.task, self.components.len())?;
        unique_nonempty(
            &self
                .components
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>(),
            "Fusion component names",
        )?;
        unique_nonempty(&self.required_features(), "Qualified fusion features")?;
        for c in &self.components {
            c.model.validate()?;
            if c.model.manifest.split_plan_sha256 != self.split_plan.fingerprint()?
                || c.model.manifest.task != self.split_plan.task
                || c.model.manifest.classes != self.split_plan.classes
            {
                return Err(invalid(
                    "Fusion component model uses a different SplitPlan/task/class order.",
                ));
            }
            training::validate_oof(
                &c.oof,
                &self.split_plan,
                &c.base_model_name,
                c.model.fitted.fit_unit,
                c.oof_sources.iter().map(|f| (f.fold, &f.partition, &f.fit)),
            )?;
        }
        if self.meta_feature_names != feature_names(&self.components, &self.split_plan.classes) {
            return Err(invalid(
                "Meta feature order differs from named component/class order.",
            ));
        }
        let (x, target, _) = meta_input(self)?;
        if self.meta_training_sha256
            != digest(&(
                &self.meta_feature_names,
                &x,
                &target,
                self.components.iter().map(|c| &c.oof).collect::<Vec<_>>(),
            ))?
        {
            return Err(invalid(
                "Saved meta training matrix/source checksum differs from verified OOF inputs.",
            ));
        }
        match (&self.config.method, &self.meta_model) {
            (FusionMethod::Weighted { .. }, None) => (),
            (
                FusionMethod::Logistic {
                    lambda,
                    tolerance,
                    max_iterations,
                },
                Some(model),
            ) => {
                model.validate()?;
                if model.spec
                    != (ModelSpec::LR {
                        lambda: *lambda,
                        tolerance: *tolerance,
                        max_iterations: *max_iterations,
                    })
                    || model.feature_count != x.cols
                    || model.classes() != Some(self.split_plan.classes.as_slice())
                {
                    return Err(invalid("Meta model is incompatible with its declared Logistic configuration or input order."));
                }
            }
            _ => return Err(invalid("Unexpected or missing fitted meta model.")),
        }
        Ok(())
    }
}
impl FittedState for FittedFusion {
    fn validate(&self, manifest: &ModelManifest) -> RadResult<()> {
        self.validate_state()?;
        if manifest
            .training_configuration
            .get("model_kind")
            .and_then(|v| v.as_str())
            != Some("fusion")
        {
            return Err(invalid("Unknown or incompatible saved fusion model type."));
        }
        if manifest.task != self.split_plan.task
            || manifest.classes != self.split_plan.classes
            || manifest.threshold != self.config.threshold
            || manifest.feature_names != self.required_features()
            || manifest.split_plan_sha256 != self.split_plan.fingerprint()?
        {
            return Err(invalid(
                "Fusion model manifest differs from its typed fitted state.",
            ));
        }
        Ok(())
    }
}

fn fused_table(
    state: &FittedFusion,
    tables: &[PredictionTable],
    source: &str,
    keep_fold: bool,
) -> RadResult<PredictionTable> {
    let names: Vec<_> = state.components.iter().map(|c| c.name.clone()).collect();
    let (x, ids, cohorts, folds) = aligned_matrix(tables, &names)?;
    let classes = &state.split_plan.classes;
    let probabilities = match &state.config.method {
        FusionMethod::Weighted { weights } => {
            let sum = weights.iter().sum::<f64>();
            let maps: Vec<BTreeMap<&str, &PredictionRow>> = tables
                .iter()
                .map(|t| t.rows.iter().map(|r| (r.patient_id.as_str(), r)).collect())
                .collect();
            let values: Vec<_> = ids
                .iter()
                .flat_map(|id| {
                    (0..classes.len()).map(|k| {
                        maps.iter()
                            .zip(weights)
                            .map(|(m, w)| m[id.as_str()].probabilities[k] * w)
                            .sum::<f64>()
                            / sum
                    })
                })
                .collect();
            NumericBuffer {
                rows: ids.len(),
                cols: classes.len(),
                missing: vec![false; values.len()],
                values,
            }
        }
        FusionMethod::Logistic { .. } => match state
            .meta_model
            .as_ref()
            .ok_or_else(|| invalid("Missing fitted meta model."))?
            .predict(&x)?
        {
            ModelPrediction::Classification { probabilities, .. } => probabilities,
            _ => {
                return Err(invalid(
                    "A Logistic fusion model returned a regression value.",
                ))
            }
        },
    };
    let rows = ids
        .iter()
        .enumerate()
        .map(|(i, id)| PredictionRow {
            sample_id: id.clone(),
            patient_id: id.clone(),
            slide_id: None,
            tile_id: None,
            cohort: cohorts[i].clone(),
            fold: if keep_fold { folds[i] } else { None },
            training_source: source.into(),
            probabilities: probabilities.values[i * classes.len()..(i + 1) * classes.len()]
                .to_vec(),
            value: None,
        })
        .collect();
    let table = PredictionTable {
        task: state.split_plan.task,
        unit: PredictionUnit::Patient,
        classes: classes.clone(),
        rows,
    };
    table.validate()?;
    Ok(table)
}

pub fn fit_fusion(inputs: &[FusionInput<'_>], config: &FusionConfig) -> RadResult<FusionResult> {
    if inputs.len() < 2 {
        return Err(invalid("Supply at least two named trained components."));
    }
    let plan = &inputs[0].training.split_plan;
    plan.validate()?;
    config.validate(plan.task, inputs.len())?;
    unique_nonempty(
        &inputs
            .iter()
            .map(|i| i.name.to_string())
            .collect::<Vec<_>>(),
        "Fusion component names",
    )?;
    let mut components = Vec::new();
    let mut final_tables = Vec::new();
    for input in inputs {
        input.training.validate()?;
        plan.assert_shared(&input.training.split_plan)?;
        let run = input
            .training
            .models
            .iter()
            .find(|m| m.name == input.model_name)
            .ok_or_else(|| invalid("Requested fusion base model is absent."))?;
        run.validate_provenance(plan)?;
        final_tables.push(aggregate_patient_predictions(&run.predictions)?);
        components.push(FittedComponent {
            name: input.name.into(),
            base_model_name: input.model_name.into(),
            model: run.model.clone(),
            oof: run.oof.clone(),
            oof_sources: run
                .outer_folds
                .iter()
                .map(|f| FusionOofSource {
                    fold: f.fold,
                    partition: f.partition.clone(),
                    fit: f.fit.clone(),
                })
                .collect(),
        });
    }
    let mut state = FittedFusion {
        config: config.clone(),
        split_plan: plan.clone(),
        meta_feature_names: feature_names(&components, &plan.classes),
        components,
        meta_training_sha256: String::new(),
        meta_model: None,
    };
    let (x, target, oof_tables) = meta_input(&state)?;
    state.meta_training_sha256 = digest(&(
        &state.meta_feature_names,
        &x,
        &target,
        state.components.iter().map(|c| &c.oof).collect::<Vec<_>>(),
    ))?;
    if let FusionMethod::Logistic {
        lambda,
        tolerance,
        max_iterations,
    } = config.method
    {
        state.meta_model = Some(models::fit(
            &ModelSpec::LR {
                lambda,
                tolerance,
                max_iterations,
            },
            &x,
            &target,
        )?);
    }
    let manifest = ModelManifest {
        format: "radiomics.rad.json".into(),
        schema_version: MODEL_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        task: plan.task,
        feature_names: state.required_features(),
        classes: plan.classes.clone(),
        threshold: config.threshold,
        split_plan_sha256: plan.fingerprint()?,
        training_configuration: BTreeMap::from([
            ("model_kind".into(), "fusion".into()),
            ("prediction_unit".into(), "patient".into()),
            ("fusion".into(), serde_json::to_value(config)?),
            ("meta_source".into(), "patient_aligned_base_oof".into()),
            (
                "implementation_sha256".into(),
                crate::workflow::implementation_sha256().into(),
            ),
        ]),
    };
    let model = ModelBundle {
        manifest,
        fitted: state,
    };
    model.validate()?;
    let checksum = digest(&model)?;
    let weighted = matches!(config.method, FusionMethod::Weighted { .. });
    let training = fused_table(
        &model.fitted,
        &oof_tables,
        &format!(
            "{}:{checksum}",
            if weighted {
                "oof_fixed_weight"
            } else {
                "apparent_meta_training_from_oof"
            }
        ),
        weighted,
    )?;
    let predictions = fused_table(
        &model.fitted,
        &final_tables,
        &format!("fusion_final:{checksum}"),
        false,
    )?;
    Ok(FusionResult {
        model,
        training,
        training_prediction_type: if weighted {
            "oof_fixed_weight"
        } else {
            "apparent_meta_training"
        }
        .into(),
        predictions,
    })
}

pub fn predict_fusion(
    bundle: &ModelBundle<FittedFusion>,
    data: &BTreeMap<String, StudyData>,
) -> RadResult<PredictionTable> {
    bundle.validate()?;
    if data.keys().cloned().collect::<BTreeSet<_>>()
        != bundle
            .fitted
            .components
            .iter()
            .map(|c| c.name.clone())
            .collect()
    {
        return Err(invalid(
            "Fusion prediction requires exactly the fitted named component inputs.",
        ));
    }
    let tables: Vec<_> = bundle
        .fitted
        .components
        .iter()
        .map(|c| aggregate_patient_predictions(&predict(&c.model, &data[&c.name])?))
        .collect::<RadResult<_>>()?;
    fused_table(
        &bundle.fitted,
        &tables,
        &format!("fusion_final:{}", digest(bundle)?),
        false,
    )
}

/// Reuse already computed genuine base predictions without refitting any model.
/// Each table retains the exact fitted base model source and prediction unit.
pub fn predict_fusion_tables(
    bundle: &ModelBundle<FittedFusion>,
    tables: &BTreeMap<String, PredictionTable>,
) -> RadResult<PredictionTable> {
    bundle.validate()?;
    if tables.keys().cloned().collect::<BTreeSet<_>>()
        != bundle
            .fitted
            .components
            .iter()
            .map(|c| c.name.clone())
            .collect()
    {
        return Err(invalid(
            "Fusion prediction requires exactly the fitted component names.",
        ));
    }
    let mut aggregated = Vec::new();
    for component in &bundle.fitted.components {
        let table = &tables[&component.name];
        table.validate()?;
        let source = format!("final:{}", digest(&component.model)?);
        let training = component.model.fitted.training_patients();
        if table.unit != component.model.fitted.fit_unit
            || table.task != component.model.manifest.task
            || table.classes != component.model.manifest.classes
            || table.rows.iter().any(|r| {
                r.fold.is_some()
                    || r.training_source != source
                    || r.cohort != "train" && training.contains(&r.patient_id)
            })
        {
            return Err(RadiomicsError::new(ErrorCode::Leakage,"Provided base probabilities do not have the fitted component model/unit/source/cohort provenance."));
        }
        aggregated.push(aggregate_patient_predictions(table)?);
    }
    fused_table(
        &bundle.fitted,
        &aggregated,
        &format!("fusion_final:{}", digest(bundle)?),
        false,
    )
}
