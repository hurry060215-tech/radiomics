use super::{digest, invalid, pipeline, FittedPipeline, PipelineSpec, StudyData};
use crate::{
    bundle::{ModelBundle, ModelManifest, MODEL_SCHEMA_VERSION},
    contracts::*,
    errors::{ErrorCode, RadResult, RadiomicsError},
    models::{FitReport, ModelPrediction, ModelTarget},
    preprocessing::{FittedPreprocessor, FittedSelector},
    split::{FoldPartition, SplitPlan},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreUnit {
    Sample,
    Patient,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGrid {
    pub name: String,
    pub candidates: Vec<PipelineSpec>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub task: Task,
    pub classes: Vec<String>,
    pub fit_unit: PredictionUnit,
    pub scoring_unit: ScoreUnit,
    pub threshold: Option<f64>,
    pub selection_tie_atol: f64,
    pub models: Vec<ModelGrid>,
}
impl TrainingConfig {
    pub fn validate(&self) -> RadResult<()> {
        unique_nonempty(&self.classes, "Training classes")?;
        if (self.task == Task::Binary && self.classes.len() != 2)
            || (self.task == Task::Multiclass && self.classes.len() < 3)
            || (self.task == Task::Regression && !self.classes.is_empty())
        {
            return Err(invalid("Task and explicit training class order disagree."));
        }
        if self.task == Task::Binary {
            if self
                .threshold
                .is_none_or(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            {
                return Err(invalid(
                    "Binary training requires an explicit threshold in [0,1].",
                ));
            }
        } else if self.threshold.is_some() {
            return Err(invalid(
                "A binary threshold does not apply to multiclass or regression.",
            ));
        }
        if !self.selection_tie_atol.is_finite()
            || self.selection_tie_atol < 0.
            || self.models.is_empty()
        {
            return Err(invalid(
                "Provide a nonempty model grid and nonnegative finite selection_tie_atol.",
            ));
        }
        unique_nonempty(
            &self
                .models
                .iter()
                .map(|m| m.name.clone())
                .collect::<Vec<_>>(),
            "Model names",
        )?;
        for model in &self.models {
            if !model
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
                || model.candidates.is_empty()
            {
                return Err(invalid("Model names must be safe alphanumeric/hyphen/underscore identifiers, with nonempty explicit candidate lists."));
            }
            for candidate in &model.candidates {
                candidate.validate()?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitAudit {
    pub spec: PipelineSpec,
    pub training_patients: BTreeSet<String>,
    pub training_sample_ids: Vec<String>,
    pub preprocessing: FittedPreprocessor,
    pub selector: Option<FittedSelector>,
    pub selected_features: Vec<String>,
    pub model_report: FitReport,
    pub model_state_sha256: String,
    pub provenance_id: String,
}
impl FitAudit {
    fn new(fitted: &FittedPipeline, plan: &SplitPlan, purpose: &str) -> RadResult<Self> {
        fitted.validate_state()?;
        let mut result = Self {
            spec: fitted.spec.clone(),
            training_patients: fitted.training_patients(),
            training_sample_ids: fitted.training_samples.keys().cloned().collect(),
            preprocessing: fitted.preprocessing.clone(),
            selector: fitted.selector.clone(),
            selected_features: fitted.selected_features.clone(),
            model_report: fitted.model.report.clone(),
            model_state_sha256: fitted.state_sha256()?,
            provenance_id: String::new(),
        };
        result.provenance_id = result.expected_id(plan, purpose)?;
        Ok(result)
    }
    fn expected_id(&self, plan: &SplitPlan, purpose: &str) -> RadResult<String> {
        digest(&(
            purpose,
            plan.fingerprint()?,
            &self.spec,
            &self.training_patients,
            &self.training_sample_ids,
            &self.preprocessing,
            &self.selector,
            &self.selected_features,
            &self.model_report,
            &self.model_state_sha256,
        ))
    }
    fn validate(
        &self,
        plan: &SplitPlan,
        population: &BTreeSet<String>,
        purpose: &str,
    ) -> RadResult<()> {
        self.spec.validate()?;
        self.preprocessing.validate()?;
        if let Some(selector) = &self.selector {
            selector.validate()?;
        }
        self.model_report.validate(matches!(
            self.spec.model,
            crate::models::ModelSpec::LR { .. }
                | crate::models::ModelSpec::Linear { .. }
                | crate::models::ModelSpec::Ridge { .. }
        ))?;
        if self.preprocessing.config != self.spec.preprocessing
            || self.selector.as_ref().map(|s| &s.config) != self.spec.selector.as_ref()
        {
            return Err(invalid(
                "Fit audit preprocessing/selection differs from the declared pipeline.",
            ));
        }
        unique_nonempty(&self.training_sample_ids, "Audit training sample IDs")?;
        let expected: Vec<_> = plan
            .sample_patients
            .iter()
            .filter(|(_, p)| population.contains(*p))
            .map(|(s, _)| s.clone())
            .collect();
        if self.training_patients != *population
            || self.training_sample_ids != expected
            || self.preprocessing.training_rows != expected.len()
            || self.model_state_sha256.len() != 64
            || self.provenance_id != self.expected_id(plan, purpose)?
        {
            return Err(RadiomicsError::new(ErrorCode::Leakage,"Fit provenance does not match the actual declared training patients/sample population."));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InnerScore {
    pub fold: usize,
    pub partition: FoldPartition,
    pub fit: FitAudit,
    pub score: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateScore {
    pub candidate: usize,
    pub inner_folds: Vec<InnerScore>,
    pub mean_score: Option<f64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuningReport {
    pub candidates: Vec<CandidateScore>,
    pub selected_candidate: usize,
    pub selection_tie_atol: f64,
    pub fixed_configuration: bool,
}
impl TuningReport {
    fn validate(
        &self,
        plan: &SplitPlan,
        folds: &[FoldPartition],
        grid: &ModelGrid,
        config: &TrainingConfig,
        context: &str,
    ) -> RadResult<()> {
        if self.candidates.len() != grid.candidates.len()
            || self.selected_candidate >= self.candidates.len()
            || self.selection_tie_atol != config.selection_tie_atol
            || self.fixed_configuration != (grid.candidates.len() == 1)
        {
            return Err(invalid(
                "Tuning report differs from its explicit grid or tie policy.",
            ));
        }
        let mut best = 0;
        for (i, candidate) in self.candidates.iter().enumerate() {
            if candidate.candidate != i {
                return Err(invalid(
                    "Tuning candidate order differs from the requested grid.",
                ));
            }
            if self.fixed_configuration {
                if !candidate.inner_folds.is_empty()
                    || candidate.mean_score.is_some()
                    || self.selected_candidate != 0
                {
                    return Err(invalid(
                        "Fixed configurations must not claim invented inner CV scores.",
                    ));
                }
                continue;
            }
            if candidate.inner_folds.len() != folds.len() {
                return Err(invalid("Tuning report is missing an inner fold."));
            }
            for (j, (inner, partition)) in candidate.inner_folds.iter().zip(folds).enumerate() {
                if inner.fold != j
                    || &inner.partition != partition
                    || inner.fit.spec != grid.candidates[i]
                    || !valid_score(config.task, inner.score)
                {
                    return Err(RadiomicsError::new(ErrorCode::Leakage,"Inner tuning result does not use the exact saved patient partition and candidate."));
                }
                inner.fit.validate(
                    plan,
                    &partition.training_patients,
                    &format!("{context}/candidate-{i}/inner-{j}"),
                )?;
            }
            let mean =
                candidate.inner_folds.iter().map(|f| f.score).sum::<f64>() / folds.len() as f64;
            if candidate.mean_score != Some(mean) {
                return Err(invalid(
                    "Stored inner CV mean differs from its fold metrics.",
                ));
            }
            if i > 0
                && better(
                    config.task,
                    mean,
                    self.candidates[best]
                        .mean_score
                        .ok_or_else(|| invalid("Missing previous candidate metric."))?,
                    config.selection_tie_atol,
                )
            {
                best = i;
            }
        }
        if !self.fixed_configuration && self.selected_candidate != best {
            return Err(invalid(
                "Selected candidate violates the saved comparison metric/tie rule.",
            ));
        }
        Ok(())
    }
}
fn valid_score(task: Task, score: f64) -> bool {
    score.is_finite() && score >= 0. && (task == Task::Regression || score <= 1.)
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldResult {
    pub fold: usize,
    pub partition: FoldPartition,
    pub tuning: TuningReport,
    pub fit: FitAudit,
    pub score: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRun {
    pub name: String,
    pub split_plan_sha256: String,
    pub outer_folds: Vec<FoldResult>,
    pub mean_cv_score: f64,
    pub final_tuning: TuningReport,
    pub model: ModelBundle<FittedPipeline>,
    pub oof: PredictionTable,
    pub predictions: PredictionTable,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingResult {
    pub config: TrainingConfig,
    pub split_plan: SplitPlan,
    pub models: Vec<ModelRun>,
    pub best_model: String,
}

pub(super) fn prediction_table(
    data: &StudyData,
    prediction: &ModelPrediction,
    unit: PredictionUnit,
    source: &str,
    fold: Option<usize>,
) -> RadResult<PredictionTable> {
    data.validate_unit(unit)?;
    let (task, classes) = match prediction {
        ModelPrediction::Classification { classes, .. } => (
            if classes.len() == 2 {
                Task::Binary
            } else {
                Task::Multiclass
            },
            classes.clone(),
        ),
        ModelPrediction::Regression { .. } => (Task::Regression, vec![]),
    };
    let mut rows = Vec::new();
    match prediction {
        ModelPrediction::Classification { probabilities, .. } => {
            probabilities.validate()?;
            if probabilities.rows != data.samples.len() || probabilities.cols != classes.len() {
                return Err(invalid(
                    "Prediction matrix differs from sample/class dimensions.",
                ));
            }
            for (i, s) in data.samples.iter().enumerate() {
                rows.push(PredictionRow {
                    sample_id: s.sample_id.clone(),
                    patient_id: s.patient_id.clone(),
                    slide_id: s.slide_id.clone(),
                    tile_id: (unit == PredictionUnit::Tile).then(|| s.sample_id.clone()),
                    cohort: data.cohorts[i].clone(),
                    fold,
                    training_source: source.into(),
                    probabilities: probabilities.values
                        [i * probabilities.cols..(i + 1) * probabilities.cols]
                        .to_vec(),
                    value: None,
                });
            }
        }
        ModelPrediction::Regression { values } => {
            if values.len() != data.samples.len() {
                return Err(invalid(
                    "Regression prediction length differs from samples.",
                ));
            }
            for (i, s) in data.samples.iter().enumerate() {
                rows.push(PredictionRow {
                    sample_id: s.sample_id.clone(),
                    patient_id: s.patient_id.clone(),
                    slide_id: s.slide_id.clone(),
                    tile_id: (unit == PredictionUnit::Tile).then(|| s.sample_id.clone()),
                    cohort: data.cohorts[i].clone(),
                    fold,
                    training_source: source.into(),
                    probabilities: vec![],
                    value: Some(values[i]),
                });
            }
        }
    }
    let table = PredictionTable {
        task,
        unit,
        classes,
        rows,
    };
    table.validate()?;
    Ok(table)
}

/// Equal sample weighting within each patient; labels must agree, never majority-voted.
pub(super) fn score_input(
    data: &StudyData,
    prediction: &ModelPrediction,
    task: Task,
    classes: &[String],
    unit: ScoreUnit,
) -> RadResult<(ModelTarget, ModelPrediction)> {
    let target = data.target(task, classes)?;
    if unit == ScoreUnit::Sample {
        return Ok((target, prediction.clone()));
    }
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, s) in data.samples.iter().enumerate() {
        groups.entry(&s.patient_id).or_default().push(i);
    }
    match (target, prediction) {
        (
            ModelTarget::Classification { classes, labels },
            ModelPrediction::Classification {
                classes: pred_classes,
                probabilities,
            },
        ) => {
            if &classes != pred_classes
                || probabilities.rows != data.samples.len()
                || probabilities.cols != classes.len()
            {
                return Err(invalid(
                    "Scoring classes or prediction dimensions disagree.",
                ));
            }
            probabilities.validate()?;
            let mut y = Vec::new();
            let mut values = Vec::new();
            for rows in groups.values() {
                let label = labels[rows[0]];
                if rows.iter().any(|&i| labels[i] != label) {
                    return Err(invalid("Patient-level scoring requires consistent outcomes within each patient; select sample scoring explicitly for lesion-level targets."));
                }
                y.push(label);
                for j in 0..classes.len() {
                    values.push(
                        rows.iter()
                            .map(|&i| probabilities.values[i * classes.len() + j])
                            .sum::<f64>()
                            / rows.len() as f64,
                    );
                }
            }
            Ok((
                ModelTarget::Classification {
                    classes: classes.clone(),
                    labels: y,
                },
                ModelPrediction::Classification {
                    classes: classes.clone(),
                    probabilities: NumericBuffer {
                        rows: groups.len(),
                        cols: classes.len(),
                        missing: vec![false; values.len()],
                        values,
                    },
                },
            ))
        }
        (ModelTarget::Regression { values: target }, ModelPrediction::Regression { values }) => {
            if values.len() != data.samples.len() {
                return Err(invalid("Regression scoring dimensions disagree."));
            }
            let mut y = Vec::new();
            let mut p = Vec::new();
            for rows in groups.values() {
                let value = target[rows[0]];
                if rows.iter().any(|&i| target[i] != value) {
                    return Err(invalid("Patient-level scoring requires consistent regression outcomes within each patient."));
                }
                y.push(value);
                p.push(rows.iter().map(|&i| values[i]).sum::<f64>() / rows.len() as f64);
            }
            Ok((
                ModelTarget::Regression { values: y },
                ModelPrediction::Regression { values: p },
            ))
        }
        _ => Err(invalid("Scoring task differs from model prediction type.")),
    }
}
fn score(
    data: &StudyData,
    prediction: &ModelPrediction,
    config: &TrainingConfig,
) -> RadResult<f64> {
    let (target, prediction) = score_input(
        data,
        prediction,
        config.task,
        &config.classes,
        config.scoring_unit,
    )?;
    crate::evaluation::model_score(config.task, &target, &prediction)
}
fn better(task: Task, new: f64, old: f64, tolerance: f64) -> bool {
    if task == Task::Regression {
        new < old - tolerance
    } else {
        new > old + tolerance
    }
}

fn tune(
    data: &StudyData,
    plan: &SplitPlan,
    folds: &[FoldPartition],
    grid: &ModelGrid,
    config: &TrainingConfig,
    context: &str,
) -> RadResult<TuningReport> {
    if grid.candidates.len() == 1 {
        return Ok(TuningReport {
            candidates: vec![CandidateScore {
                candidate: 0,
                inner_folds: vec![],
                mean_score: None,
            }],
            selected_candidate: 0,
            selection_tie_atol: config.selection_tie_atol,
            fixed_configuration: true,
        });
    }
    let mut candidates = Vec::new();
    let mut best = 0;
    for (index, spec) in grid.candidates.iter().enumerate() {
        let mut inner_folds = Vec::new();
        for (fold, partition) in folds.iter().enumerate() {
            let train = data.take_rows(&data.indices_for(&partition.training_patients))?;
            let valid = data.take_rows(&data.indices_for(&partition.validation_patients))?;
            let fitted = pipeline::fit(&train, spec, config.task, &config.classes, config.fit_unit)
                .map_err(|e| {
                    RadiomicsError::new(
                        e.code,
                        format!(
                            "{context}, candidate {index}, inner fold {fold}: {}",
                            e.message
                        ),
                    )
                })?;
            let prediction = fitted.predict(&valid.features)?;
            let score = score(&valid, &prediction, config)?;
            let audit = FitAudit::new(
                &fitted,
                plan,
                &format!("{context}/candidate-{index}/inner-{fold}"),
            )?;
            inner_folds.push(InnerScore {
                fold,
                partition: partition.clone(),
                fit: audit,
                score,
            });
        }
        let mean = inner_folds.iter().map(|f| f.score).sum::<f64>() / inner_folds.len() as f64;
        if !mean.is_finite() {
            return Err(invalid("CV comparison metric is undefined."));
        }
        candidates.push(CandidateScore {
            candidate: index,
            inner_folds,
            mean_score: Some(mean),
        });
        if index > 0
            && better(
                config.task,
                mean,
                candidates[best].mean_score.unwrap(),
                config.selection_tie_atol,
            )
        {
            best = index;
        }
    }
    Ok(TuningReport {
        candidates,
        selected_candidate: best,
        selection_tie_atol: config.selection_tie_atol,
        fixed_configuration: false,
    })
}

fn train_model(
    data: &StudyData,
    plan: &SplitPlan,
    grid: &ModelGrid,
    config: &TrainingConfig,
) -> RadResult<ModelRun> {
    let fingerprint = plan.fingerprint()?;
    let mut outer_folds = Vec::new();
    let mut oof_rows = Vec::new();
    for (fold, nested) in plan.outer_folds.iter().enumerate() {
        let context = format!("{}/outer-{fold}", grid.name);
        let tuning = tune(data, plan, &nested.inner, grid, config, &context)?;
        let training = data.take_rows(&data.indices_for(&nested.outer.training_patients))?;
        let validation = data.take_rows(&data.indices_for(&nested.outer.validation_patients))?;
        let fitted = pipeline::fit(
            &training,
            &grid.candidates[tuning.selected_candidate],
            config.task,
            &config.classes,
            config.fit_unit,
        )?;
        let fit = FitAudit::new(&fitted, plan, &format!("{context}/fit"))?;
        fit.validate(
            plan,
            &nested.outer.training_patients,
            &format!("{context}/fit"),
        )?;
        let prediction = fitted.predict(&validation.features)?;
        let value = score(&validation, &prediction, config)?;
        let table = prediction_table(
            &validation,
            &prediction,
            config.fit_unit,
            &format!("oof:{}", fit.provenance_id),
            Some(fold),
        )?;
        oof_rows.extend(table.rows);
        outer_folds.push(FoldResult {
            fold,
            partition: nested.outer.clone(),
            tuning,
            fit,
            score: value,
        });
    }
    oof_rows.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
    let oof = PredictionTable {
        task: config.task,
        unit: config.fit_unit,
        classes: config.classes.clone(),
        rows: oof_rows,
    };
    oof.validate()?;
    let final_tuning = tune(
        data,
        plan,
        &plan.final_inner_folds,
        grid,
        config,
        &format!("{}/final", grid.name),
    )?;
    let training = data.take_rows(&data.indices_for(&plan.cohorts["train"]))?;
    let fitted = pipeline::fit(
        &training,
        &grid.candidates[final_tuning.selected_candidate],
        config.task,
        &config.classes,
        config.fit_unit,
    )?;
    let manifest = ModelManifest {
        format: "radiomics.rad.json".into(),
        schema_version: MODEL_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        task: config.task,
        feature_names: fitted.required_features(),
        classes: config.classes.clone(),
        threshold: config.threshold,
        split_plan_sha256: fingerprint.clone(),
        training_configuration: BTreeMap::from([
            ("split_plan".into(), serde_json::to_value(plan)?),
            (
                "candidate_grid".into(),
                serde_json::to_value(&grid.candidates)?,
            ),
            ("final_tuning".into(), serde_json::to_value(&final_tuning)?),
            ("fit_unit".into(), serde_json::to_value(config.fit_unit)?),
            (
                "scoring_unit".into(),
                serde_json::to_value(config.scoring_unit)?,
            ),
            ("model_name".into(), grid.name.clone().into()),
            (
                "implementation_sha256".into(),
                crate::workflow::implementation_sha256().into(),
            ),
            (
                "selection_tie_atol".into(),
                config.selection_tie_atol.into(),
            ),
        ]),
    };
    let bundle = ModelBundle { manifest, fitted };
    bundle.validate()?;
    let prediction = bundle.fitted.predict(&data.features)?;
    let predictions = prediction_table(
        data,
        &prediction,
        config.fit_unit,
        &format!("final:{}", digest(&bundle)?),
        None,
    )?;
    let mean_cv_score = outer_folds.iter().map(|f| f.score).sum::<f64>() / outer_folds.len() as f64;
    Ok(ModelRun {
        name: grid.name.clone(),
        split_plan_sha256: fingerprint,
        outer_folds,
        mean_cv_score,
        final_tuning,
        model: bundle,
        oof,
        predictions,
    })
}

pub fn train(
    data: &StudyData,
    plan: &SplitPlan,
    config: &TrainingConfig,
    n_jobs: usize,
) -> RadResult<TrainingResult> {
    config.validate()?;
    plan.validate()?;
    if plan.task != config.task || plan.classes != config.classes || n_jobs == 0 {
        return Err(invalid(
            "Training task/classes must match SplitPlan and n_jobs must be positive.",
        ));
    }
    let data = data.canonical()?;
    data.validate_unit(config.fit_unit)?;
    plan.validate_observations(&data.observations()?)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs)
        .build()
        .map_err(|e| invalid(e.to_string()))?;
    let models: Vec<ModelRun> = pool.install(|| {
        config
            .models
            .par_iter()
            .map(|grid| train_model(&data, plan, grid, config))
            .collect::<RadResult<_>>()
    })?;
    let mut best = 0;
    for i in 1..models.len() {
        if better(
            config.task,
            models[i].mean_cv_score,
            models[best].mean_cv_score,
            config.selection_tie_atol,
        ) {
            best = i;
        }
    }
    let result = TrainingResult {
        best_model: models[best].name.clone(),
        config: config.clone(),
        split_plan: plan.clone(),
        models,
    };
    result.validate()?;
    Ok(result)
}

impl ModelRun {
    pub fn validate_provenance(&self, plan: &SplitPlan) -> RadResult<()> {
        self.model.validate()?;
        self.oof.validate()?;
        self.predictions.validate()?;
        if self.split_plan_sha256 != plan.fingerprint()?
            || self.model.manifest.split_plan_sha256 != self.split_plan_sha256
            || self.outer_folds.len() != plan.outer_folds.len()
            || self.model.manifest.task != plan.task
            || self.model.manifest.classes != plan.classes
            || [&self.oof, &self.predictions].iter().any(|t| {
                t.task != plan.task
                    || t.classes != plan.classes
                    || t.unit != self.model.fitted.fit_unit
            })
        {
            return Err(RadiomicsError::new(
                ErrorCode::Alignment,
                "Model and OOF results do not share the supplied SplitPlan.",
            ));
        }
        let expected: BTreeSet<_> = plan
            .sample_patients
            .iter()
            .filter(|(_, p)| plan.cohorts["train"].contains(*p))
            .map(|(s, _)| s.clone())
            .collect();
        if self
            .oof
            .rows
            .iter()
            .map(|r| r.sample_id.clone())
            .collect::<BTreeSet<_>>()
            != expected
            || self
                .predictions
                .rows
                .iter()
                .map(|r| r.sample_id.clone())
                .collect::<BTreeSet<_>>()
                != plan.sample_patients.keys().cloned().collect()
        {
            return Err(RadiomicsError::new(
                ErrorCode::Alignment,
                "Prediction sample populations differ from SplitPlan.",
            ));
        }
        for (index, (fold, nested)) in self.outer_folds.iter().zip(&plan.outer_folds).enumerate() {
            if fold.fold != index || fold.partition != nested.outer || !fold.score.is_finite() {
                return Err(RadiomicsError::new(
                    ErrorCode::Leakage,
                    "Fold indices do not identify the same actual patient partitions.",
                ));
            }
            fold.fit.validate(
                plan,
                &nested.outer.training_patients,
                &format!("{}/outer-{index}/fit", self.name),
            )?;
            for row in self.oof.rows.iter().filter(|r| r.fold == Some(index)) {
                if !nested.outer.validation_patients.contains(&row.patient_id)
                    || row.cohort != "train"
                    || plan.sample_patients.get(&row.sample_id) != Some(&row.patient_id)
                    || row.training_source != format!("oof:{}", fold.fit.provenance_id)
                {
                    return Err(RadiomicsError::new(
                        ErrorCode::Leakage,
                        "OOF prediction does not come from its declared patient-excluding fit.",
                    ));
                }
            }
        }
        let mean =
            self.outer_folds.iter().map(|f| f.score).sum::<f64>() / self.outer_folds.len() as f64;
        if self.mean_cv_score != mean || !valid_score(plan.task, mean) {
            return Err(invalid(
                "Stored model comparison score differs from outer-fold scores.",
            ));
        }
        if self
            .oof
            .rows
            .iter()
            .any(|r| r.fold.is_none_or(|f| f >= self.outer_folds.len()))
        {
            return Err(invalid("An OOF prediction has no valid source fold."));
        }
        let source = format!("final:{}", digest(&self.model)?);
        for row in &self.predictions.rows {
            if row.fold.is_some()
                || row.training_source != source
                || plan.sample_patients.get(&row.sample_id) != Some(&row.patient_id)
                || !plan
                    .cohorts
                    .get(&row.cohort)
                    .is_some_and(|s| s.contains(&row.patient_id))
            {
                return Err(RadiomicsError::new(
                    ErrorCode::Alignment,
                    "Final prediction identity/cohort/source differs from the fitted model.",
                ));
            }
        }
        Ok(())
    }
}
impl TrainingResult {
    pub fn validate(&self) -> RadResult<()> {
        self.config.validate()?;
        self.split_plan.validate()?;
        if self.config.task != self.split_plan.task
            || self.config.classes != self.split_plan.classes
            || self.models.len() != self.config.models.len()
        {
            return Err(invalid(
                "Training result configuration/schema differs from its SplitPlan.",
            ));
        }
        for (model, grid) in self.models.iter().zip(&self.config.models) {
            if model.name != grid.name {
                return Err(invalid(
                    "Model result order differs from the requested grid.",
                ));
            }
            model.validate_provenance(&self.split_plan)?;
            for (i, fold) in model.outer_folds.iter().enumerate() {
                fold.tuning.validate(
                    &self.split_plan,
                    &self.split_plan.outer_folds[i].inner,
                    grid,
                    &self.config,
                    &format!("{}/outer-{i}", model.name),
                )?;
                if fold.fit.spec != grid.candidates[fold.tuning.selected_candidate] {
                    return Err(invalid("Outer fit used a candidate other than the one selected inside that training fold."));
                }
            }
            model.final_tuning.validate(
                &self.split_plan,
                &self.split_plan.final_inner_folds,
                grid,
                &self.config,
                &format!("{}/final", model.name),
            )?;
            if model.model.fitted.spec != grid.candidates[model.final_tuning.selected_candidate]
                || model
                    .model
                    .manifest
                    .training_configuration
                    .get("final_tuning")
                    != Some(&serde_json::to_value(&model.final_tuning)?)
                || model
                    .model
                    .manifest
                    .training_configuration
                    .get("candidate_grid")
                    != Some(&serde_json::to_value(&grid.candidates)?)
            {
                return Err(invalid("Final fitted model or saved tuning configuration differs from the training result."));
            }
        }
        if !self.models.iter().any(|m| m.name == self.best_model) {
            return Err(invalid("Best model is absent from the training result."));
        }
        let mut best = 0;
        for i in 1..self.models.len() {
            if better(
                self.config.task,
                self.models[i].mean_cv_score,
                self.models[best].mean_cv_score,
                self.config.selection_tie_atol,
            ) {
                best = i;
            }
        }
        if self.best_model != self.models[best].name {
            return Err(invalid(
                "Best model violates the predeclared CV comparison/tie policy.",
            ));
        }
        Ok(())
    }
}

/// Standalone prediction uses the fitted schema/unit and never fits on supplied outcomes.
pub fn predict(
    bundle: &ModelBundle<FittedPipeline>,
    data: &StudyData,
) -> RadResult<PredictionTable> {
    bundle.validate()?;
    let data = data.canonical()?;
    data.validate_unit(bundle.fitted.fit_unit)?;
    let training = bundle.fitted.training_patients();
    if data
        .samples
        .iter()
        .zip(&data.cohorts)
        .any(|(s, cohort)| cohort != "train" && training.contains(&s.patient_id))
    {
        return Err(RadiomicsError::new(ErrorCode::Leakage,"An external prediction cohort overlaps fitted training patients; preserve the actual cohort identity."));
    }
    let predicted = bundle.fitted.predict(&data.features)?;
    prediction_table(
        &data,
        &predicted,
        bundle.fitted.fit_unit,
        &format!("final:{}", digest(bundle)?),
        None,
    )
}
