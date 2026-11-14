use super::common::*;
use clap::Args;
use ov_radiomics_core::{
    contracts::Task,
    errors::RadResult,
    learning::{self, TrainingConfig},
    split::SplitPlan,
};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Args, Serialize)]
pub struct TrainCommand {
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub split_plan: Option<PathBuf>,
    /// Explicit center holdout; already external cohorts remain external.
    #[arg(long = "hold-out-center")]
    pub held_out_centers: Vec<String>,
    #[arg(long, default_value = "radiomics-models")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 5)]
    pub outer_cv: usize,
    #[arg(long, default_value_t = 3)]
    pub inner_cv: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 1)]
    pub n_jobs: usize,
    #[arg(long)]
    pub overwrite: bool,
}
pub fn train(args: TrainCommand) -> RadResult<u8> {
    let config: TrainingConfig = read_json(&args.config)?;
    config.validate()?;
    if args.n_jobs == 0 {
        return Err(invalid("n_jobs must be positive."));
    }
    let mut names = vec![
        "split_plan.json".into(),
        "training.json".into(),
        "load_report.json".into(),
        "command_config.json".into(),
        "summary.json".into(),
    ];
    for index in 0..config.models.len() {
        for suffix in [
            ".rad",
            "-oof.json",
            "-oof.csv",
            "-predictions.json",
            "-predictions.csv",
            "-cv.json",
            "-cv.csv",
        ] {
            names.push(format!("model-{index:03}{suffix}").into());
        }
    }
    let output = Output::preflight(&args.output, args.overwrite, names)?;
    if args.split_plan.is_some() && !args.held_out_centers.is_empty() {
        return Err(invalid(
            "Choose a supplied SplitPlan or explicit center holdout, not both.",
        ));
    }
    let (study_spec, mut loaded) = load_study_record(&args.data, config.task)?;
    if !args.held_out_centers.is_empty() {
        let observations = ov_radiomics_core::split::center_holdout(
            &loaded.data.observations()?,
            &args.held_out_centers.iter().cloned().collect(),
        )?;
        loaded.data.cohorts = observations.into_iter().map(|row| row.cohort).collect();
    }
    let study_spec_directory = std::fs::canonicalize(parent(&args.data))?;
    let plan = if let Some(path) = &args.split_plan {
        let plan: SplitPlan = read_json(path)?;
        plan.validate()?;
        plan.validate_observations(&loaded.data.observations()?)?;
        plan
    } else {
        SplitPlan::from_cohorts(
            &loaded.data.observations()?,
            config.task,
            config.classes.clone(),
            args.outer_cv,
            args.inner_cv,
            args.seed,
        )?
    };
    let result = learning::train(&loaded.data, &plan, &config, args.n_jobs)?;
    result.validate()?;
    output.begin()?;
    output.json("split_plan.json", &plan)?;
    output.json("training.json", &result)?;
    output.json("load_report.json", &loaded.report)?;
    output.json("command_config.json",&json!({"format":"radiomics.cli.train","schema_version":1,"arguments":args,"study_spec":study_spec,"study_spec_directory":study_spec_directory,"training_config":config,"effective_seed":plan.seed,"effective_outer_folds":plan.outer_folds.len(),"provided_split_plan":args.split_plan.is_some(),"runtime":"rust"}))?;
    let metric = match config.task {
        Task::Binary => "roc_auc",
        Task::Multiclass => "macro_ovr_roc_auc",
        Task::Regression => "rmse",
    };
    let direction = if config.task == Task::Regression {
        "minimize"
    } else {
        "maximize"
    };
    let mut summaries = Vec::new();
    for (index, run) in result.models.iter().enumerate() {
        let prefix = format!("model-{index:03}");
        let model_file = format!("{prefix}.rad");
        run.model
            .save(output.checked_path(&model_file)?, args.overwrite)?;
        output.predictions(
            &format!("{prefix}-oof.json"),
            &format!("{prefix}-oof.csv"),
            &run.oof,
        )?;
        output.predictions(
            &format!("{prefix}-predictions.json"),
            &format!("{prefix}-predictions.csv"),
            &run.predictions,
        )?;
        output.json(&format!("{prefix}-cv.json"), &run.outer_folds)?;
        output.csv(
            &format!("{prefix}-cv.csv"),
            &[
                "model",
                "fold",
                "score",
                "metric",
                "score_unit",
                "training_patients_n",
                "validation_patients_n",
                "training_samples_n",
                "fit_provenance",
                "selected_candidate",
            ],
            run.outer_folds.iter().map(|fold| {
                vec![
                    run.name.clone(),
                    fold.fold.to_string(),
                    fold.score.to_string(),
                    metric.into(),
                    format!("{:?}", config.scoring_unit).to_ascii_lowercase(),
                    fold.partition.training_patients.len().to_string(),
                    fold.partition.validation_patients.len().to_string(),
                    fold.fit.training_sample_ids.len().to_string(),
                    fold.fit.provenance_id.clone(),
                    fold.tuning.selected_candidate.to_string(),
                ]
            }),
        )?;
        summaries.push(json!({"name":run.name,"model_file":model_file,"mean_cv_score":run.mean_cv_score,"metric":metric,"direction":direction,"selected_candidate":run.final_tuning.selected_candidate,"split_plan_sha256":run.split_plan_sha256,"classes":run.model.manifest.classes,"fit_unit":run.model.fitted.fit_unit}));
    }
    let best = result
        .models
        .iter()
        .position(|m| m.name == result.best_model)
        .ok_or_else(|| schema("Core training result did not contain its declared best model."))?;
    output.json("summary.json",&json!({"format":"radiomics.cli.training-summary","schema_version":1,"task":config.task,"classes":config.classes,"fit_unit":config.fit_unit,"score_unit":config.scoring_unit,"best_model":result.best_model,"best_model_file":format!("model-{best:03}.rad"),"selection_metric":metric,"selection_direction":direction,"models":summaries}))?;
    println!(
        "{}",
        json!({"status":"complete","command":"train","best_model":result.best_model,"output":args.output,"models":result.models.len()})
    );
    Ok(0)
}
