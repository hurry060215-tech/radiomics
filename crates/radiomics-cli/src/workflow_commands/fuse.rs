use super::common::*;
use clap::Args;
use ov_radiomics_core::{
    contracts::unique_nonempty,
    errors::RadResult,
    learning::{self, FusionConfig, FusionInput, TrainingResult},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Args, Serialize)]
pub struct FuseCommand {
    #[arg(long)]
    pub components: PathBuf,
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long, default_value = "radiomics-fusion")]
    pub output: PathBuf,
    #[arg(long)]
    pub overwrite: bool,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ComponentInput {
    name: String,
    training: PathBuf,
    model_name: String,
}
pub fn fuse(args: FuseCommand) -> RadResult<u8> {
    let components: Vec<ComponentInput> = read_json(&args.components)?;
    unique_nonempty(
        &components
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>(),
        "Fusion component names",
    )?;
    if components.len() < 2
        || components
            .iter()
            .any(|c| c.model_name.trim().is_empty() || c.training.as_os_str().is_empty())
    {
        return Err(invalid("Fusion needs at least two named training results with explicit nonempty model names and paths."));
    }
    let config: FusionConfig = read_json(&args.config)?;
    let output = Output::preflight(
        &args.output,
        args.overwrite,
        [
            "fusion.rad",
            "fusion_result.json",
            "training-predictions.json",
            "training-predictions.csv",
            "predictions.json",
            "predictions.csv",
            "fusion_config.json",
            "component_inputs.json",
            "summary.json",
        ]
        .map(PathBuf::from),
    )?;
    let resolved_components: Vec<_> = components
        .iter()
        .map(|c| {
            Ok(ComponentInput {
                name: c.name.clone(),
                training: std::fs::canonicalize(resolve(parent(&args.components), &c.training))?,
                model_name: c.model_name.clone(),
            })
        })
        .collect::<RadResult<_>>()?;
    let training: Vec<TrainingResult> = resolved_components
        .iter()
        .map(|c| read_json(&c.training))
        .collect::<RadResult<_>>()?;
    for result in &training {
        result.validate()?;
    }
    let inputs: Vec<_> = components
        .iter()
        .zip(&training)
        .map(|(c, result)| FusionInput {
            name: &c.name,
            training: result,
            model_name: &c.model_name,
        })
        .collect();
    let result = learning::fit_fusion(&inputs, &config)?;
    result.model.validate()?;
    output.begin()?;
    result
        .model
        .save(output.checked_path("fusion.rad")?, args.overwrite)?;
    output.json("fusion_result.json", &result)?;
    output.predictions(
        "training-predictions.json",
        "training-predictions.csv",
        &result.training,
    )?;
    output.predictions("predictions.json", "predictions.csv", &result.predictions)?;
    output.json("fusion_config.json", &config)?;
    output.json("component_inputs.json", &resolved_components)?;
    output.json("summary.json",&json!({"format":"radiomics.cli.fusion","schema_version":1,"task":result.model.manifest.task,"classes":result.model.manifest.classes,"unit":result.predictions.unit,"model_file":"fusion.rad","training_prediction_type":result.training_prediction_type,"config":config,"components":components.iter().map(|c|c.name.as_str()).collect::<Vec<_>>(),"split_plan_sha256":result.model.manifest.split_plan_sha256,"meta_training_sha256":result.model.fitted.meta_training_sha256,"runtime":"rust"}))?;
    println!(
        "{}",
        json!({"status":"complete","command":"fuse","output":args.output,"training_prediction_type":result.training_prediction_type,"components":components.len()})
    );
    Ok(0)
}
