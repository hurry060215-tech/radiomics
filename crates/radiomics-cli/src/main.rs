use clap::{Parser, Subcommand};
mod workflow_commands;
use ov_radiomics_core::{
    contracts::MaskVolume,
    errors::{ErrorCode, RadResult, RadiomicsError},
    imaging,
    workflow::{self, BatchStatus, ExtractConfig},
};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::ExitCode,
};

#[derive(Parser)]
#[command(
    name = "radiomics",
    version,
    about = "Independent native radiomics workflows; no Python runtime required"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Process ordinary patches, existing embeddings, masks and true tile maps.
    Pathology(workflow_commands::PathologyCommand),
    /// Fit grouped native pipelines from keyed CSV/Parquet study inputs.
    Train(workflow_commands::TrainCommand),
    /// Predict from a portable native pipeline or fusion model, without refitting.
    Predict(workflow_commands::PredictCommand),
    /// Evaluate matching predictions at their declared sample or patient level.
    Evaluate(workflow_commands::EvaluateCommand),
    /// Fit a patient probability fusion using verified shared OOF partitions.
    Fuse(workflow_commands::FuseCommand),
    /// Diagnose scalar NIfTI image/mask grids and selected labels without modifying inputs.
    Check {
        #[arg(long)]
        cohort: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1")]
        labels: Vec<u32>,
        /// Write a new JSON file. Without this flag the report goes to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        overwrite: bool,
    },
    /// Prepare images and extract the explicit python_golden_v1 Original feature subset.
    Extract {
        #[arg(long)]
        cohort: PathBuf,
        /// Strict JSON ExtractConfig. Bin width or bin count must be explicit.
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value = "radiomics-output")]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        n_jobs: usize,
        /// Optional content-validated feature cache; no cache is used by default.
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        overwrite: bool,
    },
}

fn run(cli: Cli) -> RadResult<u8> {
    match cli.command {
        Command::Pathology(args) => workflow_commands::pathology(args),
        Command::Train(args) => workflow_commands::train(args),
        Command::Predict(args) => workflow_commands::predict(args),
        Command::Evaluate(args) => workflow_commands::evaluate(args),
        Command::Fuse(args) => workflow_commands::fuse(args),
        Command::Check {
            cohort,
            labels,
            output,
            overwrite,
        } => {
            let samples = workflow::read_cohort(cohort)?;
            let mut report = Vec::new();
            let mut failed = false;
            for sample in samples {
                let checked =
                    (|| -> RadResult<(MaskVolume, ov_radiomics_core::contracts::Geometry)> {
                        let image = imaging::read_nifti(&sample.image)?;
                        let mask = imaging::read_mask(&sample.mask)?;
                        image.geometry.aligned_with(&mask.geometry)?;
                        let selected = sample
                            .roi_label
                            .map_or_else(|| labels.clone(), |label| vec![label]);
                        Ok((imaging::select_labels(&mask, &selected)?, image.geometry))
                    })();
                let row = match checked {
                    Ok((mask, geometry)) => {
                        serde_json::json!({"sample_id":sample.metadata.sample_id,"patient_id":sample.metadata.patient_id,"ok":true,"roi_voxels":mask.labels.iter().filter(|&&v|v!=0).count(),"geometry":geometry})
                    }
                    Err(error) => {
                        failed = true;
                        serde_json::json!({"sample_id":sample.metadata.sample_id,"patient_id":sample.metadata.patient_id,"ok":false,"error":error})
                    }
                };
                report.push(row);
            }
            report.sort_by(|a, b| a["sample_id"].as_str().cmp(&b["sample_id"].as_str()));
            let bytes = serde_json::to_vec_pretty(&report)?;
            if let Some(path) = output {
                let mut options = OpenOptions::new();
                options.write(true);
                if overwrite {
                    options.create(true).truncate(true);
                } else {
                    options.create_new(true);
                }
                let mut file = options.open(path)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
            } else {
                println!(
                    "{}",
                    String::from_utf8(bytes)
                        .map_err(|e| RadiomicsError::new(ErrorCode::Schema, e.to_string()))?
                );
            }
            Ok(if failed { 2 } else { 0 })
        }
        Command::Extract {
            cohort,
            config,
            output,
            n_jobs,
            cache,
            overwrite,
        } => {
            if output.exists() && !overwrite {
                return Err(RadiomicsError::new(
                    ErrorCode::InvalidInput,
                    "Output directory exists; choose a new location or pass --overwrite.",
                ));
            }
            let config: ExtractConfig = serde_json::from_slice(&fs::read(config)?)?;
            let samples = workflow::read_cohort(cohort)?;
            let batch = workflow::extract_batch(&samples, &config, n_jobs, cache.as_deref())?;
            workflow::export_extraction(&batch, &output, overwrite)?;
            println!(
                "{}",
                serde_json::json!({"status":batch.status,"successes":batch.records.len(),"failures":batch.failures.len(),"output":output,"implementation_sha256":batch.implementation_sha256})
            );
            Ok(if batch.status == BatchStatus::Complete {
                0
            } else {
                2
            })
        }
    }
}
fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
