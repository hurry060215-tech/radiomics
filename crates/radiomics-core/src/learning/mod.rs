//! Patient-scoped native learning workflows and portable fitted pipelines.
mod data;
mod fusion;
mod io_spec;
mod pipeline;
mod training;
pub use data::{Outcome, StudyData};
pub use fusion::{
    aggregate_patient_predictions, fit_fusion, predict_fusion, predict_fusion_tables,
    FittedComponent, FittedFusion, FusionConfig, FusionInput, FusionMethod, FusionOofSource,
    FusionResult,
};
pub use io_spec::{ClinicalJoinKey, FeatureTableManifest, JoinPolicy, StudyTableSpec};
pub use pipeline::{FittedPipeline, PipelineSpec};
pub use training::{
    predict, train, CandidateScore, FitAudit, FoldResult, InnerScore, ModelGrid, ModelRun,
    ScoreUnit, TrainingConfig, TrainingResult, TuningReport,
};

use crate::errors::{ErrorCode, RadiomicsError};
use serde::Serialize;
use sha2::{Digest, Sha256};
fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn digest<T: Serialize>(value: &T) -> crate::errors::RadResult<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}
