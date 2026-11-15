//! Structured errors shared by native workflows and language bindings.
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidInput,
    Geometry,
    Unsupported,
    Io,
    Schema,
    Alignment,
    Leakage,
    Convergence,
    EmptyResult,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadiomicsError {
    pub code: ErrorCode,
    pub message: String,
}

impl RadiomicsError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for RadiomicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}
impl Error for RadiomicsError {}
impl From<std::io::Error> for RadiomicsError {
    fn from(error: std::io::Error) -> Self {
        Self::new(ErrorCode::Io, error.to_string())
    }
}
impl From<serde_json::Error> for RadiomicsError {
    fn from(error: serde_json::Error) -> Self {
        Self::new(ErrorCode::Schema, error.to_string())
    }
}
pub type RadResult<T> = Result<T, RadiomicsError>;
