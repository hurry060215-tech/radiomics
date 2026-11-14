//! Versioned `.rad` JSON container. Fitted states are typed Rust values, never executable code.
use crate::contracts::{unique_nonempty, Task};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

pub const MODEL_SCHEMA_VERSION: u32 = 1;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub format: String,
    pub schema_version: u32,
    pub package_version: String,
    pub task: Task,
    pub feature_names: Vec<String>,
    pub classes: Vec<String>,
    pub threshold: Option<f64>,
    pub split_plan_sha256: String,
    pub training_configuration: BTreeMap<String, serde_json::Value>,
}
impl ModelManifest {
    pub fn validate(&self) -> RadResult<()> {
        let invalid = |s| RadiomicsError::new(ErrorCode::Schema, s);
        if self.format != "radiomics.rad.json" || self.schema_version != MODEL_SCHEMA_VERSION {
            return Err(invalid("Unknown model format or schema version."));
        }
        unique_nonempty(&self.feature_names, "Model feature names")?;
        unique_nonempty(&self.classes, "Model classes")?;
        if self.feature_names.is_empty()
            || self.package_version.is_empty()
            || self.split_plan_sha256.len() != 64
            || !self
                .split_plan_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid(
                "Model requires features, package version and a SHA256 SplitPlan fingerprint.",
            ));
        }
        match self.task {
            Task::Binary
                if self.classes.len() != 2
                    || self
                        .threshold
                        .is_none_or(|t| !t.is_finite() || !(0.0..=1.0).contains(&t)) =>
            {
                return Err(invalid(
                    "Binary model requires two classes and a finite probability threshold.",
                ))
            }
            Task::Multiclass if self.classes.len() < 3 || self.threshold.is_some() => {
                return Err(invalid(
                    "Multiclass model requires at least three classes and no binary threshold.",
                ))
            }
            Task::Regression if !self.classes.is_empty() || self.threshold.is_some() => {
                return Err(invalid(
                    "Regression model cannot have classes or a probability threshold.",
                ))
            }
            _ => (),
        }
        Ok(())
    }
    /// Return input column indices in fitted feature order; never infer alignment by width.
    pub fn column_order(&self, supplied: &[String]) -> RadResult<Vec<usize>> {
        self.validate()?;
        unique_nonempty(supplied, "Input feature names")?;
        self.feature_names
            .iter()
            .map(|name| {
                supplied.iter().position(|n| n == name).ok_or_else(|| {
                    RadiomicsError::new(
                        ErrorCode::Alignment,
                        format!("Missing fitted feature: {name}"),
                    )
                })
            })
            .collect()
    }
}

/// Each implemented pipeline state validates its numerical buffers, types and convergence.
pub trait FittedState: Serialize + DeserializeOwned {
    fn validate(&self, manifest: &ModelManifest) -> RadResult<()>;
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBundle<S> {
    pub manifest: ModelManifest,
    pub fitted: S,
}
impl<S: FittedState> ModelBundle<S> {
    pub fn validate(&self) -> RadResult<()> {
        self.manifest.validate()?;
        self.fitted.validate(&self.manifest)
    }
    pub fn save(&self, path: impl AsRef<Path>, overwrite: bool) -> RadResult<()> {
        self.validate()?;
        let path = path.as_ref();
        // Serialize before opening any output, so malformed states cannot truncate a model.
        let bytes = serde_json::to_vec(self)?;
        if !overwrite {
            let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        } else {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| RadiomicsError::new(ErrorCode::Io, e.to_string()))?
                .as_nanos();
            let temp = parent.join(format!(".radiomics-{}-{nonce}.tmp", std::process::id()));
            let result = (|| -> RadResult<()> {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temp)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
                drop(file);
                fs::rename(&temp, path)?;
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(temp);
            }
            result?;
        }
        Ok(())
    }
    pub fn load(path: impl AsRef<Path>) -> RadResult<Self> {
        let bytes = fs::read(path)?;
        // Inspect the version before decoding a state whose schema might not be understood.
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        if value
            .pointer("/manifest/schema_version")
            .and_then(|v| v.as_u64())
            != Some(MODEL_SCHEMA_VERSION.into())
        {
            return Err(RadiomicsError::new(
                ErrorCode::Schema,
                "Unknown or absent .rad schema version.",
            ));
        }
        let bundle: Self = serde_json::from_value(value)?;
        bundle.validate()?;
        Ok(bundle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Hand-defined typed fitted state tests the container without claiming model fitting support.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct State {
        coefficient: Vec<f64>,
        intercept: f64,
        converged: bool,
    }
    impl FittedState for State {
        fn validate(&self, manifest: &ModelManifest) -> RadResult<()> {
            if !self.converged
                || self.coefficient.len() != manifest.feature_names.len()
                || !self.intercept.is_finite()
                || self.coefficient.iter().any(|v| !v.is_finite())
            {
                return Err(RadiomicsError::new(
                    ErrorCode::Convergence,
                    "Invalid fitted state or optimizer did not converge.",
                ));
            }
            Ok(())
        }
    }
    fn bundle() -> ModelBundle<State> {
        ModelBundle {
            manifest: ModelManifest {
                format: "radiomics.rad.json".into(),
                schema_version: 1,
                package_version: env!("CARGO_PKG_VERSION").into(),
                task: Task::Regression,
                feature_names: vec!["a".into(), "b".into()],
                classes: vec![],
                threshold: None,
                split_plan_sha256: "a".repeat(64),
                training_configuration: BTreeMap::new(),
            },
            fitted: State {
                coefficient: vec![1., 2.],
                intercept: 3.,
                converged: true,
            },
        }
    }
    #[test]
    fn round_trip_and_explicit_overwrite() {
        let file =
            std::env::temp_dir().join(format!("radiomics-container-{}.rad", std::process::id()));
        let b = bundle();
        b.save(&file, true).unwrap();
        assert_eq!(ModelBundle::<State>::load(&file).unwrap(), b);
        assert!(b.save(&file, false).is_err());
        let mut newer = b.clone();
        newer.fitted.intercept = 5.;
        newer.save(&file, true).unwrap();
        assert_eq!(ModelBundle::<State>::load(&file).unwrap(), newer);
        fs::remove_file(file).unwrap();
    }
    #[test]
    fn reject_unknown_schema_bad_state_and_missing_feature() {
        let mut b = bundle();
        assert_eq!(
            b.manifest.column_order(&["b".into(), "a".into()]).unwrap(),
            vec![1, 0]
        );
        assert!(b.manifest.column_order(&["b".into()]).is_err());
        b.manifest.schema_version = 2;
        assert!(b.validate().is_err());
        b.manifest.schema_version = 1;
        b.fitted.converged = false;
        assert!(b.validate().is_err());
        b.fitted.converged = true;
        b.fitted.coefficient[0] = f64::NAN;
        assert!(b.validate().is_err());
    }
}
