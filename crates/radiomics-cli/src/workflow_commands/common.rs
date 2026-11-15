use ov_radiomics_core::{
    contracts::*,
    errors::{ErrorCode, RadResult, RadiomicsError},
    learning::StudyTableSpec,
    table_io::{self, LoadedStudy},
};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{BufReader, Write},
    path::{Component, Path, PathBuf},
};

pub(super) fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
pub(super) fn schema(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Schema, message)
}
pub(super) fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}
pub(super) fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
pub(super) fn read_json<T: DeserializeOwned>(path: &Path) -> RadResult<T> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}
pub(super) fn load_study_record(
    path: &Path,
    task: Task,
) -> RadResult<(StudyTableSpec, LoadedStudy)> {
    let spec: StudyTableSpec = read_json(path)?;
    let loaded = table_io::read_study(&spec, task, parent(path))?;
    Ok((spec, loaded))
}
fn csv_error(error: csv::Error) -> RadiomicsError {
    invalid(format!("CSV export: {error}"))
}

pub(super) struct Output {
    directory: PathBuf,
    overwrite: bool,
    files: BTreeSet<PathBuf>,
}
impl Output {
    pub fn preflight(
        directory: &Path,
        overwrite: bool,
        files: impl IntoIterator<Item = PathBuf>,
    ) -> RadResult<Self> {
        if directory.exists() && (!overwrite || !directory.is_dir()) {
            return Err(invalid("Output directory exists or is not a directory; choose a new location or explicitly pass --overwrite."));
        }
        let files: BTreeSet<_> = files.into_iter().collect();
        for relative in &files {
            if relative.as_os_str().is_empty()
                || relative
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(schema("Command output filenames must be fixed relative paths without parent/root components."));
            }
            let path = directory.join(relative);
            if path.exists()
                && (!overwrite
                    || !path.is_file()
                    || fs::symlink_metadata(&path)?.file_type().is_symlink())
            {
                return Err(invalid(format!("Known output '{}' exists without overwrite permission or is not an ordinary file.",path.display())));
            }
        }
        Ok(Self {
            directory: directory.to_owned(),
            overwrite,
            files,
        })
    }
    pub fn begin(&self) -> RadResult<()> {
        fs::create_dir_all(&self.directory)?;
        for name in &self.files {
            let path = self.directory.join(name);
            fs::create_dir_all(parent(&path))?;
            self.checked_path(name)?;
        }
        Ok(())
    }
    pub fn checked_path(&self, relative: impl AsRef<Path>) -> RadResult<PathBuf> {
        let relative = relative.as_ref();
        if !self.files.contains(relative) {
            return Err(schema(
                "Attempted to write a file outside this command's declared output set.",
            ));
        }
        let path = self.directory.join(relative);
        if self.directory.exists() && parent(&path).exists() {
            let root = fs::canonicalize(&self.directory)?;
            let target_parent = fs::canonicalize(parent(&path))?;
            if !target_parent.starts_with(root) {
                return Err(invalid(
                    "A known output parent resolves outside the requested output directory.",
                ));
            }
        }
        Ok(path)
    }
    fn write(&self, name: &str, write: impl FnOnce(&mut File) -> RadResult<()>) -> RadResult<()> {
        let path = self.checked_path(name)?;
        if path.exists() && !self.overwrite {
            return Err(invalid(format!(
                "Output '{}' was created before publishing; overwrite was not authorized.",
                path.display()
            )));
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| RadiomicsError::new(ErrorCode::Io, e.to_string()))?
            .as_nanos();
        let temporary =
            parent(&path).join(format!(".radiomics-cli-{}-{stamp}.tmp", std::process::id()));
        let result = (|| -> RadResult<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            write(&mut file)?;
            file.flush()?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }
    pub fn json<T: Serialize>(&self, name: &str, value: &T) -> RadResult<()> {
        self.write(name, |file| {
            serde_json::to_writer_pretty(file, value)?;
            Ok(())
        })
    }
    pub fn csv(
        &self,
        name: &str,
        headers: &[&str],
        rows: impl IntoIterator<Item = Vec<String>>,
    ) -> RadResult<()> {
        self.write(name, |file| {
            let mut writer = csv::Writer::from_writer(file);
            writer.write_record(headers).map_err(csv_error)?;
            for row in rows {
                writer.write_record(row).map_err(csv_error)?;
            }
            writer.flush()?;
            Ok(())
        })
    }
    pub fn predictions(
        &self,
        json_name: &str,
        csv_name: &str,
        table: &PredictionTable,
    ) -> RadResult<()> {
        table.validate()?;
        self.json(json_name, table)?;
        let mut headers = vec![
            "sample_id".to_string(),
            "patient_id".into(),
            "slide_id".into(),
            "tile_id".into(),
            "prediction_unit".into(),
            "cohort".into(),
            "fold".into(),
            "training_source".into(),
            "value".into(),
        ];
        headers.extend(
            table
                .classes
                .iter()
                .map(|class| format!("probability:{class}")),
        );
        let header_refs: Vec<_> = headers.iter().map(String::as_str).collect();
        let unit = match table.unit {
            PredictionUnit::Sample => "sample",
            PredictionUnit::Tile => "tile",
            PredictionUnit::Slide => "slide",
            PredictionUnit::Patient => "patient",
        };
        self.csv(
            csv_name,
            &header_refs,
            table.rows.iter().map(|row| {
                let mut values = vec![
                    row.sample_id.clone(),
                    row.patient_id.clone(),
                    row.slide_id.clone().unwrap_or_default(),
                    row.tile_id.clone().unwrap_or_default(),
                    unit.into(),
                    row.cohort.clone(),
                    row.fold.map(|v| v.to_string()).unwrap_or_default(),
                    row.training_source.clone(),
                    row.value.map(|v| v.to_string()).unwrap_or_default(),
                ];
                values.extend(row.probabilities.iter().map(ToString::to_string));
                values
            }),
        )
    }
}
