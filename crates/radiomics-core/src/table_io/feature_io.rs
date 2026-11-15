use super::*;
use parquet::basic::{Compression, LogicalType, Repetition, Type as PhysicalType};
use parquet::data_type::{ByteArray, ByteArrayType, DoubleType};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;
use std::fs::OpenOptions;
use std::sync::Arc;

pub const FEATURE_MANIFEST_KEY: &str = "radiomics.feature_table_manifest";
pub fn feature_manifest_path(path: impl AsRef<Path>) -> PathBuf {
    let mut name = path.as_ref().as_os_str().to_owned();
    name.push(".manifest.json");
    PathBuf::from(name)
}
fn embedded_manifest(path: &Path) -> RadResult<Option<FeatureTableManifest>> {
    let reader = SerializedFileReader::new(File::open(path)?).map_err(parquet_error)?;
    let entries: Vec<_> = reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .into_iter()
        .flatten()
        .filter(|kv| kv.key == FEATURE_MANIFEST_KEY)
        .collect();
    if entries.len() > 1 {
        return Err(schema("Parquet contains duplicate feature manifest keys."));
    }
    if let Some(entry) = entries.first() {
        let manifest: FeatureTableManifest = serde_json::from_str(
            entry
                .value
                .as_deref()
                .ok_or_else(|| schema("Embedded feature manifest is empty."))?,
        )?;
        manifest.validate()?;
        if manifest.table_sha256.is_some() {
            return Err(schema(
                "Embedded feature manifest must have table_sha256=None to avoid circular hashing.",
            ));
        }
        Ok(Some(manifest))
    } else {
        Ok(None)
    }
}
pub fn read_feature_table(
    path: impl AsRef<Path>,
    manifest_path: Option<&Path>,
) -> RadResult<FeatureTable> {
    let path = resolve(Path::new("."), path.as_ref())?;
    let embedded = if format(&path)? == TableFormat::Parquet {
        embedded_manifest(&path)?
    } else {
        None
    };
    let sidecar = manifest_path
        .map(Path::to_owned)
        .unwrap_or_else(|| feature_manifest_path(&path));
    let external = if manifest_path.is_some() || sidecar.exists() {
        let manifest: FeatureTableManifest = serde_json::from_reader(File::open(&sidecar)?)?;
        manifest.validate()?;
        Some(manifest)
    } else {
        None
    };
    if let (Some(a), Some(b)) = (&embedded, &external) {
        let mut b = b.clone();
        b.table_sha256 = None;
        if a != &b {
            return Err(schema("Embedded and external feature manifests disagree."));
        }
    }
    let manifest=external.or(embedded).ok_or_else(||schema("Feature table reading requires its versioned manifest; sample metadata/definitions must not be guessed from row counts or column names."))?;
    let projection = Projection {
        numeric: manifest.features.iter().map(|f| f.name.clone()).collect(),
        text: BTreeMap::from([("sample_id".into(), TextRole::Id)]),
        missing_values: vec![manifest.csv_null_value.clone().unwrap_or_default()],
        nan_as_missing: false,
    };
    let raw = read_raw(&path, "feature_table", &projection)?;
    let ids = required_ids(&raw, "sample_id")?;
    check_manifest(&manifest, &path, &raw.headers, &ids)?;
    let lookup: BTreeMap<_, _> = manifest
        .samples
        .iter()
        .map(|s| (s.sample_id.as_str(), s))
        .collect();
    let samples = ids.iter().map(|id| lookup[id.as_str()].clone()).collect();
    let table = FeatureTable {
        data: raw.numeric,
        samples,
        features: manifest.features,
        effective_parameters: manifest.effective_parameters,
        quality: manifest.quality,
    };
    table.validate()?;
    Ok(table)
}
fn manifest(table: &FeatureTable, csv_null_value: Option<String>) -> FeatureTableManifest {
    FeatureTableManifest {
        format: "radiomics.feature-table".into(),
        schema_version: 1,
        samples: table.samples.clone(),
        features: table.features.clone(),
        effective_parameters: table.effective_parameters.clone(),
        quality: table.quality.clone(),
        csv_null_value,
        table_sha256: None,
    }
}
fn unique_temp(path: &Path, kind: &str) -> RadResult<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| RadiomicsError::new(ErrorCode::Io, e.to_string()))?
        .as_nanos();
    Ok(parent(path).join(format!(
        ".radiomics-{kind}-{}-{stamp}.tmp",
        std::process::id()
    )))
}
fn validate_destination(path: &Path, overwrite: bool) -> RadResult<()> {
    if path.exists() && (!overwrite || !path.is_file()) {
        return Err(invalid(format!(
            "Output '{}' exists or is not a file; use a new location or explicit overwrite.",
            path.display()
        )));
    }
    Ok(())
}
fn csv_write(table: &FeatureTable, path: &Path, null: &str) -> RadResult<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = csv::Writer::from_writer(file);
    writer
        .write_record(
            std::iter::once("sample_id").chain(table.features.iter().map(|f| f.name.as_str())),
        )
        .map_err(csv_error)?;
    for (i, sample) in table.samples.iter().enumerate() {
        writer.write_field(&sample.sample_id).map_err(csv_error)?;
        for j in 0..table.data.cols {
            let index = i * table.data.cols + j;
            if table.data.missing[index] {
                writer.write_field(null).map_err(csv_error)?;
            } else {
                writer
                    .write_field(table.data.values[index].to_string())
                    .map_err(csv_error)?;
            }
        }
        writer
            .write_record(std::iter::empty::<&str>())
            .map_err(csv_error)?;
    }
    writer.flush()?;
    Ok(())
}
pub(super) fn parquet_write(
    table: &FeatureTable,
    path: &Path,
    manifest: &FeatureTableManifest,
    compression: Compression,
) -> RadResult<()> {
    let mut fields = vec![Arc::new(
        Type::primitive_type_builder("sample_id", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(Some(LogicalType::String))
            .build()
            .map_err(parquet_error)?,
    )];
    for feature in &table.features {
        fields.push(Arc::new(
            Type::primitive_type_builder(&feature.name, PhysicalType::DOUBLE)
                .with_repetition(Repetition::OPTIONAL)
                .build()
                .map_err(parquet_error)?,
        ));
    }
    let parquet_schema = Arc::new(
        Type::group_type_builder("radiomics_features")
            .with_fields(fields)
            .build()
            .map_err(parquet_error)?,
    );
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(compression)
            .set_key_value_metadata(Some(vec![KeyValue {
                key: FEATURE_MANIFEST_KEY.into(),
                value: Some(serde_json::to_string(manifest)?),
            }]))
            .build(),
    );
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer =
        SerializedFileWriter::new(file, parquet_schema, props).map_err(parquet_error)?;
    for start in (0..table.data.rows).step_by(4096) {
        let stop = (start + 4096).min(table.data.rows);
        let mut group = writer.next_row_group().map_err(parquet_error)?;
        let mut id = group
            .next_column()
            .map_err(parquet_error)?
            .ok_or_else(|| schema("Parquet writer is missing the ID column."))?;
        let ids: Vec<_> = table.samples[start..stop]
            .iter()
            .map(|s| ByteArray::from(s.sample_id.as_str()))
            .collect();
        id.typed::<ByteArrayType>()
            .write_batch(&ids, None, None)
            .map_err(parquet_error)?;
        id.close().map_err(parquet_error)?;
        for j in 0..table.data.cols {
            let mut column = group
                .next_column()
                .map_err(parquet_error)?
                .ok_or_else(|| schema("Parquet writer is missing a feature column."))?;
            let mut values = Vec::with_capacity(stop - start);
            let mut definitions = Vec::with_capacity(stop - start);
            for i in start..stop {
                let index = i * table.data.cols + j;
                definitions.push(i16::from(!table.data.missing[index]));
                if !table.data.missing[index] {
                    values.push(table.data.values[index]);
                }
            }
            column
                .typed::<DoubleType>()
                .write_batch(&values, Some(&definitions), None)
                .map_err(parquet_error)?;
            column.close().map_err(parquet_error)?;
        }
        group.close().map_err(parquet_error)?;
    }
    writer.close().map_err(parquet_error)?;
    Ok(())
}
/// Write a new table plus a small manifest sidecar. Parquet also embeds the same
/// manifest with no circular table hash. All validation precedes output writes.
pub fn write_feature_table(
    table: &FeatureTable,
    path: impl AsRef<Path>,
    overwrite: bool,
    csv_null_value: &str,
) -> RadResult<FeatureTableManifest> {
    table.validate()?;
    let path = path.as_ref();
    let kind = format(path)?;
    let names: Vec<_> = std::iter::once("sample_id".to_string())
        .chain(table.features.iter().map(|f| f.name.clone()))
        .collect();
    unique_nonempty(&names, "Written feature column names")?;
    if kind == TableFormat::Csv {
        if table.samples.iter().any(|s| s.sample_id == csv_null_value) {
            return Err(invalid(
                "A real sample ID equals the declared CSV null token.",
            ));
        }
        if csv_null_value
            .parse::<f64>()
            .ok()
            .is_some_and(|v| v.is_finite() && v.to_string() == csv_null_value)
            && table
                .data
                .values
                .iter()
                .zip(&table.data.missing)
                .any(|(v, m)| !m && v.to_string() == csv_null_value)
        {
            return Err(invalid(
                "A real numeric value equals the declared CSV null token.",
            ));
        }
    }
    let sidecar = feature_manifest_path(path);
    validate_destination(path, overwrite)?;
    validate_destination(&sidecar, overwrite)?;
    fs::create_dir_all(parent(path))?;
    let temp = unique_temp(path, "table")?;
    let manifest_temp = unique_temp(&sidecar, "manifest")?;
    let mut manifest = manifest(
        table,
        (kind == TableFormat::Csv).then(|| csv_null_value.to_string()),
    );
    manifest.validate()?;
    let result = (|| -> RadResult<()> {
        match kind {
            TableFormat::Csv => csv_write(table, &temp, csv_null_value)?,
            TableFormat::Parquet => parquet_write(table, &temp, &manifest, Compression::SNAPPY)?,
        }
        manifest.table_sha256 = Some(sha256_file(&temp)?);
        let out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&manifest_temp)?;
        serde_json::to_writer_pretty(out, &manifest)?;
        fs::rename(&temp, path)?;
        fs::rename(&manifest_temp, &sidecar)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
        let _ = fs::remove_file(&manifest_temp);
    }
    result?;
    Ok(manifest)
}
