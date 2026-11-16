//! Shared image-cohort workflow used by the CLI and the Python boundary.
use crate::{
    contracts::*,
    errors::{ErrorCode, RadResult, RadiomicsError},
    features, imaging,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

pub const fn implementation_sha256() -> &'static str {
    env!("RAD_IMPLEMENTATION_SHA256")
}
fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn csv_error(e: csv::Error) -> RadiomicsError {
    invalid(format!("CSV: {e}"))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortSample {
    pub metadata: SampleMetadata,
    pub image: PathBuf,
    pub mask: PathBuf,
    pub organ_mask: Option<PathBuf>,
    #[serde(default)]
    pub roi_label: Option<u32>,
}

/// Paths are resolved relative to the cohort CSV, never the shell's current directory.
pub fn read_cohort(path: impl AsRef<Path>) -> RadResult<Vec<CohortSample>> {
    let path = path.as_ref();
    let base = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let base = fs::canonicalize(base)?;
    let mut reader = csv::Reader::from_path(path).map_err(csv_error)?;
    let headers = reader.headers().map_err(csv_error)?.clone();
    unique_nonempty(
        &headers.iter().map(str::to_string).collect::<Vec<_>>(),
        "Cohort column names",
    )?;
    for required in ["sample_id", "patient_id", "image", "mask"] {
        if !headers.iter().any(|s| s == required) {
            return Err(invalid(format!("Missing cohort column: {required}")));
        }
    }
    let mut result = Vec::new();
    for record in reader.records() {
        let record = record.map_err(csv_error)?;
        let get = |name: &str| {
            headers
                .iter()
                .position(|s| s == name)
                .and_then(|i| record.get(i))
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let required = |name: &str| {
            get(name).ok_or_else(|| {
                invalid(format!("Missing {name} on cohort row {}", result.len() + 2))
            })
        };
        let resolve = |value: String| {
            let p = PathBuf::from(value);
            if p.is_absolute() {
                p
            } else {
                base.join(p)
            }
        };
        result.push(CohortSample {
            metadata: SampleMetadata {
                sample_id: required("sample_id")?,
                patient_id: required("patient_id")?,
                slide_id: get("slide_id"),
                lesion_id: get("lesion_id"),
                modality: get("modality"),
                region: get("region"),
                timepoint: get("timepoint"),
                center: get("center"),
            },
            image: resolve(required("image")?),
            mask: resolve(required("mask")?),
            organ_mask: get("organ_mask").map(resolve),
            roi_label: get("roi_label")
                .map(|label| {
                    label
                        .parse::<u32>()
                        .map_err(|_| invalid("roi_label must be a nonnegative integer."))
                })
                .transpose()?,
        });
    }
    validate_cohort(&result)?;
    Ok(result)
}
pub fn validate_cohort(samples: &[CohortSample]) -> RadResult<()> {
    if samples.is_empty() {
        return Err(invalid("Cohort is empty."));
    }
    unique_nonempty(
        &samples
            .iter()
            .map(|s| s.metadata.sample_id.clone())
            .collect::<Vec<_>>(),
        "Sample IDs",
    )?;
    for sample in samples {
        sample.metadata.validate()?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticsScope {
    WholeImage,
    Roi,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum PrepareStep {
    ResampleSpacing {
        spacing_mm: [f64; 3],
        interpolation: imaging::Interpolation,
        outside: f64,
    },
    CropVoxels {
        margin_voxels: [usize; 3],
    },
    CropMm {
        margin_mm: [f64; 3],
    },
    ClipRange {
        lower: f64,
        upper: f64,
    },
    ClipPercentile {
        lower: f64,
        upper: f64,
        statistics: StatisticsScope,
    },
    Normalize {
        method: imaging::Normalization,
        statistics: StatisticsScope,
    },
    LargestComponent {
        connectivity: imaging::Connectivity,
    },
    Peritumoral {
        inner_mm: f64,
        outer_mm: f64,
    },
    Resegment {
        lower: f64,
        upper: f64,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractConfig {
    pub extraction: ExtractionConfig,
    pub labels: Vec<u32>,
    #[serde(default)]
    pub preparation: Vec<PrepareStep>,
    #[serde(default)]
    pub families: Option<Vec<features::FeatureFamily>>,
}
impl ExtractConfig {
    pub fn validate(&self) -> RadResult<()> {
        self.extraction.validate()?;
        if let Some(families) = &self.families {
            features::validate_families(families)?;
        }
        if self.labels.is_empty()
            || self.labels.contains(&0)
            || self.labels.iter().collect::<BTreeSet<_>>().len() != self.labels.len()
        {
            return Err(invalid("Select unique, nonzero ROI labels."));
        }
        for (i, step) in self.preparation.iter().enumerate() {
            if matches!(step, PrepareStep::Resegment { .. }) && i + 1 != self.preparation.len() {
                return Err(invalid("Resegment must be the final preparation step, preserving the morphological ROI."));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractedSample {
    pub metadata: SampleMetadata,
    pub features: BTreeMap<String, Option<f64>>,
    pub quality: Vec<QualityRecord>,
    pub input_sha256: BTreeMap<String, String>,
    pub cache_key: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailedSample {
    pub metadata: SampleMetadata,
    pub error: RadiomicsError,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Complete,
    Partial,
    Empty,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionBatch {
    pub status: BatchStatus,
    pub table: Option<FeatureTable>,
    pub records: Vec<ExtractedSample>,
    pub failures: Vec<FailedSample>,
    pub config: ExtractConfig,
    pub implementation_sha256: String,
    #[serde(default)]
    pub cache_summary: CacheSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheSummary {
    pub enabled: bool,
    pub hits: usize,
    pub misses: usize,
    pub rejected: usize,
    pub write_failures: usize,
}
#[derive(Default)]
struct CacheCounters {
    hits: AtomicUsize,
    misses: AtomicUsize,
    rejected: AtomicUsize,
    write_failures: AtomicUsize,
}

fn effective_extract_config(
    sample: &CohortSample,
    config: &ExtractConfig,
) -> RadResult<ExtractConfig> {
    let mut effective = config.clone();
    if let Some(label) = sample.roi_label {
        effective.labels = vec![label];
    }
    effective.validate()?;
    Ok(effective)
}
pub fn prepare_sample(
    sample: &CohortSample,
    config: &ExtractConfig,
) -> RadResult<(ImageSample, Option<MaskVolume>, Vec<QualityRecord>)> {
    let effective = effective_extract_config(sample, config)?;
    let config = &effective;
    let mut image = imaging::read_nifti(&sample.image)?;
    let source_mask = imaging::read_mask(&sample.mask)?;
    image.geometry.aligned_with(&source_mask.geometry)?;
    let mut mask = imaging::select_labels(&source_mask, &config.labels)?;
    let mut organ = sample
        .organ_mask
        .as_ref()
        .map(imaging::read_mask)
        .transpose()?;
    if let Some(organ) = &organ {
        image.geometry.aligned_with(&organ.geometry)?;
    }
    let mut intensity_mask = None;
    let mut quality = image.geometry.diagnostics.clone();
    quality.extend(mask.geometry.diagnostics.clone());
    quality.push(QualityRecord {
        code: "effective_roi_labels".into(),
        message: serde_json::to_string(&config.labels)?,
        severity: "info".into(),
        sample_id: Some(sample.metadata.sample_id.clone()),
    });
    for step in &config.preparation {
        match *step {
            PrepareStep::ResampleSpacing {
                spacing_mm,
                interpolation,
                outside,
            } => {
                let grid = imaging::spacing_grid(&image.geometry, spacing_mm)?;
                image = imaging::resample_like(&image, &grid, interpolation, outside)?;
                mask = imaging::resample_mask_like(&mask, &grid, 0)?;
                organ = organ
                    .map(|o| imaging::resample_mask_like(&o, &grid, 0))
                    .transpose()?;
            }
            PrepareStep::CropVoxels { margin_voxels } => {
                let bounds = imaging::roi_bounds_margin_voxels(&mask, margin_voxels)?;
                image = imaging::crop_image(&image, bounds)?;
                mask = imaging::crop_mask(&mask, bounds)?;
                organ = organ.map(|o| imaging::crop_mask(&o, bounds)).transpose()?;
            }
            PrepareStep::CropMm { margin_mm } => {
                let bounds = imaging::roi_bounds_margin_mm(&mask, margin_mm)?;
                image = imaging::crop_image(&image, bounds)?;
                mask = imaging::crop_mask(&mask, bounds)?;
                organ = organ.map(|o| imaging::crop_mask(&o, bounds)).transpose()?;
            }
            PrepareStep::ClipRange { lower, upper } => {
                image = imaging::clip_range(&image, lower, upper)?
            }
            PrepareStep::ClipPercentile {
                lower,
                upper,
                statistics,
            } => {
                image = imaging::clip_percentile(
                    &image,
                    lower,
                    upper,
                    if statistics == StatisticsScope::Roi {
                        Some(&mask)
                    } else {
                        None
                    },
                )?
            }
            PrepareStep::Normalize { method, statistics } => {
                let fitted = imaging::fit_intensity_transform(
                    &image,
                    method,
                    if statistics == StatisticsScope::Roi {
                        Some(&mask)
                    } else {
                        None
                    },
                )?;
                image = fitted.apply(&image)?;
                quality.push(QualityRecord {
                    code: "intensity_transform".into(),
                    message: serde_json::to_string(&fitted)?,
                    severity: "info".into(),
                    sample_id: Some(sample.metadata.sample_id.clone()),
                });
            }
            PrepareStep::LargestComponent { connectivity } => {
                mask = imaging::largest_component(&mask, connectivity)?
            }
            PrepareStep::Peritumoral { inner_mm, outer_mm } => {
                mask = imaging::peritumoral_ring(&mask, inner_mm, outer_mm, organ.as_ref())?
            }
            PrepareStep::Resegment { lower, upper } => {
                intensity_mask = Some(imaging::resegment(&image, &mask, lower, upper)?)
            }
        }
    }
    for q in &mut quality {
        q.sample_id = Some(sample.metadata.sample_id.clone());
    }
    let prepared = ImageSample {
        metadata: sample.metadata.clone(),
        image,
        mask,
    };
    prepared.validate()?;
    Ok((prepared, intensity_mask, quality))
}

pub fn file_sha256(path: impl AsRef<Path>) -> RadResult<String> {
    let mut source = File::open(path)?;
    let mut digest = Sha256::new();
    let mut block = [0u8; 65536];
    loop {
        let count = source.read(&mut block)?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheEntry {
    schema: u32,
    key: String,
    payload_sha256: String,
    payload: ExtractedSample,
}
fn payload_hash(payload: &ExtractedSample) -> RadResult<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(payload)?)
    ))
}
fn extract_one(
    sample: &CohortSample,
    config: &ExtractConfig,
    cache: Option<&Path>,
    counters: &CacheCounters,
    threads: usize,
) -> RadResult<ExtractedSample> {
    let effective = effective_extract_config(sample, config)?;
    let mut hashes = BTreeMap::from([
        ("image".into(), file_sha256(&sample.image)?),
        ("mask".into(), file_sha256(&sample.mask)?),
    ]);
    if let Some(organ) = &sample.organ_mask {
        hashes.insert("organ_mask".into(), file_sha256(organ)?);
    }
    let key = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(
            1u32,
            &hashes,
            &effective,
            &sample.metadata,
            implementation_sha256()
        ))?)
    );
    let cache_file = cache.map(|dir| dir.join(format!("{key}.json")));
    let mut cache_diagnostic = None;
    if let Some(path) = &cache_file {
        if path.exists() {
            let candidate = (|| -> RadResult<ExtractedSample> {
                let entry: CacheEntry = serde_json::from_slice(&fs::read(path)?)?;
                if entry.schema != 1
                    || entry.key != key
                    || entry.payload.cache_key != key
                    || entry.payload.metadata != sample.metadata
                    || entry.payload.input_sha256 != hashes
                    || entry.payload_sha256 != payload_hash(&entry.payload)?
                {
                    return Err(invalid("Cache content validation failed."));
                }
                if entry.payload.features.is_empty()
                    || entry
                        .payload
                        .features
                        .values()
                        .flatten()
                        .any(|v| !v.is_finite())
                {
                    return Err(invalid("Invalid cached feature payload."));
                }
                Ok(entry.payload)
            })();
            match candidate {
                Ok(payload) => {
                    counters.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(payload);
                }
                Err(e) => {
                    counters.rejected.fetch_add(1, Ordering::Relaxed);
                    cache_diagnostic = Some(e.to_string());
                }
            }
        }
    }
    if cache.is_some() {
        counters.misses.fetch_add(1, Ordering::Relaxed);
    }
    let (prepared, intensity_mask, mut quality) = prepare_sample(sample, &effective)?;
    if let Some(message) = cache_diagnostic {
        quality.push(QualityRecord {
            code: "cache_rejected".into(),
            message,
            severity: "warning".into(),
            sample_id: Some(sample.metadata.sample_id.clone()),
        });
    }
    let output = features::extract_families_with_threads(
        &prepared.image,
        &prepared.mask,
        intensity_mask.as_ref(),
        1,
        &effective.extraction,
        effective
            .families
            .as_deref()
            .unwrap_or(&features::ALL_FAMILIES),
        threads,
    )?;
    let mut features = BTreeMap::new();
    for (name, value) in output {
        if !value.is_finite() {
            quality.push(QualityRecord {
                code: "undefined_feature".into(),
                message: format!("{name}: {value}"),
                severity: "warning".into(),
                sample_id: Some(sample.metadata.sample_id.clone()),
            });
        }
        features.insert(name, value.is_finite().then_some(value));
    }
    let mut record = ExtractedSample {
        metadata: sample.metadata.clone(),
        features,
        quality,
        input_sha256: hashes,
        cache_key: key.clone(),
    };
    if file_sha256(&sample.image)? != record.input_sha256["image"]
        || file_sha256(&sample.mask)? != record.input_sha256["mask"]
        || sample
            .organ_mask
            .as_ref()
            .map(file_sha256)
            .transpose()?
            .is_some_and(|hash| Some(&hash) != record.input_sha256.get("organ_mask"))
    {
        return Err(invalid("Input content changed during extraction; no result/cache entry is accepted for a moving input."));
    }
    if let Some(path) = cache_file {
        let entry = CacheEntry {
            schema: 1,
            key,
            payload_sha256: payload_hash(&record)?,
            payload: record.clone(),
        };
        let write = (|| -> RadResult<()> {
            let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
            f.write_all(&serde_json::to_vec(&entry)?)?;
            f.sync_all()?;
            Ok(())
        })();
        if let Err(e) = write {
            counters.write_failures.fetch_add(1, Ordering::Relaxed);
            record.quality.push(QualityRecord {
                code: "cache_write_failed".into(),
                message: e.to_string(),
                severity: "warning".into(),
                sample_id: Some(sample.metadata.sample_id.clone()),
            });
        }
    }
    Ok(record)
}

pub fn extract_batch(
    samples: &[CohortSample],
    config: &ExtractConfig,
    n_jobs: usize,
    cache: Option<&Path>,
) -> RadResult<ExtractionBatch> {
    extract_batch_with_threads(samples, config, n_jobs, cache, 1)
}

pub fn extract_batch_with_threads(
    samples: &[CohortSample],
    config: &ExtractConfig,
    n_jobs: usize,
    cache: Option<&Path>,
    threads: usize,
) -> RadResult<ExtractionBatch> {
    validate_cohort(samples)?;
    config.validate()?;
    features::validate_thread_budget(n_jobs, threads)?;
    if let Some(dir) = cache {
        fs::create_dir_all(dir)?;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs)
        .build()
        .map_err(|e| invalid(e.to_string()))?;
    let counters = CacheCounters::default();
    let results: Vec<_> = pool.install(|| {
        samples
            .par_iter()
            .map(|sample| extract_one(sample, config, cache, &counters, threads))
            .collect()
    });
    let mut records = Vec::new();
    let mut failures = Vec::new();
    for (sample, result) in samples.iter().zip(results) {
        match result {
            Ok(row) => records.push(row),
            Err(error) => failures.push(FailedSample {
                metadata: sample.metadata.clone(),
                error,
            }),
        }
    }
    records.sort_by(|a, b| a.metadata.sample_id.cmp(&b.metadata.sample_id));
    failures.sort_by(|a, b| a.metadata.sample_id.cmp(&b.metadata.sample_id));
    let table = if records.is_empty() {
        None
    } else {
        let names: Vec<_> = records
            .iter()
            .flat_map(|r| r.features.keys().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut values = Vec::new();
        let mut missing = Vec::new();
        for row in &records {
            for name in &names {
                let value = row.features.get(name).copied().flatten();
                values.push(value.unwrap_or(0.));
                missing.push(value.is_none());
            }
        }
        let table = FeatureTable {
            data: NumericBuffer {
                rows: records.len(),
                cols: names.len(),
                values,
                missing,
            },
            samples: records.iter().map(|r| r.metadata.clone()).collect(),
            features: names
                .iter()
                .map(|name| FeatureDefinition {
                    name: name.clone(),
                    definition: "python_golden_v1 explicit Original subset; see features-native.md"
                        .into(),
                    unit: name
                        .rsplit_once('_')
                        .filter(|(_, u)| {
                            matches!(
                                *u,
                                "mm" | "mm2"
                                    | "mm3"
                                    | "native"
                                    | "native2"
                                    | "native3"
                                    | "index"
                                    | "voxels"
                            )
                        })
                        .map(|(_, u)| u.to_string()),
                })
                .collect(),
            effective_parameters: BTreeMap::from([
                ("workflow".into(), serde_json::to_value(config)?),
                (
                    "implementation_sha256".into(),
                    implementation_sha256().into(),
                ),
            ]),
            quality: records.iter().flat_map(|r| r.quality.clone()).collect(),
        };
        table.validate()?;
        Some(table)
    };
    let status = if records.is_empty() {
        BatchStatus::Empty
    } else if failures.is_empty() {
        BatchStatus::Complete
    } else {
        BatchStatus::Partial
    };
    Ok(ExtractionBatch {
        status,
        table,
        records,
        failures,
        config: config.clone(),
        implementation_sha256: implementation_sha256().into(),
        cache_summary: CacheSummary {
            enabled: cache.is_some(),
            hits: counters.hits.load(Ordering::Relaxed),
            misses: counters.misses.load(Ordering::Relaxed),
            rejected: counters.rejected.load(Ordering::Relaxed),
            write_failures: counters.write_failures.load(Ordering::Relaxed),
        },
    })
}

pub fn export_extraction(
    batch: &ExtractionBatch,
    directory: impl AsRef<Path>,
    overwrite: bool,
) -> RadResult<()> {
    let directory = directory.as_ref();
    if directory.exists() && !overwrite {
        return Err(invalid(
            "Output directory exists; choose a new location or explicitly allow overwrite.",
        ));
    }
    fs::create_dir_all(directory)?;
    let mut writer = csv::Writer::from_path(directory.join("features.csv")).map_err(csv_error)?;
    let names: Vec<_> = batch
        .table
        .as_ref()
        .map(|t| t.features.iter().map(|f| f.name.clone()).collect())
        .unwrap_or_default();
    writer
        .write_record(std::iter::once("sample_id".to_string()).chain(names.iter().cloned()))
        .map_err(csv_error)?;
    for row in &batch.records {
        writer
            .write_record(
                std::iter::once(row.metadata.sample_id.clone()).chain(names.iter().map(|n| {
                    row.features
                        .get(n)
                        .copied()
                        .flatten()
                        .map(|v| v.to_string())
                        .unwrap_or_default()
                })),
            )
            .map_err(csv_error)?;
    }
    writer.flush()?;
    let mut metadata = csv::Writer::from_path(directory.join("metadata.csv")).map_err(csv_error)?;
    for sample in batch
        .records
        .iter()
        .map(|r| &r.metadata)
        .chain(batch.failures.iter().map(|r| &r.metadata))
    {
        metadata.serialize(sample).map_err(csv_error)?;
    }
    metadata.flush()?;
    let mut failure = csv::Writer::from_path(directory.join("failures.csv")).map_err(csv_error)?;
    failure
        .write_record(["sample_id", "patient_id", "code", "message"])
        .map_err(csv_error)?;
    for row in &batch.failures {
        failure
            .write_record([
                row.metadata.sample_id.clone(),
                row.metadata.patient_id.clone(),
                format!("{:?}", row.error.code),
                row.error.message.clone(),
            ])
            .map_err(csv_error)?;
    }
    failure.flush()?;
    fs::write(
        directory.join("extraction.json"),
        serde_json::to_vec_pretty(batch)?,
    )?;
    Ok(())
}
