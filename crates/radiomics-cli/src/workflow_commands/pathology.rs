//! Python-free command orchestration over the common pathology kernels.
use super::common::*;
use clap::{Args, Subcommand};
use ov_radiomics_core::{
    contracts::*, errors::RadResult, imaging, overlay, pathology_aggregate as aggregate,
    pathology_pixels as pixels, table_io, workflow,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Args)]
pub struct PathologyCommand {
    #[command(subcommand)]
    pub command: PathologyAction,
}
#[derive(Debug, Subcommand)]
pub enum PathologyAction {
    /// Stream ordinary-image patches through explicit native QC/features and aggregation.
    Patches(PatchesCommand),
    /// Aggregate an already ID-aligned TileTable to slides and patients.
    Aggregate(AggregateCommand),
    /// Rasterize actual tile predictions/features/scores in bounded-memory blocks.
    Overlay(OverlayCommand),
}
#[derive(Debug, Args)]
pub struct PatchesCommand {
    #[arg(long)]
    pub cohort: PathBuf,
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long, default_value = "pathology-patches")]
    pub output: PathBuf,
    #[arg(long)]
    pub overwrite: bool,
}
#[derive(Debug, Args)]
pub struct AggregateCommand {
    #[arg(long)]
    pub table: PathBuf,
    #[arg(long)]
    pub config: PathBuf,
    /// Optional native CSV/Parquet FeatureTable, joined by sample_id == tile_id.
    #[arg(long)]
    pub embeddings: Option<PathBuf>,
    #[arg(long, requires = "embeddings")]
    pub embedding_manifest: Option<PathBuf>,
    #[arg(long, default_value = "pathology-aggregate")]
    pub output: PathBuf,
    #[arg(long)]
    pub overwrite: bool,
}
#[derive(Debug, Args)]
pub struct OverlayCommand {
    #[arg(long)]
    pub table: PathBuf,
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long, default_value = "pathology-overlay")]
    pub output: PathBuf,
    #[arg(long)]
    pub overwrite: bool,
}
pub fn pathology(args: PathologyCommand) -> RadResult<u8> {
    match args.command {
        PathologyAction::Patches(args) => patches(args),
        PathologyAction::Aggregate(args) => aggregate_command(args),
        PathologyAction::Overlay(args) => overlay_command(args),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SlideInput {
    image: PathBuf,
    /// Explicit co-gridded Gray8/Gray16 input, never an implicit RGB conversion.
    texture_image: Option<PathBuf>,
    #[serde(default)]
    mask: Option<PathBuf>,
    metadata: pixels::SlideMetadata,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchConfig {
    schema_version: u32,
    tiling: pixels::TileSpec,
    qc: Option<pixels::QcConfig>,
    color: Option<pixels::ColorConfig>,
    texture: Option<ExtractionConfig>,
    #[serde(default)]
    mask_analysis: Option<MaskConfig>,
    tissue_only: bool,
    /// Explicit ordered scalar schema; circular hue angles are not scalar choices.
    features: Vec<PatchFeature>,
    aggregation: aggregate::AggregateConfig,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MaskConfig {
    kind: pixels::MaskKind,
    connectivity: imaging::Connectivity,
    classes: Vec<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum PatchFeature {
    Qc { score: String },
    ColorStatistic { channel: String, statistic: String },
    ColorHistogram { channel: String, bin: usize },
    HueResultantLength,
    HueCircularVariance,
    Texture { name: String },
    PatchObjectCount,
    ObjectSizeStatistic { statistic: String },
    AnnotationFraction { label: u32 },
}
impl PatchFeature {
    fn name(&self) -> String {
        match self {
            Self::Qc { score } => format!("qc_{score}"),
            Self::ColorStatistic { channel, statistic } => format!("color_{channel}_{statistic}"),
            Self::ColorHistogram { channel, bin } => {
                format!("color_{channel}_histogram_probability_{bin}")
            }
            Self::HueResultantLength => "color_H_resultant_length".into(),
            Self::HueCircularVariance => "color_H_circular_variance".into(),
            Self::Texture { name } => name.clone(),
            Self::PatchObjectCount => "patch_object_count".into(),
            Self::ObjectSizeStatistic { statistic } => format!("patch_object_area_px_{statistic}"),
            Self::AnnotationFraction { label } => format!("annotation_fraction_label_{label}"),
        }
    }
    fn value(
        &self,
        qc: Option<&pixels::PatchQc>,
        color: Option<&pixels::ColorFeatures>,
        texture: Option<&BTreeMap<String, f64>>,
        objects: Option<&pixels::ObjectFeatures>,
        annotation: Option<&pixels::AnnotationCoverage>,
    ) -> Option<f64> {
        match self {
            Self::Qc { score } => qc.and_then(|q| q.scores.get(score).copied().flatten()),
            Self::ColorStatistic { channel, statistic } => color
                .and_then(|c| c.channels.get(channel))
                .and_then(|c| c.statistics.as_ref())
                .and_then(|s| match statistic.as_str() {
                    "mean" => Some(s.mean),
                    "std" => Some(s.std),
                    "minimum" => Some(s.minimum),
                    "maximum" => Some(s.maximum),
                    "q25" => Some(s.q25),
                    "q50" => Some(s.q50),
                    "q75" => Some(s.q75),
                    _ => None,
                }),
            Self::ColorHistogram { channel, bin } => color
                .and_then(|c| {
                    if channel == "H" {
                        c.hue.as_ref().map(|h| &h.histogram)
                    } else {
                        c.channels.get(channel).map(|c| &c.histogram)
                    }
                })
                .and_then(|h| h.probabilities.as_ref())
                .and_then(|p| p.get(*bin).copied()),
            Self::HueResultantLength => color
                .and_then(|c| c.hue.as_ref())
                .and_then(|h| h.resultant_length),
            Self::HueCircularVariance => color
                .and_then(|c| c.hue.as_ref())
                .and_then(|h| h.circular_variance),
            Self::Texture { name } => texture.and_then(|t| t.get(name).copied()),
            Self::PatchObjectCount => objects.and_then(|o| {
                ((o.object_count as u128) <= (1u128 << 53)).then_some(o.object_count as f64)
            }),
            Self::ObjectSizeStatistic { statistic } => objects
                .and_then(|o| o.size_distribution.as_ref())
                .and_then(|s| match statistic.as_str() {
                    "mean" => Some(s.mean),
                    "std" => Some(s.std),
                    "minimum" => Some(s.minimum),
                    "maximum" => Some(s.maximum),
                    "q25" => Some(s.q25),
                    "q50" => Some(s.q50),
                    "q75" => Some(s.q75),
                    _ => None,
                }),
            Self::AnnotationFraction { label } => {
                annotation.and_then(|a| a.fractions.get(&label.to_string()).copied())
            }
        }
        .filter(|v| v.is_finite())
    }
}
impl PatchConfig {
    fn validate(&self) -> RadResult<()> {
        if self.schema_version != 1 || self.features.is_empty() {
            return Err(schema(
                "Patch config requires schema_version=1 and explicit nonempty features.",
            ));
        }
        if self.tiling.patch_size.contains(&0) || self.tiling.stride.contains(&0) {
            return Err(invalid("Patch size and stride must be positive."));
        }
        if self.tissue_only && self.qc.is_none() {
            return Err(invalid(
                "tissue_only requires an explicit RGB QC configuration.",
            ));
        }
        if let Some(config) = &self.qc {
            config.validate()?;
        }
        if let Some(config) = &self.color {
            config.validate()?;
        }
        self.aggregation.validate()?;
        if let Some(mask) = &self.mask_analysis {
            if !matches!(
                mask.connectivity,
                imaging::Connectivity::Four | imaging::Connectivity::Eight
            ) {
                return Err(invalid(
                    "Pathology semantic/instance masks require explicit four/eight connectivity.",
                ));
            }
            pixels::annotation_coverage(1, 1, &[0], [0, 0, 1, 1], &mask.classes)?;
        }
        let template = if let Some(config) = &self.texture {
            config.validate()?;
            Some(pixels::texture_features(
                &RasterImage {
                    width: 2,
                    height: 2,
                    pixels: RasterPixels::Gray8(vec![1, 2, 3, 4]),
                },
                [0, 0, 2, 2],
                None,
                config,
            )?)
        } else {
            None
        };
        let names: Vec<_> = self.features.iter().map(PatchFeature::name).collect();
        unique_nonempty(&names, "Selected patch features")?;
        for feature in &self.features {
            let valid = match feature {
                PatchFeature::Qc { score } => {
                    self.qc.is_some()
                        && [
                            "white_fraction",
                            "low_saturation_fraction",
                            "black_border_fraction",
                            "tissue_fraction",
                            "laplacian_variance",
                        ]
                        .contains(&score.as_str())
                }
                PatchFeature::ColorStatistic { channel, statistic } => {
                    self.color.is_some()
                        && ["R", "G", "B", "S", "V", "L", "a", "b"].contains(&channel.as_str())
                        && ["mean", "std", "minimum", "maximum", "q25", "q50", "q75"]
                            .contains(&statistic.as_str())
                }
                PatchFeature::ColorHistogram { channel, bin } => self
                    .color
                    .as_ref()
                    .and_then(|c| c.fixed_edges.get(channel))
                    .is_some_and(|e| *bin < e.len() - 1),
                PatchFeature::HueResultantLength | PatchFeature::HueCircularVariance => {
                    self.color.is_some()
                }
                PatchFeature::Texture { name } => {
                    template.as_ref().is_some_and(|t| t.contains_key(name))
                }
                PatchFeature::PatchObjectCount => self.mask_analysis.is_some(),
                PatchFeature::ObjectSizeStatistic { statistic } => {
                    self.mask_analysis.is_some()
                        && ["mean", "std", "minimum", "maximum", "q25", "q50", "q75"]
                            .contains(&statistic.as_str())
                }
                PatchFeature::AnnotationFraction { label } => self
                    .mask_analysis
                    .as_ref()
                    .is_some_and(|m| m.classes.contains(label)),
            };
            if !valid {
                return Err(invalid(format!("Unknown/inapplicable explicit patch feature {}. Hue mean angles cannot enter linear aggregation.",feature.name())));
            }
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
struct SlideResult {
    patient_id: String,
    slide_id: String,
    status: String,
    tiles: usize,
    accepted_tiles: usize,
    image_sha256: Option<String>,
    texture_sha256: Option<String>,
    mask_sha256: Option<String>,
    error: Option<String>,
}
fn scalar_table(
    table: &TileTable,
    config: &PatchConfig,
    quality: Vec<QualityRecord>,
) -> FeatureTable {
    FeatureTable {
        data: table
            .features
            .clone()
            .expect("Patch config guarantees feature schema"),
        samples: table
            .tiles
            .iter()
            .map(|t| SampleMetadata {
                sample_id: t.tile_id.clone(),
                patient_id: t.patient_id.clone(),
                slide_id: Some(t.slide_id.clone()),
                lesion_id: None,
                modality: None,
                region: None,
                timepoint: None,
                center: None,
            })
            .collect(),
        features: config
            .features
            .iter()
            .map(|f| FeatureDefinition {
                name: f.name(),
                definition: serde_json::to_string(f).expect("typed feature serializes"),
                unit: None,
            })
            .collect(),
        effective_parameters: [
            ("config".into(), json!(config)),
            (
                "implementation_sha256".into(),
                json!(workflow::implementation_sha256()),
            ),
            ("coordinate_space".into(), json!("level0_pixel_edges")),
            ("object_count_scope".into(), json!("patch_local_only")),
        ]
        .into_iter()
        .collect(),
        quality,
    }
}
fn table_names(prefix: &str) -> Vec<PathBuf> {
    [
        "json",
        "csv",
        "csv.manifest.json",
        "parquet",
        "parquet.manifest.json",
    ]
    .into_iter()
    .map(|ext| format!("{prefix}.{ext}").into())
    .collect()
}
fn write_table(
    output: &Output,
    prefix: &str,
    table: &FeatureTable,
    overwrite: bool,
) -> RadResult<()> {
    table.validate()?;
    output.json(&format!("{prefix}.json"), table)?;
    for ext in ["csv", "parquet"] {
        table_io::write_feature_table(
            table,
            output.checked_path(format!("{prefix}.{ext}"))?,
            overwrite,
            "\\N",
        )?;
    }
    Ok(())
}
fn json_line(writer: &mut BufWriter<File>, value: &Value) -> RadResult<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}
fn stream_file(output: &Output, name: &str, overwrite: bool) -> RadResult<BufWriter<File>> {
    let mut options = OpenOptions::new();
    options.write(true);
    if overwrite {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    Ok(BufWriter::new(options.open(output.checked_path(name)?)?))
}
fn read_hashed_json<T: serde::de::DeserializeOwned>(path: &Path) -> RadResult<(T, String)> {
    let digest = workflow::file_sha256(path)?;
    let value = read_json(path)?;
    if workflow::file_sha256(path)? != digest {
        return Err(invalid("Input JSON content changed while it was read."));
    }
    Ok((value, digest))
}
fn separate_output_inputs(directory: &Path, inputs: &[&Path]) -> RadResult<()> {
    if directory.exists() {
        let root = std::fs::canonicalize(directory)?;
        for input in inputs {
            if std::fs::canonicalize(input)?.starts_with(&root) {
                return Err(invalid("Output directory must not contain command input files, including with --overwrite."));
            }
        }
    }
    Ok(())
}
fn patches(args: PatchesCommand) -> RadResult<u8> {
    let (config, config_sha256): (PatchConfig, _) = read_hashed_json(&args.config)?;
    config.validate()?;
    let (slides, cohort_sha256): (Vec<SlideInput>, _) = read_hashed_json(&args.cohort)?;
    if slides.is_empty() {
        return Err(invalid("Pathology cohort is empty."));
    }
    unique_nonempty(
        &slides
            .iter()
            .map(|s| s.metadata.slide_id.clone())
            .collect::<Vec<_>>(),
        "Global slide IDs",
    )?;
    let mut table = TileTable {
        tiles: vec![],
        features: Some(NumericBuffer {
            rows: 0,
            cols: config.features.len(),
            values: vec![],
            missing: vec![],
        }),
        feature_names: config.features.iter().map(PatchFeature::name).collect(),
        slide_status: BTreeMap::new(),
        slide_patients: BTreeMap::new(),
    };
    for slide in &slides {
        if slide.metadata.patient_id.trim().is_empty() {
            return Err(invalid(
                "Patient identity is required even for failed/zero-tile slides.",
            ));
        }
        table.slide_patients.insert(
            slide.metadata.slide_id.clone(),
            slide.metadata.patient_id.clone(),
        );
        table
            .slide_status
            .insert(slide.metadata.slide_id.clone(), "pending".into());
    }
    let mut files: Vec<PathBuf> = [
        "tiles.json",
        "qc.jsonl",
        "patch-details.jsonl",
        "summary.json",
        "manifest.json",
        "aggregation.json",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    for prefix in ["tile-features", "slide-features", "patient-features"] {
        files.extend(table_names(prefix));
    }
    separate_output_inputs(&args.output, &[&args.cohort, &args.config])?;
    let output = Output::preflight(&args.output, args.overwrite, files)?;
    output.begin()?;
    let mut qc_writer = stream_file(&output, "qc.jsonl", args.overwrite)?;
    let mut details = stream_file(&output, "patch-details.jsonl", args.overwrite)?;
    let mut results = vec![];
    let mut quality = vec![];
    let base = parent(&args.cohort);
    for slide in &slides {
        let start = table.tiles.len();
        let mut result = SlideResult {
            patient_id: slide.metadata.patient_id.clone(),
            slide_id: slide.metadata.slide_id.clone(),
            status: "complete".into(),
            tiles: 0,
            accepted_tiles: 0,
            image_sha256: None,
            texture_sha256: None,
            mask_sha256: None,
            error: None,
        };
        let processing = (|| -> RadResult<()> {
            let image_path = resolve(base, &slide.image);
            result.image_sha256 = Some(workflow::file_sha256(&image_path)?);
            let image = imaging::read_raster(&image_path)?;
            let texture_path = slide.texture_image.as_ref().map(|p| resolve(base, p));
            let texture_image = if let Some(path) = &texture_path {
                result.texture_sha256 = Some(workflow::file_sha256(path)?);
                Some(imaging::read_raster(path)?)
            } else {
                None
            };
            let mask_path = slide.mask.as_ref().map(|p| resolve(base, p));
            let mask_image = if let Some(path) = &mask_path {
                if config.mask_analysis.is_none() {
                    return Err(invalid("A supplied mask requires explicit mask_analysis kind/connectivity/classes."));
                }
                result.mask_sha256 = Some(workflow::file_sha256(path)?);
                let mask = imaging::read_raster(path)?;
                if [image.width, image.height] != [mask.width, mask.height]
                    || matches!(mask.pixels, RasterPixels::Rgb8(_))
                {
                    return Err(invalid("Mask must be explicit co-gridded Gray8/Gray16 labels matching primary image dimensions; no RGB conversion or resampling."));
                }
                Some(mask)
            } else {
                if config.mask_analysis.is_some() {
                    return Err(invalid(
                        "mask_analysis requires an explicit co-gridded mask for every slide.",
                    ));
                }
                None
            };
            if let Some(texture) = &texture_image {
                if [image.width, image.height] != [texture.width, texture.height]
                    || matches!(texture.pixels, RasterPixels::Rgb8(_))
                {
                    return Err(invalid("Explicit texture image must be co-gridded Gray8/Gray16 with matching decoded dimensions."));
                }
            }
            if config.texture.is_some()
                && texture_image.is_none()
                && matches!(image.pixels, RasterPixels::Rgb8(_))
            {
                return Err(invalid("RGB texture extraction requires an explicit co-gridded grayscale texture_image; no conversion is inferred."));
            }
            let mut gray_tiles = texture_image
                .as_ref()
                .map(|g| pixels::tile_image(g, &config.tiling, &slide.metadata))
                .transpose()?;
            let mut mask_tiles = mask_image
                .as_ref()
                .map(|m| pixels::tile_image(m, &config.tiling, &slide.metadata))
                .transpose()?;
            for patch in pixels::tile_image(&image, &config.tiling, &slide.metadata)? {
                let mut patch = patch?;
                let texture_patch = gray_tiles
                    .as_mut()
                    .map(|tiles| {
                        tiles.next().ok_or_else(|| {
                            invalid("Explicit grayscale patch stream ended before primary image.")
                        })
                    })
                    .transpose()?
                    .transpose()?;
                if texture_patch
                    .as_ref()
                    .is_some_and(|p| p.record != patch.record)
                {
                    return Err(invalid(
                        "Explicit grayscale tile geometry/identity differs from primary tile.",
                    ));
                }
                let mask_patch = mask_tiles
                    .as_mut()
                    .map(|tiles| {
                        tiles
                            .next()
                            .ok_or_else(|| invalid("Mask tile stream ended before primary image."))
                    })
                    .transpose()?
                    .transpose()?;
                if mask_patch
                    .as_ref()
                    .is_some_and(|p| p.record != patch.record)
                {
                    return Err(invalid(
                        "Mask tile geometry/identity differs from primary tile.",
                    ));
                }
                let labels = mask_patch.as_ref().map(|p| match &p.image.pixels {
                    RasterPixels::Gray8(v) => v.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(),
                    RasterPixels::Gray16(v) => v.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(),
                    RasterPixels::Rgb8(_) => unreachable!("RGB masks rejected before tiling"),
                });
                let overlapping = config
                    .tiling
                    .stride
                    .iter()
                    .zip(config.tiling.patch_size)
                    .any(|(s, p)| *s < p);
                let objects = labels
                    .as_ref()
                    .zip(config.mask_analysis.as_ref())
                    .map(|(labels, c)| {
                        pixels::object_features_for_tile(
                            labels,
                            &patch.record,
                            c.kind,
                            c.connectivity,
                            overlapping,
                        )
                    })
                    .transpose()?;
                let annotation = labels
                    .as_ref()
                    .zip(config.mask_analysis.as_ref())
                    .map(|(labels, c)| {
                        pixels::annotation_coverage(
                            patch.record.patch_width,
                            patch.record.patch_height,
                            labels,
                            patch.record.valid_pixels,
                            &c.classes,
                        )
                    })
                    .transpose()?;
                let qc = config
                    .qc
                    .as_ref()
                    .map(|c| pixels::qc_patch(&patch.image, patch.record.valid_pixels, c))
                    .transpose()?;
                if let Some(qc) = &qc {
                    qc.apply_to_record(&mut patch.record)?;
                }
                let tissue = if config.tissue_only {
                    qc.as_ref().map(|q| q.tissue_mask.as_slice())
                } else {
                    None
                };
                let color = config
                    .color
                    .as_ref()
                    .map(|c| {
                        pixels::color_features(&patch.image, patch.record.valid_pixels, c, tissue)
                    })
                    .transpose()?;
                let texture = config
                    .texture
                    .as_ref()
                    .map(|c| {
                        let source = texture_patch.as_ref().unwrap_or(&patch);
                        pixels::texture_features(
                            &source.image,
                            patch.record.valid_pixels,
                            tissue,
                            c,
                        )
                    })
                    .transpose();
                let texture = match texture {
                    Ok(v) => v,
                    Err(error) => {
                        patch.record.exclusion_reasons.push("texture_error".into());
                        quality.push(QualityRecord {
                            code: "texture_error".into(),
                            message: error.to_string(),
                            severity: "error".into(),
                            sample_id: Some(patch.record.tile_id.clone()),
                        });
                        None
                    }
                };
                let data = table.features.as_mut().expect("validated schema");
                for feature in &config.features {
                    let value = feature.value(
                        qc.as_ref(),
                        color.as_ref(),
                        texture.as_ref(),
                        objects.as_ref(),
                        annotation.as_ref(),
                    );
                    data.values.push(value.unwrap_or(0.));
                    data.missing.push(value.is_none());
                    if value.is_none() {
                        quality.push(QualityRecord {
                            code: "undefined_patch_feature".into(),
                            message: format!(
                                "{} is missing/undefined; no zero was substituted.",
                                feature.name()
                            ),
                            severity: "warning".into(),
                            sample_id: Some(patch.record.tile_id.clone()),
                        });
                    }
                }
                data.rows += 1;
                json_line(
                    &mut qc_writer,
                    &json!({"tile_id":patch.record.tile_id,"patient_id":patch.record.patient_id,"slide_id":patch.record.slide_id,"scores":qc.as_ref().map(|q|&q.scores),"valid_pixels":patch.record.valid_pixels,"valid_pixel_count":qc.as_ref().map(|q|q.valid_pixels),"border_pixels":qc.as_ref().map(|q|q.border_pixels),"laplacian_pixels":qc.as_ref().map(|q|q.laplacian_pixels),"qc_accepted":qc.as_ref().map(|q|q.accepted),"stage_exclusion_reasons":patch.record.exclusion_reasons,"final_status_source":"tiles.json exclusion_reasons and slide_status"}),
                )?;
                json_line(
                    &mut details,
                    &json!({"tile_id":patch.record.tile_id,"colors":color,"objects":objects,"annotation":annotation,"texture_status":if texture.is_some(){"computed"}else{"not_available"},"note":"Hue mean is circular metadata, excluded from linear feature schema; objects are patch_local, never summed as a whole-slide total; no patch pixels retained."}),
                )?;
                table.tiles.push(patch.record);
            }
            if workflow::file_sha256(&image_path)? != result.image_sha256.clone().unwrap() {
                return Err(invalid("Source image content changed during processing."));
            }
            if let Some(path) = texture_path {
                if workflow::file_sha256(path)? != result.texture_sha256.clone().unwrap() {
                    return Err(invalid("Texture image content changed during processing."));
                }
            }
            if let Some(path) = mask_path {
                if workflow::file_sha256(path)? != result.mask_sha256.clone().unwrap() {
                    return Err(invalid("Mask content changed during processing."));
                }
            }
            Ok(())
        })();
        result.tiles = table.tiles.len() - start;
        result.accepted_tiles = table.tiles[start..]
            .iter()
            .filter(|t| t.exclusion_reasons.is_empty())
            .count();
        if let Err(error) = processing {
            result.status = "failed".into();
            result.error = Some(error.to_string());
            // Partially computed rows cannot silently count as accepted input.
            for tile in &mut table.tiles[start..] {
                tile.exclusion_reasons
                    .push("slide_processing_failed".into());
            }
            let data = table.features.as_mut().expect("validated schema");
            for index in start * data.cols..data.values.len() {
                data.values[index] = 0.;
                data.missing[index] = true;
            }
            quality.push(QualityRecord {
                code: "slide_processing_failed".into(),
                message: error.to_string(),
                severity: "error".into(),
                sample_id: None,
            });
            result.accepted_tiles = 0;
        } else if result.tiles == 0 {
            result.status = "no_tiles".into();
        } else if result.accepted_tiles == 0 {
            result.status = "all_excluded".into();
        }
        table
            .slide_status
            .insert(result.slide_id.clone(), result.status.clone());
        results.push(result);
    }
    qc_writer.flush()?;
    details.flush()?;
    table.validate()?;
    let tile_features = scalar_table(&table, &config, quality);
    tile_features.validate()?;
    let report = aggregate::aggregate(&table, &config.aggregation)?;
    let (slide_features, patient_features) = report.to_feature_tables()?;
    output.json("tiles.json", &table)?;
    output.json("aggregation.json", &report)?;
    write_table(&output, "tile-features", &tile_features, args.overwrite)?;
    write_table(&output, "slide-features", &slide_features, args.overwrite)?;
    write_table(
        &output,
        "patient-features",
        &patient_features,
        args.overwrite,
    )?;
    let partial = results.iter().any(|r| r.status != "complete");
    let summary = json!({"status":if partial{"partial_or_empty"}else{"complete"},"slides":results,"tiles":table.tiles.len(),"accepted_tiles":table.tiles.iter().filter(|t|t.exclusion_reasons.is_empty()).count(),"runtime":"rust","python_required":false});
    output.json("summary.json", &summary)?;
    output.json("manifest.json",&json!({"schema":"radiomics.pathology.patches.v1","config":config,"cohort":slides,"input_cohort_sha256":cohort_sha256,"input_config_sha256":config_sha256,"implementation_sha256":workflow::implementation_sha256(),"coordinate_space":"level0_pixel_edges","mpp_policy":"explicit optional level0 micrometers per pixel; no DPI/magnification inference","objects":"No whole-slide instance total is inferred from overlapping patches.","source_images_decoded_at_once":"one primary, optional explicit co-gridded gray image and label mask; patches streamed"}))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(if partial { 2 } else { 0 })
}

fn aggregate_command(args: AggregateCommand) -> RadResult<u8> {
    let (mut table, table_sha256): (TileTable, _) = read_hashed_json(&args.table)?;
    table.validate()?;
    let mut embedding_sha256 = None;
    let mut embedding_manifest_sha256 = None;
    if let Some(path) = &args.embeddings {
        let before = workflow::file_sha256(path)?;
        let manifest_path = args
            .embedding_manifest
            .clone()
            .unwrap_or_else(|| table_io::feature_manifest_path(path));
        let manifest_before = manifest_path
            .exists()
            .then(|| workflow::file_sha256(&manifest_path))
            .transpose()?;
        let features = table_io::read_feature_table(path, args.embedding_manifest.as_deref())?;
        let ids: Vec<_> = features
            .samples
            .iter()
            .map(|s| s.sample_id.clone())
            .collect();
        let names: Vec<_> = features.features.iter().map(|s| s.name.clone()).collect();
        table = aggregate::align_embeddings(&table, &ids, &features.data, &names)?;
        let metadata: BTreeMap<_, _> = table
            .tiles
            .iter()
            .map(|t| (t.tile_id.as_str(), t))
            .collect();
        for sample in &features.samples {
            let tile = metadata[sample.sample_id.as_str()];
            if sample.patient_id != tile.patient_id
                || sample
                    .slide_id
                    .as_ref()
                    .is_some_and(|s| s != &tile.slide_id)
            {
                return Err(invalid("Embedding FeatureTable patient/slide metadata conflicts with TileTable identity."));
            }
        }
        if workflow::file_sha256(path)? != before
            || manifest_path
                .exists()
                .then(|| workflow::file_sha256(&manifest_path))
                .transpose()?
                != manifest_before
        {
            return Err(invalid(
                "Embedding table/manifest changed while it was read.",
            ));
        }
        embedding_sha256 = Some(before);
        embedding_manifest_sha256 = manifest_before;
    }
    let (config, config_sha256): (aggregate::AggregateConfig, _) = read_hashed_json(&args.config)?;
    config.validate()?;
    let report = aggregate::aggregate(&table, &config)?;
    let (slides, patients) = report.to_feature_tables()?;
    let mut files: Vec<PathBuf> = ["aggregation.json", "manifest.json"]
        .into_iter()
        .map(Into::into)
        .collect();
    files.extend(table_names("slide-features"));
    files.extend(table_names("patient-features"));
    let mut inputs = vec![args.table.as_path(), args.config.as_path()];
    if let Some(path) = &args.embeddings {
        inputs.push(path);
    }
    if let Some(path) = &args.embedding_manifest {
        inputs.push(path);
    }
    separate_output_inputs(&args.output, &inputs)?;
    let output = Output::preflight(&args.output, args.overwrite, files)?;
    output.begin()?;
    output.json("aggregation.json", &report)?;
    write_table(&output, "slide-features", &slides, args.overwrite)?;
    write_table(&output, "patient-features", &patients, args.overwrite)?;
    output.json("manifest.json",&json!({"schema":"radiomics.pathology.aggregate.v1","config":config,"input_sha256":table_sha256,"input_config_sha256":config_sha256,"embedding_sha256":embedding_sha256,"embedding_manifest_sha256":embedding_manifest_sha256,"implementation_sha256":workflow::implementation_sha256(),"coordinate_space":"level0_pixel_edges","object_count_scope":"patch_local_only"}))?;
    let empty = report.slides.iter().any(|s| s.accepted_tiles == 0);
    println!(
        "{}",
        json!({"status":if empty{"partial_or_empty"}else{"complete"},"slides":report.slides.len(),"patients":report.patients.len(),"runtime":"rust"})
    );
    Ok(if empty { 2 } else { 0 })
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OverlayInput {
    schema_version: u32,
    canvas: overlay::Canvas,
    block_size: [usize; 2],
    clip: bool,
    source: OverlaySource,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum OverlaySource {
    Scores {
        tile_ids: Vec<String>,
        scores: Vec<f64>,
        prediction_unit: PredictionUnit,
    },
    Feature {
        slide_id: String,
        feature: String,
    },
    Predictions {
        path: PathBuf,
        slide_id: String,
        class: Option<String>,
    },
}
fn overlay_command(args: OverlayCommand) -> RadResult<u8> {
    let (table, table_sha256): (TileTable, _) = read_hashed_json(&args.table)?;
    table.validate()?;
    let (config, config_sha256): (OverlayInput, _) = read_hashed_json(&args.config)?;
    if config.schema_version != 1 {
        return Err(schema("Overlay schema_version must be 1."));
    }
    let mut predictions_sha256 = None;
    let blocks = match &config.source {
        OverlaySource::Scores {
            tile_ids,
            scores,
            prediction_unit,
        } => overlay::raster_blocks(
            &table,
            tile_ids,
            scores,
            &config.canvas,
            config.block_size,
            &overlay::OverlayConfig {
                prediction_unit: *prediction_unit,
                clip: config.clip,
            },
        )?,
        OverlaySource::Feature { slide_id, feature } => overlay::rasterize_features(
            &table,
            slide_id,
            feature,
            &config.canvas,
            config.block_size,
            config.clip,
        )?,
        OverlaySource::Predictions {
            path,
            slide_id,
            class,
        } => {
            let path = resolve(parent(&args.config), path);
            let (predictions, digest): (PredictionTable, _) = read_hashed_json(&path)?;
            predictions_sha256 = Some(digest);
            overlay::rasterize_predictions(
                &table,
                &predictions,
                slide_id,
                class.as_deref(),
                &config.canvas,
                config.block_size,
                config.clip,
            )?
        }
    };
    let mut inputs = vec![args.table.as_path(), args.config.as_path()];
    let predictions_path = match &config.source {
        OverlaySource::Predictions { path, .. } => Some(resolve(parent(&args.config), path)),
        _ => None,
    };
    if let Some(path) = &predictions_path {
        inputs.push(path);
    }
    separate_output_inputs(&args.output, &inputs)?;
    if config.block_size.contains(&0) {
        return Err(invalid("Overlay block size must be positive."));
    }
    let output = Output::preflight(
        &args.output,
        args.overwrite,
        ["manifest.json".into(), "blocks.jsonl".into()],
    )?;
    output.begin()?;
    let mut index = stream_file(&output, "blocks.jsonl", args.overwrite)?;
    let root = std::fs::canonicalize(&args.output)?;
    let mut block_count = 0usize;
    for block in blocks {
        let block = block?;
        let name = format!("block-{:012}-{:012}.json", block.y, block.x);
        // Generated basename only; do not allocate a list of every canvas block.
        if std::fs::canonicalize(&args.output)? != root {
            return Err(invalid(
                "Overlay output directory changed during execution.",
            ));
        }
        let path = args.output.join(&name);
        if path.exists() {
            let metadata = std::fs::symlink_metadata(&path)?;
            if !args.overwrite || !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(invalid("Overlay block exists without overwrite authorization or is not an ordinary file."));
            }
        }
        let mut options = OpenOptions::new();
        options.write(true);
        if args.overwrite {
            options.create(true).truncate(true);
        } else {
            options.create_new(true);
        }
        let mut writer = BufWriter::new(options.open(path)?);
        serde_json::to_writer(&mut writer, &block)?;
        writer.flush()?;
        json_line(
            &mut index,
            &json!({"path":name,"x":block.x,"y":block.y,"width":block.width,"height":block.height,"origin0":block.origin0,"pixel_size0":block.pixel_size0}),
        )?;
        block_count = block_count
            .checked_add(1)
            .ok_or_else(|| invalid("Overlay block count overflows usize."))?;
    }
    index.flush()?;
    output.json("manifest.json",&json!({"schema":"radiomics.pathology.overlay.v1","config":config,"canvas":config.canvas,"block_index":"blocks.jsonl","block_count":block_count,"input_tile_table_sha256":table_sha256,"input_config_sha256":config_sha256,"input_predictions_sha256":predictions_sha256,"implementation_sha256":workflow::implementation_sha256(),"coordinate_space":"level0_pixel_edges","sampling":"global-index pixel centers; half-open rectangles; overlap mean","uncovered":"NaN values, zero coverage and zero alpha","runtime":"rust"}))?;
    println!(
        "{}",
        json!({"status":"complete","blocks":block_count,"runtime":"rust"})
    );
    Ok(0)
}
