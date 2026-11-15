//! Thin owned-buffer/iterator boundary to the common pathology core.
use ov_radiomics_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

fn error(e: core::errors::RadiomicsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn encode<T: Serialize>(value: &T) -> core::errors::RadResult<String> {
    Ok(serde_json::to_string(value)?)
}
#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    ReadImage {
        path: String,
    },
    ValidateImage {
        image: core::contracts::RasterImage,
    },
    Qc {
        patch: core::pathology_pixels::TilePatch,
        config: core::pathology_pixels::QcConfig,
    },
    Color {
        patch: core::pathology_pixels::TilePatch,
        config: core::pathology_pixels::ColorConfig,
        tissue_mask: Option<Vec<bool>>,
    },
    Texture {
        patch: core::pathology_pixels::TilePatch,
        config: core::contracts::ExtractionConfig,
        tissue_mask: Option<Vec<bool>>,
    },
    Annotation {
        width: usize,
        height: usize,
        labels: Vec<u32>,
        rect: [usize; 4],
        classes: Vec<u32>,
    },
    Objects {
        width: usize,
        height: usize,
        labels: Vec<u32>,
        rect: [usize; 4],
        kind: core::pathology_pixels::MaskKind,
        connectivity: core::imaging::Connectivity,
        patch_mpp_um: Option<[f64; 2]>,
        overlapping_tiles: bool,
    },
    ObjectsForTile {
        labels: Vec<u32>,
        record: core::contracts::TileRecord,
        kind: core::pathology_pixels::MaskKind,
        connectivity: core::imaging::Connectivity,
        overlapping_tiles: bool,
    },
    Aggregate {
        table: core::contracts::TileTable,
        config: core::pathology_aggregate::AggregateConfig,
    },
}
pub fn execute(payload: &str) -> core::errors::RadResult<String> {
    match serde_json::from_str::<Request>(payload)? {
        Request::ReadImage { path } => encode(&core::imaging::read_raster(path)?),
        Request::ValidateImage { image } => {
            image.validate()?;
            encode(&image)
        }
        Request::Qc { mut patch, config } => {
            patch.validate()?;
            let qc =
                core::pathology_pixels::qc_patch(&patch.image, patch.record.valid_pixels, &config)?;
            qc.apply_to_record(&mut patch.record)?;
            encode(&serde_json::json!({"record":patch.record,"qc":qc}))
        }
        Request::Color {
            patch,
            config,
            tissue_mask,
        } => {
            patch.validate()?;
            encode(&core::pathology_pixels::color_features(
                &patch.image,
                patch.record.valid_pixels,
                &config,
                tissue_mask.as_deref(),
            )?)
        }
        Request::Texture {
            patch,
            config,
            tissue_mask,
        } => {
            patch.validate()?;
            let values = core::pathology_pixels::texture_features(
                &patch.image,
                patch.record.valid_pixels,
                tissue_mask.as_deref(),
                &config,
            )?;
            // Undefined scalar features keep explicit states, not JSON null/zero.
            #[derive(Serialize)]
            struct Scalar(#[serde(with = "core::statistics::float_state")] f64);
            encode(
                &values
                    .into_iter()
                    .map(|(name, value)| (name, Scalar(value)))
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
        }
        Request::Annotation {
            width,
            height,
            labels,
            rect,
            classes,
        } => encode(&core::pathology_pixels::annotation_coverage(
            width, height, &labels, rect, &classes,
        )?),
        Request::Objects {
            width,
            height,
            labels,
            rect,
            kind,
            connectivity,
            patch_mpp_um,
            overlapping_tiles,
        } => encode(&core::pathology_pixels::object_features(
            width,
            height,
            &labels,
            rect,
            kind,
            connectivity,
            patch_mpp_um,
            overlapping_tiles,
        )?),
        Request::ObjectsForTile {
            labels,
            record,
            kind,
            connectivity,
            overlapping_tiles,
        } => encode(&core::pathology_pixels::object_features_for_tile(
            &labels,
            &record,
            kind,
            connectivity,
            overlapping_tiles,
        )?),
        Request::Aggregate { table, config } => {
            let report = core::pathology_aggregate::aggregate(&table, &config)?;
            let (slides, patients) = report.to_feature_tables()?;
            encode(
                &serde_json::json!({"report":report,"slide_features":slides,"patient_features":patients}),
            )
        }
    }
}

#[pyclass]
pub struct PatchIterator {
    iterator: core::pathology_pixels::TileIterator<'static>,
}
#[pymethods]
impl PatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| {
            self.iterator
                .next()
                .map(|patch| encode(&patch?))
                .transpose()
        })
        .map_err(error)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tiling {
    image: core::contracts::RasterImage,
    spec: core::pathology_pixels::TileSpec,
    metadata: core::pathology_pixels::SlideMetadata,
}
pub fn tiles(payload: &str) -> core::errors::RadResult<PatchIterator> {
    let input: Tiling = serde_json::from_str(payload)?;
    Ok(PatchIterator {
        iterator: core::pathology_pixels::tile_image_owned(
            input.image,
            &input.spec,
            &input.metadata,
        )?,
    })
}

#[pyclass]
pub struct OverlayIterator {
    iterator: core::overlay::OverlayBlocks,
}
#[pymethods]
impl OverlayIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| {
            self.iterator
                .next()
                .map(|block| encode(&block?))
                .transpose()
        })
        .map_err(error)
    }
}
#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Overlay {
    Scores {
        table: core::contracts::TileTable,
        ids: Vec<String>,
        scores: Vec<f64>,
        canvas: core::overlay::Canvas,
        block_size: [usize; 2],
        config: core::overlay::OverlayConfig,
    },
    Features {
        table: core::contracts::TileTable,
        slide_id: String,
        feature: String,
        canvas: core::overlay::Canvas,
        block_size: [usize; 2],
        clip: bool,
    },
    Predictions {
        table: core::contracts::TileTable,
        predictions: core::contracts::PredictionTable,
        slide_id: String,
        class: Option<String>,
        canvas: core::overlay::Canvas,
        block_size: [usize; 2],
        clip: bool,
    },
}
pub fn overlay(payload: &str) -> core::errors::RadResult<OverlayIterator> {
    let iterator = match serde_json::from_str::<Overlay>(payload)? {
        Overlay::Scores {
            table,
            ids,
            scores,
            canvas,
            block_size,
            config,
        } => core::overlay::raster_blocks(&table, &ids, &scores, &canvas, block_size, &config)?,
        Overlay::Features {
            table,
            slide_id,
            feature,
            canvas,
            block_size,
            clip,
        } => core::overlay::rasterize_features(
            &table, &slide_id, &feature, &canvas, block_size, clip,
        )?,
        Overlay::Predictions {
            table,
            predictions,
            slide_id,
            class,
            canvas,
            block_size,
            clip,
        } => core::overlay::rasterize_predictions(
            &table,
            &predictions,
            &slide_id,
            class.as_deref(),
            &canvas,
            block_size,
            clip,
        )?,
    };
    Ok(OverlayIterator { iterator })
}
