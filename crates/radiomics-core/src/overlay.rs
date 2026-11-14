//! Lazy tile-level overlays with exact global-index sampling and explicit missing coverage.
use crate::contracts::{unique_nonempty, PredictionTable, PredictionUnit, Task, TileTable};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::{pathology_aggregate::valid_bounds, statistics};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub origin0: [f64; 2],
    pub pixel_size0: [f64; 2],
}
impl Canvas {
    pub fn validate(&self) -> RadResult<()> {
        if self.width == 0
            || self.height == 0
            || self.width.checked_mul(self.height).is_none()
            || self.origin0.iter().any(|v| !v.is_finite() || *v < 0.)
            || self.pixel_size0.iter().any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(invalid(
                "Canvas needs positive dimensions/scales and finite nonnegative origins.",
            ));
        }
        for (j, &size) in [self.width, self.height].iter().enumerate() {
            let end = self.origin0[j] + size as f64 * self.pixel_size0[j];
            if !end.is_finite() || end <= self.origin0[j] {
                return Err(invalid("Canvas coordinates are not representable."));
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayConfig {
    pub prediction_unit: PredictionUnit,
    pub clip: bool,
}
impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            prediction_unit: PredictionUnit::Tile,
            clip: false,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayBlock {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub origin0: [f64; 2],
    pub pixel_size0: [f64; 2],
    #[serde(with = "crate::statistics::float_vec_state")]
    pub values: Vec<f64>,
    pub coverage: Vec<u64>,
    pub missing: Vec<bool>,
    pub alpha: Vec<u8>,
}
struct Entry {
    bounds: [f64; 4],
    score: f64,
}
pub struct OverlayBlocks {
    canvas: Canvas,
    block_size: [usize; 2],
    entries: Vec<Entry>,
    x: usize,
    y: usize,
}

/// Validate IDs/metadata once, then allocate only one requested block per iterator item.
pub fn raster_blocks(
    table: &TileTable,
    score_ids: &[String],
    scores: &[f64],
    canvas: &Canvas,
    block_size: [usize; 2],
    config: &OverlayConfig,
) -> RadResult<OverlayBlocks> {
    table.validate()?;
    canvas.validate()?;
    unique_nonempty(score_ids, "Tile score IDs")?;
    if config.prediction_unit != PredictionUnit::Tile {
        return Err(invalid("Local overlays require actual tile-level scores; patient/slide probabilities cannot be broadcast."));
    }
    if block_size.contains(&0) || block_size[0].checked_mul(block_size[1]).is_none() {
        return Err(invalid(
            "Block dimensions must be positive and representable.",
        ));
    }
    if table
        .tiles
        .iter()
        .map(|t| &t.slide_id)
        .collect::<BTreeSet<_>>()
        .len()
        > 1
    {
        return Err(invalid("An overlay contains one slide only."));
    }
    let expected: BTreeSet<_> = table
        .tiles
        .iter()
        .filter(|t| t.exclusion_reasons.is_empty())
        .map(|t| t.tile_id.as_str())
        .collect();
    if score_ids.len() != scores.len()
        || score_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != expected
        || scores.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "Finite scores must match exactly the accepted tile IDs.",
        ));
    }
    let lookup: BTreeMap<_, _> = score_ids.iter().zip(scores).collect();
    let mut ordered: Vec<_> = table.tiles.iter().collect();
    ordered.sort_by(|a, b| a.tile_id.cmp(&b.tile_id));
    let mut entries = vec![];
    for tile in ordered {
        let bounds = valid_bounds(tile)?;
        if !config.clip
            && (bounds[0] < canvas.origin0[0]
                || bounds[1] < canvas.origin0[1]
                || bounds[2] > canvas.origin0[0] + canvas.width as f64 * canvas.pixel_size0[0]
                || bounds[3] > canvas.origin0[1] + canvas.height as f64 * canvas.pixel_size0[1])
        {
            return Err(invalid(
                "Tile valid rectangle lies outside canvas; request explicit clipping.",
            ));
        }
        if tile.exclusion_reasons.is_empty() {
            entries.push(Entry {
                bounds,
                score: *lookup[&tile.tile_id],
            });
        }
    }
    Ok(OverlayBlocks {
        canvas: canvas.clone(),
        block_size,
        entries,
        x: 0,
        y: 0,
    })
}
impl Iterator for OverlayBlocks {
    type Item = RadResult<OverlayBlock>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.canvas.height {
            return None;
        }
        let x = self.x;
        let y = self.y;
        let width = self.block_size[0].min(self.canvas.width - x);
        let height = self.block_size[1].min(self.canvas.height - y);
        self.x = self
            .x
            .checked_add(self.block_size[0])
            .unwrap_or(self.canvas.width);
        if self.x >= self.canvas.width {
            self.x = 0;
            self.y = self
                .y
                .checked_add(self.block_size[1])
                .unwrap_or(self.canvas.height);
        }
        Some((|| {
            let n = width
                .checked_mul(height)
                .ok_or_else(|| invalid("Block pixel count overflows."))?;
            let mut values = Vec::new();
            let mut coverage = Vec::new();
            let mut missing = Vec::new();
            let mut alpha = Vec::new();
            values
                .try_reserve_exact(n)
                .map_err(|e| invalid(format!("Block allocation failed: {e}")))?;
            coverage
                .try_reserve_exact(n)
                .map_err(|e| invalid(format!("Block allocation failed: {e}")))?;
            missing
                .try_reserve_exact(n)
                .map_err(|e| invalid(format!("Block allocation failed: {e}")))?;
            alpha
                .try_reserve_exact(n)
                .map_err(|e| invalid(format!("Block allocation failed: {e}")))?;
            let mut contributors = vec![];
            for dy in 0..height {
                for dx in 0..width {
                    // Do not derive centers from block origin: reassociation changes half-open topology.
                    let px = self.canvas.origin0[0]
                        + ((x + dx) as f64 + 0.5) * self.canvas.pixel_size0[0];
                    let py = self.canvas.origin0[1]
                        + ((y + dy) as f64 + 0.5) * self.canvas.pixel_size0[1];
                    contributors.clear();
                    for entry in &self.entries {
                        if entry.bounds[0] <= px
                            && px < entry.bounds[2]
                            && entry.bounds[1] <= py
                            && py < entry.bounds[3]
                        {
                            contributors.push(entry.score);
                        }
                    }
                    let count = u64::try_from(contributors.len())
                        .map_err(|_| invalid("Coverage count overflows u64."))?;
                    let empty = count == 0;
                    values.push(if empty {
                        f64::NAN
                    } else {
                        statistics::population_mean_variance(&contributors)?.0
                    });
                    coverage.push(count);
                    missing.push(empty);
                    alpha.push(if empty { 0 } else { 255 });
                }
            }
            Ok(OverlayBlock {
                x,
                y,
                width,
                height,
                origin0: [
                    self.canvas.origin0[0] + x as f64 * self.canvas.pixel_size0[0],
                    self.canvas.origin0[1] + y as f64 * self.canvas.pixel_size0[1],
                ],
                pixel_size0: self.canvas.pixel_size0,
                values,
                coverage,
                missing,
                alpha,
            })
        })())
    }
}
pub fn rasterize(
    table: &TileTable,
    score_ids: &[String],
    scores: &[f64],
    canvas: &Canvas,
    config: &OverlayConfig,
) -> RadResult<OverlayBlock> {
    raster_blocks(
        table,
        score_ids,
        scores,
        canvas,
        [canvas.width, canvas.height],
        config,
    )?
    .next()
    .ok_or_else(|| invalid("Empty canvas."))?
}
fn slide_metadata_only(table: &TileTable, slide_id: &str) -> RadResult<TileTable> {
    table.validate()?;
    let parent = table
        .slide_patients
        .get(slide_id)
        .ok_or_else(|| invalid("Unknown overlay slide ID."))?;
    Ok(TileTable {
        tiles: table
            .tiles
            .iter()
            .filter(|t| t.slide_id == slide_id)
            .cloned()
            .collect(),
        features: None,
        feature_names: vec![],
        slide_status: [(slide_id.into(), table.slide_status[slide_id].clone())]
            .into_iter()
            .collect(),
        slide_patients: [(slide_id.into(), parent.clone())].into_iter().collect(),
    })
}
pub fn rasterize_features(
    table: &TileTable,
    slide_id: &str,
    feature: &str,
    canvas: &Canvas,
    block_size: [usize; 2],
    clip: bool,
) -> RadResult<OverlayBlocks> {
    let subset = slide_metadata_only(table, slide_id)?;
    let data = table
        .features
        .as_ref()
        .ok_or_else(|| invalid("Tile features are missing."))?;
    let column = table
        .feature_names
        .iter()
        .position(|name| name == feature)
        .ok_or_else(|| invalid("Unknown tile feature."))?;
    let mut ids = vec![];
    let mut scores = vec![];
    for (i, tile) in table
        .tiles
        .iter()
        .enumerate()
        .filter(|(_, t)| t.slide_id == slide_id && t.exclusion_reasons.is_empty())
    {
        let position = i * data.cols + column;
        if data.missing[position] {
            return Err(invalid(
                "An accepted tile feature is missing; supply explicit finite tile scores.",
            ));
        }
        ids.push(tile.tile_id.clone());
        scores.push(data.values[position]);
    }
    raster_blocks(
        &subset,
        &ids,
        &scores,
        canvas,
        block_size,
        &OverlayConfig {
            prediction_unit: PredictionUnit::Tile,
            clip,
        },
    )
}
pub fn rasterize_predictions(
    table: &TileTable,
    predictions: &PredictionTable,
    slide_id: &str,
    class: Option<&str>,
    canvas: &Canvas,
    block_size: [usize; 2],
    clip: bool,
) -> RadResult<OverlayBlocks> {
    predictions.validate()?;
    if predictions.unit != PredictionUnit::Tile {
        return Err(invalid(
            "Local probability maps require actual tile-level predictions.",
        ));
    }
    let subset = slide_metadata_only(table, slide_id)?;
    let all_tiles: BTreeMap<_, _> = table
        .tiles
        .iter()
        .filter(|t| t.exclusion_reasons.is_empty())
        .map(|t| (&t.tile_id, t))
        .collect();
    for row in &predictions.rows {
        let tile = all_tiles
            .get(row.tile_id.as_ref().unwrap())
            .ok_or_else(|| {
                invalid("Prediction contains a tile absent from the accepted TileTable population.")
            })?;
        if row.slide_id.as_deref() != Some(tile.slide_id.as_str())
            || row.patient_id != tile.patient_id
        {
            return Err(invalid(
                "Prediction tile/slide/patient hierarchy differs from TileTable.",
            ));
        }
    }
    let class_column = if predictions.task == Task::Regression {
        if class.is_some() {
            return Err(invalid("Regression tile values cannot select a class."));
        }
        None
    } else {
        let name =
            class.ok_or_else(|| invalid("Select the explicit class probability to rasterize."))?;
        Some(
            predictions
                .classes
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| invalid("Unknown prediction class."))?,
        )
    };
    let metadata: BTreeMap<_, _> = subset
        .tiles
        .iter()
        .filter(|t| t.exclusion_reasons.is_empty())
        .map(|t| (&t.tile_id, t))
        .collect();
    let mut ids = vec![];
    let mut scores = vec![];
    for row in predictions
        .rows
        .iter()
        .filter(|r| r.slide_id.as_deref() == Some(slide_id))
    {
        let id = row.tile_id.as_ref().unwrap();
        let tile = metadata
            .get(id)
            .ok_or_else(|| invalid("Tile prediction ID is absent or excluded."))?;
        if row.patient_id != tile.patient_id {
            return Err(invalid(
                "Tile prediction patient differs from tile hierarchy.",
            ));
        }
        ids.push(id.clone());
        scores.push(class_column.map_or_else(|| row.value.unwrap(), |j| row.probabilities[j]));
    }
    raster_blocks(
        &subset,
        &ids,
        &scores,
        canvas,
        block_size,
        &OverlayConfig {
            prediction_unit: PredictionUnit::Tile,
            clip,
        },
    )
}
