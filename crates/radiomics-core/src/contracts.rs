//! Version-one native data contract. Buffers use x-fastest image and row-major table order.
//! Public contracts contain no third-party matrix or image types.
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: &str) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}

pub fn unique_nonempty(values: &[String], label: &str) -> RadResult<()> {
    if values.iter().any(|v| v.trim().is_empty())
        || values.iter().collect::<BTreeSet<_>>().len() != values.len()
    {
        return Err(invalid(&format!("{label} must be nonempty and unique.")));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityRecord {
    pub code: String,
    pub message: String,
    pub severity: String,
    pub sample_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialUnit {
    Unknown,
    Meter,
    Millimeter,
    Micron,
}
impl SpatialUnit {
    pub fn millimeter_factor(self) -> RadResult<f64> {
        match self {
            Self::Meter => Ok(1000.0),
            Self::Millimeter => Ok(1.0),
            Self::Micron => Ok(0.001),
            Self::Unknown => Err(RadiomicsError::new(
                ErrorCode::Geometry,
                "Unknown spatial units; millimeter geometry is unavailable.",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    pub shape: [usize; 3],
    pub ndim: u8,
    /// Voxel-center index to world coordinate, in the declared native spatial unit.
    pub affine: [[f64; 4]; 4],
    pub unit: SpatialUnit,
    pub diagnostics: Vec<QualityRecord>,
}
impl Geometry {
    pub fn voxel_count(&self) -> RadResult<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d).filter(|_| d > 0))
            .ok_or_else(|| invalid("Image dimensions are empty or overflow usize."))
    }
    pub fn validate(&self) -> RadResult<()> {
        self.voxel_count()?;
        if !matches!(self.ndim, 2 | 3) || (self.ndim == 2 && self.shape[2] != 1) {
            return Err(invalid("Scalar images must be explicitly 2D or 3D."));
        }
        if self.affine.iter().flatten().any(|v| !v.is_finite())
            || self.affine[3] != [0., 0., 0., 1.]
        {
            return Err(RadiomicsError::new(
                ErrorCode::Geometry,
                "Affine must be finite and homogeneous.",
            ));
        }
        let a = &self.affine;
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if !det.is_finite() || det == 0.0 {
            return Err(RadiomicsError::new(
                ErrorCode::Geometry,
                "Affine is singular.",
            ));
        }
        Ok(())
    }
    pub fn spacing_native(&self) -> [f64; 3] {
        std::array::from_fn(|j| {
            (0..3)
                .map(|i| self.affine[i][j].powi(2))
                .sum::<f64>()
                .sqrt()
        })
    }
    pub fn spacing_mm(&self) -> RadResult<[f64; 3]> {
        self.validate()?;
        let scale = self.unit.millimeter_factor()?;
        Ok(self.spacing_native().map(|s| s * scale))
    }
    pub fn world(&self, index: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|i| {
            self.affine[i][3] + (0..3).map(|j| self.affine[i][j] * index[j]).sum::<f64>()
        })
    }
    pub fn require_orthogonal(&self) -> RadResult<()> {
        self.validate()?;
        let s = self.spacing_native();
        for i in 0..3 {
            for j in i + 1..3 {
                let dot: f64 = (0..3).map(|k| self.affine[k][i] * self.affine[k][j]).sum();
                if dot.abs() > 1e-10 * s[i] * s[j] {
                    return Err(RadiomicsError::new(ErrorCode::Geometry, "Sheared affine: explicitly resample to an orthogonal grid before distance operations."));
                }
            }
        }
        Ok(())
    }
    pub fn aligned_with(&self, other: &Self) -> RadResult<()> {
        self.validate()?;
        other.validate()?;
        if self.shape != other.shape || self.ndim != other.ndim || self.unit != other.unit {
            return Err(RadiomicsError::new(
                ErrorCode::Alignment,
                "Image and mask dimensions or spatial units differ.",
            ));
        }
        let factor = match self.unit {
            SpatialUnit::Unknown => 1.0,
            unit => unit.millimeter_factor()?,
        };
        // Bound discrepancy over all voxel-center corners, rather than just affine entries.
        for corner in 0..8 {
            let index = std::array::from_fn(|j| {
                if corner & (1 << j) == 0 {
                    0.0
                } else {
                    (self.shape[j] - 1) as f64
                }
            });
            let x = self.world(index);
            let y = other.world(index);
            if (0..3).any(|j| (x[j] - y[j]).abs() * factor > 1e-6) {
                return Err(RadiomicsError::new(
                    ErrorCode::Alignment,
                    "Image and mask world-coordinate grids differ.",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageVolume {
    pub geometry: Geometry,
    pub values: Vec<f64>,
}
impl ImageVolume {
    pub fn validate(&self) -> RadResult<()> {
        self.geometry.validate()?;
        if self.values.len() != self.geometry.voxel_count()? {
            return Err(invalid("Image buffer length differs from geometry."));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskVolume {
    pub geometry: Geometry,
    pub labels: Vec<u32>,
}
impl MaskVolume {
    pub fn validate(&self) -> RadResult<()> {
        self.geometry.validate()?;
        if self.labels.len() != self.geometry.voxel_count()? {
            return Err(invalid("Mask buffer length differs from geometry."));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RasterPixels {
    Gray8(Vec<u8>),
    Gray16(Vec<u16>),
    Rgb8(Vec<u8>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RasterImage {
    pub width: usize,
    pub height: usize,
    pub pixels: RasterPixels,
}
impl RasterImage {
    pub fn validate(&self) -> RadResult<()> {
        let n = self
            .width
            .checked_mul(self.height)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("Invalid raster dimensions."))?;
        let valid = match &self.pixels {
            RasterPixels::Gray8(v) => v.len() == n,
            RasterPixels::Gray16(v) => v.len() == n,
            RasterPixels::Rgb8(v) => n.checked_mul(3) == Some(v.len()),
        };
        if !valid {
            return Err(invalid(
                "Raster buffer length differs from dimensions and pixel type.",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleMetadata {
    pub sample_id: String,
    pub patient_id: String,
    pub slide_id: Option<String>,
    pub lesion_id: Option<String>,
    pub modality: Option<String>,
    pub region: Option<String>,
    pub timepoint: Option<String>,
    pub center: Option<String>,
}
impl SampleMetadata {
    pub fn validate(&self) -> RadResult<()> {
        if self.sample_id.trim().is_empty() || self.patient_id.trim().is_empty() {
            return Err(invalid("Sample and patient IDs must be nonempty."));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageSample {
    pub metadata: SampleMetadata,
    pub image: ImageVolume,
    pub mask: MaskVolume,
}
impl ImageSample {
    pub fn validate(&self) -> RadResult<()> {
        self.metadata.validate()?;
        self.image.validate()?;
        self.mask.validate()?;
        self.image.geometry.aligned_with(&self.mask.geometry)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericBuffer {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
    /// One explicit missing flag per element. Missing values have canonical stored value zero.
    pub missing: Vec<bool>,
}
impl NumericBuffer {
    pub fn validate(&self) -> RadResult<()> {
        if self.rows.checked_mul(self.cols) != Some(self.values.len())
            || self.missing.len() != self.values.len()
        {
            return Err(invalid(
                "Numeric buffer dimensions and missing flags disagree.",
            ));
        }
        if self
            .values
            .iter()
            .zip(&self.missing)
            .any(|(&v, &missing)| !v.is_finite() || (missing && v != 0.0))
        {
            return Err(invalid(
                "Buffers require finite values and zero at explicitly missing positions.",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureDefinition {
    pub name: String,
    pub definition: String,
    pub unit: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureTable {
    pub data: NumericBuffer,
    pub samples: Vec<SampleMetadata>,
    pub features: Vec<FeatureDefinition>,
    pub effective_parameters: BTreeMap<String, serde_json::Value>,
    pub quality: Vec<QualityRecord>,
}
impl FeatureTable {
    pub fn validate(&self) -> RadResult<()> {
        self.data.validate()?;
        if self.data.rows != self.samples.len()
            || self.data.cols != self.features.len()
            || self.features.is_empty()
        {
            return Err(invalid(
                "Feature dimensions disagree or no features remain.",
            ));
        }
        for sample in &self.samples {
            sample.validate()?;
        }
        unique_nonempty(
            &self
                .samples
                .iter()
                .map(|s| s.sample_id.clone())
                .collect::<Vec<_>>(),
            "Sample IDs",
        )?;
        unique_nonempty(
            &self
                .features
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
            "Feature names",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileRecord {
    pub patient_id: String,
    pub slide_id: String,
    pub tile_id: String,
    pub x0: f64,
    pub y0: f64,
    pub width0: f64,
    pub height0: f64,
    pub patch_width: usize,
    pub patch_height: usize,
    /// [left, top, width, height] in actual patch pixels, excludes padding.
    pub valid_pixels: [usize; 4],
    pub mpp_x_um: Option<f64>,
    pub mpp_y_um: Option<f64>,
    pub source: String,
    pub qc: BTreeMap<String, f64>,
    pub exclusion_reasons: Vec<String>,
}
impl TileRecord {
    pub fn validate(&self) -> RadResult<()> {
        if [&self.patient_id, &self.slide_id, &self.tile_id]
            .iter()
            .any(|s| s.trim().is_empty())
            || !self.x0.is_finite()
            || !self.y0.is_finite()
            || self.x0 < 0.0
            || self.y0 < 0.0
            || [self.width0, self.height0]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || self.patch_width == 0
            || self.patch_height == 0
        {
            return Err(invalid(
                "Tile IDs, geometry or patch dimensions are invalid.",
            ));
        }
        let [x, y, w, h] = self.valid_pixels;
        if w == 0
            || h == 0
            || x.checked_add(w).is_none_or(|v| v > self.patch_width)
            || y.checked_add(h).is_none_or(|v| v > self.patch_height)
        {
            return Err(invalid(
                "Valid pixel rectangle is empty or extends outside the patch.",
            ));
        }
        if [self.mpp_x_um, self.mpp_y_um]
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || self.qc.values().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "MPP and QC scores must be finite, and MPP positive when present.",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileTable {
    pub tiles: Vec<TileRecord>,
    pub features: Option<NumericBuffer>,
    pub feature_names: Vec<String>,
    /// Explicit statuses also preserve slides with no accepted tiles.
    pub slide_status: BTreeMap<String, String>,
}
impl TileTable {
    pub fn validate(&self) -> RadResult<()> {
        for tile in &self.tiles {
            tile.validate()?;
        }
        unique_nonempty(
            &self
                .tiles
                .iter()
                .map(|t| t.tile_id.clone())
                .collect::<Vec<_>>(),
            "Tile IDs",
        )?;
        let mut parents = BTreeMap::new();
        for t in &self.tiles {
            if parents
                .insert(&t.slide_id, &t.patient_id)
                .is_some_and(|p| p != &t.patient_id)
            {
                return Err(invalid("A slide belongs to more than one patient."));
            }
        }
        if let Some(data) = &self.features {
            data.validate()?;
            if data.rows != self.tiles.len()
                || data.cols != self.feature_names.len()
                || data.cols == 0
            {
                return Err(invalid("Tile feature dimensions disagree."));
            }
        } else if !self.feature_names.is_empty() {
            return Err(invalid("Feature names require a feature buffer."));
        }
        unique_nonempty(&self.feature_names, "Tile feature names")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Binary,
    Multiclass,
    Regression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum Discretization {
    FixedWidth { width: f64 },
    FixedCount { count: usize },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureAggregation {
    DirectionMean,
    MergeMatrices,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionConfig {
    pub profile: String,
    pub discretization: Discretization,
    pub aggregation: TextureAggregation,
    pub gldm_alpha: u32,
    pub voxel_array_shift: f64,
}
impl ExtractionConfig {
    pub fn validate(&self) -> RadResult<()> {
        if self.profile != "python_golden_v1" {
            return Err(RadiomicsError::new(
                ErrorCode::Unsupported,
                "Unknown extraction profile.",
            ));
        }
        if !self.voxel_array_shift.is_finite() {
            return Err(invalid("voxel_array_shift must be finite."));
        }
        match self.discretization {
            Discretization::FixedWidth { width } if !width.is_finite() || width <= 0.0 => {
                Err(invalid("Fixed bin width must be finite and positive."))
            }
            Discretization::FixedCount { count: 0 } => {
                Err(invalid("Fixed bin count must be positive."))
            }
            _ => Ok(()),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionUnit {
    Sample,
    Tile,
    Slide,
    Patient,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionRow {
    pub sample_id: String,
    pub patient_id: String,
    pub slide_id: Option<String>,
    pub tile_id: Option<String>,
    pub cohort: String,
    pub fold: Option<usize>,
    pub training_source: String,
    pub probabilities: Vec<f64>,
    pub value: Option<f64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionTable {
    pub task: Task,
    pub unit: PredictionUnit,
    pub classes: Vec<String>,
    pub rows: Vec<PredictionRow>,
}
impl PredictionTable {
    pub fn validate(&self) -> RadResult<()> {
        unique_nonempty(&self.classes, "Class labels")?;
        match self.task {
            Task::Regression if !self.classes.is_empty() => {
                return Err(invalid("Regression predictions cannot have classes."))
            }
            Task::Binary if self.classes.len() != 2 => {
                return Err(invalid("Binary predictions require exactly two classes."))
            }
            Task::Multiclass if self.classes.len() < 3 => {
                return Err(invalid(
                    "Multiclass predictions require at least three classes.",
                ))
            }
            _ => (),
        }
        unique_nonempty(
            &self
                .rows
                .iter()
                .map(|r| r.sample_id.clone())
                .collect::<Vec<_>>(),
            "Prediction sample IDs",
        )?;
        for row in &self.rows {
            if row.patient_id.is_empty() || row.cohort.is_empty() || row.training_source.is_empty()
            {
                return Err(invalid(
                    "Predictions require patient, cohort and training provenance.",
                ));
            }
            if self.unit == PredictionUnit::Tile
                && row.tile_id.as_ref().is_none_or(|v| v.is_empty())
            {
                return Err(invalid("Tile predictions require actual tile IDs."));
            }
            if matches!(self.unit, PredictionUnit::Slide | PredictionUnit::Tile)
                && row.slide_id.as_ref().is_none_or(|v| v.is_empty())
            {
                return Err(invalid("Slide/tile predictions require slide IDs."));
            }
            if self.task == Task::Regression {
                if !row.probabilities.is_empty() || row.value.is_none_or(|v| !v.is_finite()) {
                    return Err(invalid("Regression predictions require one finite value."));
                }
            } else if row.value.is_some()
                || row.probabilities.len() != self.classes.len()
                || row
                    .probabilities
                    .iter()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || (row.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
            {
                return Err(invalid(
                    "Class probabilities must match class order and sum to one.",
                ));
            }
        }
        Ok(())
    }
}
