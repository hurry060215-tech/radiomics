//! Lightweight pathology pixels. All statistics use the explicitly valid, unpadded region.
//! Reference: `python_golden_v1/pathology_spec.py`; no Python is executed at runtime.
use crate::contracts::{
    ExtractionConfig, Geometry, ImageVolume, MaskVolume, RasterImage, RasterPixels, SpatialUnit,
    TileRecord,
};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::{features, imaging, statistics};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn positive_pair(values: [f64; 2], label: &str) -> RadResult<()> {
    if values.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(invalid(format!(
            "{label} must contain two positive finite scales."
        )));
    }
    Ok(())
}
fn valid_rect(width: usize, height: usize, rect: [usize; 4]) -> RadResult<()> {
    let [x, y, w, h] = rect;
    if width.checked_mul(height).is_none()
        || w == 0
        || h == 0
        || x.checked_add(w).is_none_or(|v| v > width)
        || y.checked_add(h).is_none_or(|v| v > height)
    {
        return Err(invalid("Invalid valid-pixel rectangle."));
    }
    Ok(())
}
fn indices(width: usize, rect: [usize; 4]) -> impl Iterator<Item = (usize, usize, usize)> {
    let [left, top, w, h] = rect;
    (top..top + h).flat_map(move |y| (left..left + w).map(move |x| (x, y, y * width + x)))
}
fn rgb_buffer(image: &RasterImage) -> RadResult<&[u8]> {
    image.validate()?;
    match &image.pixels {RasterPixels::Rgb8(v)=>Ok(v),_=>Err(RadiomicsError::new(ErrorCode::Unsupported,"QC and color statistics require explicit sRGB RGB8 input; grayscale values are not implicitly mapped to 8-bit RGB."))}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgePolicy {
    Drop,
    Pad,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileSpec {
    pub patch_size: [usize; 2],
    pub stride: [usize; 2],
    pub edge: EdgePolicy,
    pub pad_value: u16,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlideMetadata {
    pub patient_id: String,
    pub slide_id: String,
    pub source: String,
    pub origin0: [f64; 2],
    pub pixel_size0: [f64; 2],
    pub mpp_x_um: Option<f64>,
    pub mpp_y_um: Option<f64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TilePatch {
    pub record: TileRecord,
    pub image: RasterImage,
}
impl TilePatch {
    pub fn validate(&self) -> RadResult<()> {
        self.record.validate()?;
        self.image.validate()?;
        if self.image.width != self.record.patch_width
            || self.image.height != self.record.patch_height
        {
            return Err(invalid(
                "Decoded patch shape differs from its TileRecord pixel dimensions.",
            ));
        }
        Ok(())
    }
}
pub struct TileIterator<'a> {
    image: Cow<'a, RasterImage>,
    spec: TileSpec,
    metadata: SlideMetadata,
    x: usize,
    y: usize,
}

fn tile_identifier(metadata: &SlideMetadata, x: usize, y: usize, patch_size: [usize; 2]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"radiomics-pixel-tile-v1");
    for field in [&metadata.patient_id, &metadata.slide_id, &metadata.source] {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field.as_bytes());
    }
    for coordinate in metadata.origin0.into_iter().chain(metadata.pixel_size0) {
        digest.update(coordinate.to_bits().to_le_bytes());
    }
    for dimension in [x, y, patch_size[0], patch_size[1]] {
        digest.update((dimension as u64).to_le_bytes());
    }
    format!("{}:{:x}", metadata.slide_id, digest.finalize())
}

/// Stream y-major/x-minor patches from an already decoded image. No files are written or deleted.
pub fn tile_image<'a>(
    image: &'a RasterImage,
    spec: &TileSpec,
    metadata: &SlideMetadata,
) -> RadResult<TileIterator<'a>> {
    tile_image_impl(Cow::Borrowed(image), spec, metadata)
}

/// Own one decoded source image while yielding one patch at a time across FFI.
pub fn tile_image_owned(
    image: RasterImage,
    spec: &TileSpec,
    metadata: &SlideMetadata,
) -> RadResult<TileIterator<'static>> {
    tile_image_impl(Cow::Owned(image), spec, metadata)
}

fn tile_image_impl<'a>(
    image: Cow<'a, RasterImage>,
    spec: &TileSpec,
    metadata: &SlideMetadata,
) -> RadResult<TileIterator<'a>> {
    image.validate()?;
    if spec.patch_size.iter().chain(&spec.stride).any(|v| *v == 0) {
        return Err(invalid("Patch size and stride must be positive."));
    }
    let n = spec.patch_size[0]
        .checked_mul(spec.patch_size[1])
        .ok_or_else(|| invalid("Patch dimensions overflow."))?;
    if matches!(&image.pixels, RasterPixels::Rgb8(_)) && n.checked_mul(3).is_none() {
        return Err(invalid("RGB patch dimensions overflow."));
    }
    if !matches!(&image.pixels, RasterPixels::Gray16(_)) && spec.pad_value > 255 {
        return Err(invalid("8-bit patch padding must lie in 0..255."));
    }
    if metadata.patient_id.trim().is_empty()
        || metadata.slide_id.trim().is_empty()
        || metadata.source.trim().is_empty()
    {
        return Err(invalid("Patient, slide and patch source must be nonempty."));
    }
    positive_pair(metadata.pixel_size0, "Level-0 pixel scale")?;
    if metadata.origin0.iter().any(|v| !v.is_finite() || *v < 0.)
        || [metadata.mpp_x_um, metadata.mpp_y_um]
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(invalid(
            "Slide origins must be finite/nonnegative and supplied MPP positive/finite.",
        ));
    }
    for (j, &size) in [image.width, image.height].iter().enumerate() {
        if !(metadata.origin0[j]
            + (size as f64 + spec.patch_size[j] as f64) * metadata.pixel_size0[j])
            .is_finite()
        {
            return Err(invalid("Level-0 tile coordinates overflow."));
        }
    }
    Ok(TileIterator {
        image,
        spec: spec.clone(),
        metadata: metadata.clone(),
        x: 0,
        y: 0,
    })
}
fn padded<T: Clone>(n: usize, value: T) -> RadResult<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|e| invalid(format!("Patch allocation failed: {e}")))?;
    output.resize(n, value);
    Ok(output)
}
impl Iterator for TileIterator<'_> {
    type Item = RadResult<TilePatch>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.y < self.image.height {
            let x = self.x;
            let y = self.y;
            self.x = self
                .x
                .checked_add(self.spec.stride[0])
                .unwrap_or(self.image.width);
            if self.x >= self.image.width {
                self.x = 0;
                self.y = self
                    .y
                    .checked_add(self.spec.stride[1])
                    .unwrap_or(self.image.height);
            }
            let [pw, ph] = self.spec.patch_size;
            let w = pw.min(self.image.width - x);
            let h = ph.min(self.image.height - y);
            if self.spec.edge == EdgePolicy::Drop && (w < pw || h < ph) {
                continue;
            }
            return Some((|| {
                let pixels = match &self.image.pixels {
                    RasterPixels::Gray8(values) => {
                        let mut out = padded(pw * ph, self.spec.pad_value as u8)?;
                        for row in 0..h {
                            out[row * pw..row * pw + w].copy_from_slice(
                                &values[(y + row) * self.image.width + x
                                    ..(y + row) * self.image.width + x + w],
                            );
                        }
                        RasterPixels::Gray8(out)
                    }
                    RasterPixels::Gray16(values) => {
                        let mut out = padded(pw * ph, self.spec.pad_value)?;
                        for row in 0..h {
                            out[row * pw..row * pw + w].copy_from_slice(
                                &values[(y + row) * self.image.width + x
                                    ..(y + row) * self.image.width + x + w],
                            );
                        }
                        RasterPixels::Gray16(out)
                    }
                    RasterPixels::Rgb8(values) => {
                        let mut out = padded(pw * ph * 3, self.spec.pad_value as u8)?;
                        for row in 0..h {
                            let source = ((y + row) * self.image.width + x) * 3;
                            out[row * pw * 3..(row * pw + w) * 3]
                                .copy_from_slice(&values[source..source + w * 3]);
                        }
                        RasterPixels::Rgb8(out)
                    }
                };
                let record = TileRecord {
                    patient_id: self.metadata.patient_id.clone(),
                    slide_id: self.metadata.slide_id.clone(),
                    tile_id: tile_identifier(&self.metadata, x, y, [pw, ph]),
                    x0: self.metadata.origin0[0] + x as f64 * self.metadata.pixel_size0[0],
                    y0: self.metadata.origin0[1] + y as f64 * self.metadata.pixel_size0[1],
                    width0: pw as f64 * self.metadata.pixel_size0[0],
                    height0: ph as f64 * self.metadata.pixel_size0[1],
                    patch_width: pw,
                    patch_height: ph,
                    valid_pixels: [0, 0, w, h],
                    mpp_x_um: self.metadata.mpp_x_um,
                    mpp_y_um: self.metadata.mpp_y_um,
                    source: self.metadata.source.clone(),
                    qc: BTreeMap::new(),
                    exclusion_reasons: vec![],
                };
                record.validate()?;
                Ok(TilePatch {
                    record,
                    image: RasterImage {
                        width: pw,
                        height: ph,
                        pixels,
                    },
                })
            })());
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QcConfig {
    pub white_threshold: f64,
    pub low_saturation_threshold: f64,
    pub black_threshold: f64,
    pub border_band_px: usize,
    pub tissue_s_min: f64,
    pub tissue_v_min: f64,
    pub tissue_v_max: f64,
    pub max_white_fraction: Option<f64>,
    pub max_low_saturation_fraction: Option<f64>,
    pub max_black_border_fraction: Option<f64>,
    pub min_tissue_fraction: Option<f64>,
    pub min_laplacian_variance: Option<f64>,
}
impl Default for QcConfig {
    fn default() -> Self {
        Self {
            white_threshold: 240.,
            low_saturation_threshold: 0.1,
            black_threshold: 15.,
            border_band_px: 1,
            tissue_s_min: 0.1,
            tissue_v_min: 0.05,
            tissue_v_max: 0.95,
            max_white_fraction: None,
            max_low_saturation_fraction: None,
            max_black_border_fraction: None,
            min_tissue_fraction: None,
            min_laplacian_variance: None,
        }
    }
}
impl QcConfig {
    pub fn validate(&self) -> RadResult<()> {
        if self.border_band_px == 0 {
            return Err(invalid("border_band_px must be a positive integer."));
        }
        if [self.white_threshold, self.black_threshold]
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=255.).contains(v))
        {
            return Err(invalid("RGB thresholds must lie in 0..255."));
        }
        if [
            self.low_saturation_threshold,
            self.tissue_s_min,
            self.tissue_v_min,
            self.tissue_v_max,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
            || self.tissue_v_min > self.tissue_v_max
        {
            return Err(invalid(
                "HSV thresholds must lie in 0..1, with ordered V limits.",
            ));
        }
        if [
            self.max_white_fraction,
            self.max_low_saturation_fraction,
            self.max_black_border_fraction,
            self.min_tissue_fraction,
        ]
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
            || self
                .min_laplacian_variance
                .is_some_and(|v| !v.is_finite() || v < 0.)
        {
            return Err(invalid(
                "QC limits must be finite, with fractions in [0,1] and nonnegative blur variance.",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchQc {
    pub scores: BTreeMap<String, Option<f64>>,
    pub valid_pixels: usize,
    pub border_pixels: usize,
    pub laplacian_pixels: usize,
    pub tissue_mask: Vec<bool>,
    pub exclusion_reasons: Vec<String>,
    pub accepted: bool,
}
impl PatchQc {
    /// TileRecord stores finite raw scores. Missing scores remain explicit in PatchQc.
    pub fn apply_to_record(&self, record: &mut TileRecord) -> RadResult<()> {
        record.validate()?;
        if self.valid_pixels != record.valid_pixels[2] * record.valid_pixels[3]
            || self.tissue_mask.len()
                != record
                    .patch_width
                    .checked_mul(record.patch_height)
                    .ok_or_else(|| invalid("Patch dimensions overflow."))?
        {
            return Err(invalid("QC and tile dimensions differ."));
        }
        if self.scores.values().flatten().any(|v| !v.is_finite())
            || self.accepted != self.exclusion_reasons.is_empty()
        {
            return Err(invalid("Malformed QC result."));
        }
        for (name, value) in &self.scores {
            if let Some(value) = value {
                record.qc.insert(name.clone(), *value);
            } else {
                record.qc.remove(name);
            }
        }
        for reason in &self.exclusion_reasons {
            if !record.exclusion_reasons.contains(reason) {
                record.exclusion_reasons.push(reason.clone());
            }
        }
        record.validate()
    }
}
fn hsv(rgb: [u8; 3]) -> [f64; 3] {
    let [r, g, b] = rgb.map(|c| c as f64 / 255.);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let range = max - min;
    if min == max {
        return [0., 0., max];
    }
    let rc = (max - r) / range;
    let gc = (max - g) / range;
    let bc = (max - b) / range;
    let h = if r == max {
        bc - gc
    } else if g == max {
        2. + rc - bc
    } else {
        4. + gc - rc
    };
    let h = (h / 6.).rem_euclid(1.);
    [if h >= 1. { 0. } else { h * 360. }, range / max, max]
}
/// Raw scores always returned; default configuration excludes no patches.
pub fn qc_patch(image: &RasterImage, rect: [usize; 4], config: &QcConfig) -> RadResult<PatchQc> {
    config.validate()?;
    let pixels = rgb_buffer(image)?;
    valid_rect(image.width, image.height, rect)?;
    let [left, top, w, h] = rect;
    let mut white = 0;
    let mut low_sat = 0;
    let mut black = 0;
    let mut border_count = 0;
    let mut tissue_count = 0;
    let mut tissue_mask = vec![false; image.width * image.height];
    let gray: Vec<f64> = pixels
        .chunks_exact(3)
        .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
        .collect();
    let mut laplacian = vec![];
    for (x, y, i) in indices(image.width, rect) {
        let p = [pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]];
        let [_, s, v] = hsv(p);
        white += usize::from(*p.iter().min().unwrap() as f64 >= config.white_threshold);
        low_sat += usize::from(s <= config.low_saturation_threshold);
        let border = (x - left)
            .min(y - top)
            .min(left + w - 1 - x)
            .min(top + h - 1 - y)
            < config.border_band_px;
        border_count += usize::from(border);
        black += usize::from(border && *p.iter().max().unwrap() as f64 <= config.black_threshold);
        let tissue =
            s >= config.tissue_s_min && v >= config.tissue_v_min && v <= config.tissue_v_max;
        tissue_mask[i] = tissue;
        tissue_count += usize::from(tissue);
        if x > left && x < left + w - 1 && y > top && y < top + h - 1 {
            laplacian.push(
                gray[i - 1] + gray[i + 1] + gray[i - image.width] + gray[i + image.width]
                    - 4. * gray[i],
            );
        }
    }
    let mut scores: BTreeMap<String, Option<f64>> = [
        ("white_fraction", Some(white as f64 / (w * h) as f64)),
        (
            "low_saturation_fraction",
            Some(low_sat as f64 / (w * h) as f64),
        ),
        (
            "black_border_fraction",
            Some(black as f64 / border_count as f64),
        ),
        (
            "tissue_fraction",
            Some(tissue_count as f64 / (w * h) as f64),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    scores.insert(
        "laplacian_variance".into(),
        if laplacian.is_empty() {
            None
        } else {
            Some(statistics::population_mean_variance(&laplacian)?.1)
        },
    );
    let mut exclusion_reasons = vec![];
    for (name, limit) in [
        ("white_fraction", config.max_white_fraction),
        (
            "low_saturation_fraction",
            config.max_low_saturation_fraction,
        ),
        ("black_border_fraction", config.max_black_border_fraction),
    ] {
        if let Some(limit) = limit {
            if scores[name].unwrap() > limit {
                exclusion_reasons.push(format!("{name}_above_maximum"));
            }
        }
    }
    for (name, limit) in [
        ("tissue_fraction", config.min_tissue_fraction),
        ("laplacian_variance", config.min_laplacian_variance),
    ] {
        if let Some(limit) = limit {
            match scores[name] {
                None => exclusion_reasons.push(format!("{name}_undefined")),
                Some(value) if value < limit => {
                    exclusion_reasons.push(format!("{name}_below_minimum"))
                }
                _ => (),
            }
        }
    }
    Ok(PatchQc {
        scores,
        valid_pixels: w * h,
        border_pixels: border_count,
        laplacian_pixels: laplacian.len(),
        tissue_mask,
        accepted: exclusion_reasons.is_empty(),
        exclusion_reasons,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelStatistics {
    pub mean: f64,
    pub std: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub q25: f64,
    pub q50: f64,
    pub q75: f64,
}
fn channel_statistics(values: &[f64]) -> RadResult<Option<ChannelStatistics>> {
    if values.is_empty() {
        return Ok(None);
    }
    let (mean, variance) = statistics::population_mean_variance(values)?;
    Ok(Some(ChannelStatistics {
        mean,
        std: variance.sqrt(),
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
        maximum: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        q25: statistics::quantile_linear(values, 0.25)?,
        q50: statistics::quantile_linear(values, 0.5)?,
        q75: statistics::quantile_linear(values, 0.75)?,
    }))
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub denominator: usize,
    pub probabilities: Option<Vec<f64>>,
}
fn validate_edges(edges: &[f64]) -> RadResult<()> {
    if edges.len() < 2
        || edges.iter().any(|v| !v.is_finite())
        || edges.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(invalid(
            "Fixed histogram edges must be finite and strictly increasing.",
        ));
    }
    Ok(())
}
pub fn histogram(values: &[f64], edges: &[f64]) -> RadResult<Histogram> {
    validate_edges(edges)?;
    let mut counts = vec![0u64; edges.len() - 1];
    for &v in values {
        if !v.is_finite() || v < edges[0] || v > *edges.last().unwrap() {
            return Err(invalid("Value lies outside fixed histogram edges."));
        }
        let bin = if v == *edges.last().unwrap() {
            counts.len() - 1
        } else {
            edges.partition_point(|&edge| edge <= v) - 1
        };
        counts[bin] += 1;
    }
    let probabilities = (!values.is_empty()).then(|| {
        counts
            .iter()
            .map(|&n| n as f64 / values.len() as f64)
            .collect()
    });
    Ok(Histogram {
        edges: edges.to_vec(),
        counts,
        denominator: values.len(),
        probabilities,
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorConfig {
    pub fixed_edges: BTreeMap<String, Vec<f64>>,
}
impl ColorConfig {
    pub fn validate(&self) -> RadResult<()> {
        let expected: BTreeSet<_> = ["R", "G", "B", "S", "V", "L", "a", "b", "H"]
            .into_iter()
            .map(str::to_string)
            .collect();
        if self.fixed_edges.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(invalid(
                "Provide fixed histogram edges for exactly R,G,B,S,V,L,a,b,H.",
            ));
        }
        for edges in self.fixed_edges.values() {
            validate_edges(edges)?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorChannel {
    pub statistics: Option<ChannelStatistics>,
    pub histogram: Histogram,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircularHue {
    pub count: usize,
    pub mean_degrees: Option<f64>,
    pub resultant_length: Option<f64>,
    pub circular_variance: Option<f64>,
    pub histogram: Histogram,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorFeatures {
    pub status: String,
    pub channels: BTreeMap<String, ColorChannel>,
    pub hue: Option<CircularHue>,
}
fn lab_d65(rgb: [u8; 3]) -> [f64; 3] {
    let linear = rgb.map(|c| {
        let c = c as f64 / 255.;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let matrix = [
        [506752. / 1228815., 87881. / 245763., 12673. / 70218.],
        [87098. / 409605., 175762. / 245763., 12673. / 175545.],
        [7918. / 409605., 87881. / 737289., 1001167. / 1053270.],
    ];
    let white = [0.3127 / 0.329, 1., (1. - 0.3127 - 0.329) / 0.329];
    let f = std::array::from_fn::<_, 3, _>(|i| {
        let ratio = matrix[i]
            .iter()
            .zip(linear)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / white[i];
        if ratio > 216. / 24389. {
            ratio.powf(1. / 3.)
        } else {
            (24389. / 27. * ratio + 16.) / 116.
        }
    });
    if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
        [116. * f[1] - 16., 0., 0.]
    } else {
        [
            116. * f[1] - 16.,
            500. * (f[0] - f[1]),
            200. * (f[1] - f[2]),
        ]
    }
}
pub fn color_features(
    image: &RasterImage,
    rect: [usize; 4],
    config: &ColorConfig,
    tissue_mask: Option<&[bool]>,
) -> RadResult<ColorFeatures> {
    config.validate()?;
    let pixels = rgb_buffer(image)?;
    valid_rect(image.width, image.height, rect)?;
    if tissue_mask.is_some_and(|m| m.len() != image.width * image.height) {
        return Err(invalid("Tissue mask shape differs from patch."));
    }
    let mut channels: BTreeMap<&str, Vec<f64>> = ["R", "G", "B", "S", "V", "L", "a", "b"]
        .into_iter()
        .map(|name| (name, vec![]))
        .collect();
    let mut hues = vec![];
    for (_, _, i) in
        indices(image.width, rect).filter(|(_, _, i)| tissue_mask.is_none_or(|m| m[*i]))
    {
        let p = [pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]];
        for (name, v) in ["R", "G", "B"].into_iter().zip(p) {
            channels.get_mut(name).unwrap().push(v as f64);
        }
        let [h, s, v] = hsv(p);
        channels.get_mut("S").unwrap().push(s);
        channels.get_mut("V").unwrap().push(v);
        if s > 0. {
            hues.push(h);
        }
        for (name, v) in ["L", "a", "b"].into_iter().zip(lab_d65(p)) {
            channels.get_mut(name).unwrap().push(v);
        }
    }
    if channels["R"].is_empty() {
        return Ok(ColorFeatures {
            status: "empty_statistical_mask".into(),
            channels: BTreeMap::new(),
            hue: None,
        });
    }
    let mut hue = CircularHue {
        count: hues.len(),
        mean_degrees: None,
        resultant_length: None,
        circular_variance: None,
        histogram: histogram(&hues, &config.fixed_edges["H"])?,
    };
    if !hues.is_empty() {
        let cos = hues.iter().map(|v| v.to_radians().cos()).sum::<f64>() / hues.len() as f64;
        let sin = hues.iter().map(|v| v.to_radians().sin()).sum::<f64>() / hues.len() as f64;
        let r = cos.hypot(sin).min(1.);
        hue.resultant_length = Some(r);
        hue.circular_variance = Some(1. - r);
        if r > 1e-12 {
            let mean = sin.atan2(cos).to_degrees().rem_euclid(360.);
            hue.mean_degrees = Some(if mean >= 360. { 0. } else { mean });
        }
    }
    let mut result = BTreeMap::new();
    for (name, values) in channels {
        result.insert(
            name.into(),
            ColorChannel {
                statistics: channel_statistics(&values)?,
                histogram: histogram(&values, &config.fixed_edges[name])?,
            },
        );
    }
    Ok(ColorFeatures {
        status: "ok".into(),
        channels: result,
        hue: Some(hue),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationCoverage {
    pub denominator: usize,
    pub counts: BTreeMap<String, usize>,
    pub fractions: BTreeMap<String, f64>,
}
pub fn annotation_coverage(
    width: usize,
    height: usize,
    labels: &[u32],
    rect: [usize; 4],
    classes: &[u32],
) -> RadResult<AnnotationCoverage> {
    valid_rect(width, height, rect)?;
    if width.checked_mul(height) != Some(labels.len())
        || classes.iter().collect::<BTreeSet<_>>().len() != classes.len()
    {
        return Err(invalid(
            "Mask shape differs or requested classes are duplicated.",
        ));
    }
    let denominator = rect[2] * rect[3];
    let mut raw: BTreeMap<u32, usize> = classes.iter().map(|&c| (c, 0)).collect();
    for (_, _, i) in indices(width, rect) {
        if let Some(count) = raw.get_mut(&labels[i]) {
            *count += 1;
        }
    }
    Ok(AnnotationCoverage {
        denominator,
        counts: raw.iter().map(|(c, n)| (c.to_string(), *n)).collect(),
        fractions: raw
            .into_iter()
            .map(|(c, n)| (c.to_string(), n as f64 / denominator as f64))
            .collect(),
    })
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    Instance,
    Semantic,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectRecord {
    pub object_id: u32,
    pub source_label: u32,
    pub area_px: usize,
    pub centroid_pixel_centers: [f64; 2],
    pub bbox_xywh: [usize; 4],
    pub touches_valid_boundary: bool,
    pub area_um2: Option<f64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectFeatures {
    pub status: String,
    pub mask_kind: MaskKind,
    pub object_count: usize,
    pub objects: Vec<ObjectRecord>,
    pub size_distribution: Option<ChannelStatistics>,
    pub warnings: Vec<String>,
    pub count_scope: String,
}
fn pixel_geometry(width: usize, height: usize) -> Geometry {
    Geometry {
        shape: [width, height, 1],
        ndim: 2,
        affine: [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        unit: SpatialUnit::Unknown,
        diagnostics: vec![],
    }
}
/// All counts are patch local, including instance IDs. Semantic components reuse the image engine.
#[allow(clippy::too_many_arguments)]
pub fn object_features(
    width: usize,
    height: usize,
    labels: &[u32],
    rect: [usize; 4],
    kind: MaskKind,
    connectivity: imaging::Connectivity,
    patch_mpp_um: Option<[f64; 2]>,
    overlapping_tiles: bool,
) -> RadResult<ObjectFeatures> {
    valid_rect(width, height, rect)?;
    if width.checked_mul(height) != Some(labels.len())
        || !matches!(
            connectivity,
            imaging::Connectivity::Four | imaging::Connectivity::Eight
        )
    {
        return Err(invalid(
            "Object masks need matching shape and explicit 4/8 connectivity.",
        ));
    }
    if let Some(mpp) = patch_mpp_um {
        positive_pair(mpp, "Patch-pixel MPP")?;
    }
    let mut valid_labels = vec![0u32; labels.len()];
    for (_, _, i) in indices(width, rect) {
        valid_labels[i] = labels[i];
    }
    let mut groups: BTreeMap<u32, (u32, Vec<usize>)> = BTreeMap::new();
    if kind == MaskKind::Semantic {
        let components = imaging::connected_components(
            &MaskVolume {
                geometry: pixel_geometry(width, height),
                labels: valid_labels,
            },
            connectivity,
            true,
        )?;
        for c in &components.components {
            groups.insert(c.component_id, (c.source_label.unwrap(), vec![]));
        }
        for (i, &id) in components.mask.labels.iter().enumerate() {
            if id != 0 {
                groups.get_mut(&id).unwrap().1.push(i);
            }
        }
    } else {
        for (i, &label) in valid_labels.iter().enumerate() {
            if label != 0 {
                groups
                    .entry(label)
                    .or_insert_with(|| (label, vec![]))
                    .1
                    .push(i);
            }
        }
    }
    let [left, top, w, h] = rect;
    let mut objects = vec![];
    for (id, (source_label, members)) in groups {
        let xs: Vec<f64> = members.iter().map(|i| (i % width) as f64 + 0.5).collect();
        let ys: Vec<f64> = members.iter().map(|i| (i / width) as f64 + 0.5).collect();
        let minx = members.iter().map(|i| i % width).min().unwrap();
        let maxx = members.iter().map(|i| i % width).max().unwrap();
        let miny = members.iter().map(|i| i / width).min().unwrap();
        let maxy = members.iter().map(|i| i / width).max().unwrap();
        let area_um2 = patch_mpp_um.map(|mpp| members.len() as f64 * mpp[0] * mpp[1]);
        if area_um2.is_some_and(|v| !v.is_finite()) {
            return Err(invalid("Physical object area overflows."));
        }
        objects.push(ObjectRecord {
            object_id: id,
            source_label,
            area_px: members.len(),
            centroid_pixel_centers: [
                statistics::population_mean_variance(&xs)?.0,
                statistics::population_mean_variance(&ys)?.0,
            ],
            bbox_xywh: [minx, miny, maxx - minx + 1, maxy - miny + 1],
            touches_valid_boundary: members.iter().any(|i| {
                let x = i % width;
                let y = i / width;
                x == left || x == left + w - 1 || y == top || y == top + h - 1
            }),
            area_um2,
        });
    }
    let sizes: Vec<f64> = objects.iter().map(|o| o.area_px as f64).collect();
    let warnings = if overlapping_tiles {
        vec!["patch_local_instance_counts_must_not_be_summed_across_overlapping_tiles".into()]
    } else {
        vec![]
    };
    Ok(ObjectFeatures {
        status: if objects.is_empty() {
            "empty_foreground"
        } else {
            "ok"
        }
        .into(),
        mask_kind: kind,
        object_count: objects.len(),
        size_distribution: channel_statistics(&sizes)?,
        objects,
        warnings,
        count_scope: "patch_local".into(),
    })
}
pub fn patch_pixel_mpp(record: &TileRecord) -> RadResult<Option<[f64; 2]>> {
    record.validate()?;
    match (record.mpp_x_um, record.mpp_y_um) {
        (Some(x), Some(y)) => {
            let value = [
                x * record.width0 / record.patch_width as f64,
                y * record.height0 / record.patch_height as f64,
            ];
            positive_pair(value, "Derived patch-pixel MPP")?;
            Ok(Some(value))
        }
        _ => Ok(None),
    }
}
pub fn object_features_for_tile(
    labels: &[u32],
    record: &TileRecord,
    kind: MaskKind,
    connectivity: imaging::Connectivity,
    overlapping_tiles: bool,
) -> RadResult<ObjectFeatures> {
    object_features(
        record.patch_width,
        record.patch_height,
        labels,
        record.valid_pixels,
        kind,
        connectivity,
        patch_pixel_mpp(record)?,
        overlapping_tiles,
    )
}
/// Preserve Gray8/Gray16 values and reuse the sole radiomics texture implementation.
/// RGB callers must supply an explicitly converted grayscale image.
pub fn texture_features(
    image: &RasterImage,
    rect: [usize; 4],
    tissue_mask: Option<&[bool]>,
    config: &ExtractionConfig,
) -> RadResult<BTreeMap<String, f64>> {
    image.validate()?;
    valid_rect(image.width, image.height, rect)?;
    if tissue_mask.is_some_and(|m| m.len() != image.width * image.height) {
        return Err(invalid("Tissue mask shape differs from patch."));
    }
    let values = match &image.pixels {
        RasterPixels::Gray8(v) => v.iter().map(|&v| v as f64).collect(),
        RasterPixels::Gray16(v) => v.iter().map(|&v| v as f64).collect(),
        _ => return Err(RadiomicsError::new(
            ErrorCode::Unsupported,
            "Traditional texture input must be explicit Gray8/Gray16; no implicit RGB conversion.",
        )),
    };
    let mut labels = vec![0u32; image.width * image.height];
    for (_, _, i) in indices(image.width, rect) {
        labels[i] = u32::from(tissue_mask.is_none_or(|m| m[i]));
    }
    let geometry = pixel_geometry(image.width, image.height);
    let result = features::extract(
        &ImageVolume {
            geometry: geometry.clone(),
            values,
        },
        &MaskVolume { geometry, labels },
        None,
        1,
        config,
    )?;
    // Shape is defined elsewhere; this adapter advertises only first-order and texture engine outputs.
    Ok(result
        .into_iter()
        .filter(|(name, _)| !name.starts_with("original_shape_basic_"))
        .collect())
}

#[cfg(test)]
#[path = "pathology_pixels_tests.rs"]
mod tests;
