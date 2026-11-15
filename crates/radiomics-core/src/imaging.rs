//! Native scalar-image I/O and explicitly configured geometric preparation.
//!
//! Buffers are x-fastest. NIfTI affines retain their RAS coordinate system and declared units.
#[cfg(test)]
#[path = "imaging_tests.rs"]
mod tests;
use crate::contracts::{
    Geometry, ImageVolume, MaskVolume, QualityRecord, RasterImage, RasterPixels, SpatialUnit,
};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use nifti::{NiftiObject, NiftiType, RandomAccessNiftiVolume, ReaderOptions};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::File,
    io::BufReader,
    path::Path,
};

fn error(code: ErrorCode, message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(code, message)
}
fn invalid(message: impl Into<String>) -> RadiomicsError {
    error(ErrorCode::InvalidInput, message)
}
fn diagnostic(code: &str, message: String, severity: &str) -> QualityRecord {
    QualityRecord {
        code: code.into(),
        message,
        severity: severity.into(),
        sample_id: None,
    }
}
fn index(shape: [usize; 3], p: [usize; 3]) -> usize {
    p[0] + shape[0] * (p[1] + shape[1] * p[2])
}
fn position(shape: [usize; 3], i: usize) -> [usize; 3] {
    [
        i % shape[0],
        (i / shape[0]) % shape[1],
        i / (shape[0] * shape[1]),
    ]
}

/// Read scalar 2D/3D NIfTI-1 (.nii/.nii.gz), preserving RAS affine and native spatial units.
/// No automatic correction, reorientation, resampling, or assumed millimeters is performed.
pub fn read_nifti(path: impl AsRef<Path>) -> RadResult<ImageVolume> {
    let path = path.as_ref();
    let name = path.to_string_lossy();
    if !name.ends_with(".nii") && !name.ends_with(".nii.gz") {
        return Err(error(
            ErrorCode::Unsupported,
            "Expected .nii or .nii.gz scalar NIfTI.",
        ));
    }
    let object = ReaderOptions::new()
        .read_file(path)
        .map_err(|e| error(ErrorCode::Io, format!("{}: {e}", path.display())))?;
    let header = object.header();
    if !matches!(header.dim[0], 2 | 3) {
        return Err(error(
            ErrorCode::Unsupported,
            "NIfTI must be explicitly 2D or 3D; split 4D/multichannel inputs first.",
        ));
    }
    if !matches!(
        header.data_type().map_err(|e| invalid(e.to_string()))?,
        NiftiType::Uint8
            | NiftiType::Int8
            | NiftiType::Uint16
            | NiftiType::Int16
            | NiftiType::Uint32
            | NiftiType::Int32
            | NiftiType::Uint64
            | NiftiType::Int64
            | NiftiType::Float32
            | NiftiType::Float64
    ) {
        return Err(error(
            ErrorCode::Unsupported,
            "Only real scalar NIfTI datatypes are supported; RGB/complex inputs are not scalar.",
        ));
    }
    let shape = [
        header.dim[1] as usize,
        header.dim[2] as usize,
        if header.dim[0] == 3 {
            header.dim[3] as usize
        } else {
            1
        },
    ];
    let unit = match header.xyzt_units & 7 {
        0 => SpatialUnit::Unknown,
        1 => SpatialUnit::Meter,
        2 => SpatialUnit::Millimeter,
        3 => SpatialUnit::Micron,
        _ => {
            return Err(error(
                ErrorCode::Geometry,
                "Invalid NIfTI spatial-unit code.",
            ))
        }
    };
    if header.sform_code < 0 || header.qform_code < 0 {
        return Err(error(
            ErrorCode::Geometry,
            "Negative NIfTI transformation code.",
        ));
    }
    let mut diagnostics = vec![];
    let sform = if header.sform_code > 0 {
        Some([
            header.srow_x.map(f64::from),
            header.srow_y.map(f64::from),
            header.srow_z.map(f64::from),
            [0., 0., 0., 1.],
        ])
    } else {
        None
    };
    let qform = if header.qform_code > 0 {
        Some(qform_affine(header)?)
    } else {
        None
    };
    diagnostics.push(diagnostic("nifti_transforms", serde_json::json!({"sform_code":header.sform_code,"qform_code":header.qform_code,"sform":sform,"qform":qform,"coordinate_system":"RAS"}).to_string(), "info"));
    let affine = if let Some(s) = sform {
        s
    } else if let Some(q) = qform {
        q
    } else {
        let mut a = [[0.; 4]; 4];
        a[3][3] = 1.;
        for (j, row) in a.iter_mut().enumerate().take(3) {
            let spacing = header.pixdim[j + 1] as f64;
            if !spacing.is_finite() || spacing <= 0. {
                return Err(error(
                    ErrorCode::Geometry,
                    "NIfTI fallback pixdim must be positive and finite.",
                ));
            }
            row[j] = spacing;
        }
        diagnostics.push(diagnostic("nifti_no_world_transform", "Both qform and sform are unset. A pixdim diagonal with origin zero represents the index grid; anatomical orientation is unknown.".into(), "warning"));
        a
    };
    let geometry = Geometry {
        shape,
        ndim: header.dim[0] as u8,
        affine,
        unit,
        diagnostics,
    };
    geometry.validate()?;
    let mut geometry = geometry;
    if let (Some(s), Some(q)) = (sform, qform) {
        let mut sg = geometry.clone();
        sg.affine = s;
        let mut qg = geometry.clone();
        qg.affine = q;
        if sg.aligned_with(&qg).is_err() {
            geometry.diagnostics.push(diagnostic("nifti_sform_qform_conflict", "Sform and qform differ over the image grid; sform is selected and both matrices are retained in nifti_transforms.".into(), "warning"));
        }
    }
    if unit == SpatialUnit::Unknown {
        geometry.diagnostics.push(diagnostic(
            "unknown_spatial_units",
            "Spatial units are unknown; physical millimeter outputs are unavailable.".into(),
            "warning",
        ));
    }
    if geometry.require_orthogonal().is_err() {
        geometry.diagnostics.push(diagnostic("sheared_affine", "Affine contains shear. Reading and affine resampling are allowed; orthogonal-grid distance operations require explicit resampling.".into(), "warning"));
    }
    let volume = object.volume();
    let mut values = Vec::with_capacity(geometry.voxel_count()?);
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            for x in 0..shape[0] {
                let coords = [x as u16, y as u16, z as u16];
                let value = volume
                    .get_f64(&coords[..geometry.ndim as usize])
                    .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
                values.push(value);
            }
        }
    }
    let image = ImageVolume { geometry, values };
    image.validate()?;
    Ok(image)
}

fn qform_affine(h: &nifti::NiftiHeader) -> RadResult<[[f64; 4]; 4]> {
    let mut q = [h.quatern_b as f64, h.quatern_c as f64, h.quatern_d as f64];
    let norm = q.iter().map(|v| v * v).sum::<f64>();
    if !norm.is_finite()
        || [h.quatern_x, h.quatern_y, h.quatern_z]
            .iter()
            .any(|v| !v.is_finite())
    {
        return Err(error(ErrorCode::Geometry, "Nonfinite NIfTI qform."));
    }
    let a = if 1. - norm < 1e-7 {
        let length = norm.sqrt();
        if length == 0. {
            return Err(error(ErrorCode::Geometry, "Invalid qform quaternion."));
        }
        q = q.map(|v| v / length);
        0.
    } else {
        (1. - norm).sqrt()
    };
    let [b, c, d] = q;
    let mut r = [
        [
            a * a + b * b - c * c - d * d,
            2. * (b * c - a * d),
            2. * (b * d + a * c),
            h.quatern_x as f64,
        ],
        [
            2. * (b * c + a * d),
            a * a + c * c - b * b - d * d,
            2. * (c * d - a * b),
            h.quatern_y as f64,
        ],
        [
            2. * (b * d - a * c),
            2. * (c * d + a * b),
            a * a + d * d - c * c - b * b,
            h.quatern_z as f64,
        ],
        [0., 0., 0., 1.],
    ];
    if !h.pixdim[0].is_finite() {
        return Err(error(ErrorCode::Geometry, "Nonfinite qfac."));
    }
    for j in 0..3 {
        let s = h.pixdim[j + 1] as f64;
        if !s.is_finite() || s <= 0. {
            return Err(error(
                ErrorCode::Geometry,
                "Qform spacings must be positive and finite.",
            ));
        }
        let s = if j == 2 && h.pixdim[0] < 0. { -s } else { s };
        for row in r.iter_mut().take(3) {
            row[j] *= s;
        }
    }
    Ok(r)
}

pub fn read_mask(path: impl AsRef<Path>) -> RadResult<MaskVolume> {
    let image = read_nifti(path)?;
    let mut labels = Vec::with_capacity(image.values.len());
    for value in image.values {
        if !value.is_finite() || value < 0. || value > u32::MAX as f64 || value.fract() != 0. {
            return Err(invalid(
                "Masks must contain finite nonnegative integer labels representable as u32.",
            ));
        }
        labels.push(value as u32);
    }
    Ok(MaskVolume {
        geometry: image.geometry,
        labels,
    })
}

/// Decode ordinary PNG/JPEG RGB8 or single-page TIFF Gray8/Gray16/RGB8.
/// Pixel coordinates describe stored top-left-oriented pixels. Physical MPP is never inferred.
pub fn read_raster(path: impl AsRef<Path>) -> RadResult<RasterImage> {
    let path = path.as_ref();
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let format = reader
        .format()
        .ok_or_else(|| invalid("Unknown raster format."))?;
    let result = match format {
        image::ImageFormat::Png | image::ImageFormat::Jpeg => {
            let decoded = reader
                .decode()
                .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
            if let image::DynamicImage::ImageRgb8(rgb) = decoded {
                RasterImage {
                    width: rgb.width() as usize,
                    height: rgb.height() as usize,
                    pixels: RasterPixels::Rgb8(rgb.into_raw()),
                }
            } else {
                return Err(error(
                    ErrorCode::Unsupported,
                    "PNG/JPEG input must decode as RGB8; convert explicitly before loading.",
                ));
            }
        }
        image::ImageFormat::Tiff => read_tiff(path)?,
        _ => {
            return Err(error(
                ErrorCode::Unsupported,
                "Only PNG, JPEG and single-page TIFF inputs are supported.",
            ))
        }
    };
    result.validate()?;
    Ok(result)
}
fn read_tiff(path: &Path) -> RadResult<RasterImage> {
    use tiff::{
        decoder::{Decoder, DecodingResult},
        tags::Tag,
        ColorType,
    };
    let mut decoder = Decoder::new(BufReader::new(File::open(path)?))
        .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
    if decoder.more_images()
        || decoder
            .find_tag(Tag::Unknown(330))
            .map_err(|e| error(ErrorCode::Io, e.to_string()))?
            .is_some()
    {
        return Err(error(
            ErrorCode::Unsupported,
            "Multi-page and pyramid TIFF are not supported; supply decoded ordinary images.",
        ));
    }
    if decoder
        .find_tag_unsigned::<u16>(Tag::Orientation)
        .map_err(|e| error(ErrorCode::Io, e.to_string()))?
        .is_some_and(|x| x != 1)
    {
        return Err(error(
            ErrorCode::Unsupported,
            "TIFF orientation must be top-left (1); explicitly orient the image first.",
        ));
    }
    let (w, h) = decoder
        .dimensions()
        .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
    let color = decoder
        .colortype()
        .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
    if !matches!(
        color,
        ColorType::Gray(8) | ColorType::Gray(16) | ColorType::RGB(8)
    ) {
        return Err(error(
            ErrorCode::Unsupported,
            "TIFF must be Gray8, Gray16 or RGB8.",
        ));
    }
    let pixels = match (
        color,
        decoder
            .read_image()
            .map_err(|e| error(ErrorCode::Io, e.to_string()))?,
    ) {
        (ColorType::Gray(8), DecodingResult::U8(v)) => RasterPixels::Gray8(v),
        (ColorType::Gray(16), DecodingResult::U16(v)) => RasterPixels::Gray16(v),
        (ColorType::RGB(8), DecodingResult::U8(v)) => RasterPixels::Rgb8(v),
        _ => {
            return Err(error(
                ErrorCode::Unsupported,
                "Unsupported TIFF decoded sample type.",
            ))
        }
    };
    Ok(RasterImage {
        width: w as usize,
        height: h as usize,
        pixels,
    })
}

/// Select labels into binary mask. Requested labels must all exist; background 0 is forbidden.
pub fn select_labels(mask: &MaskVolume, labels: &[u32]) -> RadResult<MaskVolume> {
    mask.validate()?;
    let wanted: BTreeSet<_> = labels.iter().copied().collect();
    if wanted.is_empty() || wanted.contains(&0) || wanted.len() != labels.len() {
        return Err(invalid("Choose unique, nonzero ROI labels."));
    }
    let present: BTreeSet<_> = mask.labels.iter().copied().collect();
    if !wanted.is_subset(&present) {
        return Err(error(
            ErrorCode::EmptyResult,
            "A requested ROI label is absent.",
        ));
    }
    Ok(MaskVolume {
        geometry: mask.geometry.clone(),
        labels: mask
            .labels
            .iter()
            .map(|x| u32::from(wanted.contains(x)))
            .collect(),
    })
}

/// Recode supplied labels, retaining labels not present in the map.
pub fn recode_labels(mask: &MaskVolume, mapping: &BTreeMap<u32, u32>) -> RadResult<MaskVolume> {
    mask.validate()?;
    if mapping.is_empty() {
        return Err(invalid("Label mapping is empty."));
    }
    Ok(MaskVolume {
        geometry: mask.geometry.clone(),
        labels: mask
            .labels
            .iter()
            .map(|x| *mapping.get(x).unwrap_or(x))
            .collect(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundingBox {
    pub start: [usize; 3],
    pub end: [usize; 3],
}
impl BoundingBox {
    fn validate(self, shape: [usize; 3]) -> RadResult<()> {
        if (0..3).any(|j| self.start[j] >= self.end[j] || self.end[j] > shape[j]) {
            Err(invalid(
                "Bounding box must be nonempty with exclusive ends inside the image.",
            ))
        } else {
            Ok(())
        }
    }
}
pub fn bounding_box(mask: &MaskVolume) -> RadResult<BoundingBox> {
    mask.validate()?;
    let shape = mask.geometry.shape;
    let mut bounds = BoundingBox {
        start: shape,
        end: [0; 3],
    };
    for (i, &label) in mask.labels.iter().enumerate().filter(|(_, x)| **x != 0) {
        let _ = label;
        let p = position(shape, i);
        for (j, &v) in p.iter().enumerate() {
            bounds.start[j] = bounds.start[j].min(v);
            bounds.end[j] = bounds.end[j].max(v + 1);
        }
    }
    if bounds.end == [0; 3] {
        return Err(error(ErrorCode::EmptyResult, "ROI is empty."));
    }
    Ok(bounds)
}
fn cropped_geometry(geometry: &Geometry, bounds: BoundingBox) -> RadResult<Geometry> {
    geometry.validate()?;
    bounds.validate(geometry.shape)?;
    let mut out = geometry.clone();
    out.shape = std::array::from_fn(|j| bounds.end[j] - bounds.start[j]);
    let origin = geometry.world(bounds.start.map(|v| v as f64));
    for (j, row) in out.affine.iter_mut().enumerate().take(3) {
        row[3] = origin[j];
    }
    out.diagnostics.push(diagnostic(
        "crop",
        serde_json::json!({"start":bounds.start,"end_exclusive":bounds.end}).to_string(),
        "info",
    ));
    Ok(out)
}
pub fn crop_image(image: &ImageVolume, bounds: BoundingBox) -> RadResult<ImageVolume> {
    image.validate()?;
    let geometry = cropped_geometry(&image.geometry, bounds)?;
    let mut values = Vec::with_capacity(geometry.voxel_count()?);
    for z in bounds.start[2]..bounds.end[2] {
        for y in bounds.start[1]..bounds.end[1] {
            let begin = index(image.geometry.shape, [bounds.start[0], y, z]);
            values.extend_from_slice(&image.values[begin..begin + geometry.shape[0]]);
        }
    }
    Ok(ImageVolume { geometry, values })
}
pub fn crop_mask(mask: &MaskVolume, bounds: BoundingBox) -> RadResult<MaskVolume> {
    mask.validate()?;
    let geometry = cropped_geometry(&mask.geometry, bounds)?;
    let mut labels = Vec::with_capacity(geometry.voxel_count()?);
    for z in bounds.start[2]..bounds.end[2] {
        for y in bounds.start[1]..bounds.end[1] {
            let begin = index(mask.geometry.shape, [bounds.start[0], y, z]);
            labels.extend_from_slice(&mask.labels[begin..begin + geometry.shape[0]]);
        }
    }
    Ok(MaskVolume { geometry, labels })
}
pub fn roi_bounds_margin_voxels(mask: &MaskVolume, margin: [usize; 3]) -> RadResult<BoundingBox> {
    let mut bounds = bounding_box(mask)?;
    if mask.geometry.ndim == 2 && margin[2] != 0 {
        return Err(invalid("2D z margin must be zero."));
    }
    for (j, &v) in margin.iter().enumerate() {
        bounds.start[j] = bounds.start[j].saturating_sub(v);
        bounds.end[j] = bounds.end[j].saturating_add(v).min(mask.geometry.shape[j]);
    }
    Ok(bounds)
}
pub fn roi_bounds_margin_mm(mask: &MaskVolume, margin_mm: [f64; 3]) -> RadResult<BoundingBox> {
    mask.geometry.require_orthogonal()?;
    let spacing = mask.geometry.spacing_mm()?;
    if margin_mm.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err(invalid(
            "Millimeter margins must be finite and nonnegative.",
        ));
    }
    roi_bounds_margin_voxels(
        mask,
        std::array::from_fn(|j| (margin_mm[j] / spacing[j]).ceil() as usize),
    )
}
/// Return a one-voxel-thick bounding box at the first largest ROI section along axis.
pub fn maximum_roi_section(mask: &MaskVolume, axis: usize) -> RadResult<BoundingBox> {
    mask.validate()?;
    if axis >= mask.geometry.ndim as usize {
        return Err(invalid("Section axis must be an active spatial axis."));
    }
    let bounds = bounding_box(mask)?;
    let mut counts = vec![0usize; mask.geometry.shape[axis]];
    for (i, &label) in mask.labels.iter().enumerate() {
        if label != 0 {
            counts[position(mask.geometry.shape, i)[axis]] += 1;
        }
    }
    let maximum = *counts.iter().max().unwrap();
    let selected = counts.iter().position(|x| *x == maximum).unwrap();
    let mut result = bounds;
    result.start[axis] = selected;
    result.end[axis] = selected + 1;
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    Nearest,
    Linear,
}
fn inverse_linear(a: &[[f64; 4]; 4]) -> [[f64; 3]; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det,
        ],
    ]
}
fn unit_ratio(source: &Geometry, target: &Geometry) -> RadResult<f64> {
    if source.unit == target.unit {
        Ok(1.)
    } else {
        Ok(target.unit.millimeter_factor()? / source.unit.millimeter_factor()?)
    }
}
fn source_position(
    source: &Geometry,
    target: &Geometry,
    p: [usize; 3],
    inverse: &[[f64; 3]; 3],
    ratio: f64,
) -> [f64; 3] {
    let world = target.world(p.map(|x| x as f64));
    std::array::from_fn(|i| {
        (0..3)
            .map(|j| inverse[i][j] * (world[j] * ratio - source.affine[j][3]))
            .sum()
    })
}
/// SITK-style domain: [-0.5,n-0.5), nearest uses floor(index+0.5), linear clamps edge centers.
fn interpolate(
    values: &[f64],
    shape: [usize; 3],
    ndim: u8,
    p: [f64; 3],
    mode: Interpolation,
    outside: f64,
) -> f64 {
    if (0..3).any(|j| !p[j].is_finite() || p[j] < -0.5 || p[j] >= shape[j] as f64 - 0.5) {
        return outside;
    }
    if mode == Interpolation::Nearest {
        return values[index(shape, p.map(|v| (v + 0.5).floor() as usize))];
    }
    let clipped = std::array::from_fn::<_, 3, _>(|j| p[j].clamp(0., (shape[j] - 1) as f64));
    let lo = clipped.map(|v| v.floor() as usize);
    let hi = std::array::from_fn::<_, 3, _>(|j| (lo[j] + 1).min(shape[j] - 1));
    let fraction = std::array::from_fn::<_, 3, _>(|j| clipped[j] - lo[j] as f64);
    let mut result = 0.;
    for corner in 0..(1 << ndim) {
        let mut point = lo;
        let mut weight = 1.;
        for j in 0..ndim as usize {
            if corner & (1 << j) == 0 {
                weight *= 1. - fraction[j];
            } else {
                weight *= fraction[j];
                point[j] = hi[j];
            }
        }
        if weight != 0. {
            result += weight * values[index(shape, point)];
        }
    }
    result
}
pub fn resample_like(
    image: &ImageVolume,
    target: &Geometry,
    mode: Interpolation,
    outside: f64,
) -> RadResult<ImageVolume> {
    image.validate()?;
    target.validate()?;
    if image.geometry.ndim != target.ndim {
        return Err(error(
            ErrorCode::Geometry,
            "Source and target dimensionality differ.",
        ));
    }
    if !outside.is_finite() {
        return Err(invalid("Outside image value must be finite."));
    }
    let ratio = unit_ratio(&image.geometry, target)?;
    let inverse = inverse_linear(&image.geometry.affine);
    let values = (0..target.voxel_count()?)
        .map(|i| {
            interpolate(
                &image.values,
                image.geometry.shape,
                image.geometry.ndim,
                source_position(
                    &image.geometry,
                    target,
                    position(target.shape, i),
                    &inverse,
                    ratio,
                ),
                mode,
                outside,
            )
        })
        .collect();
    let mut geometry = target.clone();
    geometry.diagnostics.push(diagnostic("resample_like",format!("Interpolation: {mode:?}; outside: {outside}; source diagnostics retained separately by caller."),"info"));
    Ok(ImageVolume { geometry, values })
}
pub fn resample_mask_like(
    mask: &MaskVolume,
    target: &Geometry,
    outside: u32,
) -> RadResult<MaskVolume> {
    mask.validate()?;
    target.validate()?;
    if mask.geometry.ndim != target.ndim {
        return Err(error(
            ErrorCode::Geometry,
            "Source and target dimensionality differ.",
        ));
    }
    let ratio = unit_ratio(&mask.geometry, target)?;
    let inverse = inverse_linear(&mask.geometry.affine);
    let labels = (0..target.voxel_count()?)
        .map(|i| {
            let p = source_position(
                &mask.geometry,
                target,
                position(target.shape, i),
                &inverse,
                ratio,
            );
            if (0..3).any(|j| {
                !p[j].is_finite() || p[j] < -0.5 || p[j] >= mask.geometry.shape[j] as f64 - 0.5
            }) {
                outside
            } else {
                mask.labels[index(mask.geometry.shape, p.map(|v| (v + 0.5).floor() as usize))]
            }
        })
        .collect();
    Ok(MaskVolume {
        geometry: target.clone(),
        labels,
    })
}
/// Generic grid preserving the lower outer voxel edge. This is NOT PyRadiomics' ROI grid.
pub fn spacing_grid(source: &Geometry, spacing_mm: [f64; 3]) -> RadResult<Geometry> {
    source.require_orthogonal()?;
    let old = source.spacing_mm()?;
    if spacing_mm.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err(invalid(
            "Target spacings must be positive finite millimeters.",
        ));
    }
    if source.ndim == 2 && spacing_mm[2] != old[2] {
        return Err(invalid("Inactive 2D z spacing must remain unchanged."));
    }
    let mut target = source.clone();
    for j in 0..source.ndim as usize {
        let size = (source.shape[j] as f64 * old[j] / spacing_mm[j]).ceil();
        if size < 1. || size >= usize::MAX as f64 {
            return Err(invalid("Resampled grid size is invalid or exceeds usize."));
        }
        target.shape[j] = size as usize;
        let ratio = spacing_mm[j] / old[j];
        for i in 0..3 {
            target.affine[i][3] += 0.5 * source.affine[i][j] * (ratio - 1.);
            target.affine[i][j] = source.affine[i][j] * ratio;
        }
    }
    target.validate()?;
    Ok(target)
}
pub fn resample_spacing(
    image: &ImageVolume,
    spacing_mm: [f64; 3],
    mode: Interpolation,
    outside: f64,
) -> RadResult<ImageVolume> {
    resample_like(
        image,
        &spacing_grid(&image.geometry, spacing_mm)?,
        mode,
        outside,
    )
}

fn statistical_values(image: &ImageVolume, mask: Option<&MaskVolume>) -> RadResult<Vec<f64>> {
    image.validate()?;
    if let Some(m) = mask {
        m.validate()?;
        image.geometry.aligned_with(&m.geometry)?;
    }
    let values: Vec<f64> = image
        .values
        .iter()
        .enumerate()
        .filter(|(i, _)| mask.is_none_or(|m| m.labels[*i] != 0))
        .map(|(_, v)| *v)
        .collect();
    if values.is_empty() {
        return Err(error(ErrorCode::EmptyResult, "Statistical range is empty."));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Statistical range contains NaN or infinity."));
    }
    Ok(values)
}
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let index = q * (sorted.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (index - low as f64) * (sorted[high] - sorted[low])
}
pub fn clip_range(image: &ImageVolume, lower: f64, upper: f64) -> RadResult<ImageVolume> {
    image.validate()?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(invalid("Clipping endpoints must be finite and ordered."));
    }
    if image.values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Clipping input contains NaN or infinity."));
    }
    Ok(ImageVolume {
        geometry: image.geometry.clone(),
        values: image.values.iter().map(|v| v.clamp(lower, upper)).collect(),
    })
}
/// Percentiles use NumPy's linear interpolation. Stats are fitted to an explicit mask or whole image.
pub fn clip_percentile(
    image: &ImageVolume,
    lower_percentile: f64,
    upper_percentile: f64,
    statistical_mask: Option<&MaskVolume>,
) -> RadResult<ImageVolume> {
    if !lower_percentile.is_finite()
        || !upper_percentile.is_finite()
        || lower_percentile < 0.
        || upper_percentile > 100.
        || lower_percentile > upper_percentile
    {
        return Err(invalid(
            "Percentiles must satisfy 0 <= lower <= upper <= 100.",
        ));
    }
    let mut values = statistical_values(image, statistical_mask)?;
    values.sort_by(f64::total_cmp);
    clip_range(
        image,
        quantile(&values, lower_percentile / 100.),
        quantile(&values, upper_percentile / 100.),
    )
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    ZScore,
    MinMax,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntensityTransform {
    pub method: Normalization,
    pub offset: f64,
    pub scale: f64,
    pub statistical_voxels: usize,
}
impl IntensityTransform {
    pub fn apply(&self, image: &ImageVolume) -> RadResult<ImageVolume> {
        image.validate()?;
        if !self.offset.is_finite()
            || !self.scale.is_finite()
            || self.scale <= 0.
            || image.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "Invalid intensity transform or nonfinite image value.",
            ));
        }
        Ok(ImageVolume {
            geometry: image.geometry.clone(),
            values: image
                .values
                .iter()
                .map(|v| (v - self.offset) / self.scale)
                .collect(),
        })
    }
}
/// ZScore uses population variance (ddof=0), explicitly distinct from SITK Normalize.
pub fn fit_intensity_transform(
    image: &ImageVolume,
    method: Normalization,
    statistical_mask: Option<&MaskVolume>,
) -> RadResult<IntensityTransform> {
    let values = statistical_values(image, statistical_mask)?;
    let (offset, scale) = match method {
        Normalization::ZScore => {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            (
                mean,
                (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64)
                    .sqrt(),
            )
        }
        Normalization::MinMax => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min, max - min)
        }
    };
    if !offset.is_finite() || !scale.is_finite() || scale <= 0. {
        return Err(invalid("Intensity normalization is undefined for constant or numerically overflowing statistics."));
    }
    Ok(IntensityTransform {
        method,
        offset,
        scale,
        statistical_voxels: values.len(),
    })
}
/// Keep morphology unchanged in the caller; return a distinct intensity mask using an inclusive range.
pub fn resegment(
    image: &ImageVolume,
    morphology_mask: &MaskVolume,
    lower: f64,
    upper: f64,
) -> RadResult<MaskVolume> {
    image.validate()?;
    morphology_mask.validate()?;
    image.geometry.aligned_with(&morphology_mask.geometry)?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(invalid(
            "Resegmentation endpoints must be finite and ordered.",
        ));
    }
    if image
        .values
        .iter()
        .zip(&morphology_mask.labels)
        .any(|(v, m)| *m != 0 && !v.is_finite())
    {
        return Err(invalid("ROI intensities contain NaN or infinity."));
    }
    let labels: Vec<u32> = image
        .values
        .iter()
        .zip(&morphology_mask.labels)
        .map(|(&v, &m)| if v >= lower && v <= upper { m } else { 0 })
        .collect();
    if !labels.iter().any(|&x| x != 0) {
        return Err(error(
            ErrorCode::EmptyResult,
            "Resegmentation produced an empty intensity ROI.",
        ));
    }
    Ok(MaskVolume {
        geometry: morphology_mask.geometry.clone(),
        labels,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Connectivity {
    Four,
    Eight,
    Six,
    Eighteen,
    TwentySix,
}
fn neighbors(ndim: u8, connectivity: Connectivity) -> RadResult<Vec<[isize; 3]>> {
    let allowed = match (ndim, connectivity) {
        (2, Connectivity::Four) => 1,
        (2, Connectivity::Eight) => 2,
        (3, Connectivity::Six) => 1,
        (3, Connectivity::Eighteen) => 2,
        (3, Connectivity::TwentySix) => 3,
        _ => {
            return Err(invalid(
                "Connectivity must be 4/8 for 2D or 6/18/26 for 3D.",
            ))
        }
    };
    let mut offsets = vec![];
    for z in -1isize..=1 {
        for y in -1isize..=1 {
            for x in -1isize..=1 {
                let n = x.abs() + y.abs() + z.abs();
                if n > 0 && n <= allowed && (ndim == 3 || z == 0) {
                    offsets.push([x, y, z]);
                }
            }
        }
    }
    Ok(offsets)
}
fn shifted(shape: [usize; 3], p: [usize; 3], delta: [isize; 3]) -> Option<[usize; 3]> {
    let mut result = [0; 3];
    for j in 0..3 {
        result[j] = p[j].checked_add_signed(delta[j])?;
        if result[j] >= shape[j] {
            return None;
        }
    }
    Some(result)
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub component_id: u32,
    pub source_label: Option<u32>,
    pub voxel_count: usize,
    pub bounds: BoundingBox,
    pub touches_border: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Components {
    pub mask: MaskVolume,
    pub components: Vec<Component>,
}
/// Label components in stable x-fast scan order; optionally prevent joining different original labels.
pub fn connected_components(
    mask: &MaskVolume,
    connectivity: Connectivity,
    separate_labels: bool,
) -> RadResult<Components> {
    mask.validate()?;
    let offsets = neighbors(mask.geometry.ndim, connectivity)?;
    let shape = mask.geometry.shape;
    let mut labels = vec![0u32; mask.labels.len()];
    let mut components = vec![];
    let mut queue = VecDeque::new();
    for seed in 0..labels.len() {
        if mask.labels[seed] == 0 || labels[seed] != 0 {
            continue;
        }
        let component_id = u32::try_from(components.len() + 1)
            .map_err(|_| invalid("Too many connected components for u32 labels."))?;
        let source = mask.labels[seed];
        let mut count = 0;
        let mut bounds = BoundingBox {
            start: shape,
            end: [0; 3],
        };
        let mut border = false;
        queue.push_back(seed);
        labels[seed] = component_id;
        while let Some(i) = queue.pop_front() {
            let p = position(shape, i);
            count += 1;
            for j in 0..3 {
                bounds.start[j] = bounds.start[j].min(p[j]);
                bounds.end[j] = bounds.end[j].max(p[j] + 1);
                if j < mask.geometry.ndim as usize && (p[j] == 0 || p[j] + 1 == shape[j]) {
                    border = true;
                }
            }
            for &delta in &offsets {
                if let Some(next) = shifted(shape, p, delta) {
                    let ni = index(shape, next);
                    if labels[ni] == 0
                        && mask.labels[ni] != 0
                        && (!separate_labels || mask.labels[ni] == source)
                    {
                        labels[ni] = component_id;
                        queue.push_back(ni);
                    }
                }
            }
        }
        components.push(Component {
            component_id,
            source_label: separate_labels.then_some(source),
            voxel_count: count,
            bounds,
            touches_border: border,
        });
    }
    Ok(Components {
        mask: MaskVolume {
            geometry: mask.geometry.clone(),
            labels,
        },
        components,
    })
}
pub fn largest_component(mask: &MaskVolume, connectivity: Connectivity) -> RadResult<MaskVolume> {
    let result = connected_components(mask, connectivity, false)?;
    let maximum = result
        .components
        .iter()
        .map(|c| c.voxel_count)
        .max()
        .ok_or_else(|| error(ErrorCode::EmptyResult, "ROI has no connected components."))?;
    let keep = result
        .components
        .iter()
        .find(|c| c.voxel_count == maximum)
        .unwrap()
        .component_id;
    Ok(MaskVolume {
        geometry: mask.geometry.clone(),
        labels: result
            .mask
            .labels
            .iter()
            .map(|&x| u32::from(x == keep))
            .collect(),
    })
}
/// Iterated topological binary dilation. Iteration units are voxel-neighbor steps, not millimeters.
pub fn dilate_voxels(
    mask: &MaskVolume,
    iterations: usize,
    connectivity: Connectivity,
) -> RadResult<MaskVolume> {
    mask.validate()?;
    let offsets = neighbors(mask.geometry.ndim, connectivity)?;
    if !mask.labels.iter().any(|&x| x != 0) {
        return Err(error(ErrorCode::EmptyResult, "Cannot dilate an empty ROI."));
    }
    let mut labels: Vec<u32> = mask.labels.iter().map(|&x| u32::from(x != 0)).collect();
    let mut frontier: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != 0).then_some(i))
        .collect();
    for _ in 0..iterations {
        let mut next = vec![];
        for &i in &frontier {
            for &delta in &offsets {
                if let Some(p) =
                    shifted(mask.geometry.shape, position(mask.geometry.shape, i), delta)
                {
                    let ni = index(mask.geometry.shape, p);
                    if labels[ni] == 0 {
                        labels[ni] = 1;
                        next.push(ni);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    Ok(MaskVolume {
        geometry: mask.geometry.clone(),
        labels,
    })
}

// Exact separable squared Euclidean distance transform of sampled voxel centers.
fn edt_line(f: &[f64], spacing: f64) -> Vec<f64> {
    let finite: Vec<usize> = f
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.is_finite().then_some(i))
        .collect();
    if finite.is_empty() {
        return vec![f64::INFINITY; f.len()];
    }
    let a = spacing * spacing;
    let mut v = vec![0usize; finite.len()];
    let mut z = vec![0.; finite.len() + 1];
    let mut k = 0;
    v[0] = finite[0];
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for &q in finite.iter().skip(1) {
        let mut intersection;
        loop {
            let p = v[k];
            intersection = ((f[q] + a * (q as f64).powi(2)) - (f[p] + a * (p as f64).powi(2)))
                / (2. * a * (q - p) as f64);
            if intersection > z[k] || k == 0 {
                break;
            }
            k -= 1;
        }
        k += 1;
        v[k] = q;
        z[k] = intersection;
        z[k + 1] = f64::INFINITY;
    }
    let mut result = vec![0.; f.len()];
    let mut k = 0;
    for (q, out) in result.iter_mut().enumerate() {
        while z[k + 1] < (q as f64) {
            k += 1;
        }
        *out = a * (q as f64 - v[k] as f64).powi(2) + f[v[k]];
    }
    result
}
fn squared_distance_to(mask: &MaskVolume, foreground: bool, spacing: [f64; 3]) -> Vec<f64> {
    let shape = mask.geometry.shape;
    let mut values: Vec<f64> = mask
        .labels
        .iter()
        .map(|&x| {
            if (x != 0) == foreground {
                0.
            } else {
                f64::INFINITY
            }
        })
        .collect();
    for axis in 0..mask.geometry.ndim as usize {
        for start in 0..values.len() {
            let p = position(shape, start);
            if p[axis] != 0 {
                continue;
            }
            let stride = match axis {
                0 => 1,
                1 => shape[0],
                _ => shape[0] * shape[1],
            };
            let line: Vec<f64> = (0..shape[axis])
                .map(|j| values[start + j * stride])
                .collect();
            let transformed = edt_line(&line, spacing[axis]);
            for (j, &value) in transformed.iter().enumerate() {
                values[start + j * stride] = value;
            }
        }
    }
    values
}
/// Signed center-distance ring: inner < d <= outer; negative distances are inside the ROI.
/// Outside image centers are background for the inner distance. Output is clipped to the image.
pub fn peritumoral_ring(
    mask: &MaskVolume,
    inner_mm: f64,
    outer_mm: f64,
    organ_mask: Option<&MaskVolume>,
) -> RadResult<MaskVolume> {
    mask.validate()?;
    mask.geometry.require_orthogonal()?;
    let spacing = mask.geometry.spacing_mm()?;
    if !inner_mm.is_finite() || !outer_mm.is_finite() || inner_mm >= outer_mm {
        return Err(invalid(
            "Ring radii must be finite with inner_mm < outer_mm (negative radii denote inside).",
        ));
    }
    if !mask.labels.iter().any(|&x| x != 0) {
        return Err(error(
            ErrorCode::EmptyResult,
            "Cannot construct a ring around an empty ROI.",
        ));
    }
    if let Some(organ) = organ_mask {
        organ.validate()?;
        mask.geometry.aligned_with(&organ.geometry)?;
    }
    let outside = squared_distance_to(mask, true, spacing);
    let inside = if inner_mm < 0. {
        Some(squared_distance_to(mask, false, spacing))
    } else {
        None
    };
    let shape = mask.geometry.shape;
    let labels = mask
        .labels
        .iter()
        .enumerate()
        .map(|(i, &label)| {
            if organ_mask.is_some_and(|organ| organ.labels[i] == 0) {
                return 0;
            }
            let distance = if label == 0 {
                outside[i].sqrt()
            } else if let Some(inside) = &inside {
                let p = position(shape, i);
                let border = (0..mask.geometry.ndim as usize)
                    .map(|j| ((p[j] + 1).min(shape[j] - p[j]) as f64 * spacing[j]).powi(2))
                    .fold(f64::INFINITY, f64::min);
                -inside[i].min(border).sqrt()
            } else {
                f64::NEG_INFINITY
            };
            u32::from(distance > inner_mm && distance <= outer_mm)
        })
        .collect();
    Ok(MaskVolume {
        geometry: mask.geometry.clone(),
        labels,
    })
}
