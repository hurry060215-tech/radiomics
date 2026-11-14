//! Strict normalized WSIData geometry and generation-time embedding provenance.
//! Python supplies public-API geometry/array values; all interpretation and alignment are native.
use crate::contracts::{unique_nonempty, NumericBuffer, TileRecord, TileTable};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::pathology_aggregate;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
const SCHEMA: &str = "radiomics.wsidata.tile_manifest.v1";
fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn field<'a>(value: &'a Value, key: &str) -> RadResult<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| invalid(format!("Missing normalized WSIData field {key}.")))
}
fn array(value: &Value) -> RadResult<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("Expected normalized metadata array."))
}
fn string(value: &Value) -> RadResult<&str> {
    value
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("Nonempty string metadata required."))
}
fn exact_integer_f64(n: u64) -> bool {
    let bits = 64 - n.leading_zeros();
    bits <= 53 || n & ((1u64 << (bits - 53)) - 1) == 0
}
fn real(value: &Value) -> RadResult<f64> {
    if let Some(n) = value.as_u64() {
        if !exact_integer_f64(n) {
            return Err(invalid(
                "Integer coordinate/scale is not exactly representable as float64.",
            ));
        }
    } else if let Some(n) = value.as_i64() {
        if !exact_integer_f64(n.unsigned_abs()) {
            return Err(invalid(
                "Integer coordinate/scale is not exactly representable as float64.",
            ));
        }
    }
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid("Finite real float64 metadata required."))
}
fn integer(value: &Value, positive: bool) -> RadResult<usize> {
    let value=value.as_u64().and_then(|v|usize::try_from(v).ok()).filter(|v|!positive||*v>0).ok_or_else(||invalid("Integer metadata must be representable, nonnegative, and positive when required; booleans/floats are not integers."))?;
    Ok(value)
}
fn pair_real(value: &Value, positive: bool) -> RadResult<[f64; 2]> {
    let values = array(value)?;
    if values.len() != 2 {
        return Err(invalid("Expected two metadata scales."));
    }
    let result = [real(&values[0])?, real(&values[1])?];
    if positive && result.iter().any(|v| *v <= 0.) {
        return Err(invalid("Scales must be positive and finite."));
    }
    Ok(result)
}
fn pair_integer(value: &Value) -> RadResult<[usize; 2]> {
    let values = array(value)?;
    if values.len() != 2 {
        return Err(invalid("Expected two integer dimensions."));
    }
    Ok([integer(&values[0], true)?, integer(&values[1], true)?])
}

/// Exact Python float.hex representation, including signed zero and subnormal values.
pub fn float_hex(value: f64) -> RadResult<String> {
    if !value.is_finite() {
        return Err(invalid("Nonfinite snapshot metadata."));
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { "" } else { "-" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    Ok(if exponent == 0 && mantissa == 0 {
        format!("{sign}0x0.0p+0")
    } else if exponent == 0 {
        format!("{sign}0x0.{mantissa:013x}p-1022")
    } else {
        let power = exponent - 1023;
        format!("{sign}0x1.{mantissa:013x}p{power:+}")
    })
}
pub fn canonical(value: &Value) -> RadResult<Value> {
    match value {
        Value::Number(n) if n.is_f64() => Ok(
            json!({"float64_hex":float_hex(n.as_f64().ok_or_else(||invalid("Invalid float metadata."))?)?}),
        ),
        Value::Array(values) => Ok(Value::Array(
            values
                .iter()
                .map(canonical)
                .collect::<RadResult<Vec<_>>>()?,
        )),
        Value::Object(values) => {
            let mut out = serde_json::Map::new();
            for (k, v) in values {
                out.insert(k.clone(), canonical(v)?);
            }
            Ok(Value::Object(out))
        }
        _ => Ok(value.clone()),
    }
}
fn digest(value: &Value) -> RadResult<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical(value)?)?)
    ))
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedManifest {
    pub snapshot: Value,
    pub sha256: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileGeometry {
    pub tile_id: String,
    pub bounds0: [f64; 4],
    pub patch_size: [usize; 2],
    pub valid_pixels: [usize; 4],
}

fn rectangle_bounds(value: &Value) -> RadResult<[f64; 4]> {
    let points = array(value)?;
    if points.len() != 5 {
        return Err(invalid(
            "A closed rectangle has exactly five polygon coordinates.",
        ));
    }
    let p: Vec<_> = points
        .iter()
        .map(|p| pair_real(p, false))
        .collect::<RadResult<_>>()?;
    if p[0] != p[4] {
        return Err(invalid("Polygon must close exactly."));
    }
    let lo = [
        p[..4].iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        p[..4].iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
    ];
    let hi = [
        p[..4]
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max),
        p[..4]
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    if lo[0] == hi[0]
        || lo[1] == hi[1]
        || (0..4).any(|i| p[..i].contains(&p[i]))
        || p[..4]
            .iter()
            .any(|v| !(v[0] == lo[0] || v[0] == hi[0]) || !(v[1] == lo[1] || v[1] == hi[1]))
        || p.windows(2)
            .any(|w| (w[0][0] == w[1][0]) == (w[0][1] == w[1][1]))
    {
        return Err(invalid(
            "Actual geometry must be an axis-aligned rectangle, not its bounding approximation.",
        ));
    }
    Ok([lo[0], lo[1], hi[0], hi[1]])
}
fn nominal_span_check(
    low: f64,
    high: f64,
    scale: f64,
    translation: f64,
    actual: f64,
    expected: usize,
) -> RadResult<()> {
    if !exact_integer_f64(expected as u64) {
        return Err(invalid("Nominal extent cannot be represented as float64."));
    }
    let expected = expected as f64;
    let difference = (actual - expected).abs();
    if difference == 0. {
        return Ok(());
    }
    let gamma = 6. * f64::EPSILON / (1. - 6. * f64::EPSILON);
    let terms = [
        (scale * low).abs(),
        (scale * high).abs(),
        translation.abs(),
        translation.abs(),
        expected.abs(),
    ];
    let magnitude = terms.iter().copied().fold(0., f64::max);
    let bound = (gamma * magnitude) * terms.iter().map(|v| v / magnitude).sum::<f64>();
    if !bound.is_finite() || bound > 1e-10 * expected {
        return Err(invalid("Nominal extent cannot be reliably verified at this coordinate magnitude; re-express coordinates."));
    }
    if difference > bound {
        return Err(invalid(
            "Geometry and TileSpec extent disagree beyond declared metadata roundoff.",
        ));
    }
    Ok(())
}
type InterpretedGeometry = (
    Vec<TileGeometry>,
    Value,
    [[f64; 3]; 3],
    Option<[usize; 2]>,
    Option<[f64; 2]>,
);
fn interpret(input: &Value) -> RadResult<InterpretedGeometry> {
    let spec = field(input, "tile_spec")?;
    let raw_spec = spec
        .as_object()
        .ok_or_else(|| invalid("Recorded TileSpec must be a JSON object."))?;
    let width = integer(field(spec, "width")?, true)?;
    let height = integer(field(spec, "height")?, true)?;
    integer(field(spec, "stride_width")?, true)?;
    integer(field(spec, "stride_height")?, true)?;
    if integer(field(spec, "base_level")?, false)? != 0 {
        return Err(invalid("TileSpec base_level must be zero."));
    }
    integer(field(spec, "ops_level")?, false)?;
    let downsample = real(field(spec, "base_downsample")?)?;
    let ops = real(field(spec, "ops_downsample")?)?;
    if downsample <= 0. || ops <= 0. {
        return Err(invalid(
            "Recorded downsample values must be positive finite.",
        ));
    }
    let base = [width as f64 * downsample, height as f64 * downsample];
    if base
        .iter()
        .any(|v| !v.is_finite() || v.trunc() < 1. || *v >= usize::MAX as f64)
    {
        return Err(invalid(
            "TileSpec base extent is degenerate or unrepresentable.",
        ));
    }
    let base = base.map(|v| v.trunc() as usize);
    let transform = array(field(input, "shape_to_level0")?)?;
    if transform.len() != 3 {
        return Err(invalid(
            "Expected a 3x3 explicit shape-to-level0 transform.",
        ));
    }
    let mut matrix = [[0.; 3]; 3];
    for (i, row) in transform.iter().enumerate() {
        let row = array(row)?;
        if row.len() != 3 {
            return Err(invalid("Expected a 3x3 transform."));
        }
        for (j, v) in row.iter().enumerate() {
            matrix[i][j] = real(v)?;
        }
    }
    if matrix[2] != [0., 0., 1.]
        || matrix[0][1] != 0.
        || matrix[1][0] != 0.
        || matrix[0][0] <= 0.
        || matrix[1][1] <= 0.
    {
        return Err(invalid("An explicit positive axis-aligned transform is required; rotation, shear and unproven axes are unsupported."));
    }
    let shape_value = field(input, "slide_shape_wh")?;
    let shape = if shape_value.is_null() {
        None
    } else {
        Some(pair_integer(shape_value)?)
    };
    let mpp_value = field(input, "level0_mpp_xy")?;
    let mpp = if mpp_value.is_null() {
        None
    } else {
        Some(pair_real(mpp_value, true)?)
    };
    let rows = array(field(input, "shape_rows")?)?;
    let mut output = vec![];
    for row in rows {
        let tile_id = string(field(row, "tile_id")?)?.to_owned();
        let [x0, y0, x1, y1] = rectangle_bounds(field(row, "geometry_corners")?)?;
        let bounds = [
            matrix[0][0] * x0 + matrix[0][2],
            matrix[1][1] * y0 + matrix[1][2],
            matrix[0][0] * x1 + matrix[0][2],
            matrix[1][1] * y1 + matrix[1][2],
        ];
        if bounds.iter().any(|v| !v.is_finite() || *v < 0.)
            || bounds[2] <= bounds[0]
            || bounds[3] <= bounds[1]
        {
            return Err(invalid(
                "Level-0 rectangle is negative, nonfinite or numerically degenerate.",
            ));
        }
        nominal_span_check(
            x0,
            x1,
            matrix[0][0],
            matrix[0][2],
            bounds[2] - bounds[0],
            base[0],
        )?;
        nominal_span_check(
            y0,
            y1,
            matrix[1][1],
            matrix[1][2],
            bounds[3] - bounds[1],
            base[1],
        )?;
        let valid = if let Some(valid) = row.get("valid_pixels").filter(|v| !v.is_null()) {
            let values = array(valid)?;
            if values.len() != 4 {
                return Err(invalid("valid_pixels must contain four integers."));
            }
            [
                integer(&values[0], false)?,
                integer(&values[1], false)?,
                integer(&values[2], true)?,
                integer(&values[3], true)?,
            ]
        } else {
            if shape.is_none_or(|shape| bounds[2] > shape[0] as f64 || bounds[3] > shape[1] as f64)
            {
                return Err(invalid(
                    "Edge-padding validity is unknown; supply recorded valid_pixels.",
                ));
            }
            [0, 0, width, height]
        };
        let [left, top, w, h] = valid;
        if left.checked_add(w).is_none_or(|v| v > width)
            || top.checked_add(h).is_none_or(|v| v > height)
        {
            return Err(invalid("Explicit valid rectangle extends beyond patch."));
        }
        if let Some(shape) = shape {
            let sx = (bounds[2] - bounds[0]) / width as f64;
            let sy = (bounds[3] - bounds[1]) / height as f64;
            if bounds[0] + (left + w) as f64 * sx > shape[0] as f64
                || bounds[1] + (top + h) as f64 * sy > shape[1] as f64
            {
                return Err(invalid(
                    "Declared valid pixels include padding outside recorded slide.",
                ));
            }
        }
        output.push(TileGeometry {
            tile_id,
            bounds0: bounds,
            patch_size: [width, height],
            valid_pixels: valid,
        });
    }
    unique_nonempty(
        &output.iter().map(|r| r.tile_id.clone()).collect::<Vec<_>>(),
        "Shape tile IDs",
    )?;
    let mut normalized_spec = raw_spec.clone();
    for key in ["base_downsample", "ops_downsample", "mpp"] {
        if let Some(v) = raw_spec.get(key).filter(|v| !v.is_null()) {
            let value = real(v)?;
            if value <= 0. {
                return Err(invalid(
                    "Recorded TileSpec MPP/scales must be positive finite.",
                ));
            }
            normalized_spec.insert(key.into(), json!(value));
        }
    }
    Ok((output, Value::Object(normalized_spec), matrix, shape, mpp))
}
pub fn snapshot(input: &Value) -> RadResult<SavedManifest> {
    let tile_key = string(field(input, "tile_key")?)?;
    let source = string(field(input, "source")?)?;
    let (mut rows, spec, matrix, shape, mpp) = interpret(input)?;
    rows.sort_by(|a, b| a.tile_id.cmp(&b.tile_id));
    let payload = json!({"schema":SCHEMA,"tile_key":tile_key,"tiles":rows,"tile_spec":spec,"shape_to_level0":matrix,"slide_shape_wh":shape,"level0_mpp_xy":mpp,"source":source,"nominal_span_validation":{"operations":6,"epsilon":f64::EPSILON,"max_relative_roundoff":1e-10,"coordinate_policy":"actual coordinates retained unchanged; no raster tolerance"}});
    Ok(SavedManifest {
        sha256: digest(&payload)?,
        snapshot: payload,
    })
}
/// Return source row positions in shape order after proving historical geometry association.
pub fn validate_embedding_link(
    current: &SavedManifest,
    saved: Option<&SavedManifest>,
    table_region: &str,
    tile_key: &str,
    shape_ids: &[String],
    obs_tile_ids: &[String],
    obs_row_ids: &[String],
) -> RadResult<Vec<usize>> {
    unique_nonempty(shape_ids, "Shape tile IDs")?;
    unique_nonempty(obs_tile_ids, "Embedding tile IDs")?;
    unique_nonempty(obs_row_ids, "Embedding row IDs")?;
    if table_region != tile_key
        || obs_row_ids.len() != obs_tile_ids.len()
        || shape_ids.iter().collect::<BTreeSet<_>>() != obs_tile_ids.iter().collect()
    {
        return Err(invalid(
            "Authoritative shape/table IDs must be a region-linked bijection.",
        ));
    }
    let saved = saved
        .ok_or_else(|| invalid("Embedding lacks a generation-time geometry/TileSpec manifest."))?;
    if current.snapshot.get("schema").and_then(Value::as_str) != Some(SCHEMA)
        || saved.snapshot.get("schema").and_then(Value::as_str) != Some(SCHEMA)
    {
        return Err(RadiomicsError::new(
            ErrorCode::Schema,
            "Unknown WSIData tile manifest schema.",
        ));
    }
    if digest(&current.snapshot)? != current.sha256 {
        return Err(invalid("Current normalized snapshot digest is invalid."));
    }
    if digest(&saved.snapshot)? != saved.sha256 || current.sha256 != saved.sha256 {
        return Err(invalid("Embedding geometry or TileSpec snapshot differs."));
    }
    // Canonical float tags can also occur as opaque extra JSON objects. Digest
    // equality must not erase the original normalized metadata type distinction.
    if current.snapshot != saved.snapshot {
        return Err(invalid(
            "Embedding snapshot metadata structure, type or value differs.",
        ));
    }
    let lookup: BTreeMap<_, _> = obs_tile_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    Ok(shape_ids.iter().map(|id| lookup[id]).collect())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WsiInput {
    pub geometry: Value,
    pub saved_at_generation: Option<SavedManifest>,
    pub table_region: Value,
    pub table_region_key: String,
    pub table_instance_key: String,
    pub obs_regions: Vec<String>,
    pub obs_tile_ids: Vec<String>,
    pub obs_row_ids: Vec<String>,
    pub patient_id: String,
    pub slide_id: String,
    pub feature_key: String,
    pub feature_names: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WsiAdapted {
    pub table: TileTable,
    pub provenance: Value,
}
pub fn adapt(input: &WsiInput, buffer: &NumericBuffer) -> RadResult<WsiAdapted> {
    if input.patient_id.trim().is_empty()
        || input.slide_id.trim().is_empty()
        || input.feature_key.trim().is_empty()
    {
        return Err(invalid(
            "WSIData requires explicit patient, globally unique slide ID and feature key.",
        ));
    }
    let current = snapshot(&input.geometry)?;
    let (rows, _, _, _, mpp) = interpret(&input.geometry)?;
    let raw_ids: Vec<_> = rows.iter().map(|r| r.tile_id.clone()).collect();
    let tile_key = string(field(&input.geometry, "tile_key")?)?;
    let table_region = if let Some(region) = input.table_region.as_str() {
        region
    } else {
        let regions = array(&input.table_region)?;
        if regions.len() != 1 {
            return Err(invalid(
                "WSIData embedding table must link to one declared tile region.",
            ));
        }
        string(&regions[0])?
    };
    if input.table_region_key.trim().is_empty()
        || input.table_instance_key != "tile_id"
        || input.obs_regions.len() != input.obs_tile_ids.len()
        || input.obs_regions.iter().any(|region| region != tile_key)
    {
        return Err(invalid("Recorded TableModel region/instance keys and observation regions must prove the tile-table association."));
    }
    let row_order = validate_embedding_link(
        &current,
        input.saved_at_generation.as_ref(),
        table_region,
        tile_key,
        &raw_ids,
        &input.obs_tile_ids,
        &input.obs_row_ids,
    )?;
    let canonical_id = |id: &str| {
        serde_json::to_string(&(input.slide_id.as_str(), id)).map_err(RadiomicsError::from)
    };
    let mut tiles = vec![];
    let mut mapping = BTreeMap::new();
    for row in rows {
        let id = canonical_id(&row.tile_id)?;
        mapping.insert(row.tile_id, id.clone());
        let [x0, y0, x1, y1] = row.bounds0;
        tiles.push(TileRecord {
            patient_id: input.patient_id.clone(),
            slide_id: input.slide_id.clone(),
            tile_id: id,
            x0,
            y0,
            width0: x1 - x0,
            height0: y1 - y0,
            patch_width: row.patch_size[0],
            patch_height: row.patch_size[1],
            valid_pixels: row.valid_pixels,
            mpp_x_um: mpp.map(|p| p[0]),
            mpp_y_um: mpp.map(|p| p[1]),
            source: string(field(&input.geometry, "source")?)?.into(),
            qc: BTreeMap::new(),
            exclusion_reasons: vec![],
        });
    }
    let table = TileTable {
        tiles,
        features: None,
        feature_names: vec![],
        slide_status: [(
            input.slide_id.clone(),
            if raw_ids.is_empty() {
                "no_tiles"
            } else {
                "provided"
            }
            .into(),
        )]
        .into_iter()
        .collect(),
        slide_patients: [(input.slide_id.clone(), input.patient_id.clone())]
            .into_iter()
            .collect(),
    };
    let ids = input
        .obs_tile_ids
        .iter()
        .map(|id| canonical_id(id))
        .collect::<RadResult<Vec<_>>>()?;
    let table = pathology_aggregate::align_embeddings(&table, &ids, buffer, &input.feature_names)?;
    let source_rows: Vec<_> = row_order
        .iter()
        .map(|&i| input.obs_row_ids[i].clone())
        .collect();
    Ok(WsiAdapted {
        table,
        provenance: json!({"adapter":"from_wsidata","association":"verified_generation_manifest","cache_verified":true,"geometry_manifest":current,"feature_key":input.feature_key,"tile_key":tile_key,"raw_to_canonical_tile_ids":mapping,"source_row_ids_in_shape_order":source_rows,"coordinate_space":"level0_pixel_edges","source_schema_audit_revision":"wsidata:0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286"}),
    })
}

#[cfg(test)]
#[path = "wsidata_contract_tests.rs"]
mod tests;
