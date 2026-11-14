//! Strict tile identity and train-independent tile→slide→patient summaries.
use crate::contracts::{
    unique_nonempty, FeatureDefinition, FeatureTable, NumericBuffer, QualityRecord, SampleMetadata,
    TileRecord, TileTable,
};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::{pathology_pixels::ObjectRecord, statistics};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn alignment(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Alignment, message)
}

/// Return a new table; IDs, never row count or index position, bind the features to tiles.
pub fn align_embeddings(
    table: &TileTable,
    ids: &[String],
    buffer: &NumericBuffer,
    names: &[String],
) -> RadResult<TileTable> {
    if table.features.is_some() {
        return Err(invalid(
            "TileTable already has features; call replace_embeddings explicitly.",
        ));
    }
    replace_embeddings(table, ids, buffer, names)
}
/// Explicitly replace the feature matrix in a returned copy, leaving the source table unchanged.
pub fn replace_embeddings(
    table: &TileTable,
    ids: &[String],
    buffer: &NumericBuffer,
    names: &[String],
) -> RadResult<TileTable> {
    table.validate()?;
    buffer.validate()?;
    unique_nonempty(ids, "Embedding tile IDs")?;
    unique_nonempty(names, "Embedding feature names")?;
    if buffer.rows != ids.len() || buffer.cols != names.len() || names.is_empty() {
        return Err(invalid("Embedding dimensions, names and row IDs disagree."));
    }
    let rows: BTreeMap<_, _> = ids.iter().enumerate().map(|(i, id)| (id, i)).collect();
    if rows.keys().map(|v| v.as_str()).collect::<BTreeSet<_>>()
        != table.tiles.iter().map(|t| t.tile_id.as_str()).collect()
    {
        return Err(alignment("Embedding and tile ID sets differ."));
    }
    let mut values = Vec::with_capacity(buffer.values.len());
    let mut missing = Vec::with_capacity(buffer.missing.len());
    for tile in &table.tiles {
        let i = rows[&tile.tile_id];
        let range = i * buffer.cols..(i + 1) * buffer.cols;
        values.extend_from_slice(&buffer.values[range.clone()]);
        missing.extend_from_slice(&buffer.missing[range]);
    }
    let mut result = table.clone();
    result.features = Some(NumericBuffer {
        rows: table.tiles.len(),
        cols: names.len(),
        values,
        missing,
    });
    result.feature_names = names.to_vec();
    result.validate()?;
    Ok(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryStatistic {
    Mean,
    Std,
    Max,
    Q25,
    Q50,
    Q75,
}
pub const ALL_STATISTICS: [SummaryStatistic; 6] = [
    SummaryStatistic::Mean,
    SummaryStatistic::Std,
    SummaryStatistic::Max,
    SummaryStatistic::Q25,
    SummaryStatistic::Q50,
    SummaryStatistic::Q75,
];
impl SummaryStatistic {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Std => "std",
            Self::Max => "max",
            Self::Q25 => "q25",
            Self::Q50 => "q50",
            Self::Q75 => "q75",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMode {
    Equal,
    Level0Area,
    PhysicalArea,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateConfig {
    pub tile_weight: WeightMode,
    pub patient_weight: WeightMode,
    pub patient_stats: Vec<SummaryStatistic>,
}
impl Default for AggregateConfig {
    fn default() -> Self {
        Self {
            tile_weight: WeightMode::Equal,
            patient_weight: WeightMode::Equal,
            patient_stats: ALL_STATISTICS.to_vec(),
        }
    }
}
impl AggregateConfig {
    pub fn validate(&self) -> RadResult<()> {
        if self.patient_weight == WeightMode::Level0Area
            || self.patient_stats.is_empty()
            || self.patient_stats.iter().collect::<BTreeSet<_>>().len() != self.patient_stats.len()
        {
            return Err(invalid("Patient weights must be equal or physical area; statistics must be nonempty and unique."));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub values: BTreeMap<SummaryStatistic, Option<f64>>,
    pub count: usize,
    pub weight_sum: f64,
    pub status: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observed {
    pub count: usize,
    pub weight_sum: f64,
}

/// Weighted quantiles use stable(value,ID) ordered sequential f64 total and prefix sums.
/// The order and exact >= comparison are part of the frozen algorithm, without epsilon ties.
pub fn summarize(
    values: &[Option<f64>],
    weights: Option<&[f64]>,
    ids: &[String],
) -> RadResult<Summary> {
    unique_nonempty(ids, "Summary row IDs")?;
    if ids.len() != values.len()
        || weights
            .is_some_and(|w| w.len() != values.len() || w.iter().any(|v| !v.is_finite() || *v < 0.))
        || values.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "Summary values, IDs and finite nonnegative weights must agree.",
        ));
    }
    let mut pairs: Vec<_> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let weight = weights.map_or(1., |w| w[i]);
            v.filter(|_| weight > 0.)
                .map(|v| (v, weight, ids[i].as_str()))
        })
        .collect();
    if pairs.is_empty() {
        return Ok(Summary {
            values: ALL_STATISTICS.into_iter().map(|s| (s, None)).collect(),
            count: 0,
            weight_sum: 0.,
            status: "empty".into(),
        });
    }
    let weight_sum = pairs.iter().map(|p| p.1).sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0. {
        return Err(invalid("Summary weight total overflows or underflows."));
    }
    let (mean, variance, quantiles) = if weights.is_none() {
        let x: Vec<_> = pairs.iter().map(|p| p.0).collect();
        let (mean, var) = statistics::population_mean_variance(&x)?;
        (
            mean,
            var,
            [
                statistics::quantile_linear(&x, 0.25)?,
                statistics::quantile_linear(&x, 0.5)?,
                statistics::quantile_linear(&x, 0.75)?,
            ],
        )
    } else {
        let mean = pairs.iter().map(|p| p.0 * p.1).sum::<f64>() / weight_sum;
        let variance = pairs
            .iter()
            .map(|p| p.1 * (p.0 - mean).powi(2))
            .sum::<f64>()
            / weight_sum;
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.2.cmp(b.2)));
        let mut cdf_total = 0.;
        for p in &pairs {
            cdf_total += p.1;
        }
        if !cdf_total.is_finite() {
            return Err(invalid("Quantile weight total overflows."));
        }
        let quantiles = [0.25, 0.5, 0.75].map(|q| {
            let threshold = q * cdf_total;
            let mut cumulative = 0.;
            for p in &pairs {
                cumulative += p.1;
                if cumulative >= threshold {
                    return p.0;
                }
            }
            pairs.last().unwrap().0
        });
        (mean, variance, quantiles)
    };
    let max = pairs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let result = [
        mean,
        variance.sqrt(),
        max,
        quantiles[0],
        quantiles[1],
        quantiles[2],
    ];
    if result.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "Summary statistics overflow; explicitly rescale input.",
        ));
    }
    Ok(Summary {
        values: ALL_STATISTICS.into_iter().zip(result.map(Some)).collect(),
        count: pairs.len(),
        weight_sum,
        status: "ok".into(),
    })
}

/// Valid half-open rectangle in level-0 pixel edges, independent of padding.
pub fn valid_bounds(tile: &TileRecord) -> RadResult<[f64; 4]> {
    tile.validate()?;
    let [left, top, w, h] = tile.valid_pixels;
    let sx = tile.width0 / tile.patch_width as f64;
    let sy = tile.height0 / tile.patch_height as f64;
    let bounds = [
        tile.x0 + left as f64 * sx,
        tile.y0 + top as f64 * sy,
        tile.x0 + (left + w) as f64 * sx,
        tile.y0 + (top + h) as f64 * sy,
    ];
    if bounds.iter().any(|v| !v.is_finite()) || bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
        return Err(invalid(
            "Valid rectangle coordinates are not representable.",
        ));
    }
    Ok(bounds)
}
pub fn area_weight(tile: &TileRecord, mode: WeightMode) -> RadResult<f64> {
    tile.validate()?;
    if mode == WeightMode::Equal {
        return Ok(1.);
    }
    let sx = tile.width0 / tile.patch_width as f64;
    let sy = tile.height0 / tile.patch_height as f64;
    let area = (sx * tile.valid_pixels[2] as f64) * (sy * tile.valid_pixels[3] as f64);
    if !area.is_finite() || area <= 0. {
        return Err(invalid("Valid patch area is not representable."));
    }
    if mode == WeightMode::Level0Area {
        return Ok(area);
    }
    let (x, y) = tile.mpp_x_um.zip(tile.mpp_y_um).ok_or_else(|| {
        invalid("Physical area weighting requires both recorded level-0 MPP axes.")
    })?;
    let physical = (area * x) * y;
    if !physical.is_finite() || physical <= 0. {
        return Err(invalid("Physical patch area is not representable."));
    }
    Ok(physical)
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlideAggregate {
    pub slide_id: String,
    pub patient_id: String,
    pub status: String,
    pub accepted_tiles: usize,
    pub physical_patch_area_sum: Option<f64>,
    pub features: BTreeMap<String, Option<f64>>,
    pub observed: BTreeMap<String, Observed>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatientAggregate {
    pub patient_id: String,
    pub status: String,
    pub accepted_slides: usize,
    pub features: BTreeMap<String, Option<f64>>,
    pub observed: BTreeMap<String, Observed>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HierarchyAggregate {
    pub slides: Vec<SlideAggregate>,
    pub patients: Vec<PatientAggregate>,
    pub tile_weight: WeightMode,
    pub patient_weight: WeightMode,
    pub slide_feature_names: Vec<String>,
    pub patient_feature_names: Vec<String>,
    pub weighted_quantile: Option<String>,
    pub weight_scope: String,
}

pub fn aggregate(table: &TileTable, config: &AggregateConfig) -> RadResult<HierarchyAggregate> {
    table.validate()?;
    config.validate()?;
    let data = table
        .features
        .as_ref()
        .ok_or_else(|| invalid("Aggregation requires explicitly aligned tile features."))?;
    let mut slides = vec![];
    let mut slide_names = vec![];
    for name in &table.feature_names {
        for statistic in ALL_STATISTICS {
            slide_names.push(format!("slide_tile_{}({name})", statistic.name()));
        }
    }
    for (slide_id, patient_id) in &table.slide_patients {
        let mut positions: Vec<_> = table
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, t)| &t.slide_id == slide_id && t.exclusion_reasons.is_empty())
            .map(|(i, _)| i)
            .collect();
        positions.sort_by_key(|&i| &table.tiles[i].tile_id);
        let mut slide = SlideAggregate {
            slide_id: slide_id.clone(),
            patient_id: patient_id.clone(),
            status: if positions.is_empty() {
                "no_accepted_tiles"
            } else {
                "ok"
            }
            .into(),
            accepted_tiles: positions.len(),
            physical_patch_area_sum: None,
            features: BTreeMap::new(),
            observed: BTreeMap::new(),
        };
        if config.patient_weight == WeightMode::PhysicalArea {
            let sum = positions
                .iter()
                .map(|&i| area_weight(&table.tiles[i], WeightMode::PhysicalArea))
                .collect::<RadResult<Vec<_>>>()?
                .iter()
                .sum::<f64>();
            if !sum.is_finite() {
                return Err(invalid("Slide physical area sum overflows."));
            }
            slide.physical_patch_area_sum = Some(sum);
        }
        let weights = if config.tile_weight == WeightMode::Equal {
            None
        } else {
            Some(
                positions
                    .iter()
                    .map(|&i| area_weight(&table.tiles[i], config.tile_weight))
                    .collect::<RadResult<Vec<_>>>()?,
            )
        };
        let ids: Vec<_> = positions
            .iter()
            .map(|&i| table.tiles[i].tile_id.clone())
            .collect();
        for (j, name) in table.feature_names.iter().enumerate() {
            let values: Vec<_> = positions
                .iter()
                .map(|&i| {
                    let index = i * data.cols + j;
                    (!data.missing[index]).then_some(data.values[index])
                })
                .collect();
            let summary = summarize(&values, weights.as_deref(), &ids)?;
            for statistic in ALL_STATISTICS {
                let column = format!("slide_tile_{}({name})", statistic.name());
                slide
                    .features
                    .insert(column.clone(), summary.values[&statistic]);
                slide.observed.insert(
                    column,
                    Observed {
                        count: summary.count,
                        weight_sum: summary.weight_sum,
                    },
                );
            }
        }
        slides.push(slide);
    }
    let patient_names: Vec<_> = slide_names
        .iter()
        .flat_map(|column| {
            config
                .patient_stats
                .iter()
                .map(move |s| format!("patient_{}({column})", s.name()))
        })
        .collect();
    let mut patients = vec![];
    for patient_id in table.slide_patients.values().collect::<BTreeSet<_>>() {
        let accepted: Vec<_> = slides
            .iter()
            .filter(|s| &s.patient_id == patient_id && s.status == "ok")
            .collect();
        let weights = if config.patient_weight == WeightMode::Equal {
            None
        } else {
            Some(
                accepted
                    .iter()
                    .map(|s| s.physical_patch_area_sum.unwrap())
                    .collect::<Vec<_>>(),
            )
        };
        let ids: Vec<_> = accepted.iter().map(|s| s.slide_id.clone()).collect();
        let mut patient = PatientAggregate {
            patient_id: patient_id.clone(),
            status: if accepted.is_empty() {
                "no_accepted_slides"
            } else {
                "ok"
            }
            .into(),
            accepted_slides: accepted.len(),
            features: BTreeMap::new(),
            observed: BTreeMap::new(),
        };
        for column in &slide_names {
            let values: Vec<_> = accepted.iter().map(|s| s.features[column]).collect();
            let summary = summarize(&values, weights.as_deref(), &ids)?;
            for &statistic in &config.patient_stats {
                let key = format!("patient_{}({column})", statistic.name());
                patient
                    .features
                    .insert(key.clone(), summary.values[&statistic]);
                patient.observed.insert(
                    key,
                    Observed {
                        count: summary.count,
                        weight_sum: summary.weight_sum,
                    },
                );
            }
        }
        patients.push(patient);
    }
    Ok(HierarchyAggregate {
        slides,
        patients,
        tile_weight: config.tile_weight,
        patient_weight: config.patient_weight,
        slide_feature_names: slide_names,
        patient_feature_names: patient_names,
        weighted_quantile: (config.tile_weight != WeightMode::Equal
            || config.patient_weight != WeightMode::Equal)
            .then(|| "inverted_cdf".into()),
        weight_scope: "sum_of_valid_patch_contributions_not_deduplicated_slide_area".into(),
    })
}
fn metadata(sample_id: &str, patient_id: &str, slide_id: Option<&str>) -> SampleMetadata {
    SampleMetadata {
        sample_id: sample_id.into(),
        patient_id: patient_id.into(),
        slide_id: slide_id.map(str::to_string),
        lesion_id: None,
        modality: Some("pathology".into()),
        region: None,
        timepoint: None,
        center: None,
    }
}
type AggregateFeatureRow<'a> = (SampleMetadata, &'a BTreeMap<String, Option<f64>>, String);
impl HierarchyAggregate {
    /// Convert missing values to the shared canonical zero+missing buffer, preserving empty rows.
    pub fn to_feature_tables(&self) -> RadResult<(FeatureTable, FeatureTable)> {
        let parameters: BTreeMap<_, _> = [
            (
                "tile_weight".into(),
                serde_json::to_value(self.tile_weight)?,
            ),
            (
                "patient_weight".into(),
                serde_json::to_value(self.patient_weight)?,
            ),
            (
                "weight_scope".into(),
                serde_json::to_value(&self.weight_scope)?,
            ),
        ]
        .into_iter()
        .collect();
        let build = |names: &[String],
                     rows: Vec<AggregateFeatureRow<'_>>|
         -> RadResult<FeatureTable> {
            unique_nonempty(names, "Aggregate feature names")?;
            let mut values = vec![];
            let mut missing = vec![];
            let mut samples = vec![];
            let mut quality = vec![];
            for (sample, features, status) in rows {
                for name in names {
                    let value = features
                        .get(name)
                        .ok_or_else(|| alignment("Aggregate row lacks a declared feature."))?;
                    values.push(value.unwrap_or(0.));
                    missing.push(value.is_none());
                }
                if status != "ok" {
                    quality.push(QualityRecord {
                        code: status.clone(),
                        message: format!("Pathology aggregation status: {status}"),
                        severity: "warning".into(),
                        sample_id: Some(sample.sample_id.clone()),
                    });
                }
                samples.push(sample);
            }
            let table = FeatureTable {
                data: NumericBuffer {
                    rows: samples.len(),
                    cols: names.len(),
                    values,
                    missing,
                },
                samples,
                features: names
                    .iter()
                    .map(|name| FeatureDefinition {
                        name: name.clone(),
                        definition:
                            "Explicit hierarchical summary; path records both aggregation stages."
                                .into(),
                        unit: None,
                    })
                    .collect(),
                effective_parameters: parameters.clone(),
                quality,
            };
            table.validate()?;
            Ok(table)
        };
        Ok((
            build(
                &self.slide_feature_names,
                self.slides
                    .iter()
                    .map(|s| {
                        (
                            metadata(&s.slide_id, &s.patient_id, Some(&s.slide_id)),
                            &s.features,
                            s.status.clone(),
                        )
                    })
                    .collect(),
            )?,
            build(
                &self.patient_feature_names,
                self.patients
                    .iter()
                    .map(|s| {
                        (
                            metadata(&s.patient_id, &s.patient_id, None),
                            &s.features,
                            s.status.clone(),
                        )
                    })
                    .collect(),
            )?,
        ))
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalObject {
    pub local: ObjectRecord,
    pub centroid0: [f64; 2],
    pub bbox0: [f64; 4],
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalObjects {
    pub local_coordinate_space: String,
    pub coordinate_space: String,
    pub objects: Vec<GlobalObject>,
}
pub fn global_object_context(
    tile: &TileRecord,
    objects: &[ObjectRecord],
) -> RadResult<GlobalObjects> {
    tile.validate()?;
    let sx = tile.width0 / tile.patch_width as f64;
    let sy = tile.height0 / tile.patch_height as f64;
    let [left, top, w, h] = tile.valid_pixels;
    let mut result = vec![];
    let mut ids = BTreeSet::new();
    for local in objects {
        let [x, y] = local.centroid_pixel_centers;
        let [bx, by, bw, bh] = local.bbox_xywh;
        if !x.is_finite()
            || !y.is_finite()
            || !ids.insert(local.object_id)
            || local.object_id == 0
            || local.source_label == 0
            || local.area_px == 0
            || x < left as f64 + 0.5
            || x > (left + w) as f64 - 0.5
            || y < top as f64 + 0.5
            || y > (top + h) as f64 - 0.5
            || bw == 0
            || bh == 0
            || bx < left
            || by < top
            || bx.checked_add(bw).is_none_or(|v| v > left + w)
            || by.checked_add(bh).is_none_or(|v| v > top + h)
        {
            return Err(invalid(
                "Local object lies outside the tile's valid pixel rectangle.",
            ));
        }
        let centroid0 = [tile.x0 + x * sx, tile.y0 + y * sy];
        let bbox0 = [
            tile.x0 + bx as f64 * sx,
            tile.y0 + by as f64 * sy,
            bw as f64 * sx,
            bh as f64 * sy,
        ];
        if centroid0.iter().chain(&bbox0).any(|v| !v.is_finite()) {
            return Err(invalid("Global object coordinates overflow."));
        }
        result.push(GlobalObject {
            local: local.clone(),
            centroid0,
            bbox0,
        });
    }
    Ok(GlobalObjects {
        local_coordinate_space: "patch_pixel_edges".into(),
        coordinate_space: "level0_pixel_edges".into(),
        objects: result,
    })
}

#[cfg(test)]
#[path = "pathology_aggregate_tests.rs"]
mod tests;
