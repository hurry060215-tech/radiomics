//! Strict keyed research transformations over the common FeatureTable contract.
use crate::contracts::{
    unique_nonempty, FeatureDefinition, FeatureTable, NumericBuffer, QualityRecord, SampleMetadata,
};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn alignment(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Alignment, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataField {
    SampleId,
    PatientId,
    SlideId,
    LesionId,
    Modality,
    Region,
    Timepoint,
    Center,
}
const METADATA: [MetadataField; 8] = [
    MetadataField::SampleId,
    MetadataField::PatientId,
    MetadataField::SlideId,
    MetadataField::LesionId,
    MetadataField::Modality,
    MetadataField::Region,
    MetadataField::Timepoint,
    MetadataField::Center,
];
impl MetadataField {
    pub fn value(self, sample: &SampleMetadata) -> Option<&str> {
        match self {
            Self::SampleId => Some(&sample.sample_id),
            Self::PatientId => Some(&sample.patient_id),
            Self::SlideId => sample.slide_id.as_deref(),
            Self::LesionId => sample.lesion_id.as_deref(),
            Self::Modality => sample.modality.as_deref(),
            Self::Region => sample.region.as_deref(),
            Self::Timepoint => sample.timepoint.as_deref(),
            Self::Center => sample.center.as_deref(),
        }
    }
    fn clear(self, sample: &mut SampleMetadata) -> RadResult<()> {
        match self {
            Self::Modality => sample.modality = None,
            Self::Region => sample.region = None,
            _ => return Err(invalid("Only declared modality/region metadata may vary.")),
        }
        Ok(())
    }
}
fn keyed(
    table: &FeatureTable,
    fields: &[MetadataField],
) -> RadResult<BTreeMap<Vec<String>, usize>> {
    table.validate()?;
    if table.samples.is_empty()
        || fields.is_empty()
        || fields.iter().collect::<BTreeSet<_>>().len() != fields.len()
    {
        return Err(invalid(
            "Nonempty observations and explicit unique metadata keys are required.",
        ));
    }
    let mut result = BTreeMap::new();
    for (i, sample) in table.samples.iter().enumerate() {
        let key = fields
            .iter()
            .map(|field| {
                field
                    .value(sample)
                    .filter(|value| !value.trim().is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        alignment("Join/pair key fields must be present nonempty strings.")
                    })
            })
            .collect::<RadResult<Vec<_>>>()?;
        if result.insert(key, i).is_some() {
            return Err(alignment("Duplicate explicit join/pair key."));
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedFeatureTable {
    pub prefix: String,
    pub table: FeatureTable,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcatConfig {
    pub key_fields: Vec<MetadataField>,
    pub varying_fields: Vec<MetadataField>,
    pub already_registered: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcatProvenance {
    pub source_sample_ids: BTreeMap<String, String>,
    pub source_metadata: BTreeMap<String, BTreeMap<MetadataField, Option<String>>>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcatenatedFeatures {
    pub table: FeatureTable,
    pub rows: Vec<ConcatProvenance>,
}

/// Exact key populations, anchor row order and source/feature order. No image registration.
pub fn concat(
    sources: &[NamedFeatureTable],
    config: &ConcatConfig,
) -> RadResult<ConcatenatedFeatures> {
    if sources.is_empty() || !config.already_registered {
        return Err(invalid(
            "Caller must explicitly declare required image registration already completed.",
        ));
    }
    unique_nonempty(
        &sources.iter().map(|s| s.prefix.clone()).collect::<Vec<_>>(),
        "Source prefixes",
    )?;
    if config.varying_fields.iter().collect::<BTreeSet<_>>().len() != config.varying_fields.len()
        || config
            .varying_fields
            .iter()
            .any(|field| !matches!(field, MetadataField::Modality | MetadataField::Region))
    {
        return Err(invalid(
            "Only unique explicit modality/region differences may be allowed.",
        ));
    }
    let lookups = sources
        .iter()
        .map(|source| keyed(&source.table, &config.key_fields))
        .collect::<RadResult<Vec<_>>>()?;
    let expected: BTreeSet<_> = lookups[0].keys().collect();
    if lookups
        .iter()
        .skip(1)
        .any(|lookup| lookup.keys().collect::<BTreeSet<_>>() != expected)
    {
        return Err(alignment(
            "Source key sets differ; no implicit inner join is performed.",
        ));
    }
    let features: Vec<_> = sources
        .iter()
        .flat_map(|source| {
            source
                .table
                .features
                .iter()
                .map(|feature| FeatureDefinition {
                    name: format!("{}::{}", source.prefix, feature.name),
                    definition: format!("Keyed source {}: {}", source.prefix, feature.definition),
                    unit: feature.unit.clone(),
                })
        })
        .collect();
    unique_nonempty(
        &features.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        "Prefixed feature names",
    )?;
    let width = features.len();
    let capacity = sources[0]
        .table
        .samples
        .len()
        .checked_mul(width)
        .ok_or_else(|| invalid("Concatenated dimensions overflow."))?;
    let mut values = Vec::with_capacity(capacity);
    let mut missing = Vec::with_capacity(capacity);
    let mut samples = Vec::with_capacity(sources[0].table.samples.len());
    let mut provenance = Vec::with_capacity(samples.capacity());
    let mut source_to_anchor: Vec<BTreeMap<String, String>> = vec![BTreeMap::new(); sources.len()];
    for anchor in &sources[0].table.samples {
        let key = config
            .key_fields
            .iter()
            .map(|field| field.value(anchor).unwrap().to_owned())
            .collect::<Vec<_>>();
        let indices = lookups
            .iter()
            .map(|lookup| lookup[&key])
            .collect::<Vec<_>>();
        let aligned = sources
            .iter()
            .zip(&indices)
            .map(|(source, &i)| &source.table.samples[i])
            .collect::<Vec<_>>();
        let mut metadata = anchor.clone();
        for field in METADATA
            .into_iter()
            .filter(|field| *field != MetadataField::SampleId)
        {
            if aligned
                .iter()
                .skip(1)
                .any(|sample| field.value(sample) != field.value(aligned[0]))
            {
                if !config.varying_fields.contains(&field) {
                    return Err(alignment(format!("Conflicting source metadata: {field:?}; missing and observed values are distinct.")));
                }
                field.clear(&mut metadata)?;
            }
        }
        let mut source_sample_ids = BTreeMap::new();
        let mut source_metadata = BTreeMap::new();
        for (source_index, (source, &index)) in sources.iter().zip(&indices).enumerate() {
            let input = &source.table;
            let sample = &input.samples[index];
            let span = index * input.data.cols..(index + 1) * input.data.cols;
            values.extend_from_slice(&input.data.values[span.clone()]);
            missing.extend_from_slice(&input.data.missing[span]);
            source_sample_ids.insert(source.prefix.clone(), sample.sample_id.clone());
            source_metadata.insert(
                source.prefix.clone(),
                config
                    .varying_fields
                    .iter()
                    .map(|field| (*field, field.value(sample).map(str::to_owned)))
                    .collect(),
            );
            source_to_anchor[source_index]
                .insert(sample.sample_id.clone(), anchor.sample_id.clone());
        }
        samples.push(metadata);
        provenance.push(ConcatProvenance {
            source_sample_ids,
            source_metadata,
        });
    }
    let mut parameters = BTreeMap::new();
    parameters.insert(
        "research_operation".into(),
        serde_json::json!("strict_keyed_concat_v1"),
    );
    parameters.insert("configuration".into(), serde_json::to_value(config)?);
    parameters.insert(
        "registration".into(),
        serde_json::json!("caller_declared_already_registered; no registration algorithm executed"),
    );
    parameters.insert("sources".into(),serde_json::Value::Array(sources.iter().map(|source|serde_json::json!({"prefix":source.prefix,"features":source.table.features,"effective_parameters":source.table.effective_parameters})).collect()));
    parameters.insert("row_provenance".into(), serde_json::to_value(&provenance)?);
    let mut quality = Vec::new();
    for (source, mapping) in sources.iter().zip(&source_to_anchor) {
        for record in &source.table.quality {
            let mut record = record.clone();
            record.message = format!("Source {}: {}", source.prefix, record.message);
            record.sample_id = record
                .sample_id
                .as_ref()
                .map(|id| mapping.get(id).unwrap_or(id).clone());
            quality.push(record);
        }
    }
    let table = FeatureTable {
        data: NumericBuffer {
            rows: samples.len(),
            cols: width,
            values,
            missing,
        },
        samples,
        features,
        effective_parameters: parameters,
        quality,
    };
    table.validate()?;
    Ok(ConcatenatedFeatures {
        table,
        rows: provenance,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ZeroDenominator {
    #[default]
    Error,
    Missing,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeltaConfig {
    pub baseline: String,
    pub followup: String,
    pub zero_denominator: ZeroDenominator,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeltaPair {
    pub key: Vec<String>,
    pub baseline_sample_id: String,
    pub followup_sample_id: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeltaFeatures {
    pub table: FeatureTable,
    pub pairs: Vec<DeltaPair>,
}
const DELTA_KEYS: [MetadataField; 4] = [
    MetadataField::PatientId,
    MetadataField::LesionId,
    MetadataField::Modality,
    MetadataField::Region,
];

/// Difference, ratio and relative change. Missing propagation precedes zero division.
pub fn delta(input: &FeatureTable, config: &DeltaConfig) -> RadResult<DeltaFeatures> {
    input.validate()?;
    if config.baseline.trim().is_empty()
        || config.followup.trim().is_empty()
        || config.baseline == config.followup
    {
        return Err(invalid(
            "Distinct explicit baseline/followup timepoints are required.",
        ));
    }
    let mut partitions: [BTreeMap<Vec<String>, usize>; 2] = Default::default();
    for (index, sample) in input.samples.iter().enumerate() {
        let part = match sample.timepoint.as_deref() {
            Some(timepoint) if timepoint == config.baseline => 0,
            Some(timepoint) if timepoint == config.followup => 1,
            _ => {
                return Err(alignment(
                    "Unexpected timepoint; explicitly select the requested pair first.",
                ))
            }
        };
        let key = DELTA_KEYS
            .iter()
            .map(|field| {
                field
                    .value(sample)
                    .filter(|value| !value.trim().is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        alignment(
                            "Delta pairing requires patient, lesion, modality and region keys.",
                        )
                    })
            })
            .collect::<RadResult<Vec<_>>>()?;
        if partitions[part].insert(key, index).is_some() {
            return Err(alignment("Duplicate baseline/followup pairing key."));
        }
    }
    if partitions[0].is_empty()
        || partitions[0].keys().collect::<BTreeSet<_>>() != partitions[1].keys().collect()
    {
        return Err(alignment("Unpaired baseline/followup keys."));
    }
    let features: Vec<_> = input
        .features
        .iter()
        .flat_map(|feature| {
            ["difference", "ratio", "relative_change"]
                .into_iter()
                .map(move |method| FeatureDefinition {
                    name: format!("{}::{method}", feature.name),
                    definition: match method {
                        "difference" => format!("followup-baseline of {}", feature.definition),
                        "ratio" => format!("followup/baseline of {}", feature.definition),
                        _ => format!(
                            "(followup-baseline)/baseline of {}; fraction, not percent",
                            feature.definition
                        ),
                    },
                    unit: if method == "difference" {
                        feature.unit.clone()
                    } else {
                        Some("dimensionless".into())
                    },
                })
        })
        .collect();
    unique_nonempty(
        &features.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        "Delta feature names",
    )?;
    let capacity = partitions[0]
        .len()
        .checked_mul(features.len())
        .ok_or_else(|| invalid("Delta dimensions overflow."))?;
    let mut values = Vec::with_capacity(capacity);
    let mut missing = Vec::with_capacity(capacity);
    let mut samples = Vec::new();
    let mut pairs = Vec::new();
    let mut quality = Vec::new();
    let mut source_to_output = BTreeMap::new();
    for (key, &before_index) in &partitions[0] {
        let after_index = partitions[1][key];
        let before = &input.samples[before_index];
        let after = &input.samples[after_index];
        if before.center != after.center || before.slide_id != after.slide_id {
            return Err(alignment(
                "Paired metadata conflict outside the declared timepoint.",
            ));
        }
        let mut metadata = before.clone();
        metadata.sample_id =
            serde_json::to_string(&("delta", key, &config.baseline, &config.followup))?;
        metadata.timepoint = Some(serde_json::to_string(&(
            &config.baseline,
            &config.followup,
        ))?);
        for j in 0..input.data.cols {
            let a = before_index * input.data.cols + j;
            let b = after_index * input.data.cols + j;
            if input.data.missing[a] || input.data.missing[b] {
                values.extend([0.; 3]);
                missing.extend([true; 3]);
                quality.push(QualityRecord { code:"delta_missing_input".into(),message:format!("{}: at least one paired value is missing; all three changes are explicitly missing.",input.features[j].name),severity:"warning".into(),sample_id:Some(metadata.sample_id.clone()) });
                continue;
            }
            let baseline = input.data.values[a];
            let followup = input.data.values[b];
            let difference = followup - baseline;
            let ratios = if baseline == 0. {
                if config.zero_denominator == ZeroDenominator::Error {
                    return Err(invalid(format!(
                        "Zero baseline denominator in {} for pairing key {key:?}.",
                        input.features[j].name
                    )));
                }
                quality.push(QualityRecord { code:"delta_zero_denominator".into(),message:format!("{}: observed difference retained; ratio and relative change are missing under explicit policy.",input.features[j].name),severity:"warning".into(),sample_id:Some(metadata.sample_id.clone()) });
                [None, None]
            } else {
                [Some(followup / baseline), Some(difference / baseline)]
            };
            for value in [Some(difference), ratios[0], ratios[1]] {
                if value.is_some_and(|v| !v.is_finite()) {
                    return Err(invalid(
                        "Delta numeric overflow; results are not silently clipped.",
                    ));
                }
                values.push(value.unwrap_or(0.));
                missing.push(value.is_none());
            }
        }
        pairs.push(DeltaPair {
            key: key.clone(),
            baseline_sample_id: before.sample_id.clone(),
            followup_sample_id: after.sample_id.clone(),
        });
        source_to_output.insert(before.sample_id.clone(), metadata.sample_id.clone());
        source_to_output.insert(after.sample_id.clone(), metadata.sample_id.clone());
        samples.push(metadata);
    }
    for record in &input.quality {
        let mut record = record.clone();
        record.sample_id = record
            .sample_id
            .as_ref()
            .map(|id| source_to_output.get(id).unwrap_or(id).clone());
        quality.push(record);
    }
    let mut parameters = BTreeMap::new();
    parameters.insert(
        "research_operation".into(),
        serde_json::json!("paired_delta_v1"),
    );
    parameters.insert("configuration".into(), serde_json::to_value(config)?);
    parameters.insert("key_fields".into(), serde_json::to_value(DELTA_KEYS)?);
    parameters.insert("pairs".into(), serde_json::to_value(&pairs)?);
    parameters.insert(
        "source_parameters".into(),
        serde_json::to_value(&input.effective_parameters)?,
    );
    parameters.insert(
        "source_features".into(),
        serde_json::to_value(&input.features)?,
    );
    let table = FeatureTable {
        data: NumericBuffer {
            rows: samples.len(),
            cols: features.len(),
            values,
            missing,
        },
        samples,
        features,
        effective_parameters: parameters,
        quality,
    };
    table.validate()?;
    Ok(DeltaFeatures { table, pairs })
}
