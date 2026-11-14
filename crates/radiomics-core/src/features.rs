//! Original first-order, basic shape and distance-one texture features.
//!
//! `python_golden_v1` follows PyRadiomics 3.0.1, including non-contiguous
//! discretized gray labels and the distinction between direction averaging and
//! count-matrix merging. No Python code or external extraction engine is called.
use crate::contracts::{
    Discretization, ExtractionConfig, ImageVolume, MaskVolume, SpatialUnit, TextureAggregation,
};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: &str) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}

/// A compact matrix with explicit axis values; `counts` is row-major.
/// Missing gray labels are not silently renumbered to consecutive values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountMatrix {
    pub row_values: Vec<u32>,
    pub column_values: Vec<u32>,
    pub counts: Vec<u64>,
}
impl CountMatrix {
    fn from_entries(entries: BTreeMap<(u32, u32), u64>, rows: &[u32]) -> Self {
        let column_values: Vec<u32> = entries
            .keys()
            .map(|(_, c)| *c)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let counts = rows
            .iter()
            .flat_map(|r| {
                column_values
                    .iter()
                    .map(|c| *entries.get(&(*r, *c)).unwrap_or(&0))
            })
            .collect();
        Self {
            row_values: rows.to_vec(),
            column_values,
            counts,
        }
    }
    fn total(&self) -> f64 {
        self.counts.iter().sum::<u64>() as f64
    }
    fn entries(&self) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        self.row_values
            .iter()
            .enumerate()
            .flat_map(move |(r, &rv)| {
                self.column_values.iter().enumerate().map(move |(c, &cv)| {
                    (rv, cv, self.counts[r * self.column_values.len() + c] as f64)
                })
            })
    }
    fn merged(matrices: &[DirectionalMatrix]) -> Self {
        let rows = matrices
            .first()
            .map_or(&[][..], |m| m.matrix.row_values.as_slice());
        let mut entries = BTreeMap::new();
        for matrix in matrices {
            for (r, c, n) in matrix.matrix.entries() {
                if n > 0.0 {
                    *entries.entry((r, c)).or_insert(0) += n as u64;
                }
            }
        }
        Self::from_entries(entries, rows)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectionalMatrix {
    /// x,y,z offset in voxel indices. The sign is not a physical orientation.
    pub offset: [i32; 3],
    pub matrix: CountMatrix,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NgtdmRow {
    pub gray_level: u32,
    pub count: u64,
    pub difference_sum: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureMatrices {
    pub glcm: Vec<DirectionalMatrix>,
    pub glrlm: Vec<DirectionalMatrix>,
    pub glszm: CountMatrix,
    pub gldm: CountMatrix,
    pub ngtdm: Vec<NgtdmRow>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscretizedImage {
    pub shape: [usize; 3],
    /// Zero outside ROI, positive labels inside; x-fastest order.
    pub values: Vec<u32>,
    pub gray_levels: Vec<u32>,
}

fn validate_pair(image: &ImageVolume, mask: &MaskVolume, roi_label: u32) -> RadResult<()> {
    image.validate()?;
    mask.validate()?;
    image.geometry.aligned_with(&mask.geometry)?;
    if !mask.labels.contains(&roi_label) {
        return Err(RadiomicsError::new(
            ErrorCode::EmptyResult,
            "Requested ROI is empty or absent.",
        ));
    }
    if image
        .values
        .iter()
        .zip(&mask.labels)
        .any(|(&v, &m)| m == roi_label && !v.is_finite())
    {
        return Err(invalid("ROI intensities must be finite."));
    }
    Ok(())
}

/// Freeze binning before computing any texture. Fixed-width bins are aligned to
/// zero; fixed-count constant inputs follow NumPy histogram's expanded range.
pub fn discretize(
    image: &ImageVolume,
    mask: &MaskVolume,
    roi_label: u32,
    method: &Discretization,
) -> RadResult<DiscretizedImage> {
    validate_pair(image, mask, roi_label)?;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for (&v, &m) in image.values.iter().zip(&mask.labels) {
        if m == roi_label {
            minimum = minimum.min(v);
            maximum = maximum.max(v);
        }
    }
    match *method {
        Discretization::FixedWidth { width } if !width.is_finite() || width <= 0.0 => {
            return Err(invalid("Fixed bin width must be positive and finite."))
        }
        Discretization::FixedCount { count } if count == 0 || count > u32::MAX as usize => {
            return Err(invalid("Fixed bin count is outside the u32 label range."))
        }
        _ => (),
    }
    let mut values = vec![0; image.values.len()];
    let mut gray_levels = BTreeSet::new();
    for (i, (&v, &m)) in image.values.iter().zip(&mask.labels).enumerate() {
        if m != roi_label {
            continue;
        }
        let label = match *method {
            Discretization::FixedWidth { width } => {
                (v / width).floor() - (minimum / width).floor() + 1.0
            }
            Discretization::FixedCount { count } => {
                let (low, high) = if minimum == maximum {
                    (minimum - 0.5, maximum + 0.5)
                } else {
                    (minimum, maximum)
                };
                // np.histogram's last edge includes the maximum; np.digitize
                // receives that edge increased by one in PyRadiomics 3.0.1.
                (((v - low) / (high - low) * count as f64).floor() as usize).min(count - 1) as f64
                    + 1.0
            }
        };
        if !label.is_finite() || !(1.0..=u32::MAX as f64).contains(&label) {
            return Err(invalid(
                "Discretized gray label exceeds u32 range; adjust binning explicitly.",
            ));
        }
        values[i] = label as u32;
        gray_levels.insert(values[i]);
    }
    Ok(DiscretizedImage {
        shape: image.geometry.shape,
        values,
        gray_levels: gray_levels.into_iter().collect(),
    })
}

fn coordinates(index: usize, shape: [usize; 3]) -> [usize; 3] {
    [
        index % shape[0],
        (index / shape[0]) % shape[1],
        index / (shape[0] * shape[1]),
    ]
}
fn shifted(point: [usize; 3], offset: [i32; 3], shape: [usize; 3]) -> Option<usize> {
    let mut q = [0usize; 3];
    for d in 0..3 {
        let v = point[d] as i64 + i64::from(offset[d]);
        if v < 0 || v >= shape[d] as i64 {
            return None;
        }
        q[d] = v as usize;
    }
    Some(q[0] + shape[0] * (q[1] + shape[1] * q[2]))
}
/// Half-neighborhood in PyRadiomics C angle order, converted from z,y,x to x,y,z.
fn directions(shape: [usize; 3], full: bool) -> Vec<[i32; 3]> {
    let mut output = Vec::new();
    for z in (-1i32..=1).rev() {
        for y in (-1i32..=1).rev() {
            for x in (-1i32..=1).rev() {
                if [x, y, z] == [0, 0, 0] {
                    if !full {
                        return output;
                    } else {
                        continue;
                    }
                }
                if [x, y, z]
                    .iter()
                    .zip(shape)
                    .any(|(&v, n)| v.unsigned_abs() as usize >= n)
                {
                    continue;
                }
                output.push([x, y, z]);
            }
        }
    }
    output
}

/// Raw symmetric GLCM and run/zone/dependence counts. This diagnostic is also the
/// exact matrix path used for extraction; it is not an independent implementation.
pub fn texture_matrices(discrete: &DiscretizedImage, alpha: u32) -> RadResult<TextureMatrices> {
    if discrete
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).filter(|_| d > 0))
        != Some(discrete.values.len())
        || discrete.values.iter().all(|&v| v == 0)
    {
        return Err(invalid("Discretized image shape or ROI is invalid."));
    }
    let actual: Vec<_> = discrete
        .values
        .iter()
        .copied()
        .filter(|&v| v > 0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if actual != discrete.gray_levels {
        return Err(invalid(
            "Discretized image gray level metadata differ from buffer.",
        ));
    }
    let shape = discrete.shape;
    let half = directions(shape, false);
    let full = directions(shape, true);
    // Resolve gray labels once. Repeated BTreeMap lookups inside every pair
    // would dominate the distance-one kernels on ordinary CT volumes.
    let ranks: Vec<usize> = discrete
        .values
        .iter()
        .map(|&v| actual.binary_search(&v).unwrap_or(usize::MAX))
        .collect();
    let pair_size = actual
        .len()
        .checked_mul(actual.len())
        .ok_or_else(|| invalid("GLCM dimensions overflow usize; adjust binning."))?;
    let mut glcm = Vec::new();
    let mut glrlm = Vec::new();
    for &offset in &half {
        let mut counts = Vec::new();
        counts.try_reserve_exact(pair_size).map_err(|_| {
            invalid("GLCM allocation failed; increase bin width or decrease bin count.")
        })?;
        counts.resize(pair_size, 0u64);
        for (i, _) in discrete.values.iter().enumerate().filter(|(_, v)| **v > 0) {
            if let Some(j) = shifted(coordinates(i, shape), offset, shape) {
                let w = discrete.values[j];
                if w > 0 {
                    counts[ranks[i] * actual.len() + ranks[j]] += 1;
                    counts[ranks[j] * actual.len() + ranks[i]] += 1;
                }
            }
        }
        // GLCM keeps both actual gray-level axes, even if a direction has no
        // pairs for a particular gray. Empty directions are excluded downstream.
        let pair = CountMatrix {
            row_values: actual.clone(),
            column_values: actual.clone(),
            counts,
        };
        glcm.push(DirectionalMatrix {
            offset,
            matrix: pair,
        });
        let reverse = offset.map(|v| -v);
        let mut entries = BTreeMap::new();
        let mut multi_element = false;
        // Walk each entire geometric line exactly once. ROI holes terminate
        // individual runs but remain part of that line for multi-element status.
        for i in 0..discrete.values.len() {
            if shifted(coordinates(i, shape), reverse, shape).is_some() {
                continue;
            }
            let mut cursor = Some(i);
            let (mut gray, mut length, mut elements) = (0, 0u32, 0usize);
            while let Some(j) = cursor {
                let value = discrete.values[j];
                if value > 0 {
                    elements += 1;
                }
                if value != gray {
                    if gray > 0 {
                        *entries.entry((gray, length)).or_insert(0) += 1;
                    }
                    gray = value;
                    length = u32::from(value > 0);
                } else if value > 0 {
                    length += 1;
                }
                cursor = shifted(coordinates(j, shape), offset, shape);
            }
            if gray > 0 {
                *entries.entry((gray, length)).or_insert(0) += 1;
            }
            multi_element |= elements > 1;
        }
        if !multi_element {
            entries.clear();
        }
        glrlm.push(DirectionalMatrix {
            offset,
            matrix: CountMatrix::from_entries(entries, &actual),
        });
    }
    let mut visited = vec![false; discrete.values.len()];
    let mut zones = BTreeMap::new();
    for (start, &gray) in discrete.values.iter().enumerate() {
        if gray == 0 || visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut size = 0u32;
        while let Some(i) = stack.pop() {
            size = size
                .checked_add(1)
                .ok_or_else(|| invalid("A texture zone exceeds u32 size."))?;
            for &offset in &full {
                if let Some(j) = shifted(coordinates(i, shape), offset, shape) {
                    if !visited[j] && discrete.values[j] == gray {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        *zones.entry((gray, size)).or_insert(0) += 1;
    }
    let mut dependencies = BTreeMap::new();
    let mut differences: BTreeMap<u32, (u64, f64)> =
        actual.iter().map(|&g| (g, (0, 0.0))).collect();
    for (i, &gray) in discrete.values.iter().enumerate().filter(|(_, v)| **v > 0) {
        let (mut dependence, mut count, mut sum) = (1u32, 0u32, 0.0);
        for &offset in &full {
            if let Some(j) = shifted(coordinates(i, shape), offset, shape) {
                let other = discrete.values[j];
                if other > 0 {
                    count += 1;
                    sum += f64::from(other);
                    if gray.abs_diff(other) <= alpha {
                        dependence += 1;
                    }
                }
            }
        }
        *dependencies.entry((gray, dependence)).or_insert(0) += 1;
        // PyRadiomics C counts isolated centers with zero difference, despite
        // the prose definition saying at least one neighbor is required.
        let entry = differences.get_mut(&gray).unwrap();
        entry.0 += 1;
        if count > 0 {
            entry.1 += (f64::from(gray) - sum / f64::from(count)).abs();
        }
    }
    Ok(TextureMatrices {
        glcm,
        glrlm,
        glszm: CountMatrix::from_entries(zones, &actual),
        gldm: CountMatrix::from_entries(dependencies, &actual),
        ngtdm: differences
            .into_iter()
            .map(|(gray_level, (count, difference_sum))| NgtdmRow {
                gray_level,
                count,
                difference_sum,
            })
            .collect(),
    })
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * p;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}
fn entropy(probabilities: impl Iterator<Item = f64>) -> f64 {
    -probabilities
        .map(|p| p * (p + f64::EPSILON).log2())
        .sum::<f64>()
}
fn first_order(values: &[f64], discrete: &DiscretizedImage, shift: f64) -> BTreeMap<String, f64> {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let moments = |p: i32| values.iter().map(|v| (v - mean).powi(p)).sum::<f64>() / n;
    let variance = moments(2);
    let energy = values.iter().map(|v| (v + shift).powi(2)).sum::<f64>();
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut counts = BTreeMap::new();
    for &gray in &discrete.values {
        if gray > 0 {
            *counts.entry(gray).or_insert(0usize) += 1;
        }
    }
    [
        ("Mean", mean),
        ("Median", quantile(&sorted, 0.5)),
        ("Minimum", sorted[0]),
        ("Maximum", sorted[sorted.len() - 1]),
        ("10Percentile", quantile(&sorted, 0.1)),
        ("90Percentile", quantile(&sorted, 0.9)),
        ("Range", sorted[sorted.len() - 1] - sorted[0]),
        (
            "InterquartileRange",
            quantile(&sorted, 0.75) - quantile(&sorted, 0.25),
        ),
        ("Variance", variance),
        (
            "MeanAbsoluteDeviation",
            values.iter().map(|v| (v - mean).abs()).sum::<f64>() / n,
        ),
        (
            "Skewness",
            if variance == 0.0 {
                0.0
            } else {
                moments(3) / variance.powf(1.5)
            },
        ),
        (
            "Kurtosis",
            if variance == 0.0 {
                0.0
            } else {
                moments(4) / variance.powi(2)
            },
        ),
        ("Energy", energy),
        ("RootMeanSquared", (energy / n).sqrt()),
        ("Entropy", entropy(counts.values().map(|&c| c as f64 / n))),
        (
            "Uniformity",
            counts.values().map(|&c| (c as f64 / n).powi(2)).sum(),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (format!("original_firstorder_{k}"), v))
    .collect()
}

fn glcm_features(matrix: &CountMatrix) -> Vec<(&'static str, f64)> {
    let total = matrix.total();
    if total == 0.0 {
        return glcm_features_names()
            .into_iter()
            .map(|k| (k, f64::NAN))
            .collect();
    }
    let entries: Vec<_> = matrix
        .entries()
        .map(|(i, j, n)| (f64::from(i), f64::from(j), n / total))
        .collect();
    let ux = entries.iter().map(|(i, _, p)| i * p).sum::<f64>();
    let uy = entries.iter().map(|(_, j, p)| j * p).sum::<f64>();
    let vx = entries
        .iter()
        .map(|(i, _, p)| (i - ux).powi(2) * p)
        .sum::<f64>();
    let vy = entries
        .iter()
        .map(|(_, j, p)| (j - uy).powi(2) * p)
        .sum::<f64>();
    vec![
        ("JointAverage", ux),
        ("JointEnergy", entries.iter().map(|(_, _, p)| p * p).sum()),
        ("JointEntropy", entropy(entries.iter().map(|(_, _, p)| *p))),
        (
            "MaximumProbability",
            entries.iter().map(|(_, _, p)| *p).fold(0.0, f64::max),
        ),
        (
            "Contrast",
            entries.iter().map(|(i, j, p)| (i - j).powi(2) * p).sum(),
        ),
        (
            "Correlation",
            if vx * vy == 0.0 {
                1.0
            } else {
                entries
                    .iter()
                    .map(|(i, j, p)| (i - ux) * (j - uy) * p)
                    .sum::<f64>()
                    / ((vx * vy).sqrt() + f64::EPSILON)
            },
        ),
        (
            "DifferenceAverage",
            entries.iter().map(|(i, j, p)| (i - j).abs() * p).sum(),
        ),
        (
            "Id",
            entries
                .iter()
                .map(|(i, j, p)| p / (1.0 + (i - j).abs()))
                .sum(),
        ),
        (
            "Idm",
            entries
                .iter()
                .map(|(i, j, p)| p / (1.0 + (i - j).powi(2)))
                .sum(),
        ),
        ("SumSquares", vx),
    ]
}
fn glcm_features_names() -> Vec<&'static str> {
    vec![
        "JointAverage",
        "JointEnergy",
        "JointEntropy",
        "MaximumProbability",
        "Contrast",
        "Correlation",
        "DifferenceAverage",
        "Id",
        "Idm",
        "SumSquares",
    ]
}

/// Run/zone/dependence equations share one implementation, with explicit names
/// and feature subsets. The denominator is the count of runs/zones/dependencies.
fn size_features(matrix: &CountMatrix, family: &str, voxel_count: f64) -> Vec<(&'static str, f64)> {
    let n = matrix.total();
    let mut row_sums = vec![0.0; matrix.row_values.len()];
    let mut column_sums = vec![0.0; matrix.column_values.len()];
    for (r, row) in matrix
        .counts
        .chunks(matrix.column_values.len().max(1))
        .enumerate()
    {
        for (c, &v) in row.iter().enumerate() {
            row_sums[r] += v as f64;
            column_sums[c] += v as f64;
        }
    }
    let small = matrix
        .entries()
        .map(|(_, j, v)| v / f64::from(j).powi(2))
        .sum::<f64>()
        / n;
    let large = matrix
        .entries()
        .map(|(_, j, v)| v * f64::from(j).powi(2))
        .sum::<f64>()
        / n;
    let gln = row_sums.iter().map(|v| v * v).sum::<f64>() / n;
    let sn = column_sums.iter().map(|v| v * v).sum::<f64>() / n;
    let ent = entropy(matrix.counts.iter().map(|&v| v as f64 / n));
    match family {
        "glrlm" => vec![
            ("ShortRunEmphasis", small),
            ("LongRunEmphasis", large),
            ("GrayLevelNonUniformity", gln),
            ("GrayLevelNonUniformityNormalized", gln / n),
            ("RunLengthNonUniformity", sn),
            ("RunLengthNonUniformityNormalized", sn / n),
            ("RunPercentage", n / voxel_count),
            ("RunEntropy", ent),
        ],
        "glszm" => vec![
            ("SmallAreaEmphasis", small),
            ("LargeAreaEmphasis", large),
            ("GrayLevelNonUniformity", gln),
            ("GrayLevelNonUniformityNormalized", gln / n),
            ("SizeZoneNonUniformity", sn),
            ("SizeZoneNonUniformityNormalized", sn / n),
            ("ZonePercentage", n / voxel_count),
            ("ZoneEntropy", ent),
        ],
        _ => {
            let mean_gray = matrix
                .entries()
                .map(|(i, _, v)| f64::from(i) * v / n)
                .sum::<f64>();
            let mean_size = matrix
                .entries()
                .map(|(_, j, v)| f64::from(j) * v / n)
                .sum::<f64>();
            vec![
                ("SmallDependenceEmphasis", small),
                ("LargeDependenceEmphasis", large),
                ("GrayLevelNonUniformity", gln),
                ("DependenceNonUniformity", sn),
                ("DependenceNonUniformityNormalized", sn / n),
                (
                    "GrayLevelVariance",
                    matrix
                        .entries()
                        .map(|(i, _, v)| (f64::from(i) - mean_gray).powi(2) * v / n)
                        .sum(),
                ),
                (
                    "DependenceVariance",
                    matrix
                        .entries()
                        .map(|(_, j, v)| (f64::from(j) - mean_size).powi(2) * v / n)
                        .sum(),
                ),
                ("DependenceEntropy", ent),
            ]
        }
    }
}

fn ngtdm_features(rows: &[NgtdmRow]) -> Vec<(&'static str, f64)> {
    let n = rows.iter().map(|r| r.count).sum::<u64>() as f64;
    let probabilities: Vec<_> = rows.iter().map(|r| r.count as f64 / n).collect();
    let ps = rows
        .iter()
        .zip(&probabilities)
        .map(|(r, p)| r.difference_sum * p)
        .sum::<f64>();
    let ng = rows.iter().filter(|r| r.count > 0).count() as f64;
    let (mut contrast, mut busy, mut complexity, mut strength) = (0.0, 0.0, 0.0, 0.0);
    for (i, a) in rows.iter().enumerate() {
        for (j, b) in rows.iter().enumerate() {
            if a.count == 0 || b.count == 0 {
                continue;
            }
            let (pi, pj) = (probabilities[i], probabilities[j]);
            let delta = f64::from(a.gray_level) - f64::from(b.gray_level);
            contrast += pi * pj * delta.powi(2);
            busy += (f64::from(a.gray_level) * pi - f64::from(b.gray_level) * pj).abs();
            complexity += delta.abs() * (pi * a.difference_sum + pj * b.difference_sum) / (pi + pj);
            strength += (pi + pj) * delta.powi(2);
        }
    }
    let sum_s = rows.iter().map(|r| r.difference_sum).sum::<f64>();
    vec![
        ("Coarseness", if ps == 0.0 { 1e6 } else { 1.0 / ps }),
        (
            "Contrast",
            if ng <= 1.0 {
                0.0
            } else {
                contrast * sum_s / n / (ng * (ng - 1.0))
            },
        ),
        ("Busyness", if busy == 0.0 { 0.0 } else { ps / busy }),
        ("Complexity", complexity / n),
        (
            "Strength",
            if sum_s == 0.0 { 0.0 } else { strength / sum_s },
        ),
    ]
}

fn insert_family(
    output: &mut BTreeMap<String, f64>,
    family: &str,
    values: Vec<(&'static str, f64)>,
) {
    for (name, value) in values {
        output.insert(format!("original_{family}_{name}"), value);
    }
}
fn insert_directional(
    output: &mut BTreeMap<String, f64>,
    family: &str,
    matrices: &[DirectionalMatrix],
    aggregation: TextureAggregation,
    voxel_count: f64,
) {
    let feature_fn = |m: &CountMatrix| {
        if family == "glcm" {
            glcm_features(m)
        } else {
            size_features(m, family, voxel_count)
        }
    };
    if aggregation == TextureAggregation::MergeMatrices {
        let merged = CountMatrix::merged(matrices);
        // A merged GLRLM represents one copy of each voxel for every valid
        // direction; RunPercentage's denominator must follow that multiplicity.
        let denominator = if family == "glrlm" {
            voxel_count * matrices.iter().filter(|m| m.matrix.total() > 0.0).count() as f64
        } else {
            voxel_count
        };
        let features = if family == "glrlm" {
            size_features(&merged, family, denominator)
        } else {
            feature_fn(&merged)
        };
        insert_family(output, family, features);
    } else {
        let mut sums: BTreeMap<&'static str, (f64, usize)> = BTreeMap::new();
        for matrix in matrices.iter().filter(|m| m.matrix.total() > 0.0) {
            for (name, value) in feature_fn(&matrix.matrix) {
                let state = sums.entry(name).or_insert((0.0, 0));
                if !value.is_nan() {
                    state.0 += value;
                    state.1 += 1;
                }
            }
        }
        if sums.is_empty() {
            // Preserve undefined scalar positions when all directions are empty.
            let empty = CountMatrix {
                row_values: vec![],
                column_values: vec![],
                counts: vec![],
            };
            insert_family(
                output,
                family,
                feature_fn(&empty)
                    .into_iter()
                    .map(|(name, _)| (name, f64::NAN))
                    .collect(),
            );
        } else {
            insert_family(
                output,
                family,
                sums.into_iter()
                    .map(|(name, (sum, count))| {
                        (
                            name,
                            if count == 0 {
                                f64::NAN
                            } else {
                                sum / count as f64
                            },
                        )
                    })
                    .collect(),
            );
        }
    }
}

fn shape_basic(mask: &MaskVolume, label: u32) -> BTreeMap<String, f64> {
    let mut output = BTreeMap::new();
    let shape = mask.geometry.shape;
    let mut minimum = [usize::MAX; 3];
    let mut maximum = [0usize; 3];
    let mut center = [0.0; 3];
    let mut n = 0.0;
    for (i, &m) in mask.labels.iter().enumerate() {
        if m != label {
            continue;
        }
        let point = coordinates(i, shape);
        for d in 0..3 {
            minimum[d] = minimum[d].min(point[d]);
            maximum[d] = maximum[d].max(point[d]);
            center[d] += point[d] as f64;
        }
        n += 1.0;
    }
    output.insert("original_shape_basic_VoxelCount".into(), n);
    for value in &mut center {
        *value /= n;
    }
    let world = mask.geometry.world(center);
    let unit = match mask.geometry.unit {
        SpatialUnit::Unknown => "native",
        _ => "mm",
    };
    let factor = mask.geometry.unit.millimeter_factor().unwrap_or(1.0);
    for (d, axis) in ["X", "Y", "Z"].into_iter().enumerate() {
        output.insert(
            format!("original_shape_basic_BoundingBoxMin{axis}_index"),
            minimum[d] as f64,
        );
        output.insert(
            format!("original_shape_basic_BoundingBoxSize{axis}_voxels"),
            (maximum[d] - minimum[d] + 1) as f64,
        );
        output.insert(
            format!("original_shape_basic_Centroid{axis}_{unit}"),
            world[d] * factor,
        );
    }
    // Determinant/cross-product geometry is exact even for sheared grids; no
    // Euclidean-distance algorithm or orthogonal approximation is used here.
    let a = mask.geometry.affine;
    if mask.geometry.ndim == 3 {
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        output.insert(
            format!("original_shape_basic_Volume_{unit}3"),
            n * det.abs() * factor.powi(3),
        );
    } else {
        let cross = [
            a[1][0] * a[2][1] - a[2][0] * a[1][1],
            a[2][0] * a[0][1] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[1][0] * a[0][1],
        ];
        output.insert(
            format!("original_shape_basic_Area_{unit}2"),
            n * cross.iter().map(|v| v * v).sum::<f64>().sqrt() * factor.powi(2),
        );
    }
    output
}

/// Extract only the supported Original subset. Shape uses the morphology mask;
/// first-order/texture use the optional, separately supplied intensity mask.
/// Nonfinite ROI intensities and empty masks are errors. Undefined texture
/// features remain NaN and must become explicit missing flags at table export.
pub fn extract(
    image: &ImageVolume,
    morphology_mask: &MaskVolume,
    intensity_mask: Option<&MaskVolume>,
    roi_label: u32,
    config: &ExtractionConfig,
) -> RadResult<BTreeMap<String, f64>> {
    config.validate()?;
    validate_pair(image, morphology_mask, roi_label)?;
    let mask = intensity_mask.unwrap_or(morphology_mask);
    validate_pair(image, mask, roi_label)?;
    // Resegmentation may remove ROI voxels, but may not add voxels outside the
    // morphology ROI; this catches swapped/misaligned intensity masks.
    if mask
        .labels
        .iter()
        .zip(&morphology_mask.labels)
        .any(|(&i, &m)| i == roi_label && m != roi_label)
    {
        return Err(invalid("Intensity ROI must be a subset of morphology ROI."));
    }
    let discrete = discretize(image, mask, roi_label, &config.discretization)?;
    let values: Vec<_> = image
        .values
        .iter()
        .zip(&mask.labels)
        .filter_map(|(&v, &m)| (m == roi_label).then_some(v))
        .collect();
    let matrices = texture_matrices(&discrete, config.gldm_alpha)?;
    let n = values.len() as f64;
    let mut output = first_order(&values, &discrete, config.voxel_array_shift);
    output.extend(shape_basic(morphology_mask, roi_label));
    insert_directional(&mut output, "glcm", &matrices.glcm, config.aggregation, n);
    insert_directional(&mut output, "glrlm", &matrices.glrlm, config.aggregation, n);
    insert_family(
        &mut output,
        "glszm",
        size_features(&matrices.glszm, "glszm", n),
    );
    insert_family(
        &mut output,
        "gldm",
        size_features(&matrices.gldm, "gldm", n),
    );
    insert_family(&mut output, "ngtdm", ngtdm_features(&matrices.ngtdm));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{Geometry, SpatialUnit};
    fn sample(
        shape: [usize; 3],
        values: Vec<f64>,
        mask: Vec<u32>,
        ndim: u8,
    ) -> (ImageVolume, MaskVolume) {
        let geometry = Geometry {
            shape,
            ndim,
            affine: [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            unit: SpatialUnit::Millimeter,
            diagnostics: vec![],
        };
        (
            ImageVolume {
                geometry: geometry.clone(),
                values,
            },
            MaskVolume {
                geometry,
                labels: mask,
            },
        )
    }
    fn config() -> ExtractionConfig {
        ExtractionConfig {
            profile: "python_golden_v1".into(),
            discretization: Discretization::FixedWidth { width: 1.0 },
            aggregation: TextureAggregation::DirectionMean,
            gldm_alpha: 0,
            voxel_array_shift: 0.0,
        }
    }
    fn hand() -> (ImageVolume, MaskVolume) {
        sample([3, 2, 1], vec![1., 1., 2., 2., 3., 3.], vec![1; 6], 2)
    }
    fn check(actual: f64, reference: f64) {
        assert!(
            (actual - reference).abs() <= 1e-10 + 1e-8 * reference.abs(),
            "{actual} != {reference}"
        );
    }
    #[test]
    fn frozen_hand_firstorder_and_matrices() {
        // Independent hand fixture frozen by A in 5dbbee1 before this code.
        let (image, mask) = hand();
        let output = extract(&image, &mask, None, 1, &config()).unwrap();
        for (name, value) in [
            ("Mean", 2.0),
            ("Median", 2.0),
            ("Variance", 2.0 / 3.0),
            ("Kurtosis", 1.5),
            ("Energy", 28.0),
            ("Entropy", 3f64.log2()),
            ("InterquartileRange", 1.5),
        ] {
            check(output[&format!("original_firstorder_{name}")], value);
        }
        let discrete = discretize(&image, &mask, 1, &config().discretization).unwrap();
        let m = texture_matrices(&discrete, 0).unwrap();
        assert_eq!(
            m.glcm
                .iter()
                .find(|m| m.offset == [1, 0, 0])
                .unwrap()
                .matrix
                .counts,
            vec![2, 1, 0, 1, 0, 1, 0, 1, 2]
        );
        assert_eq!(
            m.glrlm
                .iter()
                .find(|m| m.offset == [1, 0, 0])
                .unwrap()
                .matrix
                .counts,
            vec![0, 1, 2, 0, 0, 1]
        );
        assert_eq!(m.glszm.counts, vec![0, 1, 2, 0, 0, 1]);
        assert_eq!(m.gldm.counts, vec![0, 2, 2, 0, 0, 2]);
        for (row, s) in m.ngtdm.iter().zip([2.2, 2.0 / 3.0, 2.2]) {
            assert_eq!(row.count, 2);
            check(row.difference_sum, s);
        }
    }
    #[test]
    fn missing_gray_labels_and_negative_width_boundaries() {
        let (image, mask) = sample([3, 2, 1], vec![-4., -2., -1., 0., 2., 6.], vec![1; 6], 2);
        let d = discretize(&image, &mask, 1, &Discretization::FixedWidth { width: 2.0 }).unwrap();
        assert_eq!(d.values, vec![1, 2, 2, 3, 4, 6]);
        assert_eq!(d.gray_levels, vec![1, 2, 3, 4, 6]);
        let m = texture_matrices(&d, 0).unwrap();
        assert_eq!(m.gldm.row_values, vec![1, 2, 3, 4, 6]);
    }
    #[test]
    fn direction_mean_and_merged_are_distinct() {
        let (image, mask) = hand();
        let averaged = extract(&image, &mask, None, 1, &config()).unwrap();
        let mut merged_config = config();
        merged_config.aggregation = TextureAggregation::MergeMatrices;
        let merged = extract(&image, &mask, None, 1, &merged_config).unwrap();
        assert!(
            (averaged["original_glcm_JointEnergy"] - merged["original_glcm_JointEnergy"]).abs()
                > 1e-4
        );
    }
    #[test]
    fn frozen_singleton_preserves_mode_specific_undefined_values() {
        // PyRadiomics 3.0.1 reference frozen independently in 4e424ec.
        let (image, mask) = sample(
            [3, 3, 1],
            vec![0., 0., 0., 0., 7., 0., 0., 0., 0.],
            vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
            2,
        );
        let averaged = extract(&image, &mask, None, 1, &config()).unwrap();
        for (name, value) in &averaged {
            if name.starts_with("original_glcm_") || name.starts_with("original_glrlm_") {
                assert!(value.is_nan(), "{name} must remain NaN, got {value}");
            }
        }
        assert_eq!(averaged["original_ngtdm_Coarseness"], 1e6);
        let mut merged_config = config();
        merged_config.aggregation = TextureAggregation::MergeMatrices;
        let merged = extract(&image, &mask, None, 1, &merged_config).unwrap();
        assert_eq!(merged["original_glrlm_RunEntropy"], 0.0);
        assert!(merged["original_glrlm_RunPercentage"].is_nan());
    }
    #[test]
    fn empty_nonfinite_misaligned_and_resegmentation_errors() {
        let (image, mask) = hand();
        let mut empty = mask.clone();
        empty.labels.fill(0);
        assert_eq!(
            extract(&image, &empty, None, 1, &config())
                .unwrap_err()
                .code,
            ErrorCode::EmptyResult
        );
        let mut nonfinite = image.clone();
        nonfinite.values[0] = f64::NAN;
        assert!(extract(&nonfinite, &mask, None, 1, &config()).is_err());
        let mut different = mask.clone();
        different.geometry.affine[0][3] = 1.0;
        assert_eq!(
            extract(&image, &different, None, 1, &config())
                .unwrap_err()
                .code,
            ErrorCode::Alignment
        );
        let mut subset = mask.clone();
        subset.labels[0] = 0;
        let normal = extract(&image, &mask, None, 1, &config()).unwrap();
        let resegmented = extract(&image, &mask, Some(&subset), 1, &config()).unwrap();
        assert_eq!(
            normal["original_shape_basic_VoxelCount"],
            resegmented["original_shape_basic_VoxelCount"]
        );
        assert_ne!(
            normal["original_firstorder_Mean"],
            resegmented["original_firstorder_Mean"]
        );
        assert!(extract(&image, &subset, Some(&mask), 1, &config()).is_err());
    }
    #[test]
    fn exact_affine_basic_shape_and_unknown_units() {
        let (image, mask) = hand();
        let mut image = image;
        let mut mask = mask;
        let affine = [
            [-2., 0., 0., 10.],
            [0., 3., 0., -4.],
            [0., 0., 5., 2.],
            [0., 0., 0., 1.],
        ];
        image.geometry.affine = affine;
        mask.geometry.affine = affine;
        let output = extract(&image, &mask, None, 1, &config()).unwrap();
        check(output["original_shape_basic_Area_mm2"], 36.0);
        check(output["original_shape_basic_CentroidX_mm"], 8.0);
        check(output["original_shape_basic_CentroidY_mm"], -2.5);
        image.geometry.unit = SpatialUnit::Unknown;
        mask.geometry.unit = SpatialUnit::Unknown;
        let unknown = extract(&image, &mask, None, 1, &config()).unwrap();
        assert!(!unknown.contains_key("original_shape_basic_Area_mm2"));
        assert!(unknown.contains_key("original_shape_basic_Area_native2"));
    }
}
