//! Basic deterministic Lloyd KMeans with shared fitted preprocessing and RNG.
//! Cluster indices are local mathematical labels, not cross-patient tissue identities.
use crate::contracts::{unique_nonempty, NumericBuffer, TabularData};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::preprocessing::{
    fit_preprocessor, AllMissingPolicy, FittedPreprocessor, PreprocessConfig, Scaling,
};
use crate::{random, statistics};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn convergence(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Convergence, message)
}
fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> NumericBuffer {
    NumericBuffer {
        rows,
        cols,
        missing: vec![false; values.len()],
        values,
    }
}
fn complete(data: &NumericBuffer) -> RadResult<()> {
    data.validate()?;
    if data.rows == 0 || data.cols == 0 || data.missing.iter().any(|&m| m) {
        return Err(invalid("Habitat matrices must be nonempty finite complete numeric observations; missing values are not implicitly imputed."));
    }
    Ok(())
}
fn tabular(data: &NumericBuffer, names: &[String]) -> RadResult<TabularData> {
    complete(data)?;
    unique_nonempty(names, "Habitat feature names")?;
    if data.cols != names.len() {
        return Err(invalid(
            "Habitat feature names and numerical dimensions disagree.",
        ));
    }
    Ok(TabularData {
        rows: data.rows,
        numeric_names: names.to_vec(),
        numeric: data.clone(),
        categorical_names: vec![],
        categorical_values: vec![],
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum Initialization {
    Explicit {
        centers_raw: NumericBuffer,
    },
    #[default]
    KmeansPlusPlus,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HabitatConfig {
    pub clusters: usize,
    pub scaling: Scaling,
    pub initialization: Initialization,
    pub seed: u64,
    pub tolerance: f64,
    pub max_iterations: usize,
}
impl Default for HabitatConfig {
    fn default() -> Self {
        Self {
            clusters: 2,
            scaling: Scaling::Standard,
            initialization: Initialization::KmeansPlusPlus,
            seed: 0,
            tolerance: 1e-12,
            max_iterations: 300,
        }
    }
}
impl HabitatConfig {
    pub fn validate(&self) -> RadResult<()> {
        if self.clusters == 0
            || self.max_iterations == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.
        {
            return Err(invalid(
                "KMeans needs positive cluster/iteration counts and a finite positive tolerance.",
            ));
        }
        if let Initialization::Explicit { centers_raw } = &self.initialization {
            complete(centers_raw)?;
            if centers_raw.rows != self.clusters {
                return Err(invalid(
                    "Explicit raw initial centers must match the declared cluster count.",
                ));
            }
            let mut rows = BTreeSet::new();
            for row in centers_raw.values.chunks(centers_raw.cols) {
                // Positive/negative zero have equal Euclidean locations.
                let key: Vec<_> = row
                    .iter()
                    .map(|&v| if v == 0. { 0u64 } else { v.to_bits() })
                    .collect();
                if !rows.insert(key) {
                    return Err(invalid("Duplicate explicit initial centers imply an empty cluster; no hidden relocation is performed."));
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HabitatFitReport {
    pub converged: bool,
    pub iterations: usize,
    pub inertia: f64,
    pub assignment_fixed_point: bool,
    pub centroid_fixed_point_max_abs_residual: f64,
    pub absolute_center_shift_tolerance: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedHabitat {
    pub schema_version: u32,
    pub feature_names: Vec<String>,
    pub config: HabitatConfig,
    pub preprocessing: FittedPreprocessor,
    pub centers_scaled: NumericBuffer,
    pub training_ids: Vec<String>,
    pub training_content_sha256: String,
    pub rng_scheme: String,
    pub report: HabitatFitReport,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HabitatPrediction {
    pub ids: Vec<String>,
    pub labels: Vec<usize>,
    /// Direct Euclidean distances in fitted scaled space, row×cluster order.
    pub distances: NumericBuffer,
}

fn squared_distance(a: &[f64], b: &[f64]) -> RadResult<f64> {
    let value = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
    if !value.is_finite() {
        return Err(invalid("KMeans squared Euclidean distance overflow."));
    }
    Ok(value)
}
fn assign(data: &NumericBuffer, centers: &NumericBuffer) -> RadResult<(Vec<usize>, Vec<f64>)> {
    let mut labels = Vec::with_capacity(data.rows);
    let mut distances = Vec::with_capacity(data.rows);
    for row in data.values.chunks(data.cols) {
        let mut best = 0;
        let mut smallest = f64::INFINITY;
        for (cluster, center) in centers.values.chunks(centers.cols).enumerate() {
            let distance = squared_distance(row, center)?;
            if distance < smallest {
                best = cluster;
                smallest = distance;
            }
        }
        labels.push(best);
        distances.push(smallest);
    }
    Ok((labels, distances))
}
fn member_means(
    data: &NumericBuffer,
    labels: &[usize],
    clusters: usize,
) -> RadResult<NumericBuffer> {
    let mut values = Vec::with_capacity(clusters * data.cols);
    for cluster in 0..clusters {
        let members: Vec<_> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| (label == cluster).then_some(i))
            .collect();
        if members.is_empty() {
            return Err(convergence("KMeans produced an empty assigned cluster; native fitting does not relocate centers silently."));
        }
        for column in 0..data.cols {
            let observed: Vec<_> = members
                .iter()
                .map(|&i| data.values[i * data.cols + column])
                .collect();
            values.push(statistics::population_mean_variance(&observed)?.0);
        }
    }
    Ok(matrix(clusters, data.cols, values))
}
fn initial_centers(
    data: &NumericBuffer,
    names: &[String],
    scaler: &FittedPreprocessor,
    config: &HabitatConfig,
) -> RadResult<NumericBuffer> {
    if let Initialization::Explicit { centers_raw } = &config.initialization {
        return scaler.transform(&tabular(centers_raw, names)?);
    }
    let mut rng = random::component_rng(config.seed, "habitat-kmeans-plus-plus-v1", &[]);
    let first = rng.gen_range(0..data.rows);
    let mut values = data.values[first * data.cols..(first + 1) * data.cols].to_vec();
    for count in 1..config.clusters {
        let (_, distances) = assign(data, &matrix(count, data.cols, values.clone()))?;
        let total = distances.iter().sum::<f64>();
        if !total.is_finite() {
            return Err(invalid("KMeans++ squared-distance weight sum overflow."));
        }
        if total == 0. {
            return Err(convergence("Fewer distinct observed locations than requested clusters; no empty-cluster relocation is performed."));
        }
        let draw = rng.gen::<f64>() * total;
        let mut accumulated = 0.;
        let mut chosen = None;
        for (i, &distance) in distances.iter().enumerate() {
            accumulated += distance;
            if draw < accumulated {
                chosen = Some(i);
                break;
            }
        }
        let chosen = chosen
            .or_else(|| distances.iter().rposition(|&distance| distance > 0.))
            .ok_or_else(|| convergence("KMeans++ cannot choose a distinct observed center."))?;
        values.extend_from_slice(&data.values[chosen * data.cols..(chosen + 1) * data.cols]);
    }
    Ok(matrix(config.clusters, data.cols, values))
}

/// Fit only the supplied training population. IDs canonically order floating reductions/RNG choices.
pub fn fit(
    data: &NumericBuffer,
    ids: &[String],
    feature_names: &[String],
    config: &HabitatConfig,
) -> RadResult<FittedHabitat> {
    config.validate()?;
    complete(data)?;
    unique_nonempty(ids, "Habitat training IDs")?;
    if ids.len() != data.rows || config.clusters > data.rows {
        return Err(invalid(
            "Training IDs must match rows and observations must be at least the cluster count.",
        ));
    }
    let mut order: Vec<_> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| ids[a].cmp(&ids[b]));
    let mut values = Vec::with_capacity(data.values.len());
    for &i in &order {
        values.extend_from_slice(&data.values[i * data.cols..(i + 1) * data.cols]);
    }
    let training_ids: Vec<_> = order.iter().map(|&i| ids[i].clone()).collect();
    let raw = matrix(data.rows, data.cols, values);
    let raw_table = tabular(&raw, feature_names)?;
    let preprocessing = fit_preprocessor(
        &raw_table,
        &PreprocessConfig {
            scaling: config.scaling,
            all_missing: AllMissingPolicy::Error,
        },
    )?;
    let scaled = preprocessing.transform(&raw_table)?;
    let mut variances = Vec::new();
    for column in 0..scaled.cols {
        let observations: Vec<_> = (0..scaled.rows)
            .map(|i| scaled.values[i * scaled.cols + column])
            .collect();
        variances.push(statistics::population_mean_variance(&observations)?.1);
    }
    let absolute_tolerance = statistics::population_mean_variance(&variances)?.0 * config.tolerance;
    if !absolute_tolerance.is_finite() {
        return Err(invalid("Absolute KMeans center-shift tolerance overflow."));
    }
    let mut centers = initial_centers(&scaled, feature_names, &preprocessing, config)?;
    let mut previous_labels = None;
    let mut final_report = None;
    for iteration in 1..=config.max_iterations {
        let (labels, _) = assign(&scaled, &centers)?;
        let updated = member_means(&scaled, &labels, config.clusters)?;
        let shift = squared_distance(&centers.values, &updated.values)?;
        let (updated_labels, distances) = assign(&scaled, &updated)?;
        let fixed_point = updated_labels == labels;
        let stopping = previous_labels.as_ref() == Some(&labels)
            || shift <= absolute_tolerance
            || iteration == config.max_iterations;
        centers = updated;
        if stopping && fixed_point {
            let verified = member_means(&scaled, &updated_labels, config.clusters)?;
            let residual = centers
                .values
                .iter()
                .zip(&verified.values)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            let inertia = distances.iter().sum::<f64>();
            if !inertia.is_finite() {
                return Err(invalid("KMeans inertia overflow."));
            }
            final_report = Some(HabitatFitReport {
                converged: true,
                iterations: iteration,
                inertia,
                assignment_fixed_point: true,
                centroid_fixed_point_max_abs_residual: residual,
                absolute_center_shift_tolerance: absolute_tolerance,
            });
            break;
        }
        previous_labels = Some(labels);
    }
    let report=final_report.ok_or_else(||convergence("KMeans reached max_iterations without a verified nearest-assignment/centroid fixed point; no successful model is returned or saved."))?;
    let mut digest = Sha256::new();
    digest.update(b"radiomics-habitat-training-v1\0");
    digest.update(serde_json::to_vec(&(
        &training_ids,
        feature_names,
        raw.rows,
        raw.cols,
    ))?);
    for value in &raw.values {
        digest.update(value.to_le_bytes());
    }
    let fitted = FittedHabitat {
        schema_version: 1,
        feature_names: feature_names.to_vec(),
        config: config.clone(),
        preprocessing,
        centers_scaled: centers,
        training_ids,
        training_content_sha256: format!("{:x}", digest.finalize()),
        rng_scheme: random::RNG_SCHEME.into(),
        report,
    };
    fitted.validate()?;
    Ok(fitted)
}
impl FittedHabitat {
    pub fn validate(&self) -> RadResult<()> {
        if self.schema_version != 1 {
            return Err(RadiomicsError::new(
                ErrorCode::Schema,
                "Unknown habitat state schema.",
            ));
        }
        self.config.validate()?;
        if let Initialization::Explicit { centers_raw } = &self.config.initialization {
            if centers_raw.cols != self.feature_names.len() {
                return Err(invalid(
                    "Saved raw initial centers disagree with fitted feature order.",
                ));
            }
        }
        self.preprocessing.validate()?;
        complete(&self.centers_scaled)?;
        unique_nonempty(&self.feature_names, "Habitat fitted feature names")?;
        unique_nonempty(&self.training_ids, "Habitat saved training IDs")?;
        if self.feature_names.is_empty()
            || self.centers_scaled.rows != self.config.clusters
            || self.centers_scaled.cols != self.feature_names.len()
            || self.preprocessing.output_names != self.feature_names
            || !self.preprocessing.categorical.is_empty()
            || !self.preprocessing.dropped_numeric.is_empty()
            || !self.preprocessing.dropped_categorical.is_empty()
            || self
                .preprocessing
                .numeric
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                != self
                    .feature_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
            || self.preprocessing.config.scaling != self.config.scaling
            || self.preprocessing.training_rows != self.training_ids.len()
            || self.training_ids.len() < self.config.clusters
            || self.training_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || self.training_content_sha256.len() != 64
            || !self
                .training_content_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
            || self.rng_scheme != random::RNG_SCHEME
        {
            return Err(invalid("Habitat fitted feature/schema/scaling/training identity or RNG state is inconsistent."));
        }
        let report = &self.report;
        if !report.converged || !report.assignment_fixed_point {
            return Err(convergence(
                "An unverified/nonconverged habitat state cannot be used or saved.",
            ));
        }
        let largest = self
            .centers_scaled
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0., f64::max);
        if report.iterations == 0
            || report.iterations > self.config.max_iterations
            || !report.inertia.is_finite()
            || report.inertia < 0.
            || !report.centroid_fixed_point_max_abs_residual.is_finite()
            || report.centroid_fixed_point_max_abs_residual < 0.
            || report.centroid_fixed_point_max_abs_residual > 1e-10 + 1e-8 * largest
            || !report.absolute_center_shift_tolerance.is_finite()
            || report.absolute_center_shift_tolerance < 0.
        {
            return Err(invalid(
                "Habitat convergence/objective/centroid fixed-point report is invalid.",
            ));
        }
        Ok(())
    }
    pub fn predict(
        &self,
        data: &NumericBuffer,
        ids: &[String],
        feature_names: &[String],
    ) -> RadResult<HabitatPrediction> {
        self.validate()?;
        unique_nonempty(ids, "Habitat prediction IDs")?;
        if ids.len() != data.rows {
            return Err(invalid("Habitat prediction row identities disagree."));
        }
        let scaled = self
            .preprocessing
            .transform(&tabular(data, feature_names)?)?;
        let (labels, _) = assign(&scaled, &self.centers_scaled)?;
        let count = data
            .rows
            .checked_mul(self.config.clusters)
            .ok_or_else(|| invalid("Habitat prediction dimensions overflow."))?;
        let mut distances = Vec::with_capacity(count);
        for row in scaled.values.chunks(scaled.cols) {
            for center in self.centers_scaled.values.chunks(scaled.cols) {
                distances.push(squared_distance(row, center)?.sqrt());
            }
        }
        Ok(HabitatPrediction {
            ids: ids.to_vec(),
            labels,
            distances: matrix(data.rows, self.config.clusters, distances),
        })
    }
    /// Inverse-transform centers for reporting, using the same saved scaling state.
    pub fn centers_raw(&self) -> RadResult<NumericBuffer> {
        self.validate()?;
        let values = self
            .centers_scaled
            .values
            .chunks(self.centers_scaled.cols)
            .flat_map(|row| {
                row.iter()
                    .zip(&self.preprocessing.numeric)
                    .map(|(value, state)| value * state.scale + state.center)
            })
            .collect::<Vec<_>>();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Inverse-transformed habitat centers overflow."));
        }
        Ok(matrix(
            self.centers_scaled.rows,
            self.centers_scaled.cols,
            values,
        ))
    }
}
