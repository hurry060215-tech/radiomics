//! Complete-case unpenalized binary logistic inference with separation checks.
use crate::contracts::{unique_nonempty, NumericBuffer, QualityRecord, TabularData};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use crate::models::{FitReport, ModelState, ModelTarget};
use crate::statistics::Distribution;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
fn convergence(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Convergence, message)
}
fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> NumericBuffer {
    let missing = vec![false; values.len()];
    NumericBuffer {
        rows,
        cols,
        values,
        missing,
    }
}
fn diagnostic(code: &str, message: &str) -> QualityRecord {
    QualityRecord {
        code: code.into(),
        message: message.into(),
        severity: "warning".into(),
        sample_id: None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClinicalConfig {
    pub confidence: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub rank_tolerance: f64,
}
impl Default for ClinicalConfig {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            tolerance: 1e-12,
            max_iterations: 200,
            rank_tolerance: 1e-12,
        }
    }
}
impl ClinicalConfig {
    pub fn validate(&self) -> RadResult<()> {
        if !self.confidence.is_finite()
            || self.confidence <= 0.0
            || self.confidence >= 1.0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || self.max_iterations == 0
            || !self.rank_tolerance.is_finite()
            || self.rank_tolerance <= 0.0
            || self.rank_tolerance >= 1.0
        {
            return Err(invalid(
                "Clinical confidence/rank/optimizer tolerances or iteration limit are invalid.",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparationStatus {
    Separated,
    NoSeparationDetected,
    Indeterminate,
    RankDeficient,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeparationDiagnostic {
    pub status: SeparationStatus,
    pub n_rows: usize,
    pub n_columns_including_intercept: usize,
    pub design_rank: usize,
    pub column_scales: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub scaled_beta: Vec<f64>,
    pub signed_margins: Vec<f64>,
    pub raw_solver_beta: Vec<f64>,
    pub raw_solver_margins: Vec<f64>,
    pub raw_recomputed_objective: Option<f64>,
    pub certificate_source: String,
    pub objective: Option<f64>,
    pub solver_objective: Option<f64>,
    pub objective_tolerance: f64,
    pub message: String,
}

/// A bounded LP detects the existence of a separation direction; it does not
/// classify complete versus quasi separation from one nonunique optimum.
pub fn separation_diagnostic(
    x: &NumericBuffer,
    labels: &[u8],
    rank_tolerance: f64,
) -> RadResult<SeparationDiagnostic> {
    x.validate()?;
    if x.rows == 0
        || x.missing.iter().any(|&m| m)
        || labels.len() != x.rows
        || labels.iter().any(|&y| y > 1)
        || !labels.contains(&0)
        || !labels.contains(&1)
        || !rank_tolerance.is_finite()
        || rank_tolerance <= 0.0
        || rank_tolerance >= 1.0
    {
        return Err(invalid("Separation diagnosis requires finite complete features, both binary classes and a relative rank tolerance in (0,1)."));
    }
    let d = x.cols + 1;
    let mut scales = vec![1.0; d];
    for (j, scale) in scales.iter_mut().enumerate().skip(1) {
        *scale = (0..x.rows)
            .map(|i| x.values[i * x.cols + j - 1].abs())
            .fold(0.0, f64::max);
        if *scale == 0.0 {
            *scale = 1.0;
        }
    }
    let augmented: Vec<_> = (0..x.rows)
        .flat_map(|i| {
            std::iter::once(1.0).chain((0..x.cols).map({
                let scales = &scales;
                move |j| x.values[i * x.cols + j] / scales[j + 1]
            }))
        })
        .collect();
    let svd = DMatrix::from_row_slice(x.rows, d, &augmented).svd(false, false);
    let singular_values: Vec<_> = svd.singular_values.iter().copied().collect();
    let maximum = singular_values.iter().copied().fold(0.0, f64::max);
    let rank = singular_values
        .iter()
        .filter(|&&s| s > maximum * rank_tolerance)
        .count();
    let mut output = SeparationDiagnostic {
        status: SeparationStatus::RankDeficient,
        n_rows: x.rows,
        n_columns_including_intercept: d,
        design_rank: rank,
        column_scales: scales,
        singular_values,
        scaled_beta: vec![],
        signed_margins: vec![],
        raw_solver_beta: vec![],
        raw_solver_margins: vec![],
        raw_recomputed_objective: None,
        certificate_source: "not_solved".into(),
        objective: None,
        solver_objective: None,
        objective_tolerance: x.rows as f64 * d as f64 * 1e-9,
        message:
            "Scaled augmented design is rank deficient; no unpenalized inference is permitted."
                .into(),
    };
    if rank != d {
        return Ok(output);
    }
    let signed: Vec<_> = augmented
        .chunks(d)
        .zip(labels)
        .flat_map(|(row, &label)| {
            row.iter()
                .map(move |v| v * if label == 1 { 1.0 } else { -1.0 })
        })
        .collect();
    let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Maximize);
    // beta=positive-negative, both in [0,1], is exactly beta in [-1,1].
    // This zero-start representation avoids ~1e-16 lower-bound-translation
    // artifacts for overlap. Derived beta and original margins are still
    // verified strictly below; no negative value is clipped to zero.
    let variables: Vec<_> = (0..d)
        .map(|j| {
            let coefficient = (0..x.rows).map(|i| signed[i * d + j]).sum::<f64>();
            (
                problem.add_var(coefficient, (0.0, 1.0)),
                problem.add_var(-coefficient, (0.0, 1.0)),
            )
        })
        .collect();
    for row in signed.chunks(d) {
        let constraint: Vec<_> = variables
            .iter()
            .zip(row)
            .flat_map(|(&(positive, negative), &value)| [(positive, value), (negative, -value)])
            .collect();
        problem.add_constraint(constraint.as_slice(), minilp::ComparisonOp::Ge, 0.0);
    }
    output.status = SeparationStatus::Indeterminate;
    let solution = match problem.solve() {
        Ok(solution) => solution,
        Err(error) => {
            output.message =
                format!("Separation LP did not return an optimum: {error}. Inference is refused.");
            return Ok(output);
        }
    };
    output.solver_objective = Some(solution.objective());
    output.scaled_beta = variables
        .iter()
        .map(|&(positive, negative)| solution[positive] - solution[negative])
        .collect();
    output.signed_margins = signed
        .chunks(d)
        .map(|row| {
            row.iter()
                .zip(&output.scaled_beta)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let mut objective = output.signed_margins.iter().sum::<f64>();
    output.objective = Some(objective);
    output.raw_solver_beta = output.scaled_beta.clone();
    output.raw_solver_margins = output.signed_margins.clone();
    output.raw_recomputed_objective = Some(objective);
    output.certificate_source = "solver".into();
    let bounds = output
        .scaled_beta
        .iter()
        .all(|&v| v.is_finite() && v.abs() <= 1.0 + 1e-9);
    let feasible = output
        .signed_margins
        .iter()
        .zip(augmented.chunks(d))
        .all(|(&margin, row)| {
            margin.is_finite() && margin >= -1e-9 * (1.0 + row.iter().map(|v| v.abs()).sum::<f64>())
        });
    let consistent = solution.objective().is_finite()
        && objective.is_finite()
        && (solution.objective() - objective).abs() <= output.objective_tolerance;
    if !bounds || !feasible || !consistent {
        output.message="LP certificate failed bound, row-feasibility or objective verification; inference is refused.".into();
        return Ok(output);
    }
    if solution.objective() == 0.0 {
        // The exact zero vector is a separate feasible certificate attaining
        // the solver's reported exact optimum. Preserve the raw certificate;
        // never clip its negative margins or erase a reported positive optimum.
        output.scaled_beta = vec![0.0; d];
        output.signed_margins = signed
            .chunks(d)
            .map(|row| {
                row.iter()
                    .zip(&output.scaled_beta)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        objective = output.signed_margins.iter().sum();
        if objective != 0.0 || output.signed_margins.iter().any(|&v| v != 0.0) {
            return Err(convergence(
                "The constructed zero LP certificate failed exact verification.",
            ));
        }
        output.objective = Some(objective);
        output.certificate_source = "verified_zero_optimum".into();
    }
    if output.signed_margins.iter().any(|&v| v < 0.0) {
        output.message=format!("LP has a negative recomputed margin, including within solver feasibility tolerance; objective={objective:.17e}, solver_objective={:.17e}, min_margin={:.17e}. Separation status is indeterminate and inference is refused.",solution.objective(),output.signed_margins.iter().copied().fold(f64::INFINITY,f64::min));
    } else if objective > output.objective_tolerance {
        output.status = SeparationStatus::Separated;
        output.message="Verified nonnegative-margin certificate proves a separation direction. This LP does not identify the complete/quasi subtype.".into();
    } else if objective > 0.0 || solution.objective() > 0.0 {
        output.message="A positive LP objective is below the declared sign resolution; inference is refused as indeterminate.".into();
    } else if objective == 0.0 && solution.objective() == 0.0 {
        output.status = SeparationStatus::NoSeparationDetected;
        output.message="Optimal zero margin sum: no separation direction detected within the declared resolution. A separately verified zero-optimum certificate may replace an unverifiable raw optimizer certificate, which is retained. A finite MLE still requires convergence and a valid information matrix.".into();
    } else {
        output.message="LP returned a numerically negative objective although the zero direction is feasible; inference is refused as indeterminate.".into();
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogisticInference {
    pub config: ClinicalConfig,
    pub n_input: usize,
    pub used_indices: Vec<usize>,
    pub n_used: usize,
    pub n_dropped: usize,
    pub features: Vec<String>,
    pub reference_categories: BTreeMap<String, String>,
    pub terms: Vec<String>,
    pub coefficient: Vec<f64>,
    pub covariance: NumericBuffer,
    pub information: NumericBuffer,
    pub score: Vec<f64>,
    pub scaled_score_inf: f64,
    pub standard_error: Vec<f64>,
    pub z: Vec<f64>,
    pub p_values: Vec<f64>,
    pub coefficient_ci: Vec<[f64; 2]>,
    #[serde(with = "crate::statistics::float_vec_state")]
    pub odds_ratio: Vec<f64>,
    #[serde(with = "crate::statistics::float_vec_state")]
    pub odds_ratio_ci_lower: Vec<f64>,
    #[serde(with = "crate::statistics::float_vec_state")]
    pub odds_ratio_ci_upper: Vec<f64>,
    pub apparent_probabilities: Vec<f64>,
    pub log_likelihood: f64,
    pub design_rank: usize,
    pub report: FitReport,
    pub separation: SeparationDiagnostic,
    pub diagnostics: Vec<QualityRecord>,
}
enum Feature {
    Numeric { index: usize },
    Categorical { index: usize, levels: Vec<String> },
}

impl LogisticInference {
    pub fn validate(&self) -> RadResult<()> {
        self.config.validate()?;
        self.report.validate(true)?;
        unique_nonempty(&self.features, "Clinical fitted feature names")?;
        unique_nonempty(&self.terms, "Clinical fitted terms")?;
        let d = self.terms.len();
        if d < 2
            || self.terms[0] != "const"
            || self.n_used != self.used_indices.len()
            || self.n_used.checked_add(self.n_dropped) != Some(self.n_input)
            || self.n_used <= d
            || self.used_indices.windows(2).any(|w| w[0] >= w[1])
            || self.used_indices.iter().any(|&i| i >= self.n_input)
        {
            return Err(invalid(
                "Clinical complete-case indices, counts or fitted term order are invalid.",
            ));
        }
        self.covariance.validate()?;
        self.information.validate()?;
        if self.covariance.rows != d
            || self.covariance.cols != d
            || self.information.rows != d
            || self.information.cols != d
            || self
                .covariance
                .missing
                .iter()
                .chain(&self.information.missing)
                .any(|&m| m)
        {
            return Err(invalid(
                "Clinical covariance/information dimensions or missing states are invalid.",
            ));
        }
        for values in [
            &self.coefficient,
            &self.score,
            &self.standard_error,
            &self.z,
            &self.p_values,
        ] {
            if values.len() != d || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid(
                    "Clinical coefficient-scale inference must have finite term-aligned values.",
                ));
            }
        }
        if self.coefficient_ci.len() != d
            || self
                .coefficient_ci
                .iter()
                .any(|ci| ci.iter().any(|v| !v.is_finite()) || ci[0] > ci[1])
            || self.standard_error.iter().any(|&v| v <= 0.0)
            || self.p_values.iter().any(|v| !(0.0..=1.0).contains(v))
        {
            return Err(invalid(
                "Clinical standard errors, p-values or confidence limits are invalid.",
            ));
        }
        for values in [
            &self.odds_ratio,
            &self.odds_ratio_ci_lower,
            &self.odds_ratio_ci_upper,
        ] {
            if values.len() != d || values.iter().any(|v| v.is_nan() || *v < 0.0) {
                return Err(invalid("Clinical odds-ratio states must be term aligned and nonnegative, with explicit infinity when exponentiation overflows."));
            }
        }
        if self
            .odds_ratio_ci_lower
            .iter()
            .zip(&self.odds_ratio_ci_upper)
            .any(|(a, b)| a > b)
            || self.apparent_probabilities.len() != self.n_used
            || self
                .apparent_probabilities
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !self.log_likelihood.is_finite()
        {
            return Err(invalid(
                "Clinical probabilities, odds-ratio limits or likelihood are invalid.",
            ));
        }
        if self.design_rank != d
            || self.separation.status != SeparationStatus::NoSeparationDetected
            || self.separation.design_rank != d
            || self.separation.n_rows != self.n_used
            || self.separation.n_columns_including_intercept != d
            || self.separation.objective != Some(0.0)
            || self.separation.solver_objective != Some(0.0)
        {
            return Err(convergence("Clinical inference requires full rank and a verified no-direction-detected separation diagnostic."));
        }
        if self.report.iterations > self.config.max_iterations
            || self.report.gradient_or_kkt_inf.unwrap() > self.config.tolerance
            || !self.scaled_score_inf.is_finite()
            || self.scaled_score_inf > self.config.tolerance
            || self.scaled_score_inf < 0.0
        {
            return Err(convergence("Clinical numerical gradient or independently recomputed score fails the requested convergence threshold."));
        }
        Ok(())
    }
}

pub fn logistic_inference(
    data: &TabularData,
    target: &[Option<u8>],
    features: &[String],
    reference: &BTreeMap<String, String>,
    config: &ClinicalConfig,
) -> RadResult<LogisticInference> {
    config.validate()?;
    data.validate()?;
    unique_nonempty(features, "Clinical model features")?;
    if features.is_empty() || target.len() != data.rows || target.iter().flatten().any(|&y| y > 1) {
        return Err(invalid("Clinical logistic inference requires selected features and one optional binary outcome per input row."));
    }
    if reference
        .keys()
        .any(|name| !data.categorical_names.contains(name))
    {
        return Err(invalid(
            "Reference levels can only be supplied for categorical columns.",
        ));
    }
    let mut specs = Vec::new();
    for name in features {
        if let Some(index) = data.numeric_names.iter().position(|n| n == name) {
            specs.push(Feature::Numeric { index });
        } else if let Some(index) = data.categorical_names.iter().position(|n| n == name) {
            specs.push(Feature::Categorical {
                index,
                levels: vec![],
            });
        } else {
            return Err(RadiomicsError::new(
                ErrorCode::Alignment,
                format!("Missing clinical feature '{name}'."),
            ));
        }
    }
    let used_indices: Vec<_> = (0..data.rows)
        .filter(|&i| {
            target[i].is_some()
                && specs.iter().all(|s| match s {
                    Feature::Numeric { index } => {
                        !data.numeric.missing[i * data.numeric.cols + index]
                    }
                    Feature::Categorical { index, .. } => {
                        data.categorical_values[i * data.categorical_names.len() + index].is_some()
                    }
                })
        })
        .collect();
    if used_indices.is_empty() {
        return Err(RadiomicsError::new(
            ErrorCode::EmptyResult,
            "No complete cases remain for the selected clinical model.",
        ));
    }
    let mut references = BTreeMap::new();
    let mut terms = vec!["const".into()];
    for (name, spec) in features.iter().zip(&mut specs) {
        match spec {
            Feature::Numeric { .. } => terms.push(name.clone()),
            Feature::Categorical { index, levels } => {
                let categories: BTreeSet<_> = used_indices
                    .iter()
                    .map(|&i| {
                        data.categorical_values[i * data.categorical_names.len() + *index]
                            .as_ref()
                            .unwrap()
                            .clone()
                    })
                    .collect();
                if categories.len() < 2 {
                    return Err(invalid(format!(
                        "Complete-case categorical feature '{name}' has no comparison level."
                    )));
                }
                let baseline = reference
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| categories.first().unwrap().clone());
                if !categories.contains(&baseline) {
                    return Err(invalid(format!("Reference category '{baseline}' for '{name}' is absent among complete cases.")));
                }
                references.insert(name.clone(), baseline.clone());
                *levels = categories.into_iter().filter(|c| c != &baseline).collect();
                terms.extend(levels.iter().map(|c| format!("{name}_{c}")));
            }
        }
    }
    unique_nonempty(
        &terms,
        "Clinical model terms (including intercept and categorical contrasts)",
    )?;
    let p = terms.len() - 1;
    let n = used_indices.len();
    let d = terms.len();
    if n <= d {
        return Err(invalid("Unpenalized clinical inference requires more complete cases than augmented design columns."));
    }
    let mut values = Vec::with_capacity(n * p);
    let mut labels = Vec::with_capacity(n);
    for &i in &used_indices {
        labels.push(target[i].unwrap());
        for spec in &specs {
            match spec {
                Feature::Numeric { index } => {
                    values.push(data.numeric.values[i * data.numeric.cols + index])
                }
                Feature::Categorical { index, levels } => {
                    let value = data.categorical_values[i * data.categorical_names.len() + index]
                        .as_ref()
                        .unwrap();
                    values.extend(levels.iter().map(|level| f64::from(value == level)));
                }
            }
        }
    }
    let x = matrix(n, p, values);
    let separation = separation_diagnostic(&x, &labels, config.rank_tolerance)?;
    match separation.status {
        SeparationStatus::NoSeparationDetected => (),
        SeparationStatus::RankDeficient => {
            return Err(invalid("Clinical augmented design is rank deficient."))
        }
        SeparationStatus::Separated => {
            return Err(convergence(
                "Complete or quasi separation prevents a finite unpenalized logistic MLE.",
            ))
        }
        SeparationStatus::Indeterminate => {
            return Err(convergence(format!(
                "Clinical separation diagnostic is indeterminate: {}",
                separation.message
            )))
        }
    }
    let mut means = Vec::with_capacity(p);
    let mut scales = Vec::with_capacity(p);
    for j in 0..p {
        let column: Vec<_> = (0..n).map(|i| x.values[i * p + j]).collect();
        let (mean, variance) = crate::statistics::population_mean_variance(&column)?;
        means.push(mean);
        scales.push(if variance == 0.0 {
            1.0
        } else {
            variance.sqrt()
        });
    }
    let standardized = matrix(
        n,
        p,
        x.values
            .chunks(p)
            .flat_map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (v - means[j]) / scales[j])
            })
            .collect(),
    );
    let model_target = ModelTarget::Classification {
        classes: vec!["0".into(), "1".into()],
        labels: labels.iter().map(|&y| y as usize).collect(),
    };
    let (state, report) = crate::models::solve_logistic(
        &standardized,
        &model_target,
        0.0,
        config.tolerance,
        config.max_iterations,
    )?;
    let ModelState::Logistic {
        coefficients,
        intercepts,
        ..
    } = state
    else {
        return Err(convergence(
            "Shared logistic solver returned an unexpected fitted state.",
        ));
    };
    let mut coefficient = vec![intercepts[0]];
    for j in 0..p {
        let beta = coefficients.values[j] / scales[j];
        coefficient[0] -= beta * means[j];
        coefficient.push(beta);
    }
    let mut probabilities = Vec::with_capacity(n);
    let mut information = vec![0.0; d * d];
    let mut score = vec![0.0; d];
    let mut log_likelihood = 0.0;
    for (row, &label) in x.values.chunks(p).zip(&labels) {
        let design: Vec<_> = std::iter::once(1.0).chain(row.iter().copied()).collect();
        let z = design
            .iter()
            .zip(&coefficient)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let probability = crate::models::sigmoid(z);
        probabilities.push(probability);
        log_likelihood += f64::from(label) * z - crate::models::softplus(z);
        let weight = probability * (1.0 - probability);
        for j in 0..d {
            score[j] += design[j] * (f64::from(label) - probability);
            for k in 0..d {
                information[j * d + k] += weight * design[j] * design[k];
            }
        }
    }
    if coefficient
        .iter()
        .chain(&information)
        .chain(&score)
        .chain(std::iter::once(&log_likelihood))
        .any(|v| !v.is_finite())
    {
        return Err(convergence(
            "Clinical coefficients, information, score or likelihood are not finite.",
        ));
    }
    // Recompute the score in the actual standardized fit coordinates rather
    // than trusting only the solver's own gradient report after restoring units.
    let mut scaled_score_inf = (score[0] / n as f64).abs();
    for j in 0..p {
        scaled_score_inf = scaled_score_inf
            .max(((score[j + 1] - means[j] * score[0]) / scales[j] / n as f64).abs());
    }
    if !scaled_score_inf.is_finite() || scaled_score_inf > config.tolerance {
        return Err(convergence("Independently recomputed clinical score does not meet the requested convergence tolerance."));
    }
    let column_scales = &separation.column_scales;
    let scaled_information = DMatrix::from_fn(d, d, |i, j| {
        information[i * d + j] / column_scales[i] / column_scales[j]
    });
    let singular = scaled_information.clone().svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0, f64::max);
    if singular
        .iter()
        .any(|&s| s <= maximum * config.rank_tolerance || !s.is_finite())
    {
        return Err(convergence("Observed information is numerically singular; finite unpenalized inference is not supported."));
    }
    let inverse = scaled_information
        .cholesky()
        .ok_or_else(|| convergence("Observed information is not positive definite."))?
        .inverse();
    let covariance: Vec<_> = (0..d)
        .flat_map(|i| {
            (0..d).map({
                let inverse = &inverse;
                move |j| inverse[(i, j)] / column_scales[i] / column_scales[j]
            })
        })
        .collect();
    let normal = Distribution::Normal;
    let critical = normal.quantile(0.5 + config.confidence / 2.0)?;
    let mut standard_error = Vec::with_capacity(d);
    let mut z = Vec::with_capacity(d);
    let mut p_values = Vec::with_capacity(d);
    let mut coefficient_ci = Vec::with_capacity(d);
    let mut odds_ratio = Vec::with_capacity(d);
    let mut lower = Vec::with_capacity(d);
    let mut upper = Vec::with_capacity(d);
    for j in 0..d {
        let se = covariance[j * d + j].sqrt();
        let statistic = coefficient[j] / se;
        let limits = [
            coefficient[j] - critical * se,
            coefficient[j] + critical * se,
        ];
        if !se.is_finite()
            || se <= 0.0
            || !statistic.is_finite()
            || limits.iter().any(|v| !v.is_finite())
        {
            return Err(convergence(
                "Clinical standard error, Wald statistic or coefficient limits are undefined.",
            ));
        }
        standard_error.push(se);
        z.push(statistic);
        p_values.push(2.0 * normal.sf(statistic.abs())?);
        coefficient_ci.push(limits);
        odds_ratio.push(coefficient[j].exp());
        lower.push(limits[0].exp());
        upper.push(limits[1].exp());
    }
    let mut diagnostics=vec![diagnostic("row_level_numeric_inference","This numerical kernel treats supplied complete-case rows as observations. Patient independence and analysis-unit safeguards must be enforced by the public workflow; repeated lesions are not automatically independent patients.")];
    if odds_ratio
        .iter()
        .chain(&lower)
        .chain(&upper)
        .any(|v| !v.is_finite() || *v == 0.0)
    {
        diagnostics.push(diagnostic("odds_ratio_numeric_range","Exponentiation produced explicit infinite or underflow-zero odds-ratio values; coefficient-scale inference remains available."));
    }
    let inference = LogisticInference {
        config: config.clone(),
        n_input: data.rows,
        used_indices,
        n_used: n,
        n_dropped: data.rows - n,
        features: features.to_vec(),
        reference_categories: references,
        terms,
        coefficient,
        covariance: matrix(d, d, covariance),
        information: matrix(d, d, information),
        score,
        scaled_score_inf,
        standard_error,
        z,
        p_values,
        coefficient_ci,
        odds_ratio,
        odds_ratio_ci_lower: lower,
        odds_ratio_ci_upper: upper,
        apparent_probabilities: probabilities,
        log_likelihood,
        design_rank: separation.design_rank,
        report,
        separation,
        diagnostics,
    };
    inference.validate()?;
    Ok(inference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    fn close(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 1e-10 + 1e-8 * b.abs(),
            "{a:.16e} != {b:.16e}"
        );
    }
    fn fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/statistics_reference.json"
        )))
        .unwrap()
    }
    fn clinical_data() -> (TabularData, Vec<Option<u8>>, Value) {
        let fixture = fixture();
        let clinical = &fixture["clinical_logistic"];
        let rows = clinical["rows"].as_array().unwrap();
        let mut values = Vec::new();
        let mut missing = Vec::new();
        for row in rows {
            for name in ["a", "b"] {
                values.push(row[name].as_f64().unwrap_or(0.0));
                missing.push(row[name].is_null());
            }
        }
        let data = TabularData {
            rows: rows.len(),
            numeric_names: vec!["a".into(), "b".into()],
            numeric: NumericBuffer {
                rows: rows.len(),
                cols: 2,
                values,
                missing,
            },
            categorical_names: vec!["sex".into()],
            categorical_values: rows
                .iter()
                .map(|r| r["sex"].as_str().map(str::to_string))
                .collect(),
        };
        let target = rows
            .iter()
            .map(|r| r["target"].as_u64().map(|y| y as u8))
            .collect();
        (data, target, fixture)
    }
    #[test]
    fn lp_certificate_status_and_scaled_rank_match_frozen_hand_cases() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../golden/python_golden_v1/separation_reference.json"
        )))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let rows = case["x"].as_array().unwrap();
            let cols = rows[0].as_array().unwrap().len();
            let x = matrix(
                rows.len(),
                cols,
                rows.iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                    .collect(),
            );
            let labels: Vec<_> = case["y"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let result = separation_diagnostic(&x, &labels, 1e-12).unwrap();
            println!(
                "{}: {:?}, objective {:?}, beta {:?}, margins {:?}",
                case["id"],
                result.status,
                result.objective,
                result.scaled_beta,
                result.signed_margins
            );
            assert_eq!(
                result.design_rank,
                case["scaled_augmented_rank"].as_u64().unwrap() as usize
            );
            let expected = match case["expected_diagnostic"].as_str().unwrap() {
                "separated" => SeparationStatus::Separated,
                "overlap" => SeparationStatus::NoSeparationDetected,
                _ => SeparationStatus::RankDeficient,
            };
            assert_eq!(result.status, expected);
            if expected == SeparationStatus::Separated {
                close(
                    result.objective.unwrap(),
                    case["hand_optimal_margin_sum"].as_f64().unwrap(),
                );
                assert!(result.signed_margins.iter().all(|&v| v >= 0.0));
            }
        }
    }
    #[test]
    fn complete_case_logit_matches_frozen_statsmodels_coefficients_information_and_wald() {
        let (data, target, fixture) = clinical_data();
        for (name, reference) in fixture["clinical_logistic"]["fits"].as_object().unwrap() {
            let features: Vec<_> = reference["features"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let refs = if features.contains(&"sex".into()) {
                BTreeMap::from([("sex".into(), "F".into())])
            } else {
                BTreeMap::new()
            };
            let output =
                logistic_inference(&data, &target, &features, &refs, &ClinicalConfig::default())
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            println!(
                "{name}: score {:?}, reports {:?}",
                output.score, output.report
            );
            assert_eq!(
                output.n_used,
                reference["n_used"].as_u64().unwrap() as usize
            );
            assert_eq!(
                output.n_dropped,
                reference["n_dropped"].as_u64().unwrap() as usize
            );
            let used_ids: Vec<_> = output
                .used_indices
                .iter()
                .map(|&i| {
                    fixture["clinical_logistic"]["rows"][i]["sample_id"]
                        .as_str()
                        .unwrap()
                })
                .collect();
            let expected_ids: Vec<_> = reference["used_sample_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(used_ids, expected_ids);
            assert_eq!(
                output.terms,
                reference["terms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>()
            );
            for (actual, key) in [
                (&output.coefficient, "coefficient"),
                (&output.score, "score"),
                (&output.standard_error, "standard_error"),
                (&output.z, "z"),
                (&output.p_values, "p_values"),
                (&output.odds_ratio, "odds_ratio"),
                (&output.apparent_probabilities, "apparent_probabilities"),
            ] {
                for (a, b) in actual.iter().zip(reference[key].as_array().unwrap()) {
                    close(*a, b.as_f64().unwrap());
                }
            }
            for (actual, key) in [
                (&output.covariance.values, "covariance"),
                (&output.information.values, "information"),
            ] {
                for (a, b) in actual.iter().zip(
                    reference[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| row.as_array().unwrap().iter()),
                ) {
                    close(*a, b.as_f64().unwrap());
                }
            }
            for (limits, expected) in output
                .coefficient_ci
                .iter()
                .zip(reference["coefficient_ci95"].as_array().unwrap())
            {
                close(limits[0], expected[0].as_f64().unwrap());
                close(limits[1], expected[1].as_f64().unwrap());
            }
            for ((lower, upper), expected) in output
                .odds_ratio_ci_lower
                .iter()
                .zip(&output.odds_ratio_ci_upper)
                .zip(reference["odds_ratio_ci95"].as_array().unwrap())
            {
                close(*lower, expected[0].as_f64().unwrap());
                close(*upper, expected[1].as_f64().unwrap());
            }
            close(
                output.log_likelihood,
                reference["log_likelihood"].as_f64().unwrap(),
            );
        }
    }

    #[test]
    fn clinical_inference_rejects_rank_separation_no_cases_and_nonconvergence() {
        for values in [
            vec![-3., -2., -1., 1., 2., 3.],
            vec![-2., -1., 0., 0., 1., 2.],
        ] {
            let data = TabularData {
                rows: 6,
                numeric_names: vec!["x".into()],
                numeric: matrix(6, 1, values),
                categorical_names: vec![],
                categorical_values: vec![],
            };
            let y = vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)];
            let error = logistic_inference(
                &data,
                &y,
                &["x".into()],
                &BTreeMap::new(),
                &ClinicalConfig::default(),
            )
            .unwrap_err();
            assert!(error.message.contains("separation"));
        }
        let (data, y, _) = clinical_data();
        let mut deficient = data.clone();
        deficient.numeric_names.push("duplicate_a".into());
        let values = data
            .numeric
            .values
            .chunks(2)
            .flat_map(|r| [r[0], r[1], r[0]])
            .collect();
        let missing = data
            .numeric
            .missing
            .chunks(2)
            .flat_map(|r| [r[0], r[1], r[0]])
            .collect();
        deficient.numeric = NumericBuffer {
            rows: data.rows,
            cols: 3,
            values,
            missing,
        };
        assert!(logistic_inference(
            &deficient,
            &y,
            &["a".into(), "duplicate_a".into()],
            &BTreeMap::new(),
            &ClinicalConfig::default()
        )
        .unwrap_err()
        .message
        .contains("rank deficient"));
        assert_eq!(
            logistic_inference(
                &data,
                &vec![None; data.rows],
                &["a".into()],
                &BTreeMap::new(),
                &ClinicalConfig::default()
            )
            .unwrap_err()
            .code,
            ErrorCode::EmptyResult
        );
        let short = ClinicalConfig {
            max_iterations: 1,
            ..ClinicalConfig::default()
        };
        assert_eq!(
            logistic_inference(
                &data,
                &y,
                &["a".into(), "b".into(), "sex".into()],
                &BTreeMap::new(),
                &short
            )
            .unwrap_err()
            .code,
            ErrorCode::Convergence
        );
        assert!(logistic_inference(
            &data,
            &y,
            &["sex".into()],
            &BTreeMap::from([("sex".into(), "unknown".into())]),
            &ClinicalConfig::default()
        )
        .is_err());
    }

    #[test]
    fn clinical_state_column_alignment_category_reference_and_explicit_or_overflow_are_preserved() {
        let (data, y, _) = clinical_data();
        let features = vec!["a".into(), "b".into(), "sex".into()];
        let fitted = logistic_inference(
            &data,
            &y,
            &features,
            &BTreeMap::new(),
            &ClinicalConfig::default(),
        )
        .unwrap();
        assert_eq!(fitted.reference_categories.get("sex").unwrap(), "F");
        fitted.validate().unwrap();
        let mut reordered = data.clone();
        reordered.numeric_names.reverse();
        for row in reordered.numeric.values.chunks_mut(2) {
            row.reverse();
        }
        for row in reordered.numeric.missing.chunks_mut(2) {
            row.reverse();
        }
        assert_eq!(
            fitted,
            logistic_inference(
                &reordered,
                &y,
                &features,
                &BTreeMap::new(),
                &ClinicalConfig::default()
            )
            .unwrap()
        );
        let restored: LogisticInference =
            serde_json::from_slice(&serde_json::to_vec(&fitted).unwrap()).unwrap();
        restored.validate().unwrap();
        assert_eq!(fitted, restored);
        let other = logistic_inference(
            &data,
            &y,
            &features,
            &BTreeMap::from([("sex".into(), "M".into())]),
            &ClinicalConfig::default(),
        )
        .unwrap();
        for (a, b) in fitted
            .apparent_probabilities
            .iter()
            .zip(&other.apparent_probabilities)
        {
            close(*a, *b);
        }
        let mut small_scale = data;
        for row in small_scale.numeric.values.chunks_mut(2) {
            row[0] *= 1e-6;
        }
        let overflow = logistic_inference(
            &small_scale,
            &y,
            &["a".into()],
            &BTreeMap::new(),
            &ClinicalConfig::default(),
        )
        .unwrap();
        assert!(overflow.odds_ratio.iter().any(|v| v.is_infinite()));
        let encoded = serde_json::to_vec(&overflow).unwrap();
        let restored: LogisticInference = serde_json::from_slice(&encoded).unwrap();
        restored.validate().unwrap();
        assert!(restored.odds_ratio.iter().any(|v| v.is_infinite()));
        assert!(restored
            .diagnostics
            .iter()
            .any(|r| r.code == "odds_ratio_numeric_range"));
    }
}
