//! Native predictive models and shared sparse solvers. No Python backend is called.
use crate::contracts::{unique_nonempty, NumericBuffer};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};

mod forest;
mod linear;
mod sparse;
pub use forest::{DecisionTree, MaxFeatures, TreeNode};
pub use sparse::{fit_sparse, SparseFit, SparseMethod, SparseSpec};

pub(crate) fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
pub(crate) fn convergence(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::Convergence, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnWeights {
    Uniform,
    Distance,
}

/// Regularization parameters apply to mean losses, with an unpenalized intercept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", deny_unknown_fields)]
pub enum ModelSpec {
    LR {
        lambda: f64,
        tolerance: f64,
        max_iterations: usize,
    },
    Linear {
        rank_tolerance: f64,
    },
    Ridge {
        lambda: f64,
        rank_tolerance: f64,
    },
    KNN {
        k: usize,
        weights: KnnWeights,
    },
    GaussianNB {
        priors: Option<Vec<f64>>,
        var_smoothing: f64,
    },
    RandomForest {
        trees: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: MaxFeatures,
        bootstrap: bool,
        seed: u64,
    },
}
impl ModelSpec {
    pub fn validate(&self) -> RadResult<()> {
        match self {
            Self::LR {
                lambda,
                tolerance,
                max_iterations,
            } => {
                penalty(*lambda)?;
                optimizer(*tolerance, *max_iterations)
            }
            Self::Linear { rank_tolerance } => rank_threshold(*rank_tolerance),
            Self::Ridge {
                lambda,
                rank_tolerance,
            } => {
                penalty(*lambda)?;
                rank_threshold(*rank_tolerance)
            }
            Self::KNN { k: 0, .. } => Err(invalid("KNN k must be positive.")),
            Self::KNN { .. } => Ok(()),
            Self::GaussianNB {
                priors,
                var_smoothing,
            } => {
                penalty(*var_smoothing)?;
                if let Some(p) = priors {
                    validate_priors(p)?;
                }
                Ok(())
            }
            Self::RandomForest {
                trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
                ..
            } => {
                if *trees == 0
                    || max_depth == &Some(0)
                    || *min_samples_split < 2
                    || *min_samples_leaf == 0
                    || matches!(max_features, MaxFeatures::Count { count: 0 })
                {
                    Err(invalid("Forest needs trees>0, depth>0 when set, min_samples_split>=2, min_samples_leaf>0 and positive max_features."))
                } else {
                    Ok(())
                }
            }
        }
    }
}
pub(crate) fn penalty(value: f64) -> RadResult<()> {
    if !value.is_finite() || value < 0.0 {
        Err(invalid(
            "Penalty and smoothing parameters must be finite and nonnegative.",
        ))
    } else {
        Ok(())
    }
}
fn rank_threshold(value: f64) -> RadResult<()> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        Err(invalid(
            "Relative rank tolerance must be finite and between zero and one.",
        ))
    } else {
        Ok(())
    }
}
pub(crate) fn optimizer(tolerance: f64, max_iterations: usize) -> RadResult<()> {
    if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
        Err(invalid(
            "Solver tolerance must be finite and positive, and max_iterations nonzero.",
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelTarget {
    Classification {
        classes: Vec<String>,
        labels: Vec<usize>,
    },
    Regression {
        values: Vec<f64>,
    },
}
impl ModelTarget {
    pub fn validate(&self, rows: usize) -> RadResult<()> {
        match self {
            Self::Classification { classes, labels } => {
                unique_nonempty(classes, "Model class labels")?;
                if classes.len() < 2
                    || labels.len() != rows
                    || labels.iter().any(|&l| l >= classes.len())
                {
                    return Err(invalid("Classification needs at least two ordered classes and one valid class index per row."));
                }
                let mut counts = vec![0; classes.len()];
                for &l in labels {
                    counts[l] += 1;
                }
                if counts.contains(&0) {
                    return Err(invalid(
                        "Every declared training class must be present; a fold is missing a class.",
                    ));
                }
                Ok(())
            }
            Self::Regression { values } => {
                if values.len() != rows || values.iter().any(|v| !v.is_finite()) {
                    Err(invalid("Regression requires one finite target per row."))
                } else {
                    Ok(())
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitReport {
    pub converged: bool,
    pub iterations: usize,
    /// None for estimators without a numerical optimization objective (KNN/NB).
    pub objective: Option<f64>,
    pub gradient_or_kkt_inf: Option<f64>,
}
impl FitReport {
    pub fn validate(&self, numerical: bool) -> RadResult<()> {
        if !self.converged {
            return Err(convergence(
                "A nonconverged fitted state cannot be used or saved.",
            ));
        }
        if numerical {
            if self.objective.is_none_or(|v| !v.is_finite())
                || self
                    .gradient_or_kkt_inf
                    .is_none_or(|v| !v.is_finite() || v < 0.0)
            {
                return Err(invalid("Numerically fitted models require a finite objective and a finite nonnegative gradient/KKT report."));
            }
        } else if self.objective.is_some() || self.gradient_or_kkt_inf.is_some() {
            return Err(invalid(
                "This estimator has no optimization objective or gradient.",
            ));
        }
        Ok(())
    }
    fn direct() -> Self {
        Self {
            converged: true,
            iterations: 0,
            objective: None,
            gradient_or_kkt_inf: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelState {
    Logistic {
        classes: Vec<String>,
        coefficients: NumericBuffer,
        intercepts: Vec<f64>,
    },
    Linear {
        coefficients: Vec<f64>,
        intercept: f64,
        rank: usize,
    },
    Knn {
        training: NumericBuffer,
        target: ModelTarget,
    },
    GaussianNb {
        classes: Vec<String>,
        means: NumericBuffer,
        variances: NumericBuffer,
        priors: Vec<f64>,
        epsilon: f64,
    },
    Forest {
        classes: Vec<String>,
        trees: Vec<DecisionTree>,
        rng_scheme: String,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedModel {
    pub spec: ModelSpec,
    pub feature_count: usize,
    pub state: ModelState,
    pub report: FitReport,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelPrediction {
    Classification {
        classes: Vec<String>,
        probabilities: NumericBuffer,
    },
    Regression {
        values: Vec<f64>,
    },
}

pub(crate) fn design(x: &NumericBuffer, training: bool) -> RadResult<()> {
    x.validate()?;
    if x.cols == 0 || (training && x.rows == 0) || x.missing.iter().any(|&m| m) {
        return Err(invalid(
            "Models require at least one feature and no missing values; training requires rows.",
        ));
    }
    Ok(())
}
pub(crate) fn buffer(rows: usize, cols: usize, values: Vec<f64>) -> NumericBuffer {
    let missing = vec![false; values.len()];
    NumericBuffer {
        rows,
        cols,
        values,
        missing,
    }
}
fn validate_classes(classes: &[String]) -> RadResult<()> {
    unique_nonempty(classes, "Fitted classes")?;
    if classes.len() < 2 {
        Err(invalid("Fitted classification needs at least two classes."))
    } else {
        Ok(())
    }
}
fn validate_priors(priors: &[f64]) -> RadResult<()> {
    if priors.len() < 2
        || priors
            .iter()
            .any(|&p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        || (priors.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        Err(invalid(
            "Class priors must be finite probabilities in class order and sum to one.",
        ))
    } else {
        Ok(())
    }
}

impl FittedModel {
    pub fn classes(&self) -> Option<&[String]> {
        match &self.state {
            ModelState::Logistic { classes, .. } | ModelState::GaussianNb { classes, .. } => {
                Some(classes)
            }
            ModelState::Knn {
                target: ModelTarget::Classification { classes, .. },
                ..
            } => Some(classes),
            ModelState::Forest { classes, .. } if !classes.is_empty() => Some(classes),
            _ => None,
        }
    }
    pub fn validate(&self) -> RadResult<()> {
        self.spec.validate()?;
        if self.feature_count == 0 {
            return Err(invalid("Fitted model has zero features."));
        }
        match (&self.spec, &self.state) {
            (
                ModelSpec::LR {
                    tolerance,
                    max_iterations,
                    ..
                },
                ModelState::Logistic {
                    classes,
                    coefficients,
                    intercepts,
                },
            ) => {
                validate_classes(classes)?;
                design(coefficients, true)?;
                let outputs = if classes.len() == 2 { 1 } else { classes.len() };
                if coefficients.cols != self.feature_count
                    || coefficients.rows != outputs
                    || intercepts.len() != outputs
                    || intercepts.iter().any(|v| !v.is_finite())
                {
                    return Err(invalid(
                        "Logistic fitted dimensions or intercepts are invalid.",
                    ));
                }
                self.report.validate(true)?;
                if self.report.iterations > *max_iterations
                    || self.report.gradient_or_kkt_inf.unwrap() > *tolerance
                {
                    return Err(convergence(
                        "Logistic fitted state does not meet its recorded convergence tolerance.",
                    ));
                }
            }
            (
                ModelSpec::Linear { .. } | ModelSpec::Ridge { .. },
                ModelState::Linear {
                    coefficients,
                    intercept,
                    rank,
                },
            ) => {
                if coefficients.len() != self.feature_count
                    || coefficients.iter().any(|v| !v.is_finite())
                    || !intercept.is_finite()
                    || *rank == 0
                    || *rank > self.feature_count + 1
                {
                    return Err(invalid(
                        "Linear fitted dimensions, rank or values are invalid.",
                    ));
                }
                if matches!(
                    self.spec,
                    ModelSpec::Linear { .. } | ModelSpec::Ridge { lambda: 0.0, .. }
                ) && *rank != self.feature_count + 1
                {
                    return Err(invalid(
                        "Unpenalized regression fitted design must have full column rank.",
                    ));
                }
                self.report.validate(true)?;
            }
            (ModelSpec::KNN { k, .. }, ModelState::Knn { training, target }) => {
                design(training, true)?;
                target.validate(training.rows)?;
                if training.cols != self.feature_count || *k > training.rows {
                    return Err(invalid(
                        "KNN fitted dimensions or k disagree with training rows.",
                    ));
                }
                self.report.validate(false)?;
            }
            (
                ModelSpec::GaussianNB { .. },
                ModelState::GaussianNb {
                    classes,
                    means,
                    variances,
                    priors,
                    epsilon,
                },
            ) => {
                validate_classes(classes)?;
                design(means, true)?;
                design(variances, true)?;
                validate_priors(priors)?;
                if means.rows != classes.len()
                    || means.cols != self.feature_count
                    || means.rows != variances.rows
                    || means.cols != variances.cols
                    || priors.len() != classes.len()
                    || variances.values.iter().any(|&v| v <= 0.0)
                    || !epsilon.is_finite()
                    || *epsilon < 0.0
                {
                    return Err(invalid(
                        "GaussianNB fitted means, variances or priors are invalid.",
                    ));
                }
                self.report.validate(false)?;
            }
            (
                ModelSpec::RandomForest {
                    trees: expected,
                    max_features,
                    ..
                },
                ModelState::Forest {
                    classes,
                    trees,
                    rng_scheme,
                },
            ) => {
                if !classes.is_empty() {
                    validate_classes(classes)?;
                }
                if trees.len() != *expected || rng_scheme != crate::random::RNG_SCHEME {
                    return Err(invalid(
                        "Forest tree count or recorded RNG scheme is invalid.",
                    ));
                }
                max_features.count(self.feature_count)?;
                for tree in trees {
                    tree.validate(self.feature_count, classes)?;
                }
                self.report.validate(false)?;
            }
            _ => {
                return Err(invalid(
                    "Model specification and fitted state type disagree.",
                ))
            }
        }
        Ok(())
    }
    pub fn predict(&self, x: &NumericBuffer) -> RadResult<ModelPrediction> {
        self.validate()?;
        design(x, false)?;
        if x.cols != self.feature_count {
            return Err(invalid(
                "Prediction feature count differs from fitted model.",
            ));
        }
        match &self.state {
            ModelState::Linear {
                coefficients,
                intercept,
                ..
            } => {
                let values: Vec<_> = x
                    .values
                    .chunks(x.cols)
                    .map(|row| {
                        row.iter()
                            .zip(coefficients)
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            + intercept
                    })
                    .collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid(
                        "Regression prediction overflowed; check input scale.",
                    ));
                }
                Ok(ModelPrediction::Regression { values })
            }
            ModelState::Logistic {
                classes,
                coefficients,
                intercepts,
            } => {
                let mut probabilities = Vec::with_capacity(x.rows * classes.len());
                for row in x.values.chunks(x.cols) {
                    let logits: Vec<_> = coefficients
                        .values
                        .chunks(x.cols)
                        .zip(intercepts)
                        .map(|(coef, b)| row.iter().zip(coef).map(|(a, b)| a * b).sum::<f64>() + b)
                        .collect();
                    if logits.iter().any(|v| !v.is_finite()) {
                        return Err(invalid(
                            "Logistic prediction overflowed; check input scale.",
                        ));
                    }
                    if classes.len() == 2 {
                        let p = sigmoid(logits[0]);
                        probabilities.extend([1.0 - p, p]);
                    } else {
                        probabilities.extend(softmax(&logits));
                    }
                }
                Ok(ModelPrediction::Classification {
                    classes: classes.clone(),
                    probabilities: buffer(x.rows, classes.len(), probabilities),
                })
            }
            ModelState::Knn { training, target } => predict_knn(&self.spec, training, target, x),
            ModelState::GaussianNb {
                classes,
                means,
                variances,
                priors,
                ..
            } => {
                let mut probabilities = Vec::with_capacity(x.rows * classes.len());
                for row in x.values.chunks(x.cols) {
                    let mut logits = Vec::with_capacity(classes.len());
                    for (c, &prior) in priors.iter().enumerate() {
                        let mut logp = prior.ln();
                        for (j, &v) in row.iter().enumerate() {
                            let variance = variances.values[c * x.cols + j];
                            let residual = v - means.values[c * x.cols + j];
                            logp -= 0.5
                                * ((2.0 * std::f64::consts::PI * variance).ln()
                                    + residual * residual / variance);
                        }
                        logits.push(logp);
                    }
                    if logits.iter().any(|v| v.is_nan()) || logits.iter().all(|v| !v.is_finite()) {
                        return Err(invalid("GaussianNB likelihood underflowed or overflowed for all classes; check input scale."));
                    }
                    probabilities.extend(softmax(&logits));
                }
                Ok(ModelPrediction::Classification {
                    classes: classes.clone(),
                    probabilities: buffer(x.rows, classes.len(), probabilities),
                })
            }
            ModelState::Forest { classes, trees, .. } => forest::predict(trees, classes, x),
        }
    }
}

impl crate::bundle::FittedState for FittedModel {
    fn validate(&self, manifest: &crate::bundle::ModelManifest) -> RadResult<()> {
        FittedModel::validate(self)?;
        manifest.validate()?;
        if manifest.feature_names.len() != self.feature_count
            || self.classes().unwrap_or(&[]) != manifest.classes.as_slice()
        {
            return Err(invalid(
                "Model manifest feature or class order disagrees with the typed fitted state.",
            ));
        }
        let valid_task = match manifest.task {
            crate::contracts::Task::Regression => self.classes().is_none(),
            crate::contracts::Task::Binary => self.classes().is_some_and(|c| c.len() == 2),
            crate::contracts::Task::Multiclass => self.classes().is_some_and(|c| c.len() >= 3),
        };
        if !valid_task {
            return Err(invalid(
                "Model manifest task disagrees with the fitted state.",
            ));
        }
        Ok(())
    }
}

pub fn fit(spec: &ModelSpec, x: &NumericBuffer, target: &ModelTarget) -> RadResult<FittedModel> {
    spec.validate()?;
    design(x, true)?;
    target.validate(x.rows)?;
    let (state, report) = match spec {
        ModelSpec::LR {
            lambda,
            tolerance,
            max_iterations,
        } => linear::fit_logistic(x, target, *lambda, *tolerance, *max_iterations)?,
        ModelSpec::Linear { rank_tolerance } => {
            linear::fit_regression(x, target, 0.0, *rank_tolerance)?
        }
        ModelSpec::Ridge {
            lambda,
            rank_tolerance,
        } => linear::fit_regression(x, target, *lambda, *rank_tolerance)?,
        ModelSpec::KNN { k, .. } => {
            if *k > x.rows {
                return Err(invalid("KNN k cannot exceed the number of training rows."));
            }
            (
                ModelState::Knn {
                    training: x.clone(),
                    target: target.clone(),
                },
                FitReport::direct(),
            )
        }
        ModelSpec::GaussianNB {
            priors,
            var_smoothing,
        } => fit_nb(x, target, priors, *var_smoothing)?,
        ModelSpec::RandomForest { .. } => forest::fit(spec, x, target)?,
    };
    let model = FittedModel {
        spec: spec.clone(),
        feature_count: x.cols,
        state,
        report,
    };
    model.validate()?;
    Ok(model)
}

fn fit_nb(
    x: &NumericBuffer,
    target: &ModelTarget,
    configured: &Option<Vec<f64>>,
    smoothing: f64,
) -> RadResult<(ModelState, FitReport)> {
    let ModelTarget::Classification { classes, labels } = target else {
        return Err(invalid("GaussianNB supports classification only."));
    };
    if configured
        .as_ref()
        .is_some_and(|p| p.len() != classes.len())
    {
        return Err(invalid(
            "GaussianNB priors must match declared class order.",
        ));
    }
    let mut overall_mean = vec![0.0; x.cols];
    for row in x.values.chunks(x.cols) {
        for (mean, &v) in overall_mean.iter_mut().zip(row) {
            *mean += v / x.rows as f64;
        }
    }
    let mut overall_variance = vec![0.0; x.cols];
    for row in x.values.chunks(x.cols) {
        for (j, &v) in row.iter().enumerate() {
            overall_variance[j] += (v - overall_mean[j]).powi(2) / x.rows as f64;
        }
    }
    let epsilon = smoothing * overall_variance.into_iter().fold(0.0, f64::max);
    let mut counts = vec![0usize; classes.len()];
    let mut means = vec![0.0; classes.len() * x.cols];
    for (row, &c) in x.values.chunks(x.cols).zip(labels) {
        counts[c] += 1;
        for (j, &v) in row.iter().enumerate() {
            means[c * x.cols + j] += v;
        }
    }
    for (c, &count) in counts.iter().enumerate() {
        for j in 0..x.cols {
            means[c * x.cols + j] /= count as f64;
        }
    }
    let mut variances = vec![0.0; means.len()];
    for (row, &c) in x.values.chunks(x.cols).zip(labels) {
        for (j, &v) in row.iter().enumerate() {
            variances[c * x.cols + j] += (v - means[c * x.cols + j]).powi(2) / counts[c] as f64;
        }
    }
    for variance in &mut variances {
        *variance += epsilon;
    }
    if variances.iter().any(|&v| v <= 0.0 || !v.is_finite()) || means.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("GaussianNB has an undefined class variance; configure positive smoothing on a varying training design."));
    }
    let priors = configured
        .clone()
        .unwrap_or_else(|| counts.iter().map(|&n| n as f64 / x.rows as f64).collect());
    Ok((
        ModelState::GaussianNb {
            classes: classes.clone(),
            means: buffer(classes.len(), x.cols, means),
            variances: buffer(classes.len(), x.cols, variances),
            priors,
            epsilon,
        },
        FitReport::direct(),
    ))
}

fn predict_knn(
    spec: &ModelSpec,
    training: &NumericBuffer,
    target: &ModelTarget,
    x: &NumericBuffer,
) -> RadResult<ModelPrediction> {
    let ModelSpec::KNN { k, weights } = spec else {
        return Err(invalid("KNN prediction needs a KNN specification."));
    };
    let columns = match target {
        ModelTarget::Classification { classes, .. } => classes.len(),
        _ => 1,
    };
    let mut values = vec![0.0; x.rows * columns];
    for (r, row) in x.values.chunks(x.cols).enumerate() {
        let mut distances: Vec<_> = training
            .values
            .chunks(x.cols)
            .enumerate()
            .map(|(i, other)| {
                (
                    row.iter()
                        .zip(other)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                    i,
                )
            })
            .collect();
        distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        distances.truncate(*k);
        let zeros = distances.iter().any(|(distance, _)| *distance == 0.0);
        let mut total = 0.0;
        for (distance, i) in distances {
            let weight = match weights {
                KnnWeights::Uniform => 1.0,
                KnnWeights::Distance if zeros => f64::from(distance == 0.0),
                KnnWeights::Distance => 1.0 / distance,
            };
            total += weight;
            match target {
                ModelTarget::Classification { labels, .. } => {
                    values[r * columns + labels[i]] += weight
                }
                ModelTarget::Regression { values: targets } => values[r] += weight * targets[i],
            }
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid(
                "KNN distances or weights are not representable; check input scale.",
            ));
        }
        for j in 0..columns {
            values[r * columns + j] /= total;
        }
    }
    match target {
        ModelTarget::Classification { classes, .. } => Ok(ModelPrediction::Classification {
            classes: classes.clone(),
            probabilities: buffer(x.rows, columns, values),
        }),
        ModelTarget::Regression { .. } => Ok(ModelPrediction::Regression { values }),
    }
}

pub(crate) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}
pub(crate) fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<_> = logits.iter().map(|&z| (z - max).exp()).collect();
    let total = p.iter().sum::<f64>();
    for v in &mut p {
        *v /= total;
    }
    p
}
pub(crate) fn norm_inf(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests;
