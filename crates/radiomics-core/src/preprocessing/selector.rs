use super::*;
use crate::contracts::Task;
use crate::models::{fit_sparse, ModelTarget, SparseFit, SparseSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UnivariateMethod {
    None,
    #[default]
    Auto,
    MannWhitney,
    KruskalWallis,
    Pearson,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorConfig {
    pub variance_threshold: f64,
    pub corr_threshold: Option<f64>,
    pub univariate: UnivariateMethod,
    pub p_threshold: f64,
    pub adjust_bh: bool,
    pub sparse: Option<SparseSpec>,
    pub coefficient_threshold: f64,
}
impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            variance_threshold: 0.0,
            corr_threshold: Some(0.9),
            univariate: UnivariateMethod::Auto,
            p_threshold: 0.05,
            adjust_bh: false,
            sparse: None,
            coefficient_threshold: 1e-6,
        }
    }
}
impl SelectorConfig {
    pub fn validate(&self) -> RadResult<()> {
        if !self.variance_threshold.is_finite()
            || self.variance_threshold < 0.0
            || self
                .corr_threshold
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            || !self.p_threshold.is_finite()
            || self.p_threshold <= 0.0
            || self.p_threshold > 1.0
            || !self.coefficient_threshold.is_finite()
            || self.coefficient_threshold < 0.0
        {
            return Err(invalid(
                "Selection variance/correlation/p-value/coefficient thresholds are invalid.",
            ));
        }
        if let Some(s) = &self.sparse {
            s.validate()?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarianceSource {
    InputBuffer,
    ExplicitUnscaled,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnivariateResult {
    pub feature_index: usize,
    pub statistic: f64,
    pub p_value: f64,
    pub adjusted_p_value: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelationDrop {
    pub removed_index: usize,
    pub retained_index: usize,
    pub correlation: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedSelector {
    pub config: SelectorConfig,
    pub task: Task,
    pub classes: Vec<String>,
    pub feature_names: Vec<String>,
    pub variance_source: VarianceSource,
    pub variances: Vec<f64>,
    pub constant_features: Vec<usize>,
    pub resolved_univariate: UnivariateMethod,
    pub univariate_results: Vec<UnivariateResult>,
    pub after_variance: Vec<usize>,
    pub after_univariate: Vec<usize>,
    pub after_correlation: Vec<usize>,
    pub correlation_drops: Vec<CorrelationDrop>,
    pub selected: Vec<usize>,
    pub sparse: Option<SparseFit>,
}
fn resolve(task: Task, method: UnivariateMethod) -> RadResult<UnivariateMethod> {
    let method = match (task, method) {
        (Task::Binary, UnivariateMethod::Auto) => UnivariateMethod::MannWhitney,
        (Task::Multiclass, UnivariateMethod::Auto) => UnivariateMethod::KruskalWallis,
        (Task::Regression, UnivariateMethod::Auto) => UnivariateMethod::Pearson,
        (_, m) => m,
    };
    if matches!(
        (task, method),
        (
            Task::Regression,
            UnivariateMethod::MannWhitney | UnivariateMethod::KruskalWallis
        ) | (Task::Binary | Task::Multiclass, UnivariateMethod::Pearson)
            | (Task::Multiclass, UnivariateMethod::MannWhitney)
    ) {
        return Err(invalid("Univariate method is incompatible with the declared task; multiclass class indices are not continuous targets."));
    }
    Ok(method)
}
fn column(x: &NumericBuffer, index: usize) -> Vec<f64> {
    (0..x.rows).map(|i| x.values[i * x.cols + index]).collect()
}
fn take_columns(x: &NumericBuffer, columns: &[usize]) -> NumericBuffer {
    let values: Vec<_> = (0..x.rows)
        .flat_map(|i| columns.iter().map(move |&j| x.values[i * x.cols + j]))
        .collect();
    NumericBuffer {
        rows: x.rows,
        cols: columns.len(),
        missing: vec![false; values.len()],
        values,
    }
}
fn zero(stage: &str) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::EmptyResult,format!("Zero features remain after {stage}; change the explicit study configuration rather than automatically relaxing thresholds."))
}

/// Variance filtering uses the explicitly supplied unscaled variances when
/// preprocessing has already scaled x. Every other stage uses this fitted x.
/// Fixed sparse alpha is fit once; tuning/refitting belongs to grouped CV in G.
pub fn fit_selector(
    x: &NumericBuffer,
    names: &[String],
    target: &ModelTarget,
    config: &SelectorConfig,
    variance_reference: Option<&[f64]>,
) -> RadResult<FittedSelector> {
    config.validate()?;
    crate::models::design(x, true)?;
    unique_nonempty(names, "Selector feature names")?;
    target.validate(x.rows)?;
    if x.cols != names.len() {
        return Err(invalid("Selector feature names and matrix width disagree."));
    }
    let (task, classes) = match target {
        ModelTarget::Classification { classes, .. } => (
            if classes.len() == 2 {
                Task::Binary
            } else {
                Task::Multiclass
            },
            classes.clone(),
        ),
        ModelTarget::Regression { .. } => (Task::Regression, vec![]),
    };
    let method = resolve(task, config.univariate)?;
    let variances = if let Some(reference) = variance_reference {
        if reference.len() != names.len() || reference.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(invalid("Explicit unscaled variance reference must match feature names and contain finite nonnegative values."));
        }
        reference.to_vec()
    } else {
        (0..x.cols)
            .map(|j| crate::statistics::population_mean_variance(&column(x, j)).map(|(_, v)| v))
            .collect::<RadResult<Vec<_>>>()?
    };
    let constant_features: Vec<_> = (0..x.cols)
        .filter(|&j| (1..x.rows).all(|i| x.values[i * x.cols + j] == x.values[j]))
        .collect();
    let after_variance: Vec<_> = (0..x.cols)
        .filter(|&j| variances[j] > config.variance_threshold && !constant_features.contains(&j))
        .collect();
    if after_variance.is_empty() {
        return Err(zero("constant/low-variance filtering"));
    }
    let mut univariate_results = Vec::new();
    if method != UnivariateMethod::None {
        for &j in &after_variance {
            let values = column(x, j);
            let result = match (method, target) {
                (UnivariateMethod::MannWhitney, ModelTarget::Classification { labels, .. }) => {
                    let a = values
                        .iter()
                        .zip(labels)
                        .filter_map(|(&v, &l)| (l == 0).then_some(v))
                        .collect::<Vec<_>>();
                    let b = values
                        .iter()
                        .zip(labels)
                        .filter_map(|(&v, &l)| (l == 1).then_some(v))
                        .collect::<Vec<_>>();
                    crate::statistics::mann_whitney_u(&a, &b)?
                }
                (
                    UnivariateMethod::KruskalWallis,
                    ModelTarget::Classification { classes, labels },
                ) => {
                    let mut groups = vec![Vec::new(); classes.len()];
                    for (&v, &l) in values.iter().zip(labels) {
                        groups[l].push(v);
                    }
                    crate::statistics::kruskal_wallis(&groups)?
                }
                (UnivariateMethod::Pearson, ModelTarget::Regression { values: y }) => {
                    crate::statistics::pearson(&values, y)?
                }
                _ => return Err(invalid("Univariate task resolution failed.")),
            };
            univariate_results.push(UnivariateResult {
                feature_index: j,
                statistic: result.statistic,
                p_value: result.p_value,
                adjusted_p_value: result.p_value,
            });
        }
        if config.adjust_bh {
            let adjusted = crate::statistics::benjamini_hochberg(
                &univariate_results
                    .iter()
                    .map(|r| r.p_value)
                    .collect::<Vec<_>>(),
            )?;
            for (r, p) in univariate_results.iter_mut().zip(adjusted) {
                r.adjusted_p_value = p;
            }
        }
    }
    let after_univariate: Vec<_> = if method == UnivariateMethod::None {
        after_variance.clone()
    } else {
        univariate_results
            .iter()
            .filter_map(|r| (r.adjusted_p_value < config.p_threshold).then_some(r.feature_index))
            .collect()
    };
    if after_univariate.is_empty() {
        return Err(zero("univariate/BH filtering"));
    }
    let mut after_correlation = Vec::new();
    let mut correlation_drops = Vec::new();
    for &j in &after_univariate {
        let mut redundant = false;
        if let Some(threshold) = config.corr_threshold {
            for &k in &after_correlation {
                let r = crate::statistics::pearson(&column(x, j), &column(x, k))?.statistic;
                if r.abs() > threshold {
                    correlation_drops.push(CorrelationDrop {
                        removed_index: j,
                        retained_index: k,
                        correlation: r,
                    });
                    redundant = true;
                    break;
                }
            }
        }
        if !redundant {
            after_correlation.push(j);
        }
    }
    let sparse = if let Some(spec) = &config.sparse {
        Some(fit_sparse(
            spec,
            &take_columns(x, &after_correlation),
            target,
        )?)
    } else {
        None
    };
    let selected = if let Some(sparse) = &sparse {
        after_correlation
            .iter()
            .enumerate()
            .filter_map(|(j, &original)| {
                (0..sparse.coefficients.rows)
                    .any(|c| {
                        sparse.coefficients.values[c * sparse.coefficients.cols + j].abs()
                            > config.coefficient_threshold
                    })
                    .then_some(original)
            })
            .collect::<Vec<_>>()
    } else {
        after_correlation.clone()
    };
    if selected.is_empty() {
        return Err(zero("sparse coefficient filtering"));
    }
    let fitted = FittedSelector {
        config: config.clone(),
        task,
        classes,
        feature_names: names.to_vec(),
        variance_source: if variance_reference.is_some() {
            VarianceSource::ExplicitUnscaled
        } else {
            VarianceSource::InputBuffer
        },
        variances,
        constant_features,
        resolved_univariate: method,
        univariate_results,
        after_variance,
        after_univariate,
        after_correlation,
        correlation_drops,
        selected,
        sparse,
    };
    fitted.validate()?;
    Ok(fitted)
}

fn ordered_subset(values: &[usize], parent: &[usize]) -> bool {
    values.windows(2).all(|w| w[0] < w[1]) && values.iter().all(|v| parent.binary_search(v).is_ok())
}
impl FittedSelector {
    pub fn validate(&self) -> RadResult<()> {
        self.config.validate()?;
        unique_nonempty(&self.feature_names, "Fitted selector names")?;
        unique_nonempty(&self.classes, "Selector class labels")?;
        if self.feature_names.is_empty()
            || self.variances.len() != self.feature_names.len()
            || self.variances.iter().any(|&v| !v.is_finite() || v < 0.0)
            || self.selected.is_empty()
        {
            return Err(invalid(
                "Fitted selector dimensions or variance state are invalid.",
            ));
        }
        if match self.task {
            Task::Binary => self.classes.len() != 2,
            Task::Multiclass => self.classes.len() < 3,
            Task::Regression => !self.classes.is_empty(),
        } {
            return Err(invalid("Selector task/class state disagrees."));
        }
        if resolve(self.task, self.config.univariate)? != self.resolved_univariate {
            return Err(invalid(
                "Selector resolved test disagrees with task/configuration.",
            ));
        }
        let all: Vec<_> = (0..self.feature_names.len()).collect();
        if !ordered_subset(&self.constant_features, &all) {
            return Err(invalid("Selector constant-feature indices are invalid."));
        }
        let expected: Vec<_> = all
            .into_iter()
            .filter(|&j| {
                self.variances[j] > self.config.variance_threshold
                    && !self.constant_features.contains(&j)
            })
            .collect();
        if expected != self.after_variance
            || !ordered_subset(&self.after_univariate, &self.after_variance)
            || !ordered_subset(&self.after_correlation, &self.after_univariate)
            || !ordered_subset(&self.selected, &self.after_correlation)
        {
            return Err(invalid(
                "Selector stages are not ordered nested subsets of fitted feature indices.",
            ));
        }
        if self.resolved_univariate == UnivariateMethod::None {
            if !self.univariate_results.is_empty() || self.after_univariate != self.after_variance {
                return Err(invalid(
                    "Disabled univariate selection contains fitted test state.",
                ));
            }
        } else {
            if self.univariate_results.len() != self.after_variance.len() {
                return Err(invalid(
                    "Selector test result count differs from tested features.",
                ));
            }
            for (r, &j) in self.univariate_results.iter().zip(&self.after_variance) {
                if r.feature_index != j
                    || !r.statistic.is_finite()
                    || !r.p_value.is_finite()
                    || !(0.0..=1.0).contains(&r.p_value)
                    || !r.adjusted_p_value.is_finite()
                    || !(0.0..=1.0).contains(&r.adjusted_p_value)
                {
                    return Err(invalid(
                        "Selector test result index or numerical state is invalid.",
                    ));
                }
            }
            let passed: Vec<_> = self
                .univariate_results
                .iter()
                .filter_map(|r| {
                    (r.adjusted_p_value < self.config.p_threshold).then_some(r.feature_index)
                })
                .collect();
            if passed != self.after_univariate {
                return Err(invalid(
                    "Univariate retained features disagree with fitted p-values.",
                ));
            }
        }
        let mut dropped = Vec::new();
        for r in &self.correlation_drops {
            if !self.after_univariate.contains(&r.removed_index)
                || !self.after_correlation.contains(&r.retained_index)
                || r.retained_index >= r.removed_index
                || !r.correlation.is_finite()
                || r.correlation.abs() > 1.0 + 1e-12
                || self
                    .config
                    .corr_threshold
                    .is_none_or(|t| r.correlation.abs() <= t)
            {
                return Err(invalid("Fitted correlation exclusion is invalid."));
            }
            dropped.push(r.removed_index);
        }
        dropped.sort_unstable();
        if dropped.windows(2).any(|w| w[0] == w[1])
            || self
                .after_univariate
                .iter()
                .copied()
                .filter(|j| !dropped.contains(j))
                .collect::<Vec<_>>()
                != self.after_correlation
        {
            return Err(invalid("Correlation retained/excluded sets disagree."));
        }
        match (&self.config.sparse, &self.sparse) {
            (None, None) => {
                if self.selected != self.after_correlation {
                    return Err(invalid(
                        "Disabled sparse selection changed retained features.",
                    ));
                }
            }
            (Some(spec), Some(sparse)) => {
                sparse.validate()?;
                if &sparse.spec != spec
                    || sparse.feature_count != self.after_correlation.len()
                    || sparse.classes != self.classes
                {
                    return Err(invalid("Sparse fitted task, specification or feature count disagrees with selector state."));
                }
                let expected: Vec<_> = self
                    .after_correlation
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &original)| {
                        (0..sparse.coefficients.rows)
                            .any(|c| {
                                sparse.coefficients.values[c * sparse.coefficients.cols + j].abs()
                                    > self.config.coefficient_threshold
                            })
                            .then_some(original)
                    })
                    .collect();
                if expected != self.selected {
                    return Err(invalid(
                        "Sparse selected features disagree with fitted coefficient threshold.",
                    ));
                }
            }
            _ => return Err(invalid("Sparse fitted state and configuration disagree.")),
        }
        Ok(())
    }
    pub fn selected_names(&self) -> RadResult<Vec<String>> {
        self.validate()?;
        Ok(self
            .selected
            .iter()
            .map(|&j| self.feature_names[j].clone())
            .collect())
    }
    pub fn transform(&self, x: &NumericBuffer, names: &[String]) -> RadResult<NumericBuffer> {
        self.validate()?;
        crate::models::design(x, false)?;
        unique_nonempty(names, "Prediction feature names")?;
        if x.cols != names.len() {
            return Err(invalid(
                "Prediction names and numerical buffer width disagree.",
            ));
        }
        let positions: Vec<_> = self
            .selected
            .iter()
            .map(|&j| {
                names
                    .iter()
                    .position(|name| name == &self.feature_names[j])
                    .ok_or_else(|| {
                        RadiomicsError::new(
                            ErrorCode::Alignment,
                            format!(
                                "Missing selected prediction feature '{}'.",
                                self.feature_names[j]
                            ),
                        )
                    })
            })
            .collect::<RadResult<_>>()?;
        Ok(take_columns(x, &positions))
    }
}
