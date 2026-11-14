use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMethod {
    SquaredLasso,
    BinaryL1Logistic,
    OvrL1Logistic,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseSpec {
    pub method: SparseMethod,
    pub alpha: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}
impl SparseSpec {
    pub fn validate(&self) -> RadResult<()> {
        penalty(self.alpha)?;
        optimizer(self.tolerance, self.max_iterations)
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseFit {
    pub spec: SparseSpec,
    pub feature_count: usize,
    pub classes: Vec<String>,
    /// One row for squared loss/binary L1; one row per ordered OvR class.
    pub coefficients: NumericBuffer,
    pub intercepts: Vec<f64>,
    /// Union of mathematically nonzero coefficient columns, in input order.
    pub support: Vec<usize>,
    pub per_class_reports: Vec<FitReport>,
}
impl SparseFit {
    pub fn validate(&self) -> RadResult<()> {
        self.spec.validate()?;
        design(&self.coefficients, true)?;
        let outputs = match self.spec.method {
            SparseMethod::OvrL1Logistic => {
                validate_classes(&self.classes)?;
                self.classes.len()
            }
            SparseMethod::BinaryL1Logistic => {
                if self.classes.len() != 2 {
                    return Err(invalid("Binary L1 Logistic state requires two classes."));
                }
                validate_classes(&self.classes)?;
                1
            }
            SparseMethod::SquaredLasso => {
                if !self.classes.is_empty() {
                    if self.classes.len() != 2 {
                        return Err(invalid(
                            "Squared LASSO may represent only regression or binary labels.",
                        ));
                    }
                    validate_classes(&self.classes)?;
                }
                1
            }
        };
        if self.feature_count == 0
            || self.coefficients.cols != self.feature_count
            || self.coefficients.rows != outputs
            || self.intercepts.len() != outputs
            || self.intercepts.iter().any(|v| !v.is_finite())
            || self.per_class_reports.len() != outputs
        {
            return Err(invalid(
                "Sparse fitted dimensions or intercepts are invalid.",
            ));
        }
        for report in &self.per_class_reports {
            report.validate(true)?;
            if report.iterations > self.spec.max_iterations
                || report.gradient_or_kkt_inf.unwrap() > self.spec.tolerance
            {
                return Err(convergence(
                    "Sparse fitted state fails its recorded KKT tolerance.",
                ));
            }
        }
        let expected: Vec<_> = (0..self.feature_count)
            .filter(|&j| {
                (0..outputs).any(|c| self.coefficients.values[c * self.feature_count + j] != 0.0)
            })
            .collect();
        if expected != self.support {
            return Err(invalid(
                "Sparse fitted support disagrees with coefficient state.",
            ));
        }
        Ok(())
    }
    /// Linear scores are not probabilities, including squared-loss binary LASSO.
    pub fn linear_scores(&self, x: &NumericBuffer) -> RadResult<NumericBuffer> {
        self.validate()?;
        design(x, false)?;
        if x.cols != self.feature_count {
            return Err(invalid(
                "Sparse score feature count differs from fitted input.",
            ));
        }
        let mut values = Vec::with_capacity(x.rows * self.coefficients.rows);
        for row in x.values.chunks(x.cols) {
            for (coef, b) in self
                .coefficients
                .values
                .chunks(x.cols)
                .zip(&self.intercepts)
            {
                let score = row.iter().zip(coef).map(|(a, b)| a * b).sum::<f64>() + b;
                if !score.is_finite() {
                    return Err(invalid("Sparse score overflowed; check input scale."));
                }
                values.push(score);
            }
        }
        Ok(buffer(x.rows, self.coefficients.rows, values))
    }
}

pub fn fit_sparse(
    spec: &SparseSpec,
    x: &NumericBuffer,
    target: &ModelTarget,
) -> RadResult<SparseFit> {
    spec.validate()?;
    design(x, true)?;
    target.validate(x.rows)?;
    let (classes,targets): (Vec<String>,Vec<Vec<f64>>) = match (&spec.method,target) {
        (SparseMethod::SquaredLasso,ModelTarget::Regression{values})=>(vec![],vec![values.clone()]),
        (SparseMethod::SquaredLasso,ModelTarget::Classification{classes,labels}) if classes.len()==2=>(classes.clone(),vec![labels.iter().map(|&v|v as f64).collect()]),
        (SparseMethod::BinaryL1Logistic,ModelTarget::Classification{classes,labels}) if classes.len()==2=>(classes.clone(),vec![labels.iter().map(|&v|v as f64).collect()]),
        (SparseMethod::OvrL1Logistic,ModelTarget::Classification{classes,labels})=>(classes.clone(),(0..classes.len()).map(|c|labels.iter().map(|&v|f64::from(v==c)).collect()).collect()),
        _=>return Err(invalid("Sparse method and task disagree; multiclass cannot use squared-loss LASSO on numeric class indices.")),
    };
    let mut coefficients = Vec::with_capacity(targets.len() * x.cols);
    let mut intercepts = Vec::new();
    let mut reports = Vec::new();
    for y in &targets {
        let (coefficient, intercept, report) = if spec.method == SparseMethod::SquaredLasso {
            lasso(x, y, spec)?
        } else {
            l1_logistic(x, y, spec)?
        };
        coefficients.extend(coefficient);
        intercepts.push(intercept);
        reports.push(report);
    }
    let support = (0..x.cols)
        .filter(|&j| (0..targets.len()).any(|c| coefficients[c * x.cols + j] != 0.0))
        .collect();
    let fitted = SparseFit {
        spec: spec.clone(),
        feature_count: x.cols,
        classes,
        coefficients: buffer(targets.len(), x.cols, coefficients),
        intercepts,
        support,
        per_class_reports: reports,
    };
    fitted.validate()?;
    Ok(fitted)
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}
fn kkt(gradient: &[f64], coefficients: &[f64], alpha: f64, intercept_gradient: f64) -> f64 {
    gradient
        .iter()
        .zip(coefficients)
        .map(|(&g, &w)| {
            if w == 0.0 {
                (g.abs() - alpha).max(0.0)
            } else {
                (g + alpha * w.signum()).abs()
            }
        })
        .fold(intercept_gradient.abs(), f64::max)
}
fn lasso(x: &NumericBuffer, y: &[f64], spec: &SparseSpec) -> RadResult<(Vec<f64>, f64, FitReport)> {
    let n = x.rows as f64;
    let p = x.cols;
    let ymean = y.iter().sum::<f64>() / n;
    let mut means = vec![0.0; p];
    for row in x.values.chunks(p) {
        for (m, &v) in means.iter_mut().zip(row) {
            *m += v / n;
        }
    }
    let columns: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            (0..x.rows)
                .map(|i| x.values[i * p + j] - means[j])
                .collect()
        })
        .collect();
    let norms: Vec<_> = columns
        .iter()
        .map(|column| column.iter().map(|v| v * v).sum::<f64>() / n)
        .collect();
    let mut residual: Vec<_> = y.iter().map(|v| ymean - v).collect();
    let mut w = vec![0.0; p];
    for iteration in 0..=spec.max_iterations {
        let gradient: Vec<_> = columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .zip(&residual)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / n
            })
            .collect();
        if gradient.iter().any(|v| !v.is_finite()) {
            return Err(convergence(
                "LASSO gradient became nonfinite; check input scale.",
            ));
        }
        let maximum = kkt(&gradient, &w, spec.alpha, residual.iter().sum::<f64>() / n);
        if maximum <= spec.tolerance {
            let intercept = ymean - means.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
            let objective = residual.iter().map(|r| r * r).sum::<f64>() / (2.0 * n)
                + spec.alpha * w.iter().map(|v| v.abs()).sum::<f64>();
            return Ok((
                w,
                intercept,
                FitReport {
                    converged: true,
                    iterations: iteration,
                    objective: Some(objective),
                    gradient_or_kkt_inf: Some(maximum),
                },
            ));
        }
        if iteration == spec.max_iterations {
            break;
        }
        for j in 0..p {
            if norms[j] == 0.0 {
                continue;
            }
            let correlation = columns[j]
                .iter()
                .zip(&residual)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n;
            let candidate = soft_threshold(w[j] - correlation / norms[j], spec.alpha / norms[j]);
            let delta = candidate - w[j];
            for (r, &v) in residual.iter_mut().zip(&columns[j]) {
                *r += delta * v;
            }
            w[j] = candidate;
        }
        if residual.iter().any(|v| !v.is_finite()) {
            return Err(convergence(
                "LASSO residual became nonfinite; check input scaling.",
            ));
        }
    }
    Err(convergence(format!(
        "Squared-loss LASSO did not meet KKT tolerance in {} iterations.",
        spec.max_iterations
    )))
}

fn binary_value(logits: &[f64], y: &[f64], w: &[f64], alpha: f64) -> f64 {
    logits
        .iter()
        .zip(y)
        .map(|(&z, &y)| softplus(z) - y * z)
        .sum::<f64>()
        / y.len() as f64
        + alpha * w.iter().map(|v| v.abs()).sum::<f64>()
}
fn l1_logistic(
    x: &NumericBuffer,
    y: &[f64],
    spec: &SparseSpec,
) -> RadResult<(Vec<f64>, f64, FitReport)> {
    let n = x.rows as f64;
    let p = x.cols;
    let mut w = vec![0.0; p];
    let mut b = 0.0;
    let mut logits = vec![0.0; x.rows];
    // Cyclic coordinate Newton updates with a composite Armijo line search.
    // The intercept coordinate is updated by the same numerical loop but never penalized.
    for iteration in 0..=spec.max_iterations {
        let residual: Vec<_> = logits
            .iter()
            .zip(y)
            .map(|(&z, &y)| sigmoid(z) - y)
            .collect();
        let gradient: Vec<_> = (0..p)
            .map(|j| {
                (0..x.rows)
                    .map(|i| residual[i] * x.values[i * p + j])
                    .sum::<f64>()
                    / n
            })
            .collect();
        if gradient.iter().any(|v| !v.is_finite()) || residual.iter().any(|v| !v.is_finite()) {
            return Err(convergence(
                "L1 Logistic gradient became nonfinite; check input scale.",
            ));
        }
        let maximum = kkt(&gradient, &w, spec.alpha, residual.iter().sum::<f64>() / n);
        if maximum <= spec.tolerance {
            return Ok((
                w.clone(),
                b,
                FitReport {
                    converged: true,
                    iterations: iteration,
                    objective: Some(binary_value(&logits, y, &w, spec.alpha)),
                    gradient_or_kkt_inf: Some(maximum),
                },
            ));
        }
        if iteration == spec.max_iterations {
            break;
        }
        for j in 0..=p {
            let mut g = 0.0;
            let mut h = 0.0;
            for (i, &z) in logits.iter().enumerate() {
                let value = if j == p { 1.0 } else { x.values[i * p + j] };
                let probability = sigmoid(z);
                g += (probability - y[i]) * value / n;
                h += probability * (1.0 - probability) * value * value / n;
            }
            if h == 0.0 {
                continue;
            }
            let old = if j == p { b } else { w[j] };
            let penalty = if j == p { 0.0 } else { spec.alpha };
            let candidate = soft_threshold(old - g / h, penalty / h);
            let delta = candidate - old;
            if delta == 0.0 {
                continue;
            }
            let value = binary_value(&logits, y, &w, spec.alpha);
            let slope = g * delta + penalty * (candidate.abs() - old.abs());
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..60 {
                let proposed: Vec<_> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, z)| z + step * delta * if j == p { 1.0 } else { x.values[i * p + j] })
                    .collect();
                let proposed_smooth = proposed
                    .iter()
                    .zip(y)
                    .map(|(&z, &y)| softplus(z) - y * z)
                    .sum::<f64>()
                    / n;
                let proposed_value = proposed_smooth
                    + spec.alpha * w.iter().map(|v| v.abs()).sum::<f64>()
                    + penalty * ((old + step * delta).abs() - old.abs());
                if proposed_value.is_finite() && proposed_value <= value + 1e-4 * step * slope {
                    if j == p {
                        b += step * delta;
                    } else {
                        w[j] += step * delta;
                    }
                    // Preserve exact zeros produced by the proximal update.
                    if step == 1.0 && j < p {
                        w[j] = candidate;
                    }
                    logits = proposed;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                return Err(convergence(
                    "L1 Logistic coordinate line search failed before meeting KKT tolerance.",
                ));
            }
        }
    }
    Err(convergence(format!(
        "L1 Logistic did not meet KKT tolerance in {} iterations.",
        spec.max_iterations
    )))
}
