use super::*;
use nalgebra::{DMatrix, DVector};

pub(super) fn fit_regression(
    x: &NumericBuffer,
    target: &ModelTarget,
    lambda: f64,
    rank_tolerance: f64,
) -> RadResult<(ModelState, FitReport)> {
    let ModelTarget::Regression { values: y } = target else {
        return Err(invalid("OLS and Ridge require a regression target."));
    };
    let n = x.rows as f64;
    let p = x.cols;
    let mut means = vec![0.0; p];
    for row in x.values.chunks(p) {
        for (m, &v) in means.iter_mut().zip(row) {
            *m += v / n;
        }
    }
    let ymean = y.iter().sum::<f64>() / n;
    let centered: Vec<_> = x
        .values
        .chunks(p)
        .flat_map(|row| row.iter().zip(&means).map(|(v, m)| v - m))
        .collect();
    let matrix = DMatrix::from_row_slice(x.rows, p, &centered);
    let rhs = DVector::from_iterator(x.rows, y.iter().map(|v| v - ymean));
    if centered.iter().any(|v| !v.is_finite()) || rhs.iter().any(|v| !v.is_finite()) {
        return Err(convergence(
            "Centered regression design overflowed; check input scale.",
        ));
    }
    let svd = matrix.svd(true, true);
    let maximum = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let threshold = maximum * rank_tolerance;
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&s| s > threshold && s > 0.0)
        .count()
        + 1;
    if lambda == 0.0 && (x.rows <= p || rank != p + 1) {
        return Err(invalid("OLS design is rank deficient (including intercept); select independent features or use positive Ridge regularization."));
    }
    let u = svd
        .u
        .ok_or_else(|| convergence("SVD did not return left singular vectors."))?;
    let vt = svd
        .v_t
        .ok_or_else(|| convergence("SVD did not return right singular vectors."))?;
    let mut coefficients = vec![0.0; p];
    for (k, &s) in svd.singular_values.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        let factor = if lambda == 0.0 {
            1.0 / s
        } else {
            s / (s * s + n * lambda)
        };
        let projection = u.column(k).dot(&rhs) * factor;
        for j in 0..p {
            coefficients[j] += vt[(k, j)] * projection;
        }
    }
    let intercept = ymean
        - means
            .iter()
            .zip(&coefficients)
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let residual: Vec<_> = x
        .values
        .chunks(p)
        .zip(y)
        .map(|(row, y)| {
            intercept
                + row
                    .iter()
                    .zip(&coefficients)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                - y
        })
        .collect();
    let objective = residual.iter().map(|r| r * r).sum::<f64>() / (2.0 * n)
        + lambda * coefficients.iter().map(|w| w * w).sum::<f64>() / 2.0;
    let mut gradient = vec![0.0; p + 1];
    for (row, &r) in x.values.chunks(p).zip(&residual) {
        for (j, &v) in row.iter().enumerate() {
            gradient[j] += r * v / n;
        }
        gradient[p] += r / n;
    }
    for j in 0..p {
        gradient[j] += lambda * coefficients[j];
    }
    if !objective.is_finite()
        || coefficients.iter().any(|v| !v.is_finite())
        || gradient.iter().any(|v| !v.is_finite())
    {
        return Err(convergence(
            "Linear solve produced nonfinite state; check feature scaling.",
        ));
    }
    Ok((
        ModelState::Linear {
            coefficients,
            intercept,
            rank,
        },
        FitReport {
            converged: true,
            iterations: 1,
            objective: Some(objective),
            gradient_or_kkt_inf: Some(norm_inf(&gradient)),
        },
    ))
}

pub(super) fn fit_logistic(
    x: &NumericBuffer,
    target: &ModelTarget,
    lambda: f64,
    tolerance: f64,
    max_iterations: usize,
) -> RadResult<(ModelState, FitReport)> {
    let ModelTarget::Classification { classes, labels } = target else {
        return Err(invalid(
            "Logistic Regression requires a classification target.",
        ));
    };
    let outputs = if classes.len() == 2 { 1 } else { classes.len() };
    let p = x.cols;
    let objective = |w: &[f64]| logistic_objective_gradient(x, labels, outputs, w, lambda);
    let (weights, report) = lbfgs(
        vec![0.0; outputs * (p + 1)],
        tolerance,
        max_iterations,
        objective,
    )?;
    let coefficients = weights
        .chunks(p + 1)
        .flat_map(|row| row[..p].iter().copied())
        .collect();
    let intercepts = weights.chunks(p + 1).map(|row| row[p]).collect();
    Ok((
        ModelState::Logistic {
            classes: classes.clone(),
            coefficients: buffer(outputs, p, coefficients),
            intercepts,
        },
        report,
    ))
}

fn logistic_objective_gradient(
    x: &NumericBuffer,
    labels: &[usize],
    outputs: usize,
    w: &[f64],
    lambda: f64,
) -> (f64, Vec<f64>) {
    let n = x.rows as f64;
    let width = x.cols + 1;
    let mut objective = 0.0;
    let mut gradient = vec![0.0; w.len()];
    for (row, &label) in x.values.chunks(x.cols).zip(labels) {
        let logits: Vec<_> = w
            .chunks(width)
            .map(|weights| {
                row.iter().zip(weights).map(|(a, b)| a * b).sum::<f64>() + weights[x.cols]
            })
            .collect();
        if outputs == 1 {
            let z = logits[0];
            let residual = sigmoid(z) - label as f64;
            objective += (softplus(z) - label as f64 * z) / n;
            for (j, &v) in row.iter().enumerate() {
                gradient[j] += v * residual / n;
            }
            gradient[x.cols] += residual / n;
        } else {
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probabilities = softmax(&logits);
            objective += (max + logits.iter().map(|z| (z - max).exp()).sum::<f64>().ln()
                - logits[label])
                / n;
            for (c, &probability) in probabilities.iter().enumerate() {
                let residual = probability - f64::from(c == label);
                for (j, &v) in row.iter().enumerate() {
                    gradient[c * width + j] += v * residual / n;
                }
                gradient[c * width + x.cols] += residual / n;
            }
        }
    }
    for c in 0..outputs {
        for j in 0..x.cols {
            let i = c * width + j;
            objective += lambda * w[i] * w[i] / 2.0;
            gradient[i] += lambda * w[i];
        }
    }
    (objective, gradient)
}

/// Deterministic L-BFGS with an Armijo line search and gradient-based stopping.
/// All fitted parameters share this implementation for binary and softmax LR.
fn lbfgs(
    mut w: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
    objective: impl Fn(&[f64]) -> (f64, Vec<f64>),
) -> RadResult<(Vec<f64>, FitReport)> {
    let (mut value, mut gradient) = objective(&w);
    let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    for iteration in 0..=max_iterations {
        let maximum = norm_inf(&gradient);
        if !value.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(convergence(
                "Logistic objective or gradient became nonfinite; check input scaling.",
            ));
        }
        if maximum <= tolerance {
            return Ok((
                w,
                FitReport {
                    converged: true,
                    iterations: iteration,
                    objective: Some(value),
                    gradient_or_kkt_inf: Some(maximum),
                },
            ));
        }
        if iteration == max_iterations {
            break;
        }
        let mut q = gradient.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let alpha = rho * dot(s, &q);
            for (q, y) in q.iter_mut().zip(y) {
                *q -= alpha * y;
            }
            alphas.push(alpha);
        }
        let scale = history
            .last()
            .map_or(1.0, |(s, y, _)| dot(s, y) / dot(y, y));
        let mut direction: Vec<_> = q.iter().map(|v| v * scale).collect();
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(y, &direction);
            for (d, s) in direction.iter_mut().zip(s) {
                *d += s * (alpha - beta);
            }
        }
        for v in &mut direction {
            *v = -*v;
        }
        let mut slope = dot(&gradient, &direction);
        if !slope.is_finite() || slope >= 0.0 {
            history.clear();
            direction = gradient.iter().map(|v| -v).collect();
            slope = -dot(&gradient, &gradient);
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let candidate: Vec<_> = w
                .iter()
                .zip(&direction)
                .map(|(a, b)| a + step * b)
                .collect();
            let (new_value, new_gradient) = objective(&candidate);
            if new_value.is_finite()
                && new_gradient.iter().all(|v| v.is_finite())
                && new_value <= value + 1e-4 * step * slope
            {
                accepted = Some((candidate, new_value, new_gradient));
                break;
            }
            step *= 0.5;
        }
        let Some((new_w, new_value, new_gradient)) = accepted else {
            return Err(convergence(format!(
                "Logistic line search failed with gradient infinity norm {maximum:.6e}."
            )));
        };
        let s: Vec<_> = new_w.iter().zip(&w).map(|(a, b)| a - b).collect();
        let y: Vec<_> = new_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        let sy = dot(&s, &y);
        if sy > 1e-14 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() && sy > 0.0 {
            if history.len() == 12 {
                history.remove(0);
            }
            history.push((s, y, 1.0 / sy));
        }
        w = new_w;
        value = new_value;
        gradient = new_gradient;
    }
    Err(convergence(format!("Logistic Regression did not converge in {max_iterations} iterations; gradient infinity norm {:.6e}.",norm_inf(&gradient))))
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
