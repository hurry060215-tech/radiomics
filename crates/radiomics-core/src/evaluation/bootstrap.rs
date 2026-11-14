use super::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapInterval {
    pub estimate: f64,
    pub confidence: f64,
    pub low: f64,
    pub high: f64,
    pub replicate_scores: Vec<f64>,
    pub resampling_source: String,
    pub seed: Option<u64>,
    pub rng_scheme: Option<String>,
    pub rng_namespace: Option<String>,
}
fn finish(
    estimate: f64,
    replicate_scores: Vec<f64>,
    confidence: f64,
) -> RadResult<BootstrapInterval> {
    super::inference::confidence(confidence)?;
    if replicate_scores.len() < 2 {
        return Err(invalid(
            "Bootstrap needs at least two explicit replicate index arrays.",
        ));
    }
    let alpha = (1.0 - confidence) / 2.0;
    let low = crate::statistics::quantile_linear(&replicate_scores, alpha)?;
    let high = crate::statistics::quantile_linear(&replicate_scores, 1.0 - alpha)?;
    Ok(BootstrapInterval {
        estimate,
        confidence,
        low,
        high,
        replicate_scores,
        resampling_source: "explicit_indices".into(),
        seed: None,
        rng_scheme: None,
        rng_namespace: None,
    })
}
fn indices(draws: &[Vec<usize>], n: usize) -> RadResult<()> {
    if draws.len() < 2
        || draws
            .iter()
            .any(|row| row.len() != n || row.iter().any(|&i| i >= n))
    {
        Err(invalid("Bootstrap replicate indices must each contain exactly the evaluation row count, with all indices in bounds and at least two replicates."))
    } else {
        Ok(())
    }
}
/// Deterministic binary AUC percentile interval from frozen/explicit indices.
/// No RNG is created here. Every replicate must retain both binary classes.
pub fn bootstrap_auc_explicit(
    y: &[u8],
    scores: &[f64],
    draws: &[Vec<usize>],
    confidence: f64,
) -> RadResult<BootstrapInterval> {
    super::inference::confidence(confidence)?;
    let estimate = crate::auc(y, scores).map_err(kernel)?;
    indices(draws, y.len())?;
    let mut values = Vec::with_capacity(draws.len());
    for draw in draws {
        let labels: Vec<_> = draw.iter().map(|&i| y[i]).collect();
        let probabilities: Vec<_> = draw.iter().map(|&i| scores[i]).collect();
        values.push(crate::auc(&labels, &probabilities).map_err(kernel)?);
    }
    finish(estimate, values, confidence)
}
/// Generic task comparison-metric interval. Classification draws must retain
/// every declared class; callers supply explicit stratified draws as needed.
pub fn bootstrap_model_score_explicit(
    task: Task,
    target: &ModelTarget,
    prediction: &ModelPrediction,
    draws: &[Vec<usize>],
    confidence: f64,
) -> RadResult<BootstrapInterval> {
    super::inference::confidence(confidence)?;
    let estimate = model_score(task, target, prediction)?;
    let n = match target {
        ModelTarget::Classification { labels, .. } => labels.len(),
        ModelTarget::Regression { values } => values.len(),
    };
    indices(draws, n)?;
    let mut values = Vec::with_capacity(draws.len());
    for draw in draws {
        values.push(score_for_draw(task, target, prediction, draw)?);
    }
    finish(estimate, values, confidence)
}

fn score_for_draw(
    task: Task,
    target: &ModelTarget,
    prediction: &ModelPrediction,
    draw: &[usize],
) -> RadResult<f64> {
    let n = draw.len();
    let (target, prediction) = match (target, prediction) {
        (
            ModelTarget::Classification { classes, labels },
            ModelPrediction::Classification { probabilities, .. },
        ) => {
            let data: Vec<_> = draw
                .iter()
                .flat_map(|&i| {
                    probabilities.values[i * probabilities.cols..(i + 1) * probabilities.cols]
                        .iter()
                        .copied()
                })
                .collect();
            (
                ModelTarget::Classification {
                    classes: classes.clone(),
                    labels: draw.iter().map(|&i| labels[i]).collect(),
                },
                ModelPrediction::Classification {
                    classes: classes.clone(),
                    probabilities: crate::contracts::NumericBuffer {
                        rows: n,
                        cols: probabilities.cols,
                        missing: vec![false; data.len()],
                        values: data,
                    },
                },
            )
        }
        (ModelTarget::Regression { values: y }, ModelPrediction::Regression { values: p }) => (
            ModelTarget::Regression {
                values: draw.iter().map(|&i| y[i]).collect(),
            },
            ModelPrediction::Regression {
                values: draw.iter().map(|&i| p[i]).collect(),
            },
        ),
        _ => {
            return Err(invalid(
                "Bootstrap target/prediction task resolution failed.",
            ))
        }
    };
    model_score(task, &target, &prediction)
}

/// Stratified multiclass primary-score bootstrap using the shared versioned
/// component RNG. Binary legacy sampling deliberately has its separate entry.
pub fn bootstrap_multiclass_seeded(
    target: &ModelTarget,
    prediction: &ModelPrediction,
    replicates: usize,
    seed: u64,
    confidence: f64,
) -> RadResult<BootstrapInterval> {
    seeded(
        Task::Multiclass,
        target,
        prediction,
        replicates,
        seed,
        confidence,
    )
}
/// Ordinary row bootstrap for the regression primary metric (RMSE).
pub fn bootstrap_regression_seeded(
    target: &ModelTarget,
    prediction: &ModelPrediction,
    replicates: usize,
    seed: u64,
    confidence: f64,
) -> RadResult<BootstrapInterval> {
    seeded(
        Task::Regression,
        target,
        prediction,
        replicates,
        seed,
        confidence,
    )
}
fn seeded(
    task: Task,
    target: &ModelTarget,
    prediction: &ModelPrediction,
    replicates: usize,
    seed: u64,
    confidence: f64,
) -> RadResult<BootstrapInterval> {
    super::inference::confidence(confidence)?;
    let estimate = model_score(task, target, prediction)?;
    if replicates < 2 {
        return Err(invalid("Bootstrap requires at least two replicates."));
    }
    let (n,groups,tag)=match (task,target) {
        (Task::Multiclass,ModelTarget::Classification{classes,labels})=>(labels.len(),Some((0..classes.len()).map(|c|labels.iter().enumerate().filter_map(|(i,&label)|(c==label).then_some(i)).collect::<Vec<_>>()).collect::<Vec<_>>()),1u8),
        (Task::Regression,ModelTarget::Regression{values})=>(values.len(),None,2u8),
        _=>return Err(invalid("The new component RNG entry points support explicitly multiclass or regression tasks; binary sampling uses the legacy path.")),
    };
    let namespace = "evaluation-bootstrap-v1";
    let mut values = Vec::new();
    values
        .try_reserve_exact(replicates)
        .map_err(|_| invalid("Bootstrap replicate result allocation failed."))?;
    for replicate in 0..replicates {
        let mut identifier = vec![tag];
        identifier.extend_from_slice(&(replicate as u64).to_le_bytes());
        let mut rng = crate::random::component_rng(seed, namespace, &identifier);
        let mut draw = Vec::with_capacity(n);
        if let Some(groups) = &groups {
            for group in groups {
                for _ in 0..group.len() {
                    draw.push(group[rng.gen_range(0..group.len())]);
                }
            }
        } else {
            draw.extend((0..n).map(|_| rng.gen_range(0..n)));
        }
        values.push(score_for_draw(task, target, prediction, &draw)?);
    }
    let mut report = finish(estimate, values, confidence)?;
    report.resampling_source = if task == Task::Multiclass {
        "stratified_multiclass".into()
    } else {
        "ordinary_regression".into()
    };
    report.seed = Some(seed);
    report.rng_scheme = Some(crate::random::RNG_SCHEME.into());
    report.rng_namespace = Some(namespace.into());
    Ok(report)
}
