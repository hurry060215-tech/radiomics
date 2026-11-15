use super::*;
use crate::random::{component_rng, RNG_SCHEME};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Count { count: usize },
}
impl MaxFeatures {
    pub(super) fn count(self, features: usize) -> RadResult<usize> {
        let count = match self {
            Self::All => features,
            Self::Sqrt => (features as f64).sqrt().floor() as usize,
            Self::Count { count } => count,
        };
        if count == 0 || count > features {
            Err(invalid(
                "Forest max_features must be between one and the input feature count.",
            ))
        } else {
            Ok(count)
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TreeNode {
    Leaf {
        prediction: Vec<f64>,
        samples: usize,
        impurity: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
        samples: usize,
        impurity: f64,
        impurity_decrease: f64,
    },
}
impl TreeNode {
    fn samples(&self) -> usize {
        match self {
            Self::Leaf { samples, .. } | Self::Split { samples, .. } => *samples,
        }
    }
}
/// Flat child indices permit complete topology validation without recursive
/// deserialization or an unbounded call stack on deep fitted trees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
}
impl DecisionTree {
    pub fn validate(&self, feature_count: usize, classes: &[String]) -> RadResult<()> {
        if self.nodes.is_empty() {
            return Err(invalid("A fitted tree has no root."));
        }
        let outputs = classes.len().max(1);
        let mut parents = vec![0usize; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                TreeNode::Leaf {
                    prediction,
                    samples,
                    impurity,
                } => {
                    if *samples == 0
                        || !impurity.is_finite()
                        || *impurity < 0.0
                        || prediction.len() != outputs
                        || prediction.iter().any(|v| !v.is_finite())
                        || (!classes.is_empty()
                            && (prediction.iter().any(|v| !(0.0..=1.0).contains(v))
                                || (prediction.iter().sum::<f64>() - 1.0).abs() > 1e-10))
                    {
                        return Err(invalid(
                            "Tree leaf prediction, sample count or impurity is invalid.",
                        ));
                    }
                }
                TreeNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    samples,
                    impurity,
                    impurity_decrease,
                } => {
                    if *feature >= feature_count
                        || !threshold.is_finite()
                        || *left <= i
                        || *right <= i
                        || *left >= self.nodes.len()
                        || *right >= self.nodes.len()
                        || left == right
                        || *samples == 0
                        || !impurity.is_finite()
                        || *impurity < 0.0
                        || !impurity_decrease.is_finite()
                        || *impurity_decrease < 0.0
                    {
                        return Err(invalid(
                            "Tree split geometry, topology or impurity is invalid.",
                        ));
                    }
                    if self.nodes[*left]
                        .samples()
                        .checked_add(self.nodes[*right].samples())
                        != Some(*samples)
                    {
                        return Err(invalid(
                            "Tree child sample counts do not sum to the parent.",
                        ));
                    }
                    parents[*left] += 1;
                    parents[*right] += 1;
                }
            }
        }
        if parents[0] != 0 || parents[1..].iter().any(|&n| n != 1) {
            return Err(invalid(
                "Tree contains unreachable or multiply referenced nodes.",
            ));
        }
        Ok(())
    }
    fn leaf(&self, row: &[f64]) -> &[f64] {
        let mut index = 0;
        loop {
            match &self.nodes[index] {
                TreeNode::Leaf { prediction, .. } => return prediction,
                TreeNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    index = if row[*feature] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }
}

struct NodeSummary {
    prediction: Vec<f64>,
    impurity: f64,
    counts: Vec<usize>,
    mean: f64,
}
fn summary(indices: &[usize], target: &ModelTarget) -> NodeSummary {
    let n = indices.len() as f64;
    match target {
        ModelTarget::Classification { classes, labels } => {
            let mut counts = vec![0; classes.len()];
            for &i in indices {
                counts[labels[i]] += 1;
            }
            let prediction: Vec<_> = counts.iter().map(|&n_class| n_class as f64 / n).collect();
            let impurity = (1.0 - prediction.iter().map(|p| p * p).sum::<f64>()).max(0.0);
            NodeSummary {
                prediction,
                impurity,
                counts,
                mean: 0.0,
            }
        }
        ModelTarget::Regression { values } => {
            let first = values[indices[0]];
            if indices.iter().all(|&i| values[i] == first) {
                return NodeSummary {
                    prediction: vec![first],
                    impurity: 0.0,
                    counts: vec![],
                    mean: first,
                };
            }
            let mean = indices.iter().map(|&i| values[i] / n).sum::<f64>();
            let impurity = indices
                .iter()
                .map(|&i| (values[i] - mean).powi(2) / n)
                .sum::<f64>();
            NodeSummary {
                prediction: vec![mean],
                impurity,
                counts: vec![],
                mean,
            }
        }
    }
}
struct Split {
    feature: usize,
    threshold: f64,
    gain: f64,
}

fn best_split(
    x: &NumericBuffer,
    target: &ModelTarget,
    indices: &[usize],
    node: &NodeSummary,
    features: &[usize],
    min_leaf: usize,
) -> Option<Split> {
    let n = indices.len();
    let nf = n as f64;
    let mut best: Option<Split> = None;
    for &feature in features {
        let mut sorted = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            x.values[a * x.cols + feature]
                .total_cmp(&x.values[b * x.cols + feature])
                .then(a.cmp(&b))
        });
        let mut left_counts = vec![0usize; node.counts.len()];
        let mut right_counts = node.counts.clone();
        let (mut left_sum, mut right_sum) = (0.0, 0.0);
        if let ModelTarget::Regression { values } = target {
            right_sum = indices.iter().map(|&i| values[i] - node.mean).sum();
        }
        let base = match target {
            ModelTarget::Classification { .. } => {
                node.counts.iter().map(|&c| (c as f64).powi(2)).sum::<f64>() / nf
            }
            ModelTarget::Regression { .. } => right_sum * right_sum / nf,
        };
        for cut in 1..n {
            let i = sorted[cut - 1];
            match target {
                ModelTarget::Classification { labels, .. } => {
                    left_counts[labels[i]] += 1;
                    right_counts[labels[i]] -= 1;
                }
                ModelTarget::Regression { values } => {
                    let centered = values[i] - node.mean;
                    left_sum += centered;
                    right_sum -= centered;
                }
            }
            if cut < min_leaf || n - cut < min_leaf {
                continue;
            }
            let lower = x.values[i * x.cols + feature];
            let upper = x.values[sorted[cut] * x.cols + feature];
            if lower == upper {
                continue;
            }
            let nl = cut as f64;
            let nr = (n - cut) as f64;
            let score = match target {
                ModelTarget::Classification { .. } => {
                    left_counts
                        .iter()
                        .map(|&c| (c as f64).powi(2) / nl)
                        .sum::<f64>()
                        + right_counts
                            .iter()
                            .map(|&c| (c as f64).powi(2) / nr)
                            .sum::<f64>()
                }
                ModelTarget::Regression { .. } => {
                    left_sum * left_sum / nl + right_sum * right_sum / nr
                }
            };
            let gain = (score - base) / nf;
            if !gain.is_finite() || gain <= 0.0 {
                continue;
            }
            let mut threshold = lower * 0.5 + upper * 0.5;
            if threshold >= upper {
                threshold = lower;
            }
            if best.as_ref().is_none_or(|b| {
                gain > b.gain
                    || (gain == b.gain
                        && (feature < b.feature
                            || (feature == b.feature && threshold < b.threshold)))
            }) {
                best = Some(Split {
                    feature,
                    threshold,
                    gain,
                });
            }
        }
    }
    best
}

pub(super) fn fit(
    spec: &ModelSpec,
    x: &NumericBuffer,
    target: &ModelTarget,
) -> RadResult<(ModelState, FitReport)> {
    let ModelSpec::RandomForest {
        trees,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        seed,
    } = spec
    else {
        return Err(invalid("Forest fit requires a RandomForest specification."));
    };
    let candidates = max_features.count(x.cols)?;
    let classes = match target {
        ModelTarget::Classification { classes, .. } => classes.clone(),
        _ => vec![],
    };
    let mut fitted = Vec::new();
    fitted
        .try_reserve_exact(*trees)
        .map_err(|_| invalid("Requested forest tree count cannot be allocated."))?;
    for tree_index in 0..*trees {
        let mut rng = component_rng(
            *seed,
            "random-forest-tree-v1",
            &(tree_index as u64).to_le_bytes(),
        );
        let indices: Vec<_> = if *bootstrap {
            (0..x.rows).map(|_| rng.gen_range(0..x.rows)).collect()
        } else {
            (0..x.rows).collect()
        };
        let placeholder = || TreeNode::Leaf {
            prediction: vec![],
            samples: 0,
            impurity: 0.0,
        };
        let mut tree = DecisionTree {
            nodes: vec![placeholder()],
        };
        let mut pending = vec![(0usize, indices, 0usize)];
        while let Some((index, indices, depth)) = pending.pop() {
            let node = summary(&indices, target);
            if !node.impurity.is_finite() || node.prediction.iter().any(|v| !v.is_finite()) {
                return Err(invalid(
                    "Forest impurity or leaf mean overflowed; check target scale.",
                ));
            }
            tree.nodes[index] = TreeNode::Leaf {
                prediction: node.prediction.clone(),
                samples: indices.len(),
                impurity: node.impurity,
            };
            if node.impurity == 0.0
                || indices.len() < *min_samples_split
                || indices.len() / 2 < *min_samples_leaf
                || max_depth.is_some_and(|maximum| depth >= maximum)
            {
                continue;
            }
            let mut features: Vec<_> = (0..x.cols).collect();
            if candidates < x.cols {
                for i in 0..candidates {
                    let j = rng.gen_range(i..x.cols);
                    features.swap(i, j);
                }
                features.truncate(candidates);
                features.sort_unstable();
            }
            if let Some(split) =
                best_split(x, target, &indices, &node, &features, *min_samples_leaf)
            {
                let (left, right): (Vec<_>, Vec<_>) = indices
                    .iter()
                    .copied()
                    .partition(|&i| x.values[i * x.cols + split.feature] <= split.threshold);
                if left.len() < *min_samples_leaf || right.len() < *min_samples_leaf {
                    return Err(invalid(
                        "Forest split threshold failed to reproduce its partition.",
                    ));
                }
                let left_index = tree.nodes.len();
                tree.nodes.push(placeholder());
                let right_index = tree.nodes.len();
                tree.nodes.push(placeholder());
                tree.nodes[index] = TreeNode::Split {
                    feature: split.feature,
                    threshold: split.threshold,
                    left: left_index,
                    right: right_index,
                    samples: indices.len(),
                    impurity: node.impurity,
                    impurity_decrease: split.gain,
                };
                pending.push((right_index, right, depth + 1));
                pending.push((left_index, left, depth + 1));
            }
        }
        tree.validate(x.cols, &classes)?;
        fitted.push(tree);
    }
    Ok((
        ModelState::Forest {
            classes,
            trees: fitted,
            rng_scheme: RNG_SCHEME.into(),
        },
        FitReport::direct(),
    ))
}

pub(super) fn predict(
    trees: &[DecisionTree],
    classes: &[String],
    x: &NumericBuffer,
) -> RadResult<ModelPrediction> {
    let outputs = classes.len().max(1);
    let mut values = vec![0.0; x.rows * outputs];
    for (i, row) in x.values.chunks(x.cols).enumerate() {
        for tree in trees {
            for (j, &v) in tree.leaf(row).iter().enumerate() {
                values[i * outputs + j] += v / trees.len() as f64;
            }
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Forest prediction is not finite."));
    }
    if classes.is_empty() {
        Ok(ModelPrediction::Regression { values })
    } else {
        Ok(ModelPrediction::Classification {
            classes: classes.to_vec(),
            probabilities: buffer(x.rows, classes.len(), values),
        })
    }
}
