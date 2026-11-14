//! Train-fitted imputation, categorical encoding and scaling.
use crate::contracts::{unique_nonempty, NumericBuffer, TabularData};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn invalid(message: impl Into<String>) -> RadiomicsError {
    RadiomicsError::new(ErrorCode::InvalidInput, message)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Scaling {
    #[default]
    None,
    Standard,
    #[serde(rename = "minmax")]
    MinMax,
    Robust,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AllMissingPolicy {
    #[default]
    Error,
    Drop,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PreprocessConfig {
    pub scaling: Scaling,
    pub all_missing: AllMissingPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericColumnState {
    pub name: String,
    pub median: f64,
    pub mean: f64,
    pub variance: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub q25: f64,
    pub q75: f64,
    pub center: f64,
    pub scale: f64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalColumnState {
    pub name: String,
    pub mode: String,
    pub categories: Vec<String>,
    /// Counts after training-mode imputation, in category order.
    pub counts: Vec<usize>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedPreprocessor {
    pub config: PreprocessConfig,
    pub training_rows: usize,
    pub numeric: Vec<NumericColumnState>,
    pub categorical: Vec<CategoricalColumnState>,
    pub dropped_numeric: Vec<String>,
    pub dropped_categorical: Vec<String>,
    pub output_names: Vec<String>,
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * p;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
}
fn output_names(
    numeric: &[NumericColumnState],
    categorical: &[CategoricalColumnState],
) -> Vec<String> {
    let mut names: Vec<_> = numeric.iter().map(|n| n.name.clone()).collect();
    for c in categorical {
        names.extend(
            c.categories
                .iter()
                .map(|category| format!("{}={category}", c.name)),
        );
    }
    names
}
fn scaled_parameters(column: &NumericColumnState, scaling: Scaling, rows: usize) -> (f64, f64) {
    let (center, scale) = match scaling {
        Scaling::None => (0.0, 1.0),
        Scaling::Standard => {
            let n = rows as f64;
            let epsilon = f64::EPSILON;
            let constant = column.variance
                <= n * epsilon * column.variance + (n * column.mean * epsilon).powi(2);
            (
                column.mean,
                if constant {
                    1.0
                } else {
                    column.variance.sqrt()
                },
            )
        }
        Scaling::MinMax => (column.minimum, column.maximum - column.minimum),
        Scaling::Robust => (column.median, column.q75 - column.q25),
    };
    let scale =
        if matches!(scaling, Scaling::MinMax | Scaling::Robust) && scale < 10.0 * f64::EPSILON {
            1.0
        } else {
            scale
        };
    (center, scale)
}

pub fn fit_preprocessor(
    data: &TabularData,
    config: &PreprocessConfig,
) -> RadResult<FittedPreprocessor> {
    data.validate()?;
    let mut numeric = Vec::new();
    let mut categorical = Vec::new();
    let mut dropped_numeric = Vec::new();
    let mut dropped_categorical = Vec::new();
    for (j, name) in data.numeric_names.iter().enumerate() {
        let mut observed: Vec<_> = (0..data.rows)
            .filter_map(|i| {
                let index = i * data.numeric.cols + j;
                (!data.numeric.missing[index]).then_some(data.numeric.values[index])
            })
            .collect();
        if observed.is_empty() {
            if config.all_missing == AllMissingPolicy::Error {
                return Err(invalid(format!("Training numeric feature '{name}' is entirely missing; set all_missing=drop explicitly to remove it.")));
            }
            dropped_numeric.push(name.clone());
            continue;
        }
        observed.sort_by(f64::total_cmp);
        let median = quantile(&observed, 0.5);
        let mut imputed: Vec<_> = (0..data.rows)
            .map(|i| {
                let index = i * data.numeric.cols + j;
                if data.numeric.missing[index] {
                    median
                } else {
                    data.numeric.values[index]
                }
            })
            .collect();
        let mean = imputed.iter().sum::<f64>() / data.rows as f64;
        let variance = imputed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.rows as f64;
        imputed.sort_by(f64::total_cmp);
        let mut state = NumericColumnState {
            name: name.clone(),
            median,
            mean,
            variance,
            minimum: imputed[0],
            maximum: imputed[data.rows - 1],
            q25: quantile(&imputed, 0.25),
            q75: quantile(&imputed, 0.75),
            center: 0.0,
            scale: 1.0,
        };
        (state.center, state.scale) = scaled_parameters(&state, config.scaling, data.rows);
        if [
            state.median,
            state.mean,
            state.variance,
            state.minimum,
            state.maximum,
            state.q25,
            state.q75,
            state.center,
            state.scale,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || state.scale <= 0.0
        {
            return Err(invalid(format!(
                "Training statistics overflowed for '{name}'; check input scale."
            )));
        }
        numeric.push(state);
    }
    for (j, name) in data.categorical_names.iter().enumerate() {
        let mut observed: BTreeMap<String, usize> = BTreeMap::new();
        let mut missing = 0usize;
        for i in 0..data.rows {
            if let Some(value) = &data.categorical_values[i * data.categorical_names.len() + j] {
                *observed.entry(value.clone()).or_insert(0) += 1;
            } else {
                missing += 1;
            }
        }
        if observed.is_empty() {
            if config.all_missing == AllMissingPolicy::Error {
                return Err(invalid(format!("Training categorical feature '{name}' is entirely missing; set all_missing=drop explicitly to remove it.")));
            }
            dropped_categorical.push(name.clone());
            continue;
        }
        let maximum = observed.values().copied().max().unwrap();
        let mode = observed
            .iter()
            .find(|(_, count)| **count == maximum)
            .unwrap()
            .0
            .clone();
        *observed.get_mut(&mode).unwrap() += missing;
        let (categories, counts) = observed.into_iter().unzip();
        categorical.push(CategoricalColumnState {
            name: name.clone(),
            mode,
            categories,
            counts,
        });
    }
    let names = output_names(&numeric, &categorical);
    if names.is_empty() {
        return Err(RadiomicsError::new(
            ErrorCode::EmptyResult,
            "All input features were explicitly dropped as entirely missing.",
        ));
    }
    unique_nonempty(
        &names,
        "Encoded feature names (one-hot names may collide with numeric names)",
    )?;
    let fitted = FittedPreprocessor {
        config: config.clone(),
        training_rows: data.rows,
        numeric,
        categorical,
        dropped_numeric,
        dropped_categorical,
        output_names: names,
    };
    fitted.validate()?;
    Ok(fitted)
}

impl FittedPreprocessor {
    pub fn validate(&self) -> RadResult<()> {
        if self.training_rows == 0 || self.output_names.is_empty() {
            return Err(invalid(
                "Fitted preprocessing requires training rows and nonempty output features.",
            ));
        }
        let mut all_names: Vec<_> = self
            .numeric
            .iter()
            .map(|c| c.name.clone())
            .chain(self.categorical.iter().map(|c| c.name.clone()))
            .collect();
        all_names.extend(self.dropped_numeric.clone());
        all_names.extend(self.dropped_categorical.clone());
        unique_nonempty(&all_names, "Fitted input features")?;
        if self.config.all_missing == AllMissingPolicy::Error
            && (!self.dropped_numeric.is_empty() || !self.dropped_categorical.is_empty())
        {
            return Err(invalid(
                "Dropped all-missing columns disagree with the recorded error policy.",
            ));
        }
        for c in &self.numeric {
            if [
                c.median, c.mean, c.variance, c.minimum, c.maximum, c.q25, c.q75, c.center, c.scale,
            ]
            .iter()
            .any(|v| !v.is_finite())
                || c.variance < 0.0
                || c.scale <= 0.0
                || c.minimum > c.q25
                || c.q25 > c.median
                || c.median > c.q75
                || c.q75 > c.maximum
            {
                return Err(invalid(
                    "Fitted numeric preprocessing statistics are invalid.",
                ));
            }
            if scaled_parameters(c, self.config.scaling, self.training_rows) != (c.center, c.scale)
            {
                return Err(invalid("Fitted scaling state disagrees with its recorded algorithm and training statistics."));
            }
        }
        for c in &self.categorical {
            if c.categories.is_empty()
                || c.categories.len() != c.counts.len()
                || c.counts.contains(&0)
                || c.counts.iter().try_fold(0usize, |n, &v| n.checked_add(v))
                    != Some(self.training_rows)
                || c.categories.windows(2).any(|w| w[0] >= w[1])
                || c.categories.binary_search(&c.mode).is_err()
            {
                return Err(invalid(
                    "Fitted category order, mode or training counts are invalid.",
                ));
            }
        }
        let expected = output_names(&self.numeric, &self.categorical);
        unique_nonempty(&expected, "Encoded feature names")?;
        if expected != self.output_names {
            return Err(invalid(
                "Fitted output feature order disagrees with encoded state.",
            ));
        }
        Ok(())
    }
    /// Variances after imputation/encoding and before scaling, in output order.
    /// G must pass these to fit_selector when low-variance filtering follows scaling.
    pub fn unscaled_variances(&self) -> RadResult<Vec<f64>> {
        self.validate()?;
        let mut values: Vec<_> = self.numeric.iter().map(|n| n.variance).collect();
        for c in &self.categorical {
            values.extend(c.counts.iter().map(|&n| {
                let p = n as f64 / self.training_rows as f64;
                p * (1.0 - p)
            }));
        }
        Ok(values)
    }
    pub fn transform(&self, data: &TabularData) -> RadResult<NumericBuffer> {
        self.validate()?;
        data.validate()?;
        let numeric: Vec<_> = self
            .numeric
            .iter()
            .map(|c| {
                data.numeric_names
                    .iter()
                    .position(|name| name == &c.name)
                    .ok_or_else(|| {
                        RadiomicsError::new(
                            ErrorCode::Alignment,
                            format!(
                                "Missing numeric feature '{}' or incorrect column type.",
                                c.name
                            ),
                        )
                    })
            })
            .collect::<RadResult<_>>()?;
        let categorical: Vec<_> = self
            .categorical
            .iter()
            .map(|c| {
                data.categorical_names
                    .iter()
                    .position(|name| name == &c.name)
                    .ok_or_else(|| {
                        RadiomicsError::new(
                            ErrorCode::Alignment,
                            format!(
                                "Missing categorical feature '{}' or incorrect column type.",
                                c.name
                            ),
                        )
                    })
            })
            .collect::<RadResult<_>>()?;
        let count = data
            .rows
            .checked_mul(self.output_names.len())
            .ok_or_else(|| invalid("Transformed feature dimensions overflow usize."))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| invalid("Transformed feature buffer cannot be allocated."))?;
        for i in 0..data.rows {
            for (state, &column) in self.numeric.iter().zip(&numeric) {
                let j = i * data.numeric.cols + column;
                let v = if data.numeric.missing[j] {
                    state.median
                } else {
                    data.numeric.values[j]
                };
                let transformed = (v - state.center) / state.scale;
                if !transformed.is_finite() {
                    return Err(invalid(format!(
                        "Transformed feature '{}' overflowed; check input scale.",
                        state.name
                    )));
                }
                values.push(transformed);
            }
            for (state, &column) in self.categorical.iter().zip(&categorical) {
                let value = data.categorical_values[i * data.categorical_names.len() + column]
                    .as_ref()
                    .unwrap_or(&state.mode);
                let selected = state.categories.binary_search(value).map_err(|_| {
                    RadiomicsError::new(
                        ErrorCode::Alignment,
                        format!(
                            "Unknown category '{value}' for fitted feature '{}'.",
                            state.name
                        ),
                    )
                })?;
                values.extend((0..state.categories.len()).map(|j| f64::from(j == selected)));
            }
        }
        Ok(NumericBuffer {
            rows: data.rows,
            cols: self.output_names.len(),
            missing: vec![false; values.len()],
            values,
        })
    }
}

#[cfg(test)]
mod tests;
