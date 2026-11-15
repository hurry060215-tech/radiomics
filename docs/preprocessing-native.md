# Native train-fitted table preprocessing

`preprocessing::fit_preprocessor(&TabularData, &PreprocessConfig)` fits numerical
median imputation, categorical mode imputation, full one-hot encoding and
optional numeric scaling. It returns a typed, serializable
`FittedPreprocessor`; `transform(&TabularData)` uses only those saved training
statistics. The input contains feature columns only. Sample/patient IDs, split
labels and outcomes belong to separate workflow objects.

`PreprocessConfig { scaling, all_missing }` has explicit choices:

- `Scaling::None`, `Standard`, `MinMax`, `Robust`.
- `AllMissingPolicy::Error` (default) or explicit `Drop`.

An entirely missing training column is an error under the default policy. With
explicit dropping it is recorded in `dropped_numeric` or
`dropped_categorical`. Dropping every feature is an error. Prediction data may
omit dropped columns, since they are not model inputs, but must contain every
retained input with its fitted numeric/categorical type.

Numeric medians use observed training values. Imputed training values determine
all scaling statistics. Standard scaling uses population variance, with
float64 near-constant detection matching the frozen reference. MinMax uses the
training minimum and maximum, maps the training range to 0–1, and does not clip
external values. Robust scaling centers at the median and divides by q75−q25,
using linearly interpolated quartiles; there is no Gaussian unit-variance
correction. Zero or numerically near-zero MinMax/IQR denominators become one.
All medians, means, variances, extrema, quartiles and effective center/scale
parameters are saved in typed state.

Categorical modes use the lexicographically smallest observed value when counts
tie. Category levels are sorted lexicographically, with one column for every
level and no reference-level drop. Missing prediction values use the training
mode; unseen observed categories are explicit errors. One-hot columns always
remain 0/1 and are never scaled. Output order is retained numeric input order,
then categorical input order and sorted category order.

One-hot names are `column=category`. The complete output-name vector is checked
for uniqueness, including collisions against numeric names and names produced
by other categorical columns. An ambiguous name is an error rather than a
silently renamed feature. Empty strings are ordinary categorical values;
`None` is the missing marker.

Transforms align retained inputs by name and validate types, allowing arbitrary
column reordering. Extra unused columns do not change fitted state. Serialization
stores no executable Python objects. Validation rejects malformed dimensions,
nonfinite statistics, invalid category ordering/counts and a scaling state that
disagrees with the recorded configuration.

`unscaled_variances()` returns population variances after imputation/encoding
and before scaling, in `output_names` order. The research pipeline must pass
these explicit variances to low-variance selection after scaling. Otherwise,
Standard scaling would give a tiny but varying feature variance one and change
the intended raw-variance screen. All fitting, including imputation, category
discovery and scaling, must be repeated using only the current training fold.

The independently frozen six-row table fixture verifies median/mode arithmetic,
three scaler states, training/evaluation transforms, one-hot output and raw
variances. Additional tests cover column reordering, immutable fitted state,
unknown categories, all-missing policy, type mismatch and encoded-name
collisions. Complete selection and nested-CV acceptance are separate from these
preprocessing checks.

## Native selector

```rust,ignore
let preprocessing = fit_preprocessor(&training, &preprocess_config)?;
let x = preprocessing.transform(&training)?;
let raw_variances = preprocessing.unscaled_variances()?;
let selector = fit_selector(
    &x, &preprocessing.output_names, &target, &selector_config,
    Some(&raw_variances),
)?;
let selected_x = selector.transform(&x, &preprocessing.output_names)?;
let selected_names = selector.selected_names()?;
```

`SelectorConfig` stores the raw-variance threshold, optional absolute Pearson
correlation threshold, univariate method and p-value threshold, optional BH
adjustment, optional fixed `SparseSpec`, and explicit coefficient threshold.
`fit_selector` requires a resolved, finite, missing-free numeric matrix and
unique feature names. Its optional variance reference must have one finite,
nonnegative value per input feature; it is recorded as `ExplicitUnscaled`.
Omitting it explicitly chooses population variances of the supplied matrix.

The stage order is constant/low variance → univariate/BH → absolute correlation
→ sparse coefficients. Constants are removed even when roundoff in a mean-based
variance formula would make an exactly constant buffer's variance slightly
positive. Variance must be strictly greater than its threshold. Univariate
adjusted p-values must be strictly less than their threshold. Correlation
removes a later feature only when the absolute coefficient is strictly greater
than its threshold; retained features follow original input order, including
when the correlation is negative. An empty result at any stage returns
`ErrorCode::EmptyResult` and does not relax any parameter.

`UnivariateMethod::Auto` resolves from `ModelTarget`: Mann–Whitney for binary,
Kruskal–Wallis for multiclass, and Pearson target correlation for regression.
Explicit incompatible choices are errors. All tests, distribution calculations
and Benjamini–Hochberg adjustment call the single shared `statistics` module.
This module also uses its shared mean/variance and linearly interpolated
quantile primitives for preprocessing. It contains no second CDF or test engine.

Sparse fitting calls `models::fit_sparse` once on the retained training columns,
with the explicit fixed alpha and stopping rules. For multiclass, the caller
chooses OvR L1 Logistic; squared-loss LASSO rejects multiclass class indices.
Selected sparse columns are the union, across all fitted score rows, of
coefficients whose absolute values exceed the stored `coefficient_threshold`.
The default threshold is `1e-6`; the exact solver support and this selection
threshold are distinct concepts. Hyperparameter tuning and fold construction
belong to the patient-grouped research workflow, which must refit the entire
preprocessor and selector for each training fold.

`FittedSelector` saves original feature/class order, variance source and values,
constant-feature indices, all stage index sets, univariate statistics and raw/
adjusted p-values, correlation exclusions and the optional typed sparse fit.
Validation enforces ordered nested subsets, coherent task/method choices and
nonempty final features. `selected_names()` validates the state and returns a
`RadResult<Vec<String>>`; `transform` aligns only the retained columns by name
and reports missing columns explicitly. JSON reload retains transform identity.

Frozen table tests compare all three task-specific statistics and BH values,
stable correlation exclusions and final selected names. Additional integration
tests prove that the saved raw variance screen still drops tiny features after
Standard scaling, exercise sparse solver reuse and per-class union thresholds,
and verify reordered inputs plus zero-feature failures. Full nested-CV leakage
acceptance is performed by the research workflow rather than inferred from a
single selector fit.
