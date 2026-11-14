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
