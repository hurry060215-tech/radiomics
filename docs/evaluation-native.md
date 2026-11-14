# Native evaluation

`evaluation::model_score(task, target, prediction)` returns the actual metric
for model comparison: binary ROC AUC, multiclass macro one-vs-rest ROC AUC, or
regression RMSE. It does not negate regression error; the research workflow
explicitly minimizes RMSE and maximizes AUC. Binary and class-wise AUC call the
existing single `crate::auc` implementation. The public `rmse` function is the
shared RMSE calculation for both comparison and regression reports.

Target and prediction types must agree with the task. Classification requires
the exact ordered class-name vector, every declared class in the evaluation
target, matching buffer dimensions, no missing probabilities, finite values in
0–1 and per-row probability sums within `1e-10` of one. Invalid predictions are
errors rather than silently renormalized. Regression requires equal, nonempty,
finite vectors and a representable RMSE.

Each row is the explicit evaluation unit selected by the caller. Patient/slide
aggregation and provenance checks belong to the typed prediction/research
workflow; repeated lesions cannot be treated as independent patients merely
because this numerical function accepts row arrays. Tests compare all frozen
inner/outer patient-level comparison scores for the three tasks, alongside
known hand values and invalid class-order/missing-class/probability cases.

## Reports and curves

`binary_metrics(y, scores, threshold)` reuses the core binary summary and adds
positive-class and per-class precision/recall/F1, accuracy, Brier score, clipped
log loss, AP and separately named trapezoidal PR area. Precision/recall/F1 use
the frozen zero-division value zero. PPV and NPV retain undefined NaN values when
their prediction denominator is zero, preserving the existing public semantics.

`classification_metrics` returns an exact integer confusion matrix, ordered
per-class statistics and macro/micro/support-weighted summaries. Multiclass hard
predictions use argmax, with ties assigned to the lowest declared class index.
It also returns explicit OvR binary reports, macro/weighted/micro OvR AUC and
average precision, log loss and the sum-across-classes Brier score. Binary
thresholding uses `p >= threshold`. The frozen log loss clips probabilities to
`[1e-15,1-1e-15]` and normalizes the clipped row before evaluating its true class.

`regression_metrics` provides MAE, the shared RMSE and R². Actual sklearn 1.0.2
reference behavior is retained: constant-perfect R² is one, constant-imperfect
R² is zero, and a single-observation R² is NaN. The latter is serialized as an
explicit nonfinite state rather than replaced with a finite score.

`roc_curve` emits every distinct-score point (`drop_intermediate=False`), with
the frozen sklearn 1.0.2 initial threshold `max(score)+1`. It does not replace
that convention with the infinity sentinel of later versions.
`precision_recall_curve` reverses distinct thresholds only through the first
full-recall point, then appends precision one/recall zero. Average precision is
the step integral; trapezoidal PR AUC is a different, explicitly named result.

`calibration_curve` supports explicit uniform or quantile bins. The frozen
1.0.2 uniform edges span `0` through `1+1e-8`; quantile edges use percentile
probabilities and increase only the last edge by `1e-8`. Membership follows
`digitize(right=False)-1`. It returns every input's bin index, all bin counts,
and ordered means/fractions for nonempty bins. Empty bins retain counts of zero
and are omitted from the mean arrays, matching the reference.

Golden tests compare the complete binary and multiclass reports, every ROC/PR
threshold and point, exact calibration membership/counts, both calibration
strategies, regression results and declared degenerate states against the
independently frozen patient-level reference. Numerical rows already represent
the chosen evaluation unit; aggregation is not inferred from row count.

## Binary inference and explicit bootstrap

`delong_interval` reuses the core placement-variance calculation and obtains
its normal quantile from `statistics::Distribution`. Limits are explicitly
bounded to 0–1, matching the frozen baseline. `paired_delong` reuses the paired
variance kernel and the shared normal survival function. Equal scores with zero
paired variance return z zero/p one; a nonzero AUC difference with zero variance
retains an explicit infinite z and p zero.

`decision_curve` reuses the existing sorted-threshold net-benefit kernel, keeping
threshold order and repeated thresholds. It adds the corresponding treat-all
and treat-none vectors. Thresholds must lie strictly between zero and one.

`hosmer_lemeshow` now constructs quantile bins in Rust using the shared linear
quantile primitive. Duplicate edges are removed, intervals are right-closed
with the minimum included, and only nonempty groups enter the test. It calls
the existing H–L statistic kernel and shared chi-square survival function.
Fewer than three distinct/nonempty bins or expected bin probabilities zero/one
are errors. Results expose raw quantile edges, per-row membership and the group
table; presentation rounding of interval labels belongs to the Python display
adapter, not the numerical binning algorithm.

DeLong, decision curves and H–L accept binary labels and scores only. Applying
them to a particular multiclass OvR problem requires the caller to construct
and name that explicit binary target; there is no automatic multiclass-overall
interpretation.

`bootstrap_auc_explicit` computes a percentile AUC interval from supplied index
arrays and creates no random-number generator. Each draw must have the declared
evaluation row count, valid indices and both classes. Invalid draws are errors,
not silently omitted or redrawn. Every replicate AUC calls the existing core
kernel; interval limits use shared linear quantiles.
`bootstrap_model_score_explicit` applies the same approach to the task's primary
comparison metric. Multiclass draws must retain every declared class. The
result records all replicate scores and its `explicit_indices` source.

Tests reproduce the baseline's 16 frozen bootstrap draws and the newer
patient-level fixture's 64 draws, including every replicate AUC and CI. The
baseline DeLong interval/test, DCA repeated thresholds, H–L group counts,
expected/observed values, statistic and p-value also pass the original frozen
tolerances. Seeded entry points must delegate sampling to the shared random
module; these explicit-index functions contain no separate RNG implementation.
