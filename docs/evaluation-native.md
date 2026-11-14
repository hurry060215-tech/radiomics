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
