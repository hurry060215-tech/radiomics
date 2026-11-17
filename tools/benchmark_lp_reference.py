"""Benchmark-only public sklearn composition; never imports the native project.

Matches the frozen workflow fit/tune conventions without running Golden fixture
generation, perturbation refits or objective/gradient oracles inside timed work.
"""

import warnings


def fit_pipeline(dataset, task, patients, penalty):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    population = set(patients)
    rows = [row for row in dataset["records"] if row["patient_id"] in population]
    if {row["patient_id"] for row in rows} != population:
        raise ValueError("The explicit reference fit population is incomplete.")
    data = dataset.get("_frame")
    if data is None:
        raw = np.asarray([row["values"] for row in rows], dtype=float)
        y = np.asarray(
            [row["target"] for row in rows], dtype=float if task == "regression" else int
        )
    else:
        selected = data.loc[data.patient_id.isin(population)]
        raw = selected[dataset["feature_names"]].to_numpy(dtype=float)
        y = (
            selected.target.to_numpy(dtype=float)
            if task == "regression"
            else selected.target.map(
                {label: i for i, label in enumerate(dataset["classes"])}
            ).to_numpy(dtype=int)
        )
    if task != "regression" and set(y) != set(range(len(dataset["classes"]))):
        raise ValueError("A reference training fold is missing a declared class.")
    if np.isnan(raw).all(axis=0).any():
        raise ValueError("An all-missing reference feature cannot be silently dropped.")
    imputer = SimpleImputer(strategy="median").fit(raw)
    imputed = imputer.transform(raw)
    scaler = StandardScaler().fit(imputed)
    z = scaler.transform(imputed)
    variance = imputed.var(axis=0, ddof=0)
    candidates = np.flatnonzero(variance > 0)
    if len(candidates) == 0:
        raise ValueError("Reference selection retained zero features.")
    correlation = (
        np.corrcoef(z[:, candidates], rowvar=False) if len(candidates) > 1 else np.ones((1, 1))
    )
    retained = []
    for index, column in enumerate(candidates):
        if not any(abs(correlation[index, previous]) > 0.9 for previous, _ in retained):
            retained.append((index, column))
    selected = [int(column) for _, column in retained]
    if not selected:
        raise ValueError("Reference selection retained zero features.")
    if task == "regression":
        model = Ridge(alpha=len(y) * penalty, fit_intercept=True, solver="svd")
    else:
        model = LogisticRegression(
            C=1 / (len(y) * penalty),
            penalty="l2",
            solver="newton-cg",
            fit_intercept=True,
            multi_class="multinomial" if task == "multiclass" else "ovr",
            tol=1e-7,
            max_iter=10000,
            random_state=0,
            n_jobs=1,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(z[:, selected], y)
    if task != "regression" and any(int(count) >= 10000 for count in model.n_iter_):
        raise RuntimeError("The reference estimator reached its iteration limit.")
    audit = dict(
        fitted_patient_ids=sorted(population),
        fitted_sample_ids=[row["sample_id"] for row in rows],
        training_rows=len(rows),
        feature_names=list(dataset["feature_names"]),
        medians=imputer.statistics_.tolist(),
        means=scaler.mean_.tolist(),
        scales=scaler.scale_.tolist(),
        raw_population_variances=variance.tolist(),
        selected_indices=selected,
        selected_names=[dataset["feature_names"][column] for column in selected],
        penalty=penalty,
        coefficient=model.coef_.tolist(),
        intercept=(
            model.intercept_.tolist()
            if hasattr(model.intercept_, "tolist")
            else float(model.intercept_)
        ),
        converged=True,
        iterations=None if task == "regression" else model.n_iter_.tolist(),
    )
    return dict(
        imputer=imputer,
        scaler=scaler,
        model=model,
        selected=selected,
        task=task,
        classes=dataset["classes"],
        feature_names=dataset["feature_names"],
        audit=audit,
    )


def predict(pipeline, rows, *, outcomes=False):
    import numpy as np

    raw = np.asarray([row["values"] for row in rows], dtype=float)
    transformed = pipeline["scaler"].transform(pipeline["imputer"].transform(raw))[
        :, pipeline["selected"]
    ]
    values = (
        pipeline["model"].predict(transformed)
        if pipeline["task"] == "regression"
        else pipeline["model"].predict_proba(transformed)
    )
    result = []
    for row, value in zip(rows, values):
        result.append(
            dict(
                sample_id=row["sample_id"],
                patient_id=row["patient_id"],
                cohort=row["cohort"],
                y=row.get("target") if outcomes else None,
                value=float(value) if pipeline["task"] == "regression" else None,
                probabilities=[] if pipeline["task"] == "regression" else value.tolist(),
            )
        )
    return result


def aggregate(sample):
    import numpy as np

    groups = {}
    for row in sample:
        groups.setdefault(row["patient_id"], []).append(row)
    result = []
    for patient, rows in sorted(groups.items()):
        if len({row["cohort"] for row in rows}) != 1 or len({row["y"] for row in rows}) != 1:
            raise ValueError("Reference patient prediction identities/outcomes disagree.")
        result.append(
            dict(
                patient_id=patient,
                sample_id=patient,
                cohort=rows[0]["cohort"],
                y=rows[0]["y"],
                value=(
                    None if rows[0]["value"] is None else float(np.mean([r["value"] for r in rows]))
                ),
                probabilities=(
                    np.mean([r["probabilities"] for r in rows], axis=0).tolist()
                    if rows[0]["probabilities"]
                    else []
                ),
            )
        )
    return result


def score(task, patient):
    from sklearn.metrics import mean_squared_error, roc_auc_score

    y = [row["y"] for row in patient]
    if task == "regression":
        return float(mean_squared_error(y, [row["value"] for row in patient], squared=False))
    if task == "binary":
        return float(roc_auc_score(y, [row["probabilities"][1] for row in patient]))
    return float(
        roc_auc_score(
            y,
            [row["probabilities"] for row in patient],
            multi_class="ovr",
            average="macro",
            labels=list(range(len(patient[0]["probabilities"]))),
        )
    )


def tune(dataset, task, folds, grid, tie_atol):
    candidates = []
    for penalty in grid:
        inner = []
        for index, partition in enumerate(folds):
            fitted = fit_pipeline(dataset, task, partition["training_patients"], penalty)
            validation = set(partition["validation_patients"])
            rows = [row for row in dataset["records"] if row["patient_id"] in validation]
            measured = score(task, aggregate(predict(fitted, rows, outcomes=True)))
            inner.append(
                dict(
                    fold=index,
                    training_patients=partition["training_patients"],
                    validation_patients=partition["validation_patients"],
                    fit_state=fitted["audit"],
                    score=measured,
                )
            )
        candidates.append(
            dict(
                penalty=penalty,
                inner_folds=inner,
                mean_score=sum(row["score"] for row in inner) / len(inner),
            )
        )
    best = 0
    for index in range(1, len(candidates)):
        delta = candidates[index]["mean_score"] - candidates[best]["mean_score"]
        if abs(delta) > tie_atol and (delta < 0 if task == "regression" else delta > 0):
            best = index
    return dict(
        candidates=candidates,
        selected_candidate_index=best,
        selected_lambda=grid[best],
        selection_tie_atol=tie_atol,
    )


def workflow(dataset, task, plan, *, grid=(0.05, 0.1), tie_atol=1e-12):
    outer = []
    oof = []
    for index, nested in enumerate(plan["outer_folds"]):
        tuning = tune(dataset, task, nested["inner"], grid, tie_atol)
        fitted = fit_pipeline(
            dataset, task, nested["outer"]["training_patients"], tuning["selected_lambda"]
        )
        validation = set(nested["outer"]["validation_patients"])
        predicted = predict(
            fitted,
            [row for row in dataset["records"] if row["patient_id"] in validation],
            outcomes=True,
        )
        for row in predicted:
            row.update(fold=index, prediction_type="oof")
        oof.extend(predicted)
        outer.append(
            dict(
                fold=index,
                tuning=tuning,
                fit_state=fitted["audit"],
                score=score(task, aggregate(predicted)),
            )
        )
    final_tuning = tune(dataset, task, plan["final_inner_folds"], grid, tie_atol)
    fitted = fit_pipeline(dataset, task, plan["cohorts"]["train"], final_tuning["selected_lambda"])
    predicted = predict(fitted, dataset["records"], outcomes=True)
    for row in predicted:
        row.update(
            fold=None,
            prediction_type=(
                "apparent"
                if row["cohort"] == "train"
                else "holdout"
                if row["cohort"] == "test"
                else "external"
            ),
        )
    return dict(
        pipeline=fitted,
        outer_folds=outer,
        final_tuning=final_tuning,
        oof_sample_predictions=sorted(oof, key=lambda row: row["sample_id"]),
        final_sample_predictions=predicted,
        mean_cv_score=sum(row["score"] for row in outer) / len(outer),
    )


def class_measures(y, predicted, k):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    matrix = confusion_matrix(y, predicted, labels=list(range(k)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=list(range(k)), zero_division=0
    )
    per_class = []
    for c in range(k):
        tp = int(matrix[c, c])
        fp = int(matrix[:, c].sum()) - tp
        fn = int(support[c]) - tp
        tn = len(y) - tp - fp - fn
        per_class.append(
            dict(
                class_index=c,
                support=int(support[c]),
                precision=float(precision[c]),
                recall=float(recall[c]),
                f1=float(f1[c]),
                specificity=tn / (tn + fp) if tn + fp else 0.0,
            )
        )
    averages = {}
    for method in ("macro", "micro", "weighted"):
        values = precision_recall_fscore_support(
            y, predicted, average=method, labels=list(range(k)), zero_division=0
        )
        averages[method + "_average"] = dict(
            zip(("precision", "recall", "f1"), map(float, values[:3]))
        )
    return dict(size=k, counts=matrix.ravel().astype(int).tolist()), per_class, averages


def binary_report(y, p, threshold=0.5, *, ci=False):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        auc,
        average_precision_score,
        brier_score_loss,
        log_loss,
        precision_recall_curve,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    confusion, per_class, _ = class_measures(y, (p >= threshold).astype(int), 2)
    tn, fp, fn, tp = confusion["counts"]
    precision, recall, _ = precision_recall_curve(y, p)
    value = dict(
        n=len(y),
        positive_n=int(y.sum()),
        threshold=threshold,
        accuracy=float(accuracy_score(y, p >= threshold)),
        auc=float(roc_auc_score(y, p)),
        sensitivity=tp / (tp + fn),
        specificity=tn / (tn + fp),
        ppv=tp / (tp + fp) if tp + fp else float("nan"),
        npv=tn / (tn + fn) if tn + fn else float("nan"),
        precision=per_class[1]["precision"],
        recall=per_class[1]["recall"],
        f1=per_class[1]["f1"],
        brier=float(brier_score_loss(y, p)),
        log_loss_eps1e15=float(log_loss(y, np.c_[1 - p, p], labels=[0, 1], eps=1e-15)),
        average_precision=float(average_precision_score(y, p)),
        pr_auc_trapezoidal=float(auc(recall, precision)),
        confusion=confusion,
        per_class=per_class,
    )
    if ci:
        value.update(ci_low=float("nan"), ci_high=float("nan"))
    return value


def point_report(task, classes, y, values, threshold=0.5):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=float if task == "regression" else int)
    if task == "regression":
        return dict(
            n=len(y),
            mae=float(mean_absolute_error(y, values)),
            rmse=float(mean_squared_error(y, values, squared=False)),
            r2=float(r2_score(y, values)),
        )
    p = np.asarray(values, dtype=float)
    if task == "binary":
        return binary_report(y, p[:, 1], threshold, ci=True)
    k = len(classes)
    labels = np.eye(k, dtype=int)[y.astype(int)]
    confusion, per_class, averages = class_measures(y, np.argmax(p, axis=1), k)
    ovr = [binary_report(labels[:, c], p[:, c], threshold) for c in range(k)]
    support = np.asarray([row["support"] for row in per_class], dtype=float)
    return dict(
        n=len(y),
        classes=classes,
        accuracy=float(accuracy_score(y, np.argmax(p, axis=1))),
        confusion=confusion,
        per_class=per_class,
        **averages,
        ovr=ovr,
        macro_ovr_auc=float(np.mean([row["auc"] for row in ovr])),
        weighted_ovr_auc=float(np.average([row["auc"] for row in ovr], weights=support)),
        micro_ovr_auc=float(roc_auc_score(labels.ravel(), p.ravel())),
        macro_average_precision=float(np.mean([row["average_precision"] for row in ovr])),
        weighted_average_precision=float(
            np.average([row["average_precision"] for row in ovr], weights=support)
        ),
        micro_average_precision=float(average_precision_score(labels.ravel(), p.ravel())),
        log_loss_eps1e15=float(log_loss(y, p, labels=list(range(k)), eps=1e-15)),
        multiclass_brier_sum=float(np.mean(np.sum((p - labels) ** 2, axis=1))),
    )


def evaluate(dataset, task, predictions, threshold=0.5):
    truth = {row["sample_id"]: row["target"] for row in dataset["records"]}
    identified = [dict(row, y=truth[row["sample_id"]]) for row in predictions]
    patients = aggregate(identified)
    reports = []
    for cohort in sorted({row["cohort"] for row in patients}):
        rows = [row for row in patients if row["cohort"] == cohort]
        if all(row["y"] is None for row in rows):
            reports.append(dict(cohort=cohort, status="prediction_only", n=len(rows), report=None))
            continue
        if any(row["y"] is None for row in rows):
            raise ValueError("Reference evaluation refuses partly missing outcomes.")
        measured = point_report(
            task,
            dataset["classes"],
            [row["y"] for row in rows],
            [row["value"] if task == "regression" else row["probabilities"] for row in rows],
            threshold,
        )
        reports.append(dict(cohort=cohort, status="evaluated", n=len(rows), report=measured))
    return reports


def clinical(data, confidence=0.95):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    frame = pd.DataFrame(data)
    baseline = []
    observed = frame.dropna(subset=["target"])
    for feature in ("a", "b", "sex"):
        groups = []
        for label in (0, 1):
            block = observed.loc[observed.target == label, feature]
            clean = block.dropna()
            row = dict(class_label=str(label), n=len(clean), missing=int(block.isna().sum()))
            if feature == "sex":
                row.update(
                    counts=clean.value_counts().to_dict(),
                    proportions=clean.value_counts(normalize=True).to_dict(),
                )
            else:
                row.update(
                    mean=float(clean.mean()),
                    sd=float(clean.std(ddof=1)),
                    median=float(clean.median()),
                    q1=float(clean.quantile(0.25)),
                    q3=float(clean.quantile(0.75)),
                )
            groups.append(row)
        if feature == "sex":
            table = pd.crosstab(observed[feature], observed.target).reindex(
                index=["F", "M"], columns=[0, 1], fill_value=0
            )
            statistic, pvalue, _, expected = stats.chi2_contingency(table, correction=False)
            if (expected < 5).any():
                statistic, pvalue = stats.fisher_exact(table)
                method = "fisher_2x2"
            else:
                method = "chi_square_no_yates"
        else:
            statistic, pvalue = stats.mannwhitneyu(
                observed.loc[observed.target == 0, feature].dropna(),
                observed.loc[observed.target == 1, feature].dropna(),
                alternative="two-sided",
            )
            method = "mann_whitney"
        baseline.append(
            dict(
                feature=feature,
                groups=groups,
                statistic=float(statistic),
                p_value=float(pvalue),
                method=method,
            )
        )

    def fit(features):
        complete = frame[features + ["target"]].dropna().copy()
        if "sex" in features:
            complete["sex"] = pd.Categorical(complete.sex, categories=["F", "M"])
        x = pd.get_dummies(
            complete[features], columns=["sex"] if "sex" in features else [], drop_first=True
        ).astype(float)
        x = sm.add_constant(x, has_constant="add")
        if len(x) <= x.shape[1] or np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
            raise ValueError("Reference clinical design is rank deficient.")
        model = sm.Logit(complete.target.astype(int), x).fit(disp=False, maxiter=200, tol=1e-12)
        if not model.mle_retvals["converged"]:
            raise RuntimeError("Reference clinical Logit did not converge.")
        interval = model.conf_int(alpha=1 - confidence)
        return dict(
            features=features,
            terms=x.columns.tolist(),
            coefficient=model.params.tolist(),
            standard_error=model.bse.tolist(),
            p_values=model.pvalues.tolist(),
            odds_ratio=np.exp(model.params).tolist(),
            ci_low=np.exp(interval.iloc[:, 0]).tolist(),
            ci_high=np.exp(interval.iloc[:, 1]).tolist(),
            n_used=len(complete),
            n_dropped=len(frame) - len(complete),
        )

    return dict(
        baseline=baseline,
        baseline_input_rows=len(observed),
        baseline_missing_target_rows=len(frame) - len(observed),
        univariate=[fit([feature]) for feature in ("a", "b", "sex")],
        multivariable=fit(["a", "b", "sex"]),
    )
