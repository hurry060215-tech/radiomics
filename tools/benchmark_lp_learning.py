"""Complete declared model/statistics workflows; timings belong to the controller."""

import json
from pathlib import Path


def clean(value):
    import math

    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Inf" if value > 0 else "-Inf"
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(clean(value), target, allow_nan=False, sort_keys=True, indent=2)
        target.write("\n")


def frame(dataset, task):
    import pandas as pd

    result = []
    for row in dataset["records"]:
        value = row["target"]
        if value is not None and task != "regression":
            value = dataset["classes"][value]
        result.append(
            dict(
                ID=row["sample_id"],
                patient_id=row["patient_id"],
                group=row["cohort"],
                center=row.get("center"),
                target=value,
                **dict(zip(dataset["feature_names"], row["values"])),
            )
        )
    return pd.DataFrame(result)


def native_rows(table):
    return [
        dict(
            sample_id=row["sample_id"],
            patient_id=row["patient_id"],
            cohort=row["cohort"],
            probabilities=row["probabilities"],
            value=row["value"],
            fold=row["fold"],
        )
        for row in table["rows"]
    ]


def reference_prediction_frame(rows, task, classes, model, unit, *, truth=False):
    import pandas as pd

    result = []
    for row in rows:
        kind = row.get(
            "prediction_type",
            (
                "apparent"
                if row["cohort"] == "train"
                else "holdout"
                if row["cohort"] == "test"
                else "external"
            ),
        )
        value = dict(
            ID=row["sample_id"],
            patient_id=row["patient_id"],
            slide_id=None,
            tile_id=None,
            model=model,
            group=row["cohort"],
            fold=row.get("fold"),
            prediction_unit=unit,
            prediction_type=kind,
            training_source=(
                "reference_outer:" + str(row["fold"])
                if kind == "oof"
                else "reference_final:model.joblib"
            ),
        )
        if task == "regression":
            value["prediction"] = row["value"]
        else:
            value.update(
                {"probability::" + label: p for label, p in zip(classes, row["probabilities"])}
            )
            if task == "binary":
                value["probability"] = row["probabilities"][1]
        if truth:
            y = row.get("y")
            value["y_true"] = y
            value["target_value"] = y if task == "regression" or y is None else classes[y]
        result.append(value)
    return pd.DataFrame(result)


def native_audit(run):
    def fitted(value):
        return dict(
            training_patients=value["training_patients"],
            training_sample_ids=value["training_sample_ids"],
            selected_features=value["selected_features"],
        )

    def tuning(value):
        return dict(
            selected_candidate=value["selected_candidate"],
            candidates=[
                dict(
                    mean_score=candidate["mean_score"],
                    inner_folds=[
                        dict(fold=inner["fold"], score=inner["score"], **fitted(inner["fit"]))
                        for inner in candidate["inner_folds"]
                    ],
                )
                for candidate in value["candidates"]
            ],
        )

    return dict(
        outer_folds=[
            dict(
                fold=outer["fold"],
                score=outer["score"],
                tuning=tuning(outer["tuning"]),
                **fitted(outer["fit"]),
            )
            for outer in run["outer_folds"]
        ],
        final_tuning=tuning(run["final_tuning"]),
        mean_cv_score=run["mean_cv_score"],
    )


def reference_audit(result):
    def fitted(value):
        return dict(
            training_patients=value["fitted_patient_ids"],
            training_sample_ids=value["fitted_sample_ids"],
            selected_features=value["selected_names"],
        )

    def tuning(value):
        return dict(
            selected_candidate=value["selected_candidate_index"],
            candidates=[
                dict(
                    mean_score=candidate["mean_score"],
                    inner_folds=[
                        dict(fold=inner["fold"], score=inner["score"], **fitted(inner["fit_state"]))
                        for inner in candidate["inner_folds"]
                    ],
                )
                for candidate in value["candidates"]
            ],
        )

    return dict(
        outer_folds=[
            dict(
                fold=outer["fold"],
                score=outer["score"],
                tuning=tuning(outer["tuning"]),
                **fitted(outer["fit_state"]),
            )
            for outer in result["outer_folds"]
        ],
        final_tuning=tuning(result["final_tuning"]),
        mean_cv_score=result["mean_cv_score"],
    )


def run_learning(dataset, task, plan, backend, output, stage, *, fit_unit="sample"):
    """Both sides fit every declared candidate/fold and load before predicting."""
    import pandas as pd

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    name = "Ridge" if task == "regression" else "LR"
    names = dataset["feature_names"]
    with stage("model_fit_tune"):
        data = frame(dataset, task)
    if backend == "rust":
        import ov_radiomics as rad

        with stage("model_fit_tune"):
            selected = rad.FeatureSelector(
                backend="rust", task=task, univariate=None, lasso=False, scale="standard"
            )
            result = rad.train_models(
                data,
                "target",
                names,
                backend="rust",
                task=task,
                classes=dataset["classes"],
                positive_label=dataset["classes"][1] if task == "binary" else None,
                split_plan=rad.SplitPlan.from_dict(plan),
                selector=selected,
                models=[name],
                model_params={name: [{"lambda": 0.05}, {"lambda": 0.1}]},
                random_state=42,
                fit_unit=fit_unit,
                scoring_unit="patient",
                threshold=0.5,
                n_jobs=1,
            )
        with stage("model_persist_load_and_builtin_table_exports"):
            result.save(output / "model")
            loaded = rad.load_model(output / "model/model-000.rad")
        with stage("model_predict_evaluate"):
            predicted = loaded.predict(data.drop(columns="target"))
            metrics = rad.evaluate(
                predicted,
                data=data,
                target="target",
                prediction_unit="patient",
                threshold=loaded.manifest["threshold"],
                ci_method=None,
                backend="rust",
            )
        with stage("additional_model_table_exports"):
            predicted.dataframe.to_csv(output / "unlabeled-input-predictions.csv", index=False)
            metrics.to_csv(output / "patient-metrics.csv", index=False)
            write_json(output / "metric-reports.json", metrics.attrs["native_reports"])

        def comparison():
            payload = result.to_dict()
            run = payload["models"][0]
            detailed = {row["group"]: row["report"] for row in metrics.attrs["native_reports"]}
            reports = [
                dict(
                    cohort=row["group"],
                    status=row["status"],
                    n=int(row["n"]),
                    report=detailed.get(row["group"]),
                )
                for row in metrics.to_dict(orient="records")
            ]
            artifact = dict(
                task=task,
                classes=dataset["classes"],
                feature_names=names,
                fit_unit=fit_unit,
                sample_predictions=native_rows(predicted.to_dict()),
                oof=native_rows(run["oof"]),
                patient_reports=reports,
                audit=native_audit(run),
            )
            return clean(artifact)

        return dict(comparison=comparison)
    import benchmark_lp_reference as reference
    import joblib

    with stage("model_fit_tune"):
        result = reference.workflow(dict(dataset, _frame=data), task, plan)
    with stage("model_persist_load_and_builtin_table_exports"):
        destination = output / "model"
        destination.mkdir()
        joblib.dump(
            dict(
                pipeline=result["pipeline"],
                task=task,
                classes=dataset["classes"],
                threshold=0.5,
                split_plan=plan,
                fit_unit=fit_unit,
            ),
            destination / "model.joblib",
        )
        write_json(
            destination / "training.json",
            {key: value for key, value in result.items() if key != "pipeline"},
        )
        write_json(destination / "split_plan.json", plan)
        pd.DataFrame(
            [
                dict(model=name, fold=row["fold"], score=row["score"])
                for row in result["outer_folds"]
            ]
        ).to_csv(destination / "cv_results.csv", index=False)
        reference_prediction_frame(
            result["oof_sample_predictions"], task, dataset["classes"], name, fit_unit, truth=True
        ).to_csv(destination / "oof_predictions.csv", index=False)
        reference_prediction_frame(
            result["final_sample_predictions"], task, dataset["classes"], name, fit_unit, truth=True
        ).to_csv(destination / "predictions.csv", index=False)
        write_json(
            destination / "api_config.json",
            dict(
                task=task,
                classes=dataset["classes"],
                features=names,
                fit_unit=fit_unit,
                scoring_unit="patient",
                threshold=0.5 if task == "binary" else None,
                CI=None,
            ),
        )
        loaded = joblib.load(destination / "model.joblib")
    with stage("model_predict_evaluate"):
        # Supply no outcome to prediction; truth is separately joined by ID.
        unlabeled = [
            {key: value for key, value in row.items() if key != "target"}
            for row in dataset["records"]
        ]
        predicted = reference.predict(loaded["pipeline"], unlabeled)
        reports = reference.evaluate(dataset, task, predicted, loaded["threshold"])
    with stage("additional_model_table_exports"):
        reference_prediction_frame(predicted, task, dataset["classes"], "Model", fit_unit).to_csv(
            output / "unlabeled-input-predictions.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    **(row["report"] or {}),
                    "cohort": row["cohort"],
                    "status": row["status"],
                    "n": row["n"],
                }
                for row in reports
            ]
        ).to_csv(output / "patient-metrics.csv", index=False)
        write_json(output / "metric-reports.json", reports)

    def comparison():
        artifact = dict(
            task=task,
            classes=dataset["classes"],
            feature_names=names,
            fit_unit=fit_unit,
            sample_predictions=[
                dict(
                    sample_id=row["sample_id"],
                    patient_id=row["patient_id"],
                    cohort=row["cohort"],
                    probabilities=row["probabilities"],
                    value=row["value"],
                    fold=None,
                )
                for row in predicted
            ],
            oof=[
                {
                    key: row[key]
                    for key in (
                        "sample_id",
                        "patient_id",
                        "cohort",
                        "probabilities",
                        "value",
                        "fold",
                    )
                }
                for row in result["oof_sample_predictions"]
            ],
            patient_reports=reports,
            audit=reference_audit(result),
        )
        return clean(artifact)

    return dict(comparison=comparison)


def run_clinical(rows, backend, output, stage):
    import pandas as pd

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = pd.DataFrame(rows)
    with stage("observed_baseline_and_complete_case_logistic"):
        if backend == "rust":
            import ov_radiomics as rad

            baseline = rad.baseline_table(
                data.dropna(subset=["target"]),
                "target",
                ["a", "b", "sex"],
                continuous=["a", "b"],
                categorical=["sex"],
                continuous_test="mannwhitney",
                backend="rust",
                analysis_unit="sample",
            )
            univariate = [
                rad.multivariable_logistic(
                    data,
                    "target",
                    [name],
                    categorical=[name] if name == "sex" else [],
                    reference={"sex": "F"} if name == "sex" else None,
                    backend="rust",
                    analysis_unit="sample",
                )
                for name in ("a", "b", "sex")
            ]
            multivariable = rad.multivariable_logistic(
                data,
                "target",
                ["a", "b", "sex"],
                categorical=["sex"],
                reference={"sex": "F"},
                backend="rust",
                analysis_unit="sample",
            )

            def normalize(result):
                fitted = result.model
                return dict(
                    features=fitted["features"],
                    terms=fitted["terms"],
                    coefficient=fitted["coefficient"],
                    standard_error=fitted["standard_error"],
                    p_values=fitted["p_values"],
                    odds_ratio=fitted["odds_ratio"],
                    ci_low=fitted["odds_ratio_ci_lower"],
                    ci_high=fitted["odds_ratio_ci_upper"],
                    n_used=fitted["n_used"],
                    n_dropped=fitted["n_dropped"],
                )

            baseline_rows = []
            for row in baseline.attrs["native_report"]["rows"]:
                groups = []
                for value in row["groups"]:
                    group = {
                        key: item
                        for key, item in value.items()
                        if key not in ("type", "class_index")
                    }
                    if value["type"] == "continuous":
                        group["sd"] = group.pop("sd_ddof1")
                        group["q1"] = group.pop("q25")
                        group["q3"] = group.pop("q75")
                    groups.append(group)
                baseline_rows.append(
                    dict(
                        feature=row["feature"],
                        groups=groups,
                        statistic=row["statistic"],
                        p_value=row["p_value"],
                        method=row["method"],
                    )
                )
            artifact = dict(
                baseline=baseline_rows,
                baseline_input_rows=int(data.target.notna().sum()),
                baseline_missing_target_rows=int(data.target.isna().sum()),
                univariate=[normalize(r) for r in univariate],
                multivariable=normalize(multivariable),
            )
        else:
            import benchmark_lp_reference as reference

            artifact = reference.clinical(rows)
    with stage("clinical_table_exports"):
        write_json(output / "clinical-report.json", artifact)
        pd.DataFrame(artifact["baseline"]).to_csv(output / "baseline.csv", index=False)
        for index, value in enumerate(artifact["univariate"] + [artifact["multivariable"]]):
            pd.DataFrame(
                {
                    key: value[key]
                    for key in (
                        "terms",
                        "coefficient",
                        "standard_error",
                        "p_values",
                        "odds_ratio",
                        "ci_low",
                        "ci_high",
                    )
                }
            ).to_csv(output / f"inference-{index}.csv", index=False)
    return dict(comparison=lambda: clean(artifact))
