"""Train or reload a native model from explicitly described user CSV/Parquet files.

No sample data, clinical outcomes or cohort assignments are invented by this script.
Run from an installed ov-radiomics environment; see docs/user-guide/training.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import ov_radiomics as rad


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_study(path, task, *, labeled):
    path = path.resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    # The native reader also supports inferred base columns. This user tutorial
    # deliberately requires a reviewed list so an auxiliary outcome cannot leak.
    for role in ("id_column", "patient_column", "cohort_column", "numeric_features", "categorical_features"):
        if role not in spec:
            raise ValueError("Declare the column role explicitly in the study JSON: " + role)
    if not isinstance(spec["numeric_features"], list) or not isinstance(spec["categorical_features"], list):
        raise ValueError("numeric_features and categorical_features must be explicit JSON lists.")
    if labeled and not spec.get("outcome_column"):
        raise ValueError("Training needs an explicit real outcome_column; no outcome is generated.")
    if not labeled and spec.get("outcome_column") is not None:
        raise ValueError("Prediction study must use outcome_column=null; labels are unnecessary for prediction.")
    loaded = rad.read_study(path, task=task)
    if not loaded.features:
        raise ValueError("No predictors were explicitly selected.")
    return loaded


def new_output(path):
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def prediction_exports(bundle, loaded, output, stem="predictions"):
    prediction = bundle.predict_study(loaded.to_dict()["data"])
    prediction.dataframe.to_csv(output / (stem + ".csv"), index=False)
    patient = prediction if prediction.prediction_unit == "patient" else prediction.aggregate_patients()
    patient.dataframe.to_csv(output / (stem + "-patients.csv"), index=False)
    write_json(output / (stem + ".json"), prediction.to_dict())
    return prediction


def holdout_plots(table, data, target, task, classes, output, threshold):
    """Render public/native results; do not recompute metrics with sklearn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    patient = table if table.prediction_unit == "patient" else table.aggregate_patients()
    frame = patient.dataframe
    # Patient scoring requires a single consistent outcome per patient. Native
    # evaluate has validated this before plotting. The join remains identified.
    truth = data[["patient_id", target]].drop_duplicates()
    if truth.patient_id.duplicated().any():
        raise ValueError("Patient plots require one consistent outcome per patient.")
    frame = frame.merge(truth, on="patient_id", how="left", validate="many_to_one")
    plotted = []
    for number, (cohort, rows) in enumerate(frame.groupby("group", sort=True)):
        if cohort == "train" or rows[target].isna().all():
            continue
        if rows[target].isna().any():
            raise ValueError("Partly labeled plot cohorts must be resolved explicitly.")
        prefix = "holdout-%03d" % number  # cohort names never become path fragments
        if task == "regression":
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(rows[target].astype(float), rows["prediction"], alpha=.7)
            lower = min(rows[target].astype(float).min(), rows["prediction"].min())
            upper = max(rows[target].astype(float).max(), rows["prediction"].max())
            ax.plot([lower, upper], [lower, upper], "--", color="gray")
            ax.set(xlabel="Observed outcome", ylabel="Predicted outcome", title=str(cohort))
            fig.savefig(output / (prefix + "-observed-predicted.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            labels = [classes[1]] if task == "binary" else classes
            fig, ax = plt.subplots(figsize=(6, 5))
            for label in labels:
                binary = (rows[target] == label).astype(int)
                if binary.nunique() != 2:
                    continue
                score = rows["probability"] if task == "binary" else rows["probability::" + label]
                rad.plot_roc(binary, {label: score}, ax=ax, backend="rust", title=str(cohort) + " patient ROC")
            if len(ax.lines):
                fig.savefig(output / (prefix + "-roc.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            if task == "binary" and rows[target].nunique() == 2:
                binary = (rows[target] == classes[1]).astype(int)
                ax = rad.plot_calibration(binary, rows.probability, n_bins=min(10, len(rows)), backend="rust")
                ax.figure.savefig(output / (prefix + "-calibration.png"), dpi=150, bbox_inches="tight")
                plt.close(ax.figure)
                ax = rad.plot_confusion_matrix(binary, rows.probability, threshold=threshold,
                                              class_names=classes, backend="rust")
                ax.figure.savefig(output / (prefix + "-confusion.png"), dpi=150, bbox_inches="tight")
                plt.close(ax.figure)
        plotted.append(dict(file_prefix=prefix, cohort=cohort, patients=len(rows)))
    write_json(output / "plot-cohorts.json", plotted)


def train(args):
    if args.output.exists():
        raise FileExistsError("Choose a new output directory: " + str(args.output))
    if args.task == "regression":
        if args.classes:
            raise ValueError("Regression has no class order; omit --classes.")
        classes = []
    else:
        classes = args.classes
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("Classification requires an explicit unique --classes order.")
        if args.task == "binary" and len(classes) != 2 or args.task == "multiclass" and len(classes) < 3:
            raise ValueError("The number of --classes does not match --task.")
    loaded = load_study(args.study, args.task, labeled=True)
    data = loaded.dataframe
    if args.plots and args.scoring_unit != "patient":
        raise ValueError("This tutorial's plots are patient-level; use --scoring-unit patient or omit --plots.")
    # Native table loading canonicalizes external names to ID/patient_id/group.
    result = rad.train_models(data, loaded.target, loaded.features, backend="rust", task=args.task,
                              classes=classes, positive_label=classes[1] if args.task == "binary" else None,
                              categorical=loaded.categorical, patient_id="patient_id", id_col="ID", split="group",
                              cv=args.cv, inner_cv=args.inner_cv, random_state=args.seed, n_jobs=args.n_jobs,
                              fit_unit=args.fit_unit, scoring_unit=args.scoring_unit, scale=args.scale,
                              models=args.models, threshold=args.threshold)
    report = rad.evaluate(result, backend="rust", prediction_unit=args.scoring_unit, ci_method=None)
    # Evaluate OOF separately; retain the true train cohort and OOF provenance.
    oof_reports = []
    for run in result.to_dict()["models"]:
        report_oof = rad.evaluate(rad.PredictionTable(run["oof"]), data=data, target=loaded.target,
                                  backend="rust", prediction_unit=args.scoring_unit, ci_method=None,
                                  threshold=run["model"]["manifest"]["threshold"])
        report_oof["model"] = run["name"]
        oof_reports.append(report_oof)
    output = new_output(args.output)
    result.save(output / "training")
    write_json(output / "input-report.json", loaded.report)
    report.to_csv(output / "cohort-metrics.csv", index=False)
    pd.concat(oof_reports, ignore_index=True).to_csv(output / "oof-metrics.csv", index=False)
    bundle = result.models[result.best_model]
    bundle.save(output / "best-model.rad")
    restored = rad.load_model(output / "best-model.rad")
    if args.predict_study:
        external = load_study(args.predict_study, args.task, labeled=False)
        predicted = prediction_exports(restored, external, output, "external-predictions")
        original = bundle.predict_study(external.to_dict()["data"])
        if predicted.to_dict() != original.to_dict():
            raise RuntimeError("Reloaded model predictions differ from the fitted model.")
        write_json(output / "external-input-report.json", external.report)
    if args.plots:
        best = next(run for run in result.to_dict()["models"] if run["name"] == result.best_model)
        holdout_plots(rad.PredictionTable(best["predictions"]), data, loaded.target, args.task,
                      classes, output, args.threshold)
    write_json(output / "run.json", dict(task=args.task, classes=classes, best_model=result.best_model,
               arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               backend=rad.backend_info(), split_plan_sha256=result.split_plan.fingerprint,
               metrics_unit=args.scoring_unit, ci_method=None,
               outcome_source="Explicit user study outcome_column; never generated by this script"))
    print(report.to_string(index=False))
    print("Saved", output, "best model:", result.best_model)


def predict(args):
    bundle = rad.load_model(args.model)
    loaded = load_study(args.study, bundle.manifest["task"], labeled=False)
    # Run before publishing an output directory: missing features, overlapping
    # training patients and wrong types must remain visible native errors.
    prediction = bundle.predict_study(loaded.to_dict()["data"])
    output = new_output(args.output)
    prediction.dataframe.to_csv(output / "predictions.csv", index=False)
    prediction.aggregate_patients().dataframe.to_csv(output / "predictions-patients.csv", index=False)
    write_json(output / "predictions.json", prediction.to_dict())
    write_json(output / "input-report.json", loaded.report)
    write_json(output / "run.json", dict(model=str(args.model.resolve()),
               model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), task=bundle.manifest["task"],
               classes=bundle.manifest["classes"], backend=rad.backend_info(), operation="prediction_only_no_fit"))
    print("Predicted", len(prediction.dataframe), "rows; output:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train from an explicitly labeled user study specification")
    fit.add_argument("--study", type=Path, required=True)
    fit.add_argument("--task", choices=("binary", "multiclass", "regression"), required=True)
    fit.add_argument("--classes", nargs="+", help="Exact labels, negative then positive for binary")
    fit.add_argument("--output", type=Path, required=True, help="New directory; existing output is never replaced")
    fit.add_argument("--models", nargs="+", choices=("LR", "Linear", "Ridge", "KNN", "GaussianNB", "RandomForest"))
    fit.add_argument("--cv", type=int, default=3)
    fit.add_argument("--inner-cv", type=int, default=2)
    fit.add_argument("--seed", type=int, default=20261115)
    fit.add_argument("--n-jobs", type=int, default=1)
    fit.add_argument("--scale", choices=("standard", "minmax", "robust", "none"), default="standard")
    fit.add_argument("--fit-unit", choices=("sample", "patient"), default="sample")
    fit.add_argument("--scoring-unit", choices=("sample", "patient"), default="patient")
    fit.add_argument("--threshold", type=float, default=.5, help="Predeclared binary threshold; never tuned on test outcomes")
    fit.add_argument("--predict-study", type=Path, help="Separate outcome_column=null study; fitted training patients cannot overlap")
    fit.add_argument("--plots", action="store_true", help="Patient-level held-out ROC/calibration/confusion or regression scatter")
    prediction = commands.add_parser("predict", help="Load .rad and predict an unlabeled user study without refitting")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--study", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
