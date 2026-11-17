"""Execute authored local-data lessons; keep data and results outside Git."""

import argparse
import json
import sys
from pathlib import Path

import nbformat
from run_tutorials import execute_one, git_identity, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_KIND = "local-image-mask-tabular-no-clinical-validity"
LESSONS = {
    "09_local_image_features": [
        "features.csv",
        "qa.csv",
        "failures.json",
        "source-hashes.json",
        "roi-previews.png",
        "feature-overview.png",
        "receipt.json",
    ],
    "10_local_modeling": [
        "metrics.csv",
        "oof-metrics.csv",
        "model-native55.rad",
        "model-provided107.rad",
        "validation-roc.png",
        "validation-calibration.png",
        "validation-confusion.png",
        "unlabeled-predictions.csv",
        "receipt.json",
    ],
}


def inside_git(path):
    return any((parent / ".git").exists() for parent in [path, *path.parents])


def validate_locations(data_root, output, repo_root):
    data_root, output, repo_root = map(lambda p: Path(p).resolve(), (data_root, output, repo_root))
    for path in (data_root, output):
        if path == repo_root or repo_root in path.parents or inside_git(path):
            raise ValueError("Local data and results must remain outside all Git worktrees.")
    if data_root == output or data_root in output.parents or output in data_root.parents:
        raise ValueError("Input and result directories must be separate trees.")
    if output.exists():
        raise FileExistsError("Result directory exists; choose a fresh --output.")
    for name in ("images", "masks", "label.csv", "label_multicenter.csv", "rad_features.csv"):
        path = data_root / "CT" / name
        if not path.exists():
            raise FileNotFoundError(path)
    return data_root, output, repo_root


def validate_lesson_outputs(artifacts, name):
    """Check outcome-specific gates in addition to the shared execution checks."""
    failures = []
    try:
        report = json.loads((artifacts / "receipt.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            failures.append("Notebook's workflow receipt did not pass.")
        if name == "09_local_image_features":
            empty_failures = json.loads((artifacts / "failures.json").read_text(encoding="utf-8"))
            if empty_failures != []:
                failures.append("Extraction recorded failed cases.")
            if report["source_records"] != report["extracted_records"]:
                failures.append("Some input records were not extracted.")
            if report["feature_count"] != 55 or not report["original_input_hashes_unchanged"]:
                failures.append("Feature contract or original input hash check failed.")
        else:
            exact = report["reloaded_predictions_exact"]
            if set(exact) != {"native55", "provided107"} or not all(exact.values()):
                failures.append("Saved-model predictions are not exactly reproducible.")
            if not report["input_hashes_unchanged"]:
                failures.append("Modeling input hashes changed.")
        if report["runtime"]["data_kind"] != DATA_KIND:
            failures.append("Runtime local-data marker differs from the authored notebook.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(f"Cannot validate workflow outputs: {type(error).__name__}: {error}")
    return failures


def run(data_root, output, notebooks=None, repo_root=ROOT, timeout=900):
    data_root, output, repo_root = validate_locations(data_root, output, repo_root)
    notebooks = Path(notebooks or repo_root / "website/tutorials").resolve()
    paths = [notebooks / (name + ".ipynb") for name in LESSONS]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if notebook.metadata.get("radiomics_data_kind") != DATA_KIND:
            raise ValueError("Expected the authored local-data lesson: " + path.name)
    source_bindings = {
        name: sha256(repo_root / name)
        for name in ("tools/run_local_sample_tutorials.py", "tools/run_tutorials.py")
    }
    output.mkdir(parents=True, exist_ok=False)
    kernels = output / "kernels"
    kernels.mkdir()
    records = []
    for path in paths:
        parameters = {"TUTORIAL_DATA_ROOT": str(data_root)}
        if path.stem == "10_local_modeling":
            parameters["TUTORIAL_IMAGE_FEATURES"] = str(
                output / "09_local_image_features/artifacts/features.csv"
            )
        try:
            result = execute_one(
                path,
                output,
                repo_root,
                kernels,
                timeout,
                expected_outputs=LESSONS[path.stem],
                kernel_parameters=parameters,
                source_bindings=source_bindings,
            )
            result["failures"].extend(
                validate_lesson_outputs(
                    Path(result["executed_path"]).parent / "artifacts", path.stem
                )
            )
            if result["failures"]:
                result["status"] = "failed"
        except Exception as error:
            result = {
                "notebook": path.name,
                "status": "failed",
                "failures": [f"{type(error).__name__}: {error}"],
            }
        records.append(result)
        print(path.name, result["status"], flush=True)
        if result["status"] != "passed":
            break
    status = (
        "passed"
        if len(records) == len(paths) and all(row["status"] == "passed" for row in records)
        else "failed"
    )
    receipt = {
        "schema": "radiomics.local-data-tutorial-execution.v1",
        "status": status,
        "data_kind": DATA_KIND,
        "data_root": str(data_root),
        "repository": git_identity(repo_root),
        "interpreter": sys.executable,
        "source_bindings": source_bindings,
        "distribution": "local-only; dataset redistribution permission is not inferred",
        "notebooks": records,
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notebooks", type=Path)
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per code cell in seconds")
    args = parser.parse_args()
    result = run(args.data_root, args.output, args.notebooks, timeout=args.timeout)
    print("Execution receipt:", args.output.resolve() / "receipt.json")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
