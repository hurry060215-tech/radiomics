"""Local-data privacy boundaries and two-stage execution orchestration."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_local_sample_tutorials as local  # noqa: E402


@pytest.fixture
def external_tmp():
    with tempfile.TemporaryDirectory(prefix="radiomics-local-lessons-") as directory:
        yield Path(directory).resolve()


def inputs(root):
    ct = root / "CT"
    (ct / "images").mkdir(parents=True)
    (ct / "masks").mkdir()
    for name in ("label.csv", "label_multicenter.csv", "rad_features.csv"):
        (ct / name).write_text("ID\n", encoding="utf-8")
    return root


def fixture_repo(root):
    (root / "tools").mkdir(parents=True)
    for name in ("run_local_sample_tutorials.py", "run_tutorials.py"):
        (root / "tools" / name).write_text("# orchestration fixture\n", encoding="utf-8")
    notebooks = root / "website/tutorials"
    notebooks.mkdir(parents=True)
    for name in local.LESSONS:
        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("value=1")],
            metadata={"radiomics_data_kind": local.DATA_KIND},
        )
        nbformat.write(notebook, notebooks / (name + ".ipynb"))
    return root


@pytest.mark.parametrize("private_tree", ["data", "output"])
@pytest.mark.parametrize("git_entry", ["directory", "worktree-file"])
def test_data_and_results_reject_any_git_checkout(external_tmp, private_tree, git_entry):
    checkout = external_tmp / "other-checkout"
    checkout.mkdir()
    if git_entry == "directory":
        (checkout / ".git").mkdir()
    else:
        (checkout / ".git").write_text("gitdir: ../shared-git", encoding="utf-8")
    data = checkout / "data" if private_tree == "data" else external_tmp / "data"
    output = checkout / "result" if private_tree == "output" else external_tmp / "result"
    with pytest.raises(ValueError, match="outside all Git"):
        local.validate_locations(data, output, external_tmp / "source")
    assert not output.exists()


def test_overlap_existing_output_and_missing_inputs_fail_before_writes(external_tmp):
    data = inputs(external_tmp / "data")
    repo = external_tmp / "source"
    for overlapping in (data, data / "new-results", external_tmp):
        with pytest.raises(ValueError, match="separate trees"):
            local.validate_locations(data, overlapping, repo)
    output = external_tmp / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        local.validate_locations(data, output, repo)
    missing_output = external_tmp / "untouched"
    (data / "CT/label.csv").unlink()
    with pytest.raises(FileNotFoundError):
        local.validate_locations(data, missing_output, repo)
    assert not missing_output.exists()


def test_two_stage_runner_passes_only_extracted_features_to_modeling(external_tmp, monkeypatch):
    repo = fixture_repo(external_tmp / "source")
    data = inputs(external_tmp / "data")
    output = external_tmp / "execution"
    calls = []

    def execute(path, run_root, repo_root, kernels, timeout, **kwargs):
        calls.append(path.stem)
        assert run_root == output and repo_root == repo and kernels.parent == output
        assert kwargs["expected_outputs"] == local.LESSONS[path.stem]
        assert kwargs["kernel_parameters"]["TUTORIAL_DATA_ROOT"] == str(data)
        if path.stem == "10_local_modeling":
            assert kwargs["kernel_parameters"]["TUTORIAL_IMAGE_FEATURES"] == str(
                output / "09_local_image_features/artifacts/features.csv"
            )
        assert set(kwargs["source_bindings"]) == {
            "tools/run_local_sample_tutorials.py",
            "tools/run_tutorials.py",
        }
        return {
            "status": "passed",
            "failures": [],
            "executed_path": str(output / path.stem / path.name),
        }

    monkeypatch.setattr(local, "execute_one", execute)
    monkeypatch.setattr(local, "validate_lesson_outputs", lambda *args: [])
    result = local.run(data, output, repo_root=repo)
    assert calls == list(local.LESSONS)
    assert result["status"] == "passed"
    assert json.loads((output / "receipt.json").read_text())["status"] == "passed"


def test_failed_extraction_never_starts_modeling_and_retains_failure(external_tmp, monkeypatch):
    repo = fixture_repo(external_tmp / "source")
    data = inputs(external_tmp / "data")
    output = external_tmp / "execution"
    calls = []

    def execute(path, *args, **kwargs):
        calls.append(path.stem)
        raise RuntimeError("fixture extraction failure")

    monkeypatch.setattr(local, "execute_one", execute)
    result = local.run(data, output, repo_root=repo)
    assert calls == ["09_local_image_features"]
    assert result["status"] == "failed"
    assert "fixture extraction failure" in result["notebooks"][0]["failures"][0]
    assert (output / "receipt.json").is_file()


@pytest.mark.parametrize("failed_cases", [[], [{"ID": "fixture", "error": "failed"}]])
def test_empty_failure_list_is_valid_but_recorded_failures_are_rejected(tmp_path, failed_cases):
    report = {
        "status": "passed",
        "source_records": 1,
        "extracted_records": 1,
        "feature_count": 55,
        "original_input_hashes_unchanged": True,
        "runtime": {"data_kind": local.DATA_KIND},
    }
    (tmp_path / "receipt.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "failures.json").write_text(json.dumps(failed_cases), encoding="utf-8")
    failures = local.validate_lesson_outputs(tmp_path, "09_local_image_features")
    assert bool(failures) == bool(failed_cases)


def test_committed_lessons_have_code_and_no_private_saved_outputs():
    for name in local.LESSONS:
        notebook = nbformat.read(ROOT / "website/tutorials" / (name + ".ipynb"), as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["radiomics_data_kind"] == local.DATA_KIND
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert len(code) >= 5
        assert all(cell.execution_count is None and not cell.outputs for cell in code)
        assert all(":\\" not in cell.source and ":/" not in cell.source for cell in code)
