"""Predeclaration integrity only. No model/image feature computation or timing."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import benchmark_learning_pathology as runner  # noqa: E402 - import after tool-path/optional-dependency setup


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, "Duplicate JSON key: " + key
        result[key] = value
    return result


def load():
    return json.loads(
        (ROOT / "benchmarks/v2/learning-pathology-plan.json").read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )


def test_frozen_sources_and_separate_suites_have_not_been_substituted():
    plan = load()
    assert plan["status"] == "predeclared_pending_main_review_no_measurements"
    assert plan["suites"] == ["machine_learning_statistics", "pathology"]
    evidence = runner.verify_plan_sources(
        plan, runner.file_hash(ROOT / "benchmarks/v2/learning-pathology-plan.json")
    )
    assert len(evidence) == 9
    assert sum(item["comparison"] == "raw_bytes_exact" for item in evidence.values()) == 7
    assert plan["learning"]["metrics"]["CI"] is None
    assert plan["learning"]["candidate_lambda_order"] == [0.05, 0.1]
    assert plan["execution"]["paired_repetitions"] == 7
    assert plan["execution"]["memory_budget_bytes"] == 2 * 1024**3


def test_declared_pathology_grid_padding_and_study_population_are_exact():
    plan = load()
    generator = plan["pathology"]["generator"]
    tiles = plan["pathology"]["tiling"]
    width, height = generator["width"], generator["height"]
    pw, ph = tiles["patch_size_wh"]
    sx, sy = tiles["stride_xy"]
    positions = [
        (x, y, min(pw, width - x), min(ph, height - y))
        for y in range(0, height, sy)
        for x in range(0, width, sx)
    ]
    assert len(positions) == tiles["tiles_per_slide"] == 24
    assert sum(w < pw or h < ph for _, _, w, h in positions) == tiles["padded_tiles_per_slide"] == 9
    assert positions[-1] == (120, 72, 8, 24)
    assert len(positions) * generator["total_slides"] == tiles["total_tiles"] == 600
    assert generator["study_patients"] == plan["pathology_patient_model_case"]["patient_rows"] == 12
    assert generator["total_slides"] == 25
    assert generator["total_patients"] == 13
    assert tiles["normal_tiles"] + tiles["QC_sentinel_tiles"] == 600


def test_workload_rows_and_diagnostic_stage_reporting_are_not_pooled():
    plan = load()
    rows = (
        len(plan["learning"]["primary_cases"])
        + 1
        + len(plan["pathology"]["primary_cases"])
        + 1
        + int(plan["pathology_full_scene_diagnostic"]["enabled"])
    )
    assert rows == plan["reporting"]["case_count_before_state_expansion"] == 8
    assert (
        rows * len(plan["execution"]["states"])
        == plan["reporting"]["state_expanded_case_count"]
        == 16
    )
    assert rows * 2 * 7 * 2 == plan["reporting"]["paired_worker_observations"] == 224
    assert plan["pathology_patient_model_case"]["suite"] == "machine_learning_statistics"
    assert plan["pathology_full_scene_diagnostic"]["role"].startswith(
        "Additional whole-scene diagnostic"
    )
    stages = plan["pathology_full_scene_diagnostic"]["stage_columns"]
    assert len(stages) == len(set(stages))
    assert {
        "decode_tile_qc_color",
        "id_alignment_aggregation",
        "model_fit_tune",
        "model_predict_evaluate",
        "frontend_exports",
    } <= set(stages)


def test_only_approved_public_script_eol_forms_are_accepted(tmp_path):
    plan = load()
    digest = runner.file_hash(ROOT / "benchmarks/v2/learning-pathology-plan.json")
    for name in plan["source_files_sha256"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for name in ("tools/golden_workflow.py", "tools/golden_models.py"):
        path = tmp_path / name
        lf = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(lf)
        assert (
            runner.verify_plan_sources(plan, digest, tmp_path)[name]["canonical_LF_sha256"]
            == hashlib.sha256(lf).hexdigest()
        )
        path.write_bytes(lf.replace(b"\n", b"\r\n"))
        runner.verify_plan_sources(plan, digest, tmp_path)
    path = tmp_path / "tools/golden_workflow.py"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="Unapproved"):
        runner.verify_plan_sources(plan, digest, tmp_path)
    path.write_bytes(original)
    other = tmp_path / "golden/python_golden_v1/model_inputs.json"
    other.write_bytes(other.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen benchmark"):
        runner.verify_plan_sources(plan, digest, tmp_path)
