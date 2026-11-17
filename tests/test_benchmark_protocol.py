import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark_protocol", Path(__file__).resolve().parents[1] / "tools/benchmark_protocol.py"
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_raw_pair_reporting_preserves_uncertainty_and_does_not_turn_speed_into_correctness():
    records = [
        dict(
            pair=pair,
            backend=backend,
            order=(i if pair % 2 == 0 else 1 - i),
            elapsed_seconds=(pair + 1) * scale,
            status="ok",
            correctness_checked=True,
            suite="machine_learning_statistics",
            level="public_metrics",
            dataset="fixture",
            operation="binary_metrics",
            state="warm_uncached",
            threads=1,
            input_sha256="0" * 64,
            configuration_sha256="1" * 64,
        )
        for pair in range(7)
        for i, (backend, scale) in enumerate([("python", 2.0), ("rust", 1.0)])
    ]
    report = protocol.paired_summary(records, repeats=7)
    assert report["ratio_python_over_rust"] == 2.0
    assert report["backends"]["rust"]["median_seconds"] == 4.0
    assert report["backends"]["rust"]["iqr_seconds"] == 3.0
    assert report["backends"]["rust"]["p95_seconds"] == pytest.approx(6.7)
    assert report["peak_memory_comparison"] is None
    assert report["performance_target"] == "report_only"
    records[-1]["correctness_checked"] = False
    diagnostic = protocol.paired_summary(records, repeats=7)
    assert not diagnostic["valid_speed_comparison"]
    assert "ratio_python_over_rust" not in diagnostic
    with pytest.raises(ValueError, match="both backend"):
        protocol.paired_summary(records[:-1], repeats=7)


def _records():
    return [
        dict(
            pair=pair,
            backend=backend,
            order=i if pair % 2 == 0 else 1 - i,
            elapsed_seconds=1.0,
            status="ok",
            correctness_checked=True,
            suite="machine_learning_statistics",
            level="public_metrics",
            dataset="fixture",
            operation="binary_metrics",
            state="warm_uncached",
            threads=1,
            input_sha256="0" * 64,
            configuration_sha256="1" * 64,
        )
        for pair in range(7)
        for i, backend in enumerate(("python", "rust"))
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("suite", "pathology"),
        ("level", "operator"),
        ("dataset", "different"),
        ("operation", "delong_test"),
        ("state", "cache_hit"),
        ("threads", 2),
        ("input_sha256", "2" * 64),
        ("configuration_sha256", "3" * 64),
    ],
)
def test_different_cases_cannot_be_pooled_even_with_matching_pair_numbers(field, value):
    rows = _records()
    rows[-1][field] = value
    with pytest.raises(ValueError, match="same suite"):
        protocol.paired_summary(rows, repeats=7)


def test_typed_ids_flags_and_alternating_pair_order_are_required():
    rows = _records()
    rows[-1]["order"] = 0
    with pytest.raises(ValueError, match="alternating"):
        protocol.paired_summary(rows, repeats=7)
    rows = _records()
    rows[-1]["correctness_checked"] = "false"
    with pytest.raises(ValueError, match="typed"):
        protocol.paired_summary(rows, repeats=7)
    rows = _records()
    rows[-1]["input_sha256"] = "unknown"
    with pytest.raises(ValueError, match="SHA256"):
        protocol.paired_summary(rows, repeats=7)


@pytest.mark.parametrize("failed_backend", ["python", "rust"])
def test_warmup_failure_is_retained_and_cannot_become_a_speed_claim(monkeypatch, failed_backend):
    # Load the actual runner to prove the two warmup calls are also captured.
    path = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(path))
    script = importlib.util.spec_from_file_location(
        "benchmark_metrics_test", path / "benchmark_metrics.py"
    )
    runner = importlib.util.module_from_spec(script)
    script.loader.exec_module(runner)
    import pandas as pd

    counts = {"python": 0, "rust": 0}

    def call(backend):
        counts[backend] += 1
        if backend == failed_backend and counts[backend] == 1:
            raise ValueError("deliberate unavailable warmup")
        return pd.Series({"value": 0.25})

    identity = {
        key: value
        for key, value in _records()[0].items()
        if key in protocol.CASE_FIELDS and key != "configuration_sha256"
    }
    report, raw = runner._run_case(call, identity, {"threshold": 0.5}, 7)
    assert counts == {"python": 8, "rust": 8}
    assert len(raw) == 14
    index = 0 if failed_backend == "python" else 1
    assert report["warmup_records"][index]["status"] == "failed"
    assert report["warmup_records"][index]["error"] == "ValueError: deliberate unavailable warmup"
    assert not report["valid_speed_comparison"]
    assert "ratio_python_over_rust" not in report
    assert all(not row["correctness_checked"] for row in raw)
