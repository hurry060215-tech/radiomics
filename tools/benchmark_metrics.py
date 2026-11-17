"""Warm ML/statistics public-metric microbenchmark; not a full workflow result.

Preserves the existing seed, n=100/10,000, three calls and seven alternating
pairs. Conversion and returned tables are inside timing; validation/reporting is
outside it. Every warmup/measured execution failure remains in the report.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_protocol import SCHEMA, capture_call, machine_manifest, paired_summary

import ov_radiomics as rad


def _canonical(value):
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Inf" if value > 0 else "-Inf"
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _digest(output):
    if isinstance(output, pd.DataFrame):
        payload = dict(
            kind="dataframe",
            columns=list(output.columns),
            index=list(output.index),
            values=output.to_numpy().tolist(),
        )
    else:
        payload = dict(kind="series", index=list(output.index), values=output.tolist())
    return hashlib.sha256(
        json.dumps(
            _canonical(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _run_case(call, identity, configuration, repeats):
    warmup_records = []
    warmup_outputs = {}
    for order, backend in enumerate(("python", "rust")):
        row, observed = capture_call(lambda: call(backend))
        row.update(backend=backend, order=order, output_digest=None)
        if row["status"] == "ok":
            row["output_digest"] = _digest(observed)
            warmup_outputs[backend] = observed
        warmup_records.append(row)
    warmup_error = None
    reference = warmup_outputs.get("python")
    comparator = (
        pd.testing.assert_frame_equal
        if isinstance(reference, pd.DataFrame)
        else pd.testing.assert_series_equal
    )
    if set(warmup_outputs) != {"python", "rust"}:
        warmup_error = (
            "At least one warmup execution failed; no verified reference pair is available."
        )
    else:
        try:
            comparator(reference, warmup_outputs["rust"], rtol=1e-10, atol=1e-12)
        except AssertionError as exc:
            warmup_error = str(exc)
    config_sha = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    observations = []
    for repetition in range(repeats):
        backends = ("python", "rust") if repetition % 2 == 0 else ("rust", "python")
        for order, backend in enumerate(backends):
            row, observed = capture_call(lambda: call(backend))
            row.update(
                identity,
                pair=repetition,
                order=order,
                backend=backend,
                configuration_sha256=config_sha,
                observed_concurrency=None,
                stage_seconds=None,
                peak_memory_bytes=None,
                output_digest=None,
                correctness_checked=False,
            )
            if row["status"] == "ok":
                row["output_digest"] = _digest(observed)
                if reference is not None:
                    try:
                        comparator(reference, observed, rtol=1e-10, atol=1e-12)
                        row["correctness_checked"] = warmup_error is None
                    except AssertionError as exc:
                        row["status"], row["error"] = "failed", "Numerical mismatch: " + str(exc)
            observations.append(row)
    report = dict(
        operation=identity["operation"],
        dataset=identity["dataset"],
        configuration=configuration,
        configuration_sha256=config_sha,
        warmup_records=warmup_records,
        warmup_error=warmup_error,
        **paired_summary(observations, repeats=repeats),
    )
    return report, observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/rust-metrics-benchmark.json")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.repeats < 7:
        raise ValueError("Benchmark v2 requires at least seven predeclared pairs.")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError("Choose a new output or pass --overwrite explicitly.")
    if rad.backend_info()["metrics_backend"] != "rust":
        raise RuntimeError("Build the Rust extension before benchmarking.")
    # This is an explicitly requested reference benchmark. Load its numerical
    # libraries before setting local, restored thread-pool limits so late imports
    # cannot escape the observed single-thread BLAS/OpenMP configuration.
    from threadpoolctl import threadpool_info, threadpool_limits

    from ov_radiomics import _reference_metrics

    del _reference_metrics
    rng = np.random.RandomState(2026)
    rows, raw = [], []
    manifest = machine_manifest(Path(__file__).resolve().parents[1])
    with threadpool_limits(limits=1):
        active_pools = threadpool_info()
        if any(pool.get("num_threads") != 1 for pool in active_pools):
            raise RuntimeError("Reference thread-pool limits could not be verified as one thread.")
        for n in (100, 10000):
            y = np.tile([0, 1], n // 2)
            rng.shuffle(y)
            a, b = rng.rand(2, n)
            dataset_sha256 = hashlib.sha256(
                y.astype("<i8").tobytes() + a.astype("<f8").tobytes() + b.astype("<f8").tobytes()
            ).hexdigest()
            thresholds = np.linspace(0.001, 0.999, 999)
            cases = {
                "binary_metrics": (
                    lambda backend: rad.binary_metrics(
                        y,
                        a,
                        threshold=0.5,
                        positive_label=1,
                        ci_method="delong",
                        confidence=0.95,
                        n_bootstrap=2000,
                        random_state=0,
                        backend=backend,
                    ),
                    dict(
                        threshold=0.5,
                        positive_label=1,
                        ci_method="delong",
                        confidence=0.95,
                        n_bootstrap=2000,
                        random_state=0,
                    ),
                ),
                "delong_test": (
                    lambda backend: rad.delong_test(y, a, b, positive_label=1, backend=backend),
                    dict(positive_label=1),
                ),
                "decision_curve_999_thresholds": (
                    lambda backend: rad.decision_curve(
                        y, a, thresholds=thresholds, positive_label=1, backend=backend
                    ),
                    dict(positive_label=1, thresholds=thresholds.tolist()),
                ),
            }
            for operation, (call, configuration) in cases.items():
                identity = dict(
                    suite="machine_learning_statistics",
                    level="public_metrics",
                    dataset="synthetic-randomstate-2026-n" + str(n),
                    input_sha256=dataset_sha256,
                    operation=operation,
                    state="warm_uncached",
                    threads=1,
                )
                case, observations = _run_case(call, identity, configuration, args.repeats)
                case["n"] = n
                rows.append(case)
                raw.extend(observations)
        pools_after = threadpool_info()
    result = dict(
        schema=SCHEMA,
        suite="machine_learning_statistics",
        machine=manifest,
        backend=rad.backend_info(),
        seed=2026,
        repeats=args.repeats,
        warmup_pairs=1,
        scope=(
            "Warm public metric calls including validation/conversion/output; excludes extraction,"
            " modeling and pathology."
        ),
        states_measured=["warm_uncached"],
        states_unmeasured=["cold_start", "cache_hit"],
        thread_policy=(
            "Reference BLAS/OpenMP pools locally limited to one and restored after measurement; OS"
            " observed concurrency unmeasured. Native metric algorithms are serial."
        ),
        reference_thread_pools_before=active_pools,
        reference_thread_pools_after=pools_after,
        comparison_tolerance=dict(atol=1e-12, rtol=1e-10),
        results=rows,
        raw_observations=raw,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        pd.DataFrame(
            [
                dict(
                    operation=r["operation"],
                    n=r["n"],
                    ratio_python_over_rust=r.get("ratio_python_over_rust"),
                )
                for r in rows
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
