"""Compare full public calls, including validation, conversion and DataFrame output.

Run after building the Rust extension; results do not measure image extraction or ML.
"""
import argparse
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import ov_radiomics as rad
from benchmark_protocol import machine_manifest, paired_summary, SCHEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/rust-metrics-benchmark.json')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.repeats < 7:
        raise ValueError('Benchmark v2 requires at least seven predeclared pairs.')
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError('Choose a new output or pass --overwrite explicitly.')
    if rad.backend_info()['metrics_backend'] != 'rust':
        raise RuntimeError('Build the Rust extension before benchmarking.')
    rng = np.random.RandomState(2026)
    rows, raw = [], []
    manifest = machine_manifest(Path(__file__).resolve().parents[1])
    for n in (100, 10000):
        y = np.tile([0, 1], n//2)
        rng.shuffle(y)
        a, b = rng.rand(2, n)
        dataset_sha256=hashlib.sha256(y.astype('<i8').tobytes()+a.astype('<f8').tobytes()+b.astype('<f8').tobytes()).hexdigest()
        calls = {
            'binary_metrics': lambda backend: rad.binary_metrics(y,a,backend=backend),
            'delong_test': lambda backend: rad.delong_test(y,a,b,backend=backend),
            'decision_curve_999_thresholds': lambda backend: rad.decision_curve(y,a,thresholds=np.linspace(.001,.999,999),backend=backend),
        }
        for operation, call in calls.items():
            python_result, rust_result = call('python'), call('rust')
            comparator = pd.testing.assert_frame_equal if isinstance(python_result,pd.DataFrame) else pd.testing.assert_series_equal
            warmup_error=None
            try:
                comparator(python_result,rust_result,rtol=1e-10,atol=1e-12)
            except AssertionError as exc:
                warmup_error=str(exc)
            observations=[]
            for repetition in range(args.repeats):
                # Alternate execution order to reduce warm-cache/order bias.
                for order,backend in enumerate(('python','rust') if repetition % 2 == 0 else ('rust','python')):
                    start = time.perf_counter()
                    status,error,digest,correct='ok',None,None,False
                    try:
                        observed=call(backend)
                        elapsed=time.perf_counter()-start
                        comparator(python_result,observed,rtol=1e-10,atol=1e-12)
                        digest=hashlib.sha256(observed.to_json(double_precision=15).encode()).hexdigest()
                        correct=warmup_error is None
                    except Exception as exc:
                        elapsed=time.perf_counter()-start
                        status,error='failed',f'{type(exc).__name__}: {exc}'
                    observations.append(dict(suite='machine_learning_statistics',level='public_metrics',
                        dataset='synthetic-randomstate-2026-n'+str(n),input_sha256=dataset_sha256,operation=operation,state='warm_uncached',
                        threads=1,observed_concurrency=None,pair=repetition,order=order,backend=backend,
                        elapsed_seconds=elapsed,stage_seconds=None,peak_memory_bytes=None,output_digest=digest,
                        status=status,error=error,correctness_checked=correct))
            raw.extend(observations)
            rows.append(dict(operation=operation,n=n,warmup_error=warmup_error,**paired_summary(observations,repeats=args.repeats)))
    result = dict(schema=SCHEMA,suite='machine_learning_statistics',machine=manifest,
              backend=rad.backend_info(),seed=2026,repeats=args.repeats,warmup_pairs=1,
              scope='Warm public metric calls including validation/conversion/output; excludes extraction, modeling and pathology.',
              states_measured=['warm_uncached'],states_unmeasured=['cold_start','cache_hit'],
              thread_policy='Caller must set and verify reference thread limits before launch; observed concurrency unmeasured.',
              results=rows,raw_observations=raw)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame([dict(operation=r['operation'],n=r['n'],ratio_python_over_rust=r.get('ratio_python_over_rust')) for r in rows]).to_string(index=False))


if __name__ == '__main__':
    main()
