"""Compare PR-7 and parallel candidates in fresh L2/L3 workers, outside Git.

This diagnostic never overwrites benchmark v2, Golden or earlier run evidence.
Runtime and RSS use separate workers. RSS is actual-worker lifetime resident HWM.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from profile_native_boundary import check_features, host_info, load_inputs, sha, summary
from profile_native_memory import own_memory

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import pandas as pd
    import threadpoolctl
    import ov_radiomics as rad
    from ov_radiomics._backend import resolve_backend
    native = resolve_backend('rust')
    spec = json.loads((ROOT/'benchmarks/v2/radiomics-ct-plan.json').read_text())
    reference = json.loads((args.fixtures/'bbox_crop_reference.json').read_text())
    config = spec['native_extraction']
    snapshots = [own_memory('common_imports_loaded')]
    if args.stage == 'L2':
        values = np.load(args.fixtures/'values.npy', allow_pickle=False)
        labels = np.load(args.fixtures/'mask.npy', allow_pickle=False)
    else:
        values = labels = None  # L3 must not preload the full image or crop.
    supports_threads = 'threads' in inspect.signature(rad.ImageSample.extract).parameters
    if not supports_threads and args.threads != 1:
        raise ValueError('The baseline installation only supports serial feature extraction.')
    keywords = {'threads': args.threads} if supports_threads else {}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    timings, feature_hashes = [], []
    with threadpoolctl.threadpool_limits(limits=1):
        for index in range(-args.warmups, args.repeats):
            started = time.perf_counter_ns()
            if args.stage == 'L2':
                image = rad.image.from_array(values,
                    affine=spec['dataset']['prepared_affine_ras_mm'], spatial_unit='mm')
                mask = rad.mask.from_array(labels, geometry=image.geometry)
                sample = rad.ImageSample(dict(sample_id='public_ct_pat1-R0000', patient_id='P0000'), image, mask)
                frame = pd.concat([sample.extract(config, **keywords).features], ignore_index=True)
                frame = frame[['ID']+reference['output_feature_order']].sort_values('ID').reset_index(drop=True)
            else:
                samples = rad.read_cohort(args.fixtures/'cohort-1.csv')
                result = rad.extract(samples, params=dict(extraction=config, labels=[1],
                    preparation=spec['cases'][1]['preparation']), backend='rust', n_jobs=1, **keywords)
                result.save(output/('export-%02d'%index))
                frame = result.features
            elapsed = time.perf_counter_ns()-started
            features = frame.iloc[0].to_dict()
            features.pop('ID')
            check_features(features, reference)
            # Exact repeat and cross-candidate comparisons are additional to
            # the unchanged numerical tolerances against the frozen reference.
            from hashlib import sha256
            digest = sha256(frame[reference['output_feature_order']].to_numpy(dtype='<f8').tobytes()).hexdigest()
            if index >= 0:
                timings.append(elapsed)
                feature_hashes.append(digest)
            snapshots.append(own_memory('after_call_%d'%index))
        pools = threadpoolctl.threadpool_info()
    if len(set(feature_hashes)) != 1:
        raise ValueError('Repeated feature buffers were not byte identical.')
    sources = {name:sha(Path(rad.__file__).parent/name) for name in
        ('_types.py', '_native_common.py', 'image.py', 'mask.py')}
    report = dict(stage=args.stage, kind=args.kind, pid=os.getpid(), threads=args.threads,
        python_version=sys.version,
        n_jobs=1, native_sha256=sha(native.__file__), package=rad.__file__,
        implementation_sha256=native.IMPLEMENTATION_SHA256,
        measured_python_sources_sha256=sources,
        dependencies=dict(numpy=np.__version__, pandas=pd.__version__, threadpoolctl=threadpoolctl.__version__),
        threadpools=pools, snapshots=snapshots, raw_nanoseconds=timings,
        runtime=summary(timings), feature_buffer_sha256=feature_hashes[0],
        all_66_features_match_frozen_reference=True,
        lifetime_peak_rss_bytes=max(row['lifetime_peak_rss_bytes'] for row in snapshots))
    (output/'worker.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    footer = own_memory('after_output_written')
    print(json.dumps(dict(pid=os.getpid(), stage=args.stage, kind=args.kind,
        lifetime_peak_rss_bytes=footer['lifetime_peak_rss_bytes'])))


def coordinator(args):
    if not args.acknowledge_license:
        raise PermissionError('Explicit --acknowledge-license is required for the existing CT inputs.')
    output = args.output.resolve()
    if any((path/'.git').exists() for path in (output, *output.parents)):
        raise ValueError('Keep all generated outputs outside Git.')
    load_inputs(args.fixtures.resolve(), ROOT)  # Validate every frozen hash before measurements.
    output.mkdir(parents=True, exist_ok=False)
    environments = dict(baseline=(args.baseline_python, 1), candidate=(args.candidate_python, args.threads))
    jobs = []
    # Alternate baseline/candidate order across stages and memory repetitions.
    for stage_index, stage in enumerate(('L2', 'L3')):
        order = ('baseline', 'candidate') if stage_index == 0 else ('candidate', 'baseline')
        jobs.extend((stage, 'runtime', name, 0) for name in order)
    for repetition in range(args.rss_repeats):
        for stage_index, stage in enumerate(('L2', 'L3')):
            order = ('baseline', 'candidate') if (repetition+stage_index)%2 == 0 else ('candidate', 'baseline')
            jobs.extend((stage, 'memory', name, repetition) for name in order)
    raw = []
    for stage, kind, name, repetition in jobs:
        python, threads = environments[name]
        directory = output/('%s-%s-%s-%02d'%(stage, kind, name, repetition))
        command = [str(python.resolve()), str(Path(__file__).resolve()), '--worker',
            '--fixtures', str(args.fixtures.resolve()), '--output', str(directory),
            '--stage', stage, '--kind', kind, '--threads', str(threads),
            '--repeats', str(args.repeats if kind=='runtime' else 1),
            '--warmups', str(args.warmups if kind=='runtime' else 0)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (output/(directory.name+'-stdout.txt')).write_text(result.stdout, encoding='utf-8')
        (output/(directory.name+'-stderr.txt')).write_text(result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Worker failed; its actual command and stderr are retained: '+str(command))
        record = json.loads((directory/'worker.json').read_text())
        footer = json.loads(result.stdout.strip().splitlines()[-1])
        if record['pid'] == os.getpid() or record['pid'] != footer['pid']:
            raise ValueError('RSS must belong to the actual worker PID.')
        record['lifetime_peak_rss_bytes'] = max(record['lifetime_peak_rss_bytes'], footer['lifetime_peak_rss_bytes'])
        record['footer'] = footer
        raw.append(dict(environment=name, repetition=repetition, command=command, result=record))
        with (output/'raw.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(raw[-1])+'\n')
    if len({json.dumps(row['result']['dependencies'], sort_keys=True) for row in raw}) != 1:
        raise ValueError('The compared numerical dependencies differ.')
    results = {}
    for stage in ('L2', 'L3'):
        selected = [row for row in raw if row['result']['stage']==stage]
        if len({row['result']['feature_buffer_sha256'] for row in selected}) != 1:
            raise ValueError('Baseline and candidate feature buffers differ: '+stage)
        runtime = {name:next(row['result']['runtime']['median_ms'] for row in selected
            if row['environment']==name and row['result']['kind']=='runtime') for name in environments}
        memory = {name:statistics.median(row['result']['lifetime_peak_rss_bytes'] for row in selected
            if row['environment']==name and row['result']['kind']=='memory') for name in environments}
        runtime_ratio, memory_ratio = runtime['candidate']/runtime['baseline'], memory['candidate']/memory['baseline']
        results[stage] = dict(runtime_median_ms=runtime, peak_rss_median_bytes=memory,
            candidate_runtime_ratio=runtime_ratio, candidate_memory_ratio=memory_ratio,
            speedup=1/runtime_ratio, runtime_within_1_10=runtime_ratio<=1.10,
            memory_within_1_20=memory_ratio<=1.20, target_1_5_met=1/runtime_ratio>=1.5)
    report = dict(schema='radiomics.parallel-extraction-diagnostic.v1', kind='diagnostic_not_formal_benchmark',
        driver_source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
        driver_sha256=sha(__file__), host=host_info(), threads=args.threads,
        repeats=args.repeats, warmups=args.warmups, rss_repeats=args.rss_repeats,
        fixture_reference_sha256=sha(args.fixtures/'bbox_crop_reference.json'),
        results=results, raw=raw, all_outputs_byte_equal=True,
        limitations=['Fresh-process lifetime peak RSS includes common imports/setup and is not a resettable feature-phase peak.',
            'Runtime workers use one warmup and retained warm-cache repetitions; memory workers use one cold process call.',
            'L2 preloads the fixed crop; L3 reads original NIfTI, prepares and exports through the complete API.',
            'This comparison uses the PR-7 candidate baseline and explicit candidate local threads; historical 854 observations are unchanged.',
            'No universal hardware, language or reference-backend speed claim follows from this diagnostic.'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-python', type=Path)
    parser.add_argument('--candidate-python', type=Path)
    parser.add_argument('--stage', choices=('L2', 'L3'))
    parser.add_argument('--kind', choices=('runtime', 'memory'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--rss-repeats', type=int, default=3)
    parser.add_argument('--acknowledge-license', action='store_true')
    args = parser.parse_args()
    if args.repeats<1 or args.warmups<0 or args.rss_repeats<1 or args.threads<1:
        parser.error('Use positive counts and nonnegative warmups.')
    if args.worker:
        if not args.stage or not args.kind:
            parser.error('Worker requires stage and kind.')
        worker(args)
    else:
        if not args.baseline_python or not args.candidate_python:
            parser.error('Coordinator requires both interpreters.')
        coordinator(args)


if __name__ == '__main__':
    main()
