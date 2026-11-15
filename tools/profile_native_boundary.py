"""Frozen PAT1 boundary diagnostics, never a replacement for the 854-run benchmark.

Run in the immutable native-v2 environment. All image-bearing inputs remain in
an external directory; JSON/image buffers are never printed or committed.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def summary(values):
    values = sorted(values)
    def quantile(p):
        index = (len(values)-1)*p
        lo = int(index)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(index-lo)
    return {'n': len(values), 'median_ms': statistics.median(values)/1e6,
        'iqr_ms': (quantile(.75)-quantile(.25))/1e6, 'p95_ms': quantile(.95)/1e6}


def check_features(actual, reference):
    """The frozen 66 definitions, tolerances and nonfinite locations stay fixed."""
    import math
    expected = {**reference['features'], **reference.get('basic_shape', {})}
    if set(actual) != set(expected):
        raise ValueError('Feature identity differs from the frozen CT66 reference.')
    errors = []
    for name, target in expected.items():
        value = float(actual[name])
        if isinstance(target,str):
            good = (target=='NaN' and math.isnan(value)) or (target=='Inf' and value==math.inf) or (target=='-Inf' and value==-math.inf)
        else:
            good = math.isfinite(value) and abs(value-target) <= 1e-10+1e-8*abs(target)
        if not good:
            errors.append(name)
    if errors:
        raise ValueError('Frozen CT scalar mismatch: '+', '.join(errors))


@contextmanager
def request_trace(common, types, native):
    """Actual nested Python clocks; no native/core function is replaced."""
    requests, copies = [], []
    original_request, original_copy = common.request, types._NativeObject.to_dict
    def traced(operation, **values):
        start = time.perf_counter_ns()
        envelope = dict(schema_version=1, request=dict(operation=operation, **values))
        begin = time.perf_counter_ns()
        text = common.encode_json(envelope)
        encoded = time.perf_counter_ns()
        response = native.request_json(text)
        returned = time.perf_counter_ns()
        value = json.loads(response)
        decoded = time.perf_counter_ns()
        requests.append(dict(operation=operation, encode_ns=encoded-begin,
            native_call_ns=returned-encoded, decode_ns=decoded-returned,
            request_total_ns=decoded-start, request_bytes=len(text.encode('utf-8')),
            response_bytes=len(response.encode('utf-8'))))
        return value
    def copied(self):
        started = time.perf_counter_ns()
        value = original_copy(self)
        copies.append(dict(type=type(self).__name__, nanoseconds=time.perf_counter_ns()-started))
        return value
    common.request, types._NativeObject.to_dict = traced, copied
    try:
        yield requests, copies
    finally:
        common.request, types._NativeObject.to_dict = original_request, original_copy


def load_inputs(fixtures, root):
    import numpy as np
    spec = json.loads((root/'benchmarks/v2/radiomics-ct-plan.json').read_text())
    reference = json.loads((fixtures/'bbox_crop_reference.json').read_text())
    for key, binding in spec['reference_bindings'].items():
        filename = Path(binding['repository_path']).name
        if sha(fixtures/filename) != binding['sha256']:
            raise ValueError('Frozen reference bytes mismatch: '+key)
    for name, digest in spec['dataset']['source_files_sha256'].items():
        if sha(fixtures/name) != digest:
            raise ValueError('Frozen CT/source licence bytes mismatch: '+name)
    values = np.load(fixtures/'values.npy', allow_pickle=False)
    labels = np.load(fixtures/'mask.npy', allow_pickle=False)
    if list(reversed(values.shape)) != reference['geometry']['shape_xyz']:
        raise ValueError('Prepared shape differs.')
    if hashlib.sha256(values.astype('<f8').tobytes()).hexdigest() != reference['image_f64_le_sha256']:
        raise ValueError('Prepared image buffer differs from frozen CT.')
    if hashlib.sha256(labels.astype('<u4').tobytes()).hexdigest() != reference['mask_u32_le_sha256']:
        raise ValueError('Prepared mask buffer differs from frozen CT.')
    return values, labels, spec, reference


def host_info():
    result = dict(platform=platform.platform(), machine=platform.machine(), logical_cpus=os.cpu_count())
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['cpu'] = winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                ('physical_total','physical_available','page_total','page_available','virtual_total','virtual_available','extended')]
        memory=Memory(); memory.length=ctypes.sizeof(memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        result['ram_bytes']=memory.physical_total
    return result


def run(args):
    output, fixtures = args.output.resolve(), args.fixtures.resolve()
    if any((p/'.git').exists() for p in [output,*output.parents]):
        raise ValueError('Licensed CT buffers and outputs must stay outside Git.')
    if not args.acknowledge_license:
        raise PermissionError('Explicit --acknowledge-license is required for existing CC-BY-NC-3.0 CT inputs.')
    output.mkdir(parents=True, exist_ok=False)
    # Local to this diagnostic process; no reference environment is modified.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        os.environ[key]='1'
    import numpy as np
    import pandas as pd
    import threadpoolctl
    import ov_radiomics as rad
    import ov_radiomics._native_common as common
    import ov_radiomics._types as types
    from ov_radiomics._backend import resolve_backend
    native = resolve_backend('rust')
    probe_rustc = subprocess.check_output(['rustc','-Vv'],text=True).strip()
    build = json.loads(args.build_manifest.read_text())
    if sha(native.__file__) != build['wheel_files_sha256']['ov_radiomics/_native.pyd']:
        raise ValueError('Use the fixed immutable native-v2 extension for baseline diagnostics.')
    values, labels, spec, reference = load_inputs(fixtures, ROOT)
    config = spec['native_extraction']
    affine = spec['dataset']['prepared_affine_ras_mm']
    expected_names = reference['output_feature_order']
    image = rad.image.from_array(values, affine=affine, spatial_unit='mm')
    mask = rad.mask.from_array(labels, geometry=image.geometry)
    sample = rad.ImageSample(dict(sample_id='public_ct_pat1-R0000', patient_id='P0000'), image, mask)
    envelope = dict(schema_version=1, request=dict(operation='array_features',
        image=sample._payload['image'], morphology_mask=sample._payload['mask'], intensity_mask=None,
        roi_label=1, config=config, families=None))
    text = common.encode_json(envelope)
    response = native.request_json(text)
    check_features(json.loads(response), reference)
    request_file = output/'array-request.json'
    request_file.write_bytes(text.encode('utf-8'))
    timings, traces, counts = [], [], {}
    l3_counter = 0
    def checked_table(table):
        row = table.features.iloc[0].to_dict()
        row.pop('ID')
        return row
    def full_array():
        image = rad.image.from_array(values, affine=affine, spatial_unit='mm')
        mask = rad.mask.from_array(labels, geometry=image.geometry)
        sample = rad.ImageSample(dict(sample_id='public_ct_pat1-R0000',patient_id='P0000'),image,mask)
        frame = pd.concat([sample.extract(config).features],ignore_index=True)
        return frame[['ID']+expected_names].sort_values('ID').reset_index(drop=True)
    def l3_call():
        nonlocal l3_counter
        phases = {}
        started = time.perf_counter_ns()
        samples = rad.read_cohort(fixtures/'cohort-1.csv')
        loaded = time.perf_counter_ns()
        result = rad.extract(samples,params=dict(extraction=config,labels=[1],preparation=spec['cases'][1]['preparation']),
            backend='rust',n_jobs=1)
        extracted = time.perf_counter_ns()
        result.save(output/('l3-export-%03d'%l3_counter))
        saved = time.perf_counter_ns(); l3_counter += 1
        phases.update(read_cohort_ns=loaded-started,extract_api_ns=extracted-loaded,save_ns=saved-extracted)
        return result.features, phases
    def sample_build():
        return rad.ImageSample(dict(sample_id='public_ct_pat1-R0000',patient_id='P0000'),image,mask)
    stages = [
        ('python_encode_array_envelope',lambda:common.encode_json(envelope)),
        ('pyo3_preencoded_array_call',lambda:native.request_json(text)),
        ('python_decode_66_features',lambda:json.loads(response)),
        ('python_decode_full_image_response',lambda:json.loads(common.encode_json(image._payload))),
        ('numpy_image_to_list',lambda:values.astype(float,copy=False).ravel(order='C').tolist()),
        ('numpy_mask_to_list',lambda:labels.astype(np.uint32).ravel(order='C').tolist()),
        ('python_deepcopy_image_mask',lambda:(deepcopy(image._payload),deepcopy(mask._payload))),
        ('public_image_from_array',lambda:rad.image.from_array(values,affine=affine,spatial_unit='mm')),
        ('public_mask_from_array',lambda:rad.mask.from_array(labels,geometry=image.geometry)),
        ('public_sample_from_validated_volumes',sample_build),
        ('public_extract_from_validated_sample',lambda:sample.extract(config)),
        ('public_l2_complete',full_array),
        ('instrumented_l2_complete',full_array),
        ('public_l3_complete',l3_call),
    ]
    # Prebuild full-image response outside its decode-only timing.
    image_response = common.encode_json(image._payload)
    stages[3] = ('python_decode_full_image_response',lambda:json.loads(image_response))
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        if any(pool.get('num_threads')!=1 for pool in pools):
            raise RuntimeError('Inner numerical thread pools were not limited to1.')
        for round in range(args.warmups+args.repeats):
            for offset in range(len(stages)):
                name, call = stages[(offset+round)%len(stages)]
                observations = None
                if name=='instrumented_l2_complete':
                    with request_trace(common,types,native) as observations:
                        started=time.perf_counter_ns(); value=call(); elapsed=time.perf_counter_ns()-started
                else:
                    started=time.perf_counter_ns(); value=call(); elapsed=time.perf_counter_ns()-started
                # Append before numerical checks so a failing observation and
                # all earlier observations survive a raised diagnostic error.
                with (output/'raw-python.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(round=round-args.warmups,stage=name,nanoseconds=elapsed,
                        warmup=round<args.warmups,phases=value[1] if name=='public_l3_complete' else None))+'\n')
                phases = None
                if name in ('public_l2_complete','instrumented_l2_complete'):
                    row=value.iloc[0].to_dict();row.pop('ID');check_features(row,reference)
                elif name=='public_l3_complete':
                    frame,phases=value;row=frame.iloc[0].to_dict();row.pop('ID');check_features(row,reference)
                elif name=='public_extract_from_validated_sample':
                    check_features(checked_table(value),reference)
                elif name=='pyo3_preencoded_array_call':
                    check_features(json.loads(value),reference)
                if round>=args.warmups:
                    timings.append(dict(round=round-args.warmups,stage=name,nanoseconds=elapsed,phases=phases))
                    counts[name]=counts.get(name,0)+1
                    if observations is not None:
                        traces.append(dict(round=round-args.warmups,requests=observations[0],deepcopies=observations[1]))
                # Destruction is outside the outer clock in Python. Rust black_box
                # probes include temporary-result destruction; this is disclosed.
                del value
    probe_output=output/'rust-probe.json'
    result=subprocess.run([str(args.rust_probe.resolve()),str(request_file),str(fixtures),str(probe_output),
        str(args.repeats),str(args.warmups)],capture_output=True,text=True)
    (output/'rust-stdout.txt').write_text(result.stdout,encoding='utf-8')
    (output/'rust-stderr.txt').write_text(result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Rust diagnostic failed; logs retained.')
    rust=json.loads(probe_output.read_text())
    check_features(rust['features'],reference)
    if rust['implementation_sha256'] != build['core_implementation_sha256']:
        raise ValueError('Rust probe compiled from a different feature implementation.')
    if counts != {name:args.repeats for name,_ in stages}:
        raise ValueError('A declared diagnostic observation was lost.')
    report=dict(schema='radiomics.native-boundary-diagnosis.v1',kind='diagnostic_not_formal_benchmark',
        host=host_info(),python=sys.executable,python_version=platform.python_version(),
        package_source=rad.__file__,native_path=native.__file__,native_sha256=sha(native.__file__),
        native_build_manifest_sha256=sha(args.build_manifest),native_build=build,
        probe_binary_sha256=sha(args.rust_probe),cargo_lock_sha256=sha(ROOT/'Cargo.lock'),
        probe_rustc_verbose=probe_rustc,
        probe_build_command='cargo build --release --locked -p ov-radiomics-core --example profile_native_boundary',
        probe_release_profile=dict(opt_level=3,lto='thin',codegen_units=1),
        probe_build_environment={key:os.environ.get(key) for key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS',
            'CARGO_PROFILE_RELEASE_OPT_LEVEL','CARGO_PROFILE_RELEASE_LTO','CARGO_PROFILE_RELEASE_CODEGEN_UNITS')},
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        support_sources={name:sha(ROOT/name) for name in ['tools/profile_native_boundary.py',
            'crates/radiomics-core/examples/profile_native_boundary.rs','crates/radiomics-python/src/gateway.rs',
            'crates/radiomics-core/src/features.rs']},
        repeats=args.repeats,warmups_per_stage=args.warmups,threadpools=pools,
        thread_environment={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')},
        frozen_fixture_files_sha256={name:sha(fixtures/name) for name in ['values.npy','mask.npy','image.nii.gz','mask.nii.gz','bbox_crop_reference.json']},
        prepared_shape_xyz=list(reversed(values.shape)),roi_voxels=int((labels==1).sum()),
        prepared_affine_ras_mm=affine,feature_count=66,extraction_config=config,
        request_bytes=len(text.encode()),response_bytes=len(response.encode()),
        raw=timings,instrumented_calls=traces,
        python_summaries={name:summary([r['nanoseconds'] for r in timings if r['stage']==name]) for name,_ in stages},
        rust_summaries={name:summary([r['nanoseconds_per_call'] for r in rust['raw'] if r['stage']==name]) for name in sorted({r['stage'] for r in rust['raw']})},
        all_66_features_match_frozen_ct=True,
        limitations=['Independent diagnostic stages overlap; do not add/subtract their medians or subtract official L2 and L3.',
            'Only instrumented request clocks and L3 same-call phases are nested stage observations; instrumentation has bookkeeping overhead.',
            'Pure Rust probe uses current locked dependencies; immutable native-v2 wheel uses its recorded older Cargo.lock. Core implementation SHA must match.',
            'Python outer timings exclude final temporary-result destruction; Rust probes include temporary drops.',
            'Single process warm OS-cache diagnostics, no new PyRadiomics speed comparison, allocator or per-instruction profile; no universal language claim.'])
    write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--build-manifest',type=Path,required=True)
    parser.add_argument('--rust-probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--acknowledge-license',action='store_true')
    args=parser.parse_args()
    if args.repeats<1 or args.warmups<0:parser.error('repeats must be positive; warmups nonnegative.')
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir() and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise
    print('Diagnostic report:',args.output/'report.json')


if __name__=='__main__':
    main()
