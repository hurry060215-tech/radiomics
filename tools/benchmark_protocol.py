"""Benchmark v2 evidence utilities. Standard library only; no pooled suite score."""
import hashlib
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

SCHEMA='radiomics.benchmark.v2'
CASE_FIELDS=('suite','level','dataset','operation','state','threads','input_sha256','configuration_sha256')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _command(argv,cwd):
    try:
        output=subprocess.run(argv,cwd=cwd,check=True,capture_output=True,text=True,timeout=15)
        return output.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return None


def machine_manifest(root):
    root=Path(root)
    cpu=platform.processor() or None
    memory=None
    if sys.platform=='win32':
        import ctypes
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        except OSError:
            pass
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                ('total_physical',ctypes.c_ulonglong),('available_physical',ctypes.c_ulonglong),
                ('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),
                ('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),
                ('available_extended_virtual',ctypes.c_ulonglong)]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory=status.total_physical
    else:
        try:
            memory=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        except (ValueError,OSError,AttributeError):
            pass
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    cpu=line.split(':',1)[1].strip();break
        except OSError:
            pass
    versions={}
    for name in ('numpy','pandas','SimpleITK','pyradiomics','scikit-learn','scipy','statsmodels','maturin'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    compiled={}
    for module in ('ov_radiomics._native','radiomics._cmatrices','radiomics._cshape'):
        try:
            spec=importlib.util.find_spec(module)
            origin=spec.origin if spec else None
            compiled[module]=None if origin is None or not Path(origin).is_file() else dict(path=origin,sha256=sha256_file(origin))
        except (ImportError,AttributeError,ValueError):
            compiled[module]=None
    return dict(platform=platform.platform(),cpu=cpu,logical_cpus=os.cpu_count(),
        physical_cpus=None,total_physical_memory_bytes=memory,python=platform.python_version(),
        python_executable=sys.executable,packages=versions,compiled_extensions=compiled,
        rustc_verbose=_command(['rustc','-vV'],root),git_revision=_command(['git','rev-parse','HEAD'],root),
        working_tree_status=_command(['git','status','--porcelain'],root),
        cargo_lock_sha256=sha256_file(root/'Cargo.lock'),
        benchmark_sources_sha256={name:sha256_file(root/name) for name in ('tools/benchmark_protocol.py','tools/benchmark_metrics.py')},
        thread_environment={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')},
        rust_flags={name:os.environ.get(name) for name in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET')},
        build_profile='release requested by caller; binary build provenance must be supplied separately',
        file_cache_control='uncontrolled',power_mode='unrecorded',background_load='unrecorded',
        allocator='not overridden by runner',missing_provenance=['physical core count','power mode','background load','storage description'])


def quantile_linear(values,p):
    values=sorted(values)
    if not values or not 0<=p<=1:
        raise ValueError('A nonempty sample and quantile in [0,1] are required.')
    x=(len(values)-1)*p;i=math.floor(x);j=math.ceil(x)
    return values[i]*(1-(x-i))+values[j]*(x-i)


def latency_summary(values):
    if not values or any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Latency observations must be finite, positive seconds.')
    q25,q75=quantile_linear(values,.25),quantile_linear(values,.75)
    return dict(n=len(values),median_seconds=statistics.median(values),q25_seconds=q25,q75_seconds=q75,
        iqr_seconds=q75-q25,p95_seconds=quantile_linear(values,.95),minimum_seconds=min(values),maximum_seconds=max(values),
        p95_interpretation='empirical linear quantile; seven pairs do not estimate a precise tail guarantee')


def capture_call(call):
    """Retain workload failures, including warmup, without timing comparisons/digests."""
    start=time.perf_counter()
    try:
        output=call()
    except Exception as exc:
        return dict(status='failed',elapsed_seconds=time.perf_counter()-start,
            error=f'{type(exc).__name__}: {exc}'),None
    return dict(status='ok',elapsed_seconds=time.perf_counter()-start,error=None),output


def paired_summary(records,*,repeats):
    """One predeclared case only. Failed/mismatching outputs have no speed claim."""
    if repeats<7:
        raise ValueError('Benchmark v2 requires at least seven predeclared repetition pairs.')
    by_pair={}
    case=None
    for row in records:
        if any(key not in row for key in CASE_FIELDS):
            raise ValueError('Every observation needs its complete benchmark case identity.')
        if (any(not isinstance(row[key],str) or not row[key].strip() for key in CASE_FIELDS if key!='threads')
                or type(row['threads']) is not int or row['threads']<1):
            raise ValueError('Case identities must be nonempty strings and threads a positive integer.')
        if any(len(row[key])!=64 or any(c not in '0123456789abcdef' for c in row[key])
                for key in ('input_sha256','configuration_sha256')):
            raise ValueError('Case input/configuration digests must be lowercase SHA256 strings.')
        identity=tuple(row[key] for key in CASE_FIELDS)
        if case is None:
            case=identity
        elif case!=identity:
            raise ValueError('Paired observations must have the same suite, level, dataset, operation, state, threads and input/configuration hashes.')
        pair=row['pair'];backend=row['backend']
        if (type(pair) is not int or backend not in ('python','rust') or not 0<=pair<repeats
                or backend in by_pair.setdefault(pair,{})):
            raise ValueError('Duplicate/unknown backend or invalid paired observation.')
        expected_order=(0 if backend=='python' else 1) if pair%2==0 else (1 if backend=='python' else 0)
        if type(row.get('order')) is not int or row['order']!=expected_order:
            raise ValueError('Pair order must follow the declared alternating Python/Rust sequence.')
        if type(row.get('correctness_checked')) is not bool or row.get('status') not in ('ok','failed'):
            raise ValueError('Observation status and correctness flags must be explicit typed values.')
        by_pair[pair][backend]=row
    if len(by_pair)!=repeats or any(set(row)!= {'python','rust'} for row in by_pair.values()):
        raise ValueError('Every declared pair needs both backend observations, including failures.')
    valid=all(r['status']=='ok' and r['correctness_checked'] for r in records)
    output=dict(valid_speed_comparison=valid,performance_target='report_only',
        observed_pairs=repeats,raw_observations=len(records))
    if not valid:
        output['reason']='Execution or correctness failed; timing is diagnostic only.'
        return output
    summaries={backend:latency_summary([by_pair[pair][backend]['elapsed_seconds'] for pair in range(repeats)])
        for backend in ('python','rust')}
    output.update(backends=summaries,ratio_python_over_rust=summaries['python']['median_seconds']/summaries['rust']['median_seconds'],
        paired_ratios=[by_pair[pair]['python']['elapsed_seconds']/by_pair[pair]['rust']['elapsed_seconds'] for pair in range(repeats)],
        peak_memory_comparison=None,memory_status='unmeasured; no memory target claim',
        statistical_interval=None,interval_status='not estimated; raw pairs and spread retained')
    return output
