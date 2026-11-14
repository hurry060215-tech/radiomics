"""Owned-worker timing/memory protocol. No numerical model or image algorithms."""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def process_memory(pid):
    if sys.platform=='win32':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in (
                'peak','resident','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        kernel=ctypes.windll.kernel32;api=ctypes.windll.psapi
        kernel.OpenProcess.restype=ctypes.c_void_p;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        api.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
        if not handle:
            return None,None
        try:
            counter=Counters();counter.cb=ctypes.sizeof(counter)
            if not api.GetProcessMemoryInfo(handle,ctypes.byref(counter),ctypes.sizeof(counter)):
                return None,None
            return int(counter.resident),int(counter.peak)
        finally:
            kernel.CloseHandle(handle)
    try:
        text=Path(f'/proc/{pid}/status').read_text(encoding='utf-8')
        fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in text.splitlines() if ':' in line}
        return int(fields['VmRSS'].split()[0])*1024,int(fields['VmHWM'].split()[0])*1024
    except (OSError,ValueError,KeyError):
        return None,None


def lifetime_peak():
    if sys.platform=='win32':
        return process_memory(os.getpid())[1]
    import resource
    count=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(count if sys.platform=='darwin' else count*1024)


class Stages:
    def __init__(self,measured):
        self.measured=measured;self.seconds={}

    @contextmanager
    def __call__(self,name):
        if not self.measured:
            yield
            return
        started=time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name]=self.seconds.get(name,0.)+time.perf_counter()-started


def emit_worker(result):
    # Capture the kernel high-water mark after the actual large result transport.
    print(json.dumps(result,allow_nan=False,sort_keys=True),flush=True)
    print(json.dumps(dict(worker_peak_after_transport_bytes=lifetime_peak())),flush=True)


def validate_receipt(receipt,wheel,base):
    from benchmark_learning_pathology import file_hash
    if receipt.get('schema')!='radiomics.native-build.v1' or receipt.get('source_tree_clean') is not True:
        raise ValueError('A clean immutable native build receipt is required.')
    for key,length in [('source_revision',40),('source_tree',40),('core_implementation_sha256',64),('cargo_lock_sha256',64)]:
        value=receipt.get(key)
        if not isinstance(value,str) or len(value)!=length or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing native build provenance: '+key)
    for key in ('rustc_verbose','rust_target','python_version','maturin_version'):
        if not isinstance(receipt.get(key),str) or not receipt[key]:
            raise ValueError('Missing native build provenance: '+key)
    if receipt.get('profile')!=dict(name='release',opt_level=3,lto='thin',codegen_units=1):
        raise ValueError('The native release profile differs from the frozen policy.')
    if any(key not in receipt.get('rust_flags',{}) for key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS')):
        raise ValueError('Both Rust flag states must be recorded explicitly.')
    if not receipt.get('build_commands'):
        raise ValueError('Native build command argv is missing.')
    for key in ('wheel','cli'):
        value=receipt.get(key,{})
        path=Path(value.get('path',''));path=path if path.is_absolute() else Path(base)/path
        if not path.is_file() or file_hash(path)!=value.get('sha256'):
            raise ValueError('Native '+key+' artifact does not match the immutable receipt.')
    if file_hash(wheel)!=receipt['wheel']['sha256']:
        raise ValueError('The requested wheel does not match its receipt.')


def launch(executable,script,request,request_path,stdout_path,stderr_path,*,audit,finalize,budget,timeout,measured=True):
    """All byte verification and sidecar hashing stay outside cold wall time."""
    from benchmark_learning_pathology import write_json
    try:
        before=audit(request)
    except Exception as exc:
        return dict(status='failed',error=f'PreflightAuditFailure: {type(exc).__name__}: {exc}',
            worker_started=False,peak_memory_bytes=None)
    payload={key:value for key,value in request.items() if key!='audit_environment'}
    write_json(request_path,payload)
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
    environment.update(request['plan_contents']['execution']['thread_environment'])
    environment.update(MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    process=None;observed=0;samples=0;enforced=None;started=None;wall=None
    try:
        with Path(stdout_path).open('xb') as out,Path(stderr_path).open('xb') as err:
            # This watchdog is operational even for untimed correctness probes;
            # its duration is not returned as a performance sample in that mode.
            watchdog=time.monotonic()
            if measured:
                started=time.perf_counter()
            process=subprocess.Popen([str(executable),str(script),'--worker',str(request_path)],
                cwd=Path(script).resolve().parents[1],env=environment,stdout=out,stderr=err)
            while process.poll() is None:
                resident,peak=process_memory(process.pid)
                if resident is not None:
                    samples+=1;observed=max(observed,resident,peak or 0)
                    if observed>budget:
                        enforced='OperationalFailure: fixed resident-memory budget exceeded.';process.kill();break
                if time.monotonic()-watchdog>timeout:
                    enforced='OperationalFailure: fixed observation timeout exceeded.';process.kill();break
                time.sleep(.02)
            process.wait()
            if measured:
                wall=time.perf_counter()-started
    except Exception as exc:
        cleanup_error=None
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait()
            except Exception as cleanup:
                cleanup_error=f'{type(cleanup).__name__}: {cleanup}'
        return dict(status='failed',error=f'OwnedWorkerLaunchOrMonitorFailure: {type(exc).__name__}: {exc}',
            cleanup_error=cleanup_error,worker_started=process is not None,peak_memory_bytes=observed or None,
            stdout_path=str(stdout_path),stderr_path=str(stderr_path))
    try:
        messages=[json.loads(line) for line in Path(stdout_path).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(messages)!=2 or any(not isinstance(value,dict) for value in messages):
            raise ValueError('Expected one result and one post-transport memory footer.')
        result=messages[0];footer=messages[1].get('worker_peak_after_transport_bytes')
        if type(footer) is not int or footer<=0:
            raise ValueError('No valid process lifetime resident-memory footer.')
        worker_peak=max(footer,result.get('peak_memory_bytes') or 0)
    except Exception as exc:
        result=dict(status='failed',error=f'WorkerProtocolFailure: {type(exc).__name__}: {exc}');worker_peak=0
    peak=max(observed,worker_peak)
    result.update(worker_started=True,worker_exit_code=process.returncode,worker_peak_memory_bytes=worker_peak or None,
        coordinator_peak_resident_bytes=observed or None,peak_memory_bytes=peak or None,resident_poll_samples=samples,
        resident_poll_seconds=.02,stdout_path=str(stdout_path),stderr_path=str(stderr_path))
    if process.returncode or enforced:
        result.update(status='failed',error=enforced or result.get('error') or f'Worker exited {process.returncode}.')
    if peak>budget:
        result.update(status='failed',error='OperationalFailure: full process lifetime peak exceeded the fixed budget.')
    try:
        after=audit(request);result['non_timed_audits']=dict(before=before,after=after)
        result=finalize(result,request)
    except Exception as exc:
        result.update(status='failed',error=f'PostflightAuditOrArtifactFailure: {type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    if measured:
        result['process_wall_seconds']=wall
        if request.get('case',{}).get('state')=='cold_process_total':
            result['api_elapsed_seconds']=result.get('elapsed_seconds');result['elapsed_seconds']=wall
    return result
