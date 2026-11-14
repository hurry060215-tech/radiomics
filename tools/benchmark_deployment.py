"""Independent L5 startup and wheel-payload evidence; no feature/model timings."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
from benchmark_protocol import latency_summary, sha256_file
from benchmark_radiomics import validate_build_manifest, wheel_package_files, write_json


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def footprint(directory):
    files=[]
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():
            raise ValueError('Footprint scan refuses symlink traversal.')
        if path.is_file():
            files.append(path)
    return dict(regular_file_count=len(files),logical_size_bytes=sum(path.stat().st_size for path in files),
        metric='Summed logical regular-file bytes, not physical allocation or shared/base Python storage.')


def installed_identity(purelib,expected):
    root=Path(purelib)/'ov_radiomics'
    actual={str(path.relative_to(root)).replace('\\','/'):sha256_file(path) for path in sorted(root.rglob('*'))
        if path.is_file() and path.suffix.lower() in ('.py','.pyd','.so','.dll')}
    if actual!=expected:
        raise RuntimeError('Installed package Python/native bytes differ from the immutable wheel.')
    return dict(package_path=str(root),files_verified=len(actual),files_sha256=actual)


def timed_process(argv,environment,timeout):
    started=time.perf_counter()
    try:
        result=subprocess.run(argv,env=environment,capture_output=True,timeout=timeout)
        return dict(status='ok' if result.returncode==0 else 'failed',exit_code=result.returncode,
            elapsed_seconds=time.perf_counter()-started,error=None,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        return dict(status='failed',exit_code=None,elapsed_seconds=time.perf_counter()-started,
            error='Owned subprocess exceeded fixed timeout; subprocess.run stopped it.',stdout=exc.stdout or b'',stderr=exc.stderr or b'')
    except OSError as exc:
        return dict(status='failed',exit_code=None,elapsed_seconds=time.perf_counter()-started,
            error=f'{type(exc).__name__}: {exc}',stdout=b'',stderr=b'')


def persist_logs(output,prefix,row):
    for stream in ('stdout','stderr'):
        raw=row.pop(stream)
        path=output/(prefix+'.'+stream+'.txt');path.write_bytes(raw)
        row[stream+'_path']=path.name;row[stream+'_sha256']=hashlib.sha256(raw).hexdigest()
        row[stream+'_utf8']=raw.decode('utf-8',errors='replace')
    return row


def payload_footprint(purelib):
    purelib=Path(purelib);paths=[purelib/'ov_radiomics']+list(purelib.glob('ov_radiomics-*.dist-info'))
    reports=[dict(path=str(path),**footprint(path)) for path in paths]
    return dict(paths=reports,regular_file_count=sum(row['regular_file_count'] for row in reports),
        logical_size_bytes=sum(row['logical_size_bytes'] for row in reports))


def run(args):
    plan_path=Path(args.plan);raw_plan=plan_path.read_bytes();digest=hashlib.sha256(raw_plan).hexdigest()
    if digest!=args.approved_plan_sha256:
        raise ValueError('Deployment plan differs from its exact main-reviewed SHA.')
    plan=json.loads(raw_plan)
    if plan['schema']!='radiomics.benchmark.v2.deployment_plan.v1' or plan['measurement']['repetitions_per_case']!=7:
        raise ValueError('Expected the approved independent7+7 startup plan.')
    manifest_path=Path(args.build_manifest).resolve();manifest=json.loads(manifest_path.read_bytes())
    wheel=Path(manifest['wheel']['path']);cli=Path(manifest['cli']['path'])
    validate_build_manifest(manifest,wheel,manifest_path.parent)
    for key in ('source_revision','core_implementation_sha256'):
        if manifest[key]!=plan['snapshot'][key]:
            raise ValueError('Deployment native snapshot changed: '+key)
    for key in ('wheel','cli'):
        if manifest[key]['sha256']!=plan['snapshot'][key+'_sha256']:
            raise ValueError('Deployment artifact differs from predeclared '+key)
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Deployment evidence output must be fresh; no rerun overwrite.')
    output.mkdir(parents=True)
    write_json(output/'approved-plan.json',plan);write_json(output/'native-build-manifest.json',manifest)
    started_utc=utc_now();environment=dict(os.environ)
    environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
    environment.update(plan['measurement']['thread_environment']);environment['PIP_NO_INDEX']='1'
    venv=output/'fresh-environment'
    setup_argv=[str(Path(args.base_python).resolve()),'-I','-m','venv',str(venv)]
    setup=subprocess.run(setup_argv,env=environment,capture_output=True,timeout=180)
    (output/'venv-setup.stdout.txt').write_bytes(setup.stdout);(output/'venv-setup.stderr.txt').write_bytes(setup.stderr)
    if setup.returncode!=0:
        raise RuntimeError('Untimed fresh environment creation failed; evidence preserved.')
    python=venv/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    metadata=subprocess.run([str(python),'-I','-c',"import json,sys,sysconfig;print(json.dumps({'version':sys.version,'purelib':sysconfig.get_paths()['purelib']}))"],
        env=environment,capture_output=True,check=True)
    python_info=json.loads(metadata.stdout);purelib=Path(python_info['purelib'])
    if any(purelib.glob('ov_radiomics*')):
        raise RuntimeError('Fresh environment already contains the project payload.')
    baseline=footprint(venv)
    install_argv=[str(python),'-I','-m','pip','--disable-pip-version-check','install','--no-index','--no-deps','--no-cache-dir',str(wheel)]
    installation=timed_process(install_argv,environment,180)
    installation=persist_logs(output,'payload-install',installation)
    installation.update(argv=install_argv,n=1,scope='Offline project-wheel payload only; dependencies/base Python/environment creation excluded.',
        full_dependency_install_cost_measured=False)
    write_json(output/'payload-install.json',installation)
    if installation['status']!='ok':
        raise RuntimeError('Single payload install failed; do not retry/select a faster install.')
    expected=wheel_package_files(wheel);identity=installed_identity(purelib,expected)
    post_install=footprint(venv);package_post_install=payload_footprint(purelib)
    observations=[]
    for round_index in range(7):
        cases=plan['startup_cases'] if round_index%2==0 else list(reversed(plan['startup_cases']))
        for order,case in enumerate(cases):
            argv=[str(cli) if arg=='IMMUTABLE_CLI' else str(python) if arg=='FRESH_VENV_PYTHON' else arg for arg in case['argv']]
            row=timed_process(argv,environment,plan['measurement']['timeout_seconds_per_startup'])
            stdout=row['stdout'];matched=False;validation_error=None
            if row['status']=='ok':
                try:
                    if case['id']=='cli_version_fresh_process':
                        matched=stdout.decode('utf-8').strip()=='radiomics '+plan['snapshot']['package_version']
                    else:
                        value=json.loads(stdout);matched=(value['version']==plan['snapshot']['package_version']
                            and value['backend']['metrics_backend']=='rust' and value['backend']['rust_core_version']==plan['snapshot']['package_version']
                            and value['implementation']==manifest['core_implementation_sha256'])
                except Exception as exc:
                    validation_error=f'{type(exc).__name__}: {exc}'
            row=persist_logs(output,f"round-{round_index:02}-{case['id']}",row)
            row.update(case_id=case['id'],round=round_index,order=order,argv=argv,output_matches_expected=matched,
                output_validation_error=validation_error,fresh_process=True)
            observations.append(row)
            with (output/'raw-observations.jsonl').open('a',encoding='utf-8',newline='\n') as handle:
                handle.write(json.dumps(row,ensure_ascii=True,sort_keys=True)+'\n');handle.flush()
    # Byte audits and file-count scans are deliberately outside startup elapsed.
    installed_identity(purelib,expected)
    if sha256_file(wheel)!=manifest['wheel']['sha256'] or sha256_file(cli)!=manifest['cli']['sha256']:
        raise RuntimeError('Deployment artifacts changed during observation.')
    summaries={}
    for case in plan['startup_cases']:
        rows=[row for row in observations if row['case_id']==case['id']]
        valid=len(rows)==7 and all(row['status']=='ok' and row['output_matches_expected'] for row in rows)
        summaries[case['id']]=dict(valid=valid,n=len(rows),statistics=latency_summary([row['elapsed_seconds'] for row in rows]) if valid else None)
    with zipfile.ZipFile(wheel) as archive:
        infos=[info for info in archive.infolist() if not info.is_dir()]
        wheel_info=dict(file_size_bytes=wheel.stat().st_size,zip_file_count=len(infos),uncompressed_file_bytes=sum(info.file_size for info in infos))
    report=dict(schema='radiomics.benchmark.v2.deployment_report.v1',suite='deployment',approved_plan_sha256=digest,
        started_utc=started_utc,finished_utc=utc_now(),native_build=manifest,python=python_info,platform=platform.platform(),
        setup=dict(argv=setup_argv,timed=False,system_site_packages=False),installation=installation,
        artifact_footprint=dict(wheel=wheel_info,cli=dict(file_size_bytes=cli.stat().st_size)),
        environment_footprint=dict(baseline=baseline,post_payload_install=post_install,post_startups=footprint(venv)),
        project_payload_footprint=dict(post_install=package_post_install,post_startups=payload_footprint(purelib)),
        installed_identity=identity,startup_results=summaries,raw_observations=observations,
        operational_status='passed_declared_endpoints' if all(row['valid'] for row in summaries.values()) else 'failed_or_incomplete_declared_endpoints',
        comparative_speedup=None,scope='Two distinct startup endpoints; no cross-language speed ratio or pooled suite score.',
        limitations=['One offline --no-deps wheel-payload install; no dependency resolution/download/install cost',
            'Fresh interpreter/process, not flushed OS disk/bytecode cache','Logical bytes/file counts, not filesystem allocation/shared base Python size',
            'Only Windows/native snapshot and these two endpoints measured; no algorithm or full-workflow timing',
            'Seven repeats give empirical p95 only; unrelated OS/user background processes not controlled'])
    write_json(output/'report.json',report)
    lines=['# Independent L5 deployment evidence','',f"Plan SHA256: `{digest}`.",
        '', '| Endpoint | n | Median ms | IQR ms | Empirical p95 ms | Output validated |','|---|---:|---:|---:|---:|---|']
    for name,result in summaries.items():
        stats=result['statistics']
        lines.append(f"| {name} | {result['n']} | {stats['median_seconds']*1000:.3f} | {stats['iqr_seconds']*1000:.3f} | {stats['p95_seconds']*1000:.3f} | {result['valid']} |" if stats else f"| {name} | {result['n']} | withheld | withheld | withheld | failed |")
    lines+=['',f"Single offline wheel-payload install: {installation['elapsed_seconds']:.6f}s, n=1. This excludes dependencies, downloads, resolver and environment/base-Python creation; it is not complete installation cost.",
        '',f"Wheel: {wheel_info['file_size_bytes']} compressed bytes, {wheel_info['zip_file_count']} ZIP files, {wheel_info['uncompressed_file_bytes']} uncompressed bytes. CLI: {cli.stat().st_size} bytes.",
        f"Installed project payload after startup: {report['project_payload_footprint']['post_startups']['regular_file_count']} regular files, {report['project_payload_footprint']['post_startups']['logical_size_bytes']} logical bytes.",
        '', 'The two endpoints perform different work; no CLI/Python speed ratio is computed. No language-wide speedup or combined Radiomics/ML/Pathology score is inferred.',
        '', *['- '+item for item in report['limitations']], '']
    with (output/'report.md').open('x',encoding='utf-8',newline='\n') as handle:handle.write('\n'.join(lines))
    print(json.dumps(dict(status=report['operational_status'],output=str(output),startup_results=summaries),ensure_ascii=True,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--approved-plan-sha256',required=True)
    parser.add_argument('--build-manifest',required=True);parser.add_argument('--base-python',required=True);parser.add_argument('--output',required=True)
    run(parser.parse_args())
