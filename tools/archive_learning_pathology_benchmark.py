"""Audit/archive completed observations. Never launches or reruns a workload."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def archive(source,output,plan,source_binding):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Archive output must be fresh.')
    report=load(source/'report.json');binding=load(source/'binding.json')
    raw=(source/'raw-observations.jsonl').read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    results=[case for suite in report['suites'].values() for case in suite['results']]
    if len(rows)!=224 or len(results)!=16:raise ValueError('The fixed sixteen-case /224-observation grid is incomplete.')
    if digest(plan)!=report['plan_sha256'] or digest(source_binding)!=report['source_text_binding_sha256']:
        raise ValueError('Approved original configuration bytes no longer match their report.')
    by_case={}
    for row in rows:by_case.setdefault(row['case_id'],[]).append(row)
    for result in results:
        case=result['case'];observed=by_case.get(case['id'],[])
        if len(observed)!=14 or {(r['pair'],r['backend']) for r in observed}!={(p,b) for p in range(7) for b in ('python','rust')}:
            raise ValueError('A declared pair/backend is absent or duplicated.')
        for row in observed:
            if any(row[key]!=case[key] for key in ('suite','level','dataset','operation','state','threads','input_sha256','configuration_sha256')):
                raise ValueError('An observation belongs to a different case context.')
            expected=(0 if row['backend']=='python' else 1) if row['pair']%2==0 else (1 if row['backend']=='python' else 0)
            if row['order']!=expected:raise ValueError('Backend order differs from the fixed alternating sequence.')
            if row.get('peak_memory_bytes') is not None and row['peak_memory_bytes']!=max(row.get('worker_peak_memory_bytes') or 0,row.get('coordinator_peak_resident_bytes') or 0):
                raise ValueError('The final resident-memory peak is not the observed process high-water maximum.')
            if row.get('source_text_binding_sha256')!=report['source_text_binding_sha256']:
                raise ValueError('Source representation binding is inconsistent.')
            if row['status']=='ok' and row.get('output_bytes',0)<=0:
                raise ValueError('A successful measured workflow did not retain its declared timed output bytes.')
            if row['status']=='ok' and (row.get('peak_memory_bytes') is None or row['peak_memory_bytes']>2**31):
                raise ValueError('A purportedly successful observation lacks budget-conformant memory evidence.')
    output.mkdir(parents=True)
    for name in ('report.json','binding.json','provenance.json'):
        shutil.copyfile(source/name,output/name)
    shutil.copyfile(plan,output/'approved-plan.json')
    shutil.copyfile(source_binding,output/'approved-source-text-binding.json')
    (output/'raw-observations.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    index=dict(schema='radiomics.benchmark.v2.learning_pathology_archive.v1',source_directory=str(source.resolve()),
        native_source=binding['native_build']['source_revision'],core_implementation=binding['native_build']['core_implementation_sha256'],
        raw_sha256=hashlib.sha256(raw).hexdigest(),observations=len(rows),declared_cases=len(results),
        successful_executions=sum(r['status']=='ok' for r in rows),matching_outputs=sum(r['correctness_checked'] is True for r in rows),
        peak_rss_max_bytes=max((r['peak_memory_bytes'] for r in rows if r.get('peak_memory_bytes') is not None),default=None),
        artifacts={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in sorted(output.iterdir()) if path.is_file()})
    lines=['# Independent learning and pathology benchmark results','',
        'Fixed native source `'+index['native_source']+'`. All observations, including misses/failures, are retained. '
        'These two suites have separate result rows and no pooled acceleration score.','',
        f"Grid: {len(results)} case/state combinations, {len(rows)} backend observations; successful execution {index['successful_executions']}, matching output {index['matching_outputs']}.",'']
    for suite,block in report['suites'].items():
        lines+=['## '+suite,'','| Case | State | Python median s | Native median s | Python/native ratio | Native/reference RSS | Runtime >1.10× | RSS >1.20× |',
            '|---|---|---:|---:|---:|---:|---|---|']
        for result in block['results']:
            case=result['case']
            if result['valid_speed_comparison']:
                lines.append(f"| {case['operation']} | {case['state']} | {result['backends']['python']['median_seconds']:.6g} | {result['backends']['rust']['median_seconds']:.6g} | {result['ratio_python_over_rust']:.4g} | {result['peak_memory_comparison']:.4g} | {result['runtime_target_miss']} | {result['memory_target_miss']} |")
            else:lines.append(f"| {case['operation']} | {case['state']} | diagnostic | diagnostic | withheld | inspect raw | incomplete/failure | incomplete/failure |")
        lines.append('')
    lines+=['## Interpretation and limits','',
        '- Values below one in the Python/native ratio are native slowdowns, retained rather than optimized away by changing the experiment.',
        '- Runtime1.10× and memory1.20× are engineering targets, not algorithm-correctness release gates. Per-backend median/IQR/p95/min/max/n and paired ratios are in report.json; p95 is empirical with n=7.',
        '- Each primary pathology front-end row contains QC/colors/ID alignment/hierarchy/actual tile-feature overlays/exports and excludes model fitting. The independent patient-model row belongs to ML & Statistics. The extra full-scene diagnostic retains separate stage columns; it does not replace the front-end row.',
        '- Predictive evaluation CI=None: no bootstrap/DeLong-interval performance claim. Ordinary OR/Wald intervals are included only in the separately named clinical case.',
        '- The pathology comparator is a frozen simple Python specification, not an optimized competing WSI engine. Data are synthetic; no clinical validation or WSI/GPU throughput claim.',
        '- Cold means fresh interpreter with common numeric result transport; OS disk cache and background load remain uncontrolled. Source/package/input/hash audits are outside the measured envelope.',
        '- Full-process resident memory includes imports, fixed buffers, warmup and result transport, using worker/footer/coordinator high-water maximum. This is not tracemalloc or private memory.',
        '- On this Windows host the venv executable launches a base-Python child: worker self/footer high-water measurements cover the actual interpreter, while the coordinator polls the launcher PID. Final memory checks remain valid; immediate parent memory/timeout termination is not independently validated for that child. Operational failures, if present, remain in raw records. Future process-tree supervision is separate engineering work; this grid is not replaced.',
        '- Single-thread workload only; no multicore scaling inference. Native computation/CLI deployment, Windows/Linux installation and other modules retain separately scoped acceptance evidence.','']
    (output/'report.md').write_bytes('\n'.join(lines).encode('utf-8'))
    index['artifacts']={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in sorted(output.iterdir()) if path.is_file()}
    (output/'artifact-index.json').write_bytes((json.dumps(index,indent=2)+'\n').encode('utf-8'))
    return index


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--source-binding',required=True);a=p.parse_args()
    print(json.dumps(archive(a.source,a.output,a.plan,a.source_binding),indent=2))
