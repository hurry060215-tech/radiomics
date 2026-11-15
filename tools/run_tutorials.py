"""Execute the six stepwise tutorials with fresh kernels and auditable evidence.

Run using the Python interpreter that has the native package and notebook extras:
python tools/run_tutorials.py --notebooks /fresh/generated --output /fresh/run
Neither source notebooks nor an existing result directory are overwritten.
"""
import argparse
import base64
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    '01_extract': ['python-api/features.csv', 'extraction-results.png', 'cohort.csv'],
    '02_image_clinical_binary': ['image-model/model-000.rad', 'clinical-model/model-000.rad',
        'fusion.rad', 'image-metrics.csv', 'clinical-metrics.csv', 'fusion-metrics.csv',
        'unlabeled-predictions.csv', 'heldout-patient-roc.png'],
    '03_multiclass_regression': ['multiclass/model-000.rad', 'regression/model-000.rad',
        'multiclass-metrics.csv', 'regression-metrics.csv',
        'multiclass-unlabeled-predictions.csv', 'regression-unlabeled-predictions.csv', 'task-results.png'],
    '04_regions_delta_habitat': ['delta.csv', 'delta-pairing.json', 'multimodal-multiregion.parquet',
        'provided-habitat-mask.npy', 'local-habitat.json', 'local-habitat-labels.csv', 'local-habitat.png'],
    '05_patch_patient': ['tiles.json', 'qc-colors.jsonl', 'aggregation.json',
        'patient-features.csv', 'patient-model/model-000.rad', 'patient-metrics.csv',
        'patch-objects.json', 'tile-feature-overlay/manifest.json', 'tile-feature-overlay.png'],
    '06_embedding_overlay': ['embeddings.npy', 'embedding-tile-ids.json', 'tiles.json',
        'adapter-provenance.json', 'patient-model/model-000.rad', 'tile-model/model-000.rad',
        'actual-tile-predictions.json', 'patient-broadcast-rejected.json',
        'actual-tile-score-overlay/manifest.json', 'actual-tile-score-overlay.png'],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def git_identity(root):
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'revision': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain')}


def cell_evidence(original, executed):
    """Require the original code, ordered execution, and actual PNG payloads."""
    evidence, failures, execution = [], [], 0
    if len(original.cells) != len(executed.cells):
        failures.append('Cell population changed during execution.')
    for index, before in enumerate(original.cells):
        after = executed.cells[index] if index < len(executed.cells) else {}
        if before.get('id') != after.get('id') or before.source != after.get('source'):
            failures.append(f'Cell {index} identity/source changed.')
        if before.cell_type != 'code' or not before.source.strip():
            continue
        execution += 1
        errors, images, output_types = [], [], []
        for output in after.get('outputs', []):
            output_types.append(output.get('output_type'))
            if output.get('output_type') == 'error':
                errors.append(dict(output))
            if 'image/png' in output.get('data', {}):
                try:
                    binary = base64.b64decode(output['data']['image/png'], validate=True)
                    if not binary.startswith(b'\x89PNG\r\n\x1a\n'):
                        raise ValueError('Invalid PNG signature')
                    images.append({'bytes': len(binary), 'sha256': hashlib.sha256(binary).hexdigest()})
                except Exception as error:
                    failures.append(f'Cell {index} invalid image: {error}')
        if after.get('execution_count') != execution:
            failures.append(f'Cell {index} was not executed in original order (expected {execution}).')
        if errors:
            failures.append(f'Cell {index} has error output.')
        evidence.append({'index': index, 'id': before.get('id'),
            'source_sha256': hashlib.sha256(before.source.encode('utf-8')).hexdigest(),
            'execution_count': after.get('execution_count'), 'errors': errors,
            'output_types': output_types, 'images': images})
    if not evidence:
        failures.append('No executable code cells.')
    if not any(cell['images'] for cell in evidence):
        failures.append('No actual inline PNG result.')
    return evidence, failures


def artifact_evidence(root, expected):
    files, failures = {}, []
    if not root.is_dir():
        return files, ['No artifact directory was produced.']
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            failures.append(f'Symlink output refused: {path}')
        elif path.is_file():
            files[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    for name in [*expected, 'tutorial-runtime.json']:
        if name not in files or files[name]['bytes'] == 0:
            failures.append(f'Missing/empty required output: {name}')
            continue
        path = root / name
        try:
            if path.suffix == '.csv':
                with path.open(encoding='utf-8', newline='') as handle:
                    rows = csv.reader(handle)
                    header, first = next(rows), next(rows)
                    if not header or len(header) != len(first):
                        raise ValueError('Empty or malformed CSV')
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
            elif path.suffix == '.png' and not path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Invalid PNG')
        except Exception as error:
            failures.append(f'Invalid required output {name}: {error}')
    return files, failures


def execute_one(path, run_root, repo_root, kernel_root, timeout=300, *,
                expected_outputs=None, kernel_parameters=None, source_bindings=None):
    before_hash = sha256(path)
    original = nbformat.read(path, as_version=4)
    nbformat.validate(original)
    binding = original.metadata.get('radiomics_tutorial', {})
    bindings = source_bindings if source_bindings is not None else {
        binding.get('example'): binding.get('example_sha256'),
        'examples/native/_shared.py': binding.get('shared_sha256')}
    for relative, expected_hash in bindings.items():
        if not relative:
            raise ValueError(f'Tutorial helper/source binding mismatch: {relative}')
        actual_hash = sha256(repo_root / relative)
        if source_bindings is None and binding.get('source_hash_mode') == 'utf8-lf-v1':
            actual_hash = hashlib.sha256((repo_root / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        elif source_bindings is None and binding.get('source_hash_mode') is not None:
            raise ValueError('Unknown tutorial source hash mode.')
        if actual_hash != expected_hash:
            raise ValueError(f'Tutorial helper/source binding mismatch: {relative}')
    notebook = nbformat.from_dict(json.loads(json.dumps(original)))
    # Clear stale outputs; a pre-executed notebook can never provide pass evidence.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
    case = run_root / path.stem
    case.mkdir(exist_ok=False)
    work = case / 'work'
    work.mkdir()
    artifacts = case / 'artifacts'
    kernel_dir = kernel_root / path.stem
    kernel_dir.mkdir()
    # Per-run kernelspec; no global kernel installation or changed source cells.
    prelude = [f'TUTORIAL_REPO_ROOT = {str(repo_root)!r}', f'TUTORIAL_RUN_ROOT = {str(artifacts)!r}']
    for name, value in (kernel_parameters or {}).items():
        if not name.isidentifier() or name in ('TUTORIAL_REPO_ROOT', 'TUTORIAL_RUN_ROOT'):
            raise ValueError(f'Invalid/reserved kernel parameter: {name}')
        prelude.append(f'{name} = {str(value)!r}')
    write_json(kernel_dir / 'kernel.json', {'argv': [sys.executable, '-m', 'ipykernel_launcher',
        '-f', '{connection_file}', '--IPKernelApp.exec_lines=' + json.dumps(prelude)],
        'display_name': 'Tutorial acceptance', 'language': 'python'})
    manager = AsyncKernelManager(kernel_name=path.stem,
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client = NotebookClient(notebook, km=manager, timeout=timeout, allow_errors=False,
        resources={'metadata': {'path': str(work)}})
    error = None
    started = time.perf_counter()
    try:
        client.execute(cleanup_kc=True)
    except Exception as failure:
        error = {'type': type(failure).__name__, 'message': str(failure)}
    elapsed = time.perf_counter() - started
    executed = case / path.name
    with executed.open('x', encoding='utf-8', newline='\n') as handle:
        nbformat.write(notebook, handle)
    cells, failures = cell_evidence(original, notebook)
    required = EXPECTED[path.stem] if expected_outputs is None else expected_outputs
    outputs, artifact_failures = artifact_evidence(artifacts, required)
    failures.extend(artifact_failures)
    if error:
        failures.append('Kernel execution raised an exception; see execution_error.')
    if sha256(path) != before_hash:
        failures.append('Input notebook changed during execution.')
    runtime_file = artifacts / 'tutorial-runtime.json'
    try:
        runtime = json.loads(runtime_file.read_text(encoding='utf-8'))
        if runtime.get('backend', {}).get('metrics_backend') != 'rust' or not runtime.get('native_sha256'):
            failures.append('Runtime did not prove native Rust extension identity.')
        if Path(runtime['interpreter']).resolve() != Path(sys.executable).resolve():
            failures.append('Notebook kernel used an unexpected interpreter.')
    except Exception as failure:
        runtime = None
        failures.append(f'Runtime provenance unavailable: {failure}')
    result = {'notebook': path.name, 'status': 'failed' if failures else 'passed',
        'input_path': str(path), 'input_sha256': before_hash,
        'executed_path': str(executed), 'executed_sha256': sha256(executed),
        'seconds': elapsed, 'execution_error': error, 'failures': failures,
        'kernel_prelude': prelude, 'kernel_terminated': not manager.has_kernel,
        'cells': cells, 'output_files': outputs, 'runtime': runtime}
    if manager.has_kernel:
        result['status'] = 'failed'
        result['failures'].append('Kernel was not terminated after execution.')
    write_json(case / 'receipt.json', result)
    return result


def run(notebooks, output, repo_root=ROOT, timeout=300):
    notebooks, output, repo_root = map(lambda value: Path(value).resolve(), (notebooks, output, repo_root))
    paths = [notebooks / (name + '.ipynb') for name in EXPECTED]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f'Missing tutorial inputs: {missing}')
    output.mkdir(parents=True, exist_ok=False)
    kernel_root = output / 'kernels'
    kernel_root.mkdir()
    records = []
    for path in paths:
        try:
            result = execute_one(path, output, repo_root, kernel_root, timeout)
        except Exception as error:
            result = {'notebook': path.name, 'status': 'failed', 'input_sha256': sha256(path),
                'failures': [f'{type(error).__name__}: {error}']}
        records.append(result)
        print(path.name, result['status'], flush=True)
    receipt = {'schema': 'radiomics.tutorial-execution.v1',
        'status': 'passed' if all(row['status'] == 'passed' for row in records) else 'failed',
        'dataset': 'fixed synthetic software examples, not clinical validation',
        'repository': git_identity(repo_root), 'runner_repository': git_identity(ROOT),
        'runner_sha256': sha256(__file__), 'python': sys.executable, 'platform': platform.platform(),
        'generation_tools': {name: sha256(ROOT/'tools'/name) for name in
            ('build_tutorial_notebooks.py', 'tutorial_figures.py')},
        'execution': 'fresh kernel and cwd per notebook; original ordered code, allow_errors=False; required artifacts and inline PNG checked',
        'input_dependencies': {name: sha256(repo_root/name) for name in
            ['examples/native/_shared.py', 'golden/python_golden_v1/model_inputs.json',
             'golden/python_golden_v1/pathology_inputs.json']},
        'notebooks': records}
    write_json(output / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebooks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--timeout', type=int, default=300, help='Per-code-cell timeout in seconds')
    args = parser.parse_args()
    result = run(args.notebooks, args.output, args.repo_root, args.timeout)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
