"""Execute the real public CT notebook with an external cache and explicit licence consent.

This is a separate real-data track. It never adds the CT lesson to the six
synthetic EXPECTED cases. Use the Python interpreter containing the native
package, nbclient and ipykernel. Source notebook cells are never rewritten.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nbformat

from run_tutorials import execute_one, git_identity, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_KIND = 'real-public-ct-no-clinical-outcome'
SOURCE_REVISION = '6da96021bc91faf4c0cb7fd7fa56a4225d2064a8'
REQUIRED = ['tutorial-runtime.json', 'ct-roi-and-intensity.png',
    'extraction/features.csv', 'extraction/diagnostics.csv', 'features.csv',
    'features.csv.manifest.json', 'metadata.csv', 'provenance.json']
SOURCES = ['tools/build_public_ct_notebook.py', 'examples/real/download_public_ct.py']


def validate_locations(cache, output, repo_root, acknowledge_license):
    if not acknowledge_license:
        raise PermissionError('Explicit --acknowledge-license is required for CC-BY-NC-3.0 input.')
    cache, output, repo_root = (Path(p).resolve() for p in (cache, output, repo_root))
    # Refuse both the supplied repository and any other Git checkout, including
    # .git worktree pointer files. The licensed inputs must remain external.
    if cache == repo_root or repo_root in cache.parents or any((p/'.git').exists() for p in [cache, *cache.parents]):
        raise ValueError('The public CT cache must be outside all Git worktrees.')
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError('The result directory and licensed input cache must be separate trees.')
    if output.exists():
        raise FileExistsError('Result directory already exists; provide a new --output.')
    return cache, output, repo_root


def cached_input_evidence(cache):
    return {name: {'bytes': (cache/name).stat().st_size, 'sha256': sha256(cache/name)}
        for name in ('image.nii.gz', 'mask.nii.gz', 'cohort.csv') if (cache/name).is_file()}


def validate_real_outputs(artifacts, cache, before, runtime):
    failures = []
    after = cached_input_evidence(cache)
    for name in ('image.nii.gz', 'mask.nii.gz'):
        if name not in after:
            failures.append(f'Missing actual CT input: {name}')
        elif name in before and before[name] != after[name]:
            failures.append(f'Existing cached CT bytes changed: {name}')
    try:
        provenance = json.loads((artifacts/'provenance.json').read_text(encoding='utf-8'))
        if provenance['clinical_outcome_available'] is not False:
            failures.append('The single real CT must not claim clinical outcomes.')
        if provenance['license'] != 'CC-BY-NC-3.0' or provenance['source_revision'] != SOURCE_REVISION:
            failures.append('Public data licence/source revision differs from the declared PAT1 input.')
        for name in ('image.nii.gz', 'mask.nii.gz'):
            if name not in after or provenance['input_sha256'].get(name) != after[name]['sha256']:
                failures.append(f'Input provenance SHA256 mismatch: {name}')
        with (artifacts/'features.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.reader(handle))
        if len(rows) != 2 or len(rows[0]) != 28 or len(rows[1]) != 28:
            failures.append('Expected one actual CT sample and 27 extracted features plus identity.')
    except Exception as error:
        failures.append(f'Real-data output validation failed: {type(error).__name__}: {error}')
    if not runtime or runtime.get('data_kind') != DATA_KIND:
        failures.append('Runtime did not identify the real CT/no-outcome tutorial track.')
    return after, failures


def run(notebook, cache, output, *, repo_root=ROOT, acknowledge_license=False, timeout=600):
    cache, output, repo_root = validate_locations(cache, output, repo_root, acknowledge_license)
    notebook = Path(notebook).resolve()
    original = nbformat.read(notebook, as_version=4)
    nbformat.validate(original)
    if original.metadata.get('radiomics_data_kind') != DATA_KIND:
        raise ValueError('Expected the authored real public CT/no-outcome notebook, not a synthetic case.')
    source_hashes = {relative: sha256(repo_root/relative) for relative in SOURCES}
    before = cached_input_evidence(cache)
    output.mkdir(parents=True, exist_ok=False)
    kernels = output/'kernels'
    kernels.mkdir()
    try:
        result = execute_one(notebook, output, repo_root, kernels, timeout,
            expected_outputs=REQUIRED, kernel_parameters={'TUTORIAL_DATA_ROOT': str(cache)},
            source_bindings=source_hashes)
        artifacts = Path(result['executed_path']).parent/'artifacts'
        after, failures = validate_real_outputs(artifacts, cache, before, result.get('runtime'))
        for name, expected_hash in source_hashes.items():
            if sha256(repo_root/name) != expected_hash:
                failures.append(f'Tutorial support source changed during execution: {name}')
        result['failures'].extend(failures)
        if failures:
            result['status'] = 'failed'
    except Exception as error:
        result = {'notebook': notebook.name, 'input_sha256': sha256(notebook), 'status': 'failed',
            'failures': [f'{type(error).__name__}: {error}']}
        after = cached_input_evidence(cache)
    receipt = {'schema': 'radiomics.public-ct-tutorial-execution.v1',
        'status': result['status'], 'data_kind': DATA_KIND, 'clinical_outcome_available': False,
        'license': 'CC-BY-NC-3.0', 'license_acknowledged': True, 'cache': str(cache),
        'repository': git_identity(repo_root), 'python': sys.executable,
        'runner_sha256': sha256(__file__), 'shared_runner_sha256': sha256(ROOT/'tools/run_tutorials.py'),
        'support_sources_sha256': source_hashes, 'inputs_before': before, 'inputs_after': after,
        'notebooks': [result]}
    # The shared per-notebook receipt is kernel/artifact evidence. This outer
    # receipt additionally applies the licence, real-input and 27-feature gates.
    write_json(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT/'website/tutorials/07_public_ct.ipynb')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--acknowledge-license', action='store_true', required=True)
    parser.add_argument('--timeout', type=int, default=600, help='Per-code-cell timeout in seconds')
    args = parser.parse_args()
    receipt = run(args.notebook, args.cache, args.output, repo_root=args.repo_root,
        acknowledge_license=args.acknowledge_license, timeout=args.timeout)
    print(receipt['status'])
    return 0 if receipt['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
