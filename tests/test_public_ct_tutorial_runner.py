"""Real CT runner protocol checks; these tests do not download or extract CT."""
import csv
import json
from pathlib import Path
import sys
import tempfile

import pytest

pytest.importorskip('nbclient')
nbformat = pytest.importorskip('nbformat')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_public_ct_tutorial as public
import run_tutorials as shared


@pytest.fixture
def external_tmp():
    # The production gate correctly rejects --basetemp inside a Git checkout.
    with tempfile.TemporaryDirectory(prefix='radiomics-public-ct-protocol-') as directory:
        yield Path(directory).resolve()


def fixture_inputs(tmp_path):
    cache = tmp_path/'cache'
    cache.mkdir()
    for name in ('image.nii.gz', 'mask.nii.gz'):
        (cache/name).write_bytes(('fixture protocol bytes '+name).encode())
    return cache


def fixture_outputs(artifacts, cache, **overrides):
    artifacts.mkdir(parents=True)
    provenance = {'clinical_outcome_available': False, 'license': 'CC-BY-NC-3.0',
        'source_revision': public.SOURCE_REVISION,
        'input_sha256': {name: public.sha256(cache/name) for name in ('image.nii.gz', 'mask.nii.gz')}}
    provenance.update(overrides)
    (artifacts/'provenance.json').write_text(json.dumps(provenance), encoding='utf-8')
    with (artifacts/'features.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id']+['f'+str(i) for i in range(27)])
        writer.writerow(['PAT1']+[1.]*27)


def test_licence_acknowledgement_and_external_separate_cache_are_mandatory(external_tmp):
    tmp_path = external_tmp
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'.git').write_text('gitdir: external')
    with pytest.raises(PermissionError):
        public.validate_locations(tmp_path/'cache', tmp_path/'output', repo, False)
    with pytest.raises(ValueError, match='outside all Git'):
        public.validate_locations(repo/'data', tmp_path/'output', repo, True)
    with pytest.raises(ValueError, match='separate trees'):
        public.validate_locations(tmp_path/'cache', tmp_path/'cache/results', repo, True)
    output = tmp_path/'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        public.validate_locations(tmp_path/'cache', output, repo, True)


def test_real_source_bytes_and_no_outcome_are_independent_gates(tmp_path):
    cache = fixture_inputs(tmp_path)
    before = public.cached_input_evidence(cache)
    artifacts = tmp_path/'artifacts'
    fixture_outputs(artifacts, cache)
    runtime = {'data_kind': public.DATA_KIND}
    _, failures = public.validate_real_outputs(artifacts, cache, before, runtime)
    assert failures == []
    (cache/'image.nii.gz').write_bytes(b'changed input')
    fixture = json.loads((artifacts/'provenance.json').read_text())
    fixture['clinical_outcome_available'] = True
    fixture['source_revision'] = 'unfrozen'
    (artifacts/'provenance.json').write_text(json.dumps(fixture))
    _, failures = public.validate_real_outputs(artifacts, cache, before, runtime)
    assert any('bytes changed' in f for f in failures)
    assert any('SHA256 mismatch' in f for f in failures)
    assert any('clinical outcomes' in f for f in failures)
    assert any('source revision' in f for f in failures)


def test_real_track_uses_shared_executor_without_changing_synthetic_cases(external_tmp, monkeypatch):
    tmp_path = external_tmp
    synthetic_before = dict(shared.EXPECTED)
    repo = tmp_path/'repo'
    for relative in public.SOURCES:
        path = repo/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# protocol fixture only\n')
    notebook = tmp_path/'07_public_ct.ipynb'
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('x=1')],
        metadata={'radiomics_data_kind': public.DATA_KIND})
    nbformat.write(nb, notebook)
    original_sha = public.sha256(notebook)
    cache = fixture_inputs(tmp_path)
    def fake_execute(path, output, repo_root, kernels, timeout, **kwargs):
        assert path == notebook
        assert kwargs['expected_outputs'] == public.REQUIRED
        assert kwargs['kernel_parameters'] == {'TUTORIAL_DATA_ROOT': str(cache)}
        assert set(kwargs['source_bindings']) == set(public.SOURCES)
        case = output/path.stem
        fixture_outputs(case/'artifacts', cache)
        return {'status': 'passed', 'failures': [], 'executed_path': str(case/path.name),
            'runtime': {'data_kind': public.DATA_KIND}}
    monkeypatch.setattr(public, 'execute_one', fake_execute)
    result = public.run(notebook, cache, tmp_path/'run', repo_root=repo, acknowledge_license=True)
    assert result['status']=='passed', result
    assert result['data_kind']==public.DATA_KIND
    assert result['license_acknowledged'] is True
    assert len(result['notebooks'])==1
    assert public.sha256(notebook)==original_sha
    assert shared.EXPECTED==synthetic_before


def test_synthetic_page_cannot_enter_public_ct_track(external_tmp):
    tmp_path = external_tmp
    notebook = tmp_path/'synthetic.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('x=1')]), notebook)
    with pytest.raises(ValueError, match='real public CT'):
        public.run(notebook, tmp_path/'cache', tmp_path/'run', acknowledge_license=True)
