"""Focused generation/receipt checks; actual six-kernel evidence is a separate run."""
import ast
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import build_tutorial_notebooks as build


def runner():
    pytest.importorskip('nbclient')
    pytest.importorskip('nbformat')
    import run_tutorials
    return run_tutorials


def test_notebooks_have_same_source_bindings_with_windows_or_linux_checkout(tmp_path):
    roots = [tmp_path/'lf', tmp_path/'crlf']
    for root, ending in zip(roots, (b'\n', b'\r\n')):
        folder = root/'examples/native'
        folder.mkdir(parents=True)
        for source in [*build.example_paths(ROOT), ROOT/'examples/native/_shared.py']:
            data = source.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', ending)
            (folder/source.name).write_bytes(data)
    first = build.generate_notebooks(roots[0], tmp_path/'left')
    second = build.generate_notebooks(roots[1], tmp_path/'right')
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]


def test_repeat_generation_has_identical_bytes_and_portable_inline_code(tmp_path):
    first = build.generate_notebooks(ROOT, tmp_path/'first')
    second = build.generate_notebooks(ROOT, tmp_path/'second')
    assert len(first) == len(second) == 6
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes()
        nb = json.loads(a.read_text(encoding='utf-8'))
        ids = [cell['id'] for cell in nb['cells']]
        assert len(ids) == len(set(ids))
        code = [c['source'] for c in nb['cells'] if c['cell_type']=='code']
        assert len(code)>4
        for source in code:
            ast.parse(source)
            assert 'runpy' not in source
            assert 'D:\\' not in source and 'D:/' not in source
        assert '45140bf' not in a.read_text(encoding='utf-8')
        assert not any(c.get('outputs') for c in nb['cells'])
        assert any('display(Image' in source for source in code)


def test_generation_never_silently_overwrites(tmp_path):
    paths = build.generate_notebooks(ROOT, tmp_path)
    paths[-1].write_text('sentinel', encoding='utf-8')
    with pytest.raises(FileExistsError):
        build.generate_notebooks(ROOT, tmp_path)
    assert paths[-1].read_text() == 'sentinel'


def test_future_seventh_example_is_not_implicitly_included(tmp_path):
    folder = tmp_path/'examples/native'
    folder.mkdir(parents=True)
    for path in build.example_paths(ROOT):
        (folder/path.name).write_text('pass')
    (folder/'07_real_ct.py').write_text('pass')
    assert len(build.example_paths(tmp_path)) == 6


def test_stale_or_reordered_execution_is_rejected():
    tool = runner()
    import nbformat
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('x=1'), nbformat.v4.new_code_cell('x+1')])
    executed = nbformat.from_dict(json.loads(json.dumps(original)))
    executed.cells[0].execution_count = 2
    executed.cells[1].execution_count = 1
    _, failures = tool.cell_evidence(original, executed)
    assert any('original order' in failure for failure in failures)


def test_errors_changed_source_and_invalid_png_cannot_pass():
    tool = runner()
    import nbformat
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('x=1')])
    executed = nbformat.from_dict(json.loads(json.dumps(original)))
    executed.cells[0].source = 'x=2'
    executed.cells[0].execution_count = 1
    executed.cells[0].outputs = [nbformat.v4.new_output('error', ename='ValueError', evalue='failure', traceback=[]),
        nbformat.v4.new_output('display_data', data={'image/png':'not base64'})]
    cells, failures = tool.cell_evidence(original, executed)
    assert cells[0]['errors'][0]['ename']=='ValueError'
    assert any('source changed' in f for f in failures)
    assert any('invalid image' in f for f in failures)
    assert any('error output' in f for f in failures)


def test_expected_outputs_need_nonempty_parseable_results(tmp_path):
    tool = runner()
    (tmp_path/'result.csv').write_text('header\n', encoding='utf-8')
    (tmp_path/'tutorial-runtime.json').write_text('{}', encoding='utf-8')
    files, failures = tool.artifact_evidence(tmp_path, ['result.csv', 'missing.rad'])
    assert files['result.csv']['sha256']==tool.sha256(tmp_path/'result.csv')
    assert any('Invalid required output result.csv' in f for f in failures)
    assert any('missing.rad' in f for f in failures)


def test_run_refuses_existing_results_and_missing_inputs(tmp_path):
    tool = runner()
    with pytest.raises(ValueError, match='Missing tutorial inputs'):
        tool.run(tmp_path, tmp_path/'new')
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    for name in tool.EXPECTED:
        (inputs/(name+'.ipynb')).write_text('{}')
    with pytest.raises(FileExistsError):
        tool.run(inputs, tmp_path)


def test_kernel_failure_is_preserved_with_unexecuted_cells_and_original_input(tmp_path, monkeypatch):
    tool = runner()
    import nbformat
    inputs = build.generate_notebooks(ROOT, tmp_path/'inputs')
    original_hash = tool.sha256(inputs[0])
    class Manager:
        has_kernel = False
        def __init__(self, **kwargs):
            pass
    class FailingClient:
        def __init__(self, notebook, **kwargs):
            assert kwargs['allow_errors'] is False
            self.nb = notebook
        def execute(self, **kwargs):
            assert kwargs['cleanup_kc'] is True
            self.nb.cells[2].execution_count = 1
            self.nb.cells[2].outputs = [nbformat.v4.new_output('error',
                ename='RuntimeError', evalue='deliberate failure', traceback=[])]
            raise RuntimeError('deliberate failure')
    monkeypatch.setattr(tool, 'AsyncKernelManager', Manager)
    monkeypatch.setattr(tool, 'NotebookClient', FailingClient)
    output = tmp_path/'execution'
    kernels = output/'kernels'
    kernels.mkdir(parents=True)
    result = tool.execute_one(inputs[0], output, ROOT, kernels)
    assert result['status']=='failed'
    assert result['execution_error']['message']=='deliberate failure'
    assert result['cells'][0]['errors'][0]['evalue']=='deliberate failure'
    assert result['cells'][1]['execution_count'] is None
    assert tool.sha256(inputs[0])==original_hash
    stored = json.loads((output/inputs[0].stem/'receipt.json').read_text(encoding='utf-8'))
    assert stored==result
