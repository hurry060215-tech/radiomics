"""Published PR documentation must link to files in its own source revision."""
import importlib.util
import json
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'tools/build_website_sources.py'
spec=importlib.util.spec_from_file_location('website_source_builder',PATH)
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_unmerged_documentation_source_links_are_pinned(tmp_path,monkeypatch):
    (tmp_path/'docs').mkdir()
    (tmp_path/'website/tutorials').mkdir(parents=True)
    (tmp_path/'website/tutorials/07_public_ct.ipynb').write_text('{}')
    (tmp_path/'docs/guide.md').write_text('# Guide\n\n[Evidence](new-evidence.json)\n')
    (tmp_path/'docs/new-evidence.json').write_text('{}')
    monkeypatch.setattr(builder,'PAGES',('guide.md',))
    revision='a'*40
    builder.build(tmp_path,source_ref=revision)
    rendered=(tmp_path/'website/reference/guide.md').read_text()
    assert '/blob/'+revision+'/docs/new-evidence.json' in rendered
    assert '/blob/main/' not in rendered
    manifest=json.loads((tmp_path/'website/reference/_sources.json').read_text())
    assert manifest['source_ref']==revision


def test_unpacked_source_does_not_borrow_parent_git_identity(tmp_path,monkeypatch):
    (tmp_path/'.git').mkdir()
    archive=tmp_path/'unpacked'
    archive.mkdir()
    def unexpected(*args,**kwargs):
        raise AssertionError('Do not consult a parent repository for archive identity.')
    monkeypatch.setattr(builder.subprocess,'run',unexpected)
    assert builder.source_revision(archive)=='main'
