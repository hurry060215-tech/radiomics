"""Real-file tutorial contracts; public licensed voxel data stay outside test assets."""
import importlib.util
from pathlib import Path
import sys
import pytest

DIRECTORY=Path(__file__).resolve().parents[1]/'examples/real'
sys.path.insert(0,str(DIRECTORY))


def load(name):
    spec=importlib.util.spec_from_file_location('real_example_'+name,DIRECTORY/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_patch_entry_never_guesses_ids_layout_or_physical_scale(tmp_path):
    example=load('patch_features')
    manifest=tmp_path/'patches.csv';manifest.write_text('path,patient_id,slide_id,tile_id\nimage.png,001,slide,0001\n',encoding='utf-8')
    with pytest.raises(ValueError,match='supplied x0'):
        example.load_manifest(manifest,'level0')
    rows,parents=example.load_manifest(manifest,'image-local')
    assert rows[0]['patient_id']=='001' and rows[0]['tile_id']=='0001'
    assert parents=={'slide':'001'}
    assert example.optional_number(rows[0],'mpp_x_um') is None
    manifest.write_text('path,patient_id,slide_id,tile_id\na.png,P,S,T1\nb.png,P,S,T2\n',encoding='utf-8')
    with pytest.raises(ValueError,match='one complete source-image'):
        example.load_manifest(manifest,'image-local')


def test_download_cache_refuses_git_worktree_and_unapproved_overwrite(tmp_path):
    example=load('download_public_ct')
    repository=tmp_path/'repo';repository.mkdir();(repository/'.git').write_text('worktree marker',encoding='utf-8')
    with pytest.raises(ValueError,match='outside every Git'):
        example.reject_git_output(repository/'cache')
    cache=tmp_path/'external';cache.mkdir();target=cache/'NOTICE.md'
    example.exclusive_text(target,'fixed notice\n');example.exclusive_text(target,'fixed notice\n')
    with pytest.raises(FileExistsError):example.exclusive_text(target,'different\n')
