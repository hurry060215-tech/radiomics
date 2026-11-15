"""Public CT reference consistency/hand invariants; no data download/native call."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'golden/public_ct_v1'
spec=importlib.util.spec_from_file_location('public_ct_reference_tool',ROOT/'tools/validate_public_ct.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def test_frozen_public_ct_protocol_and_source_headers_are_explicit():
    protocol=json.loads((PROFILE/'protocol.json').read_text(encoding='utf-8'))
    reference=json.loads((PROFILE/'reference.json').read_text(encoding='utf-8'))
    assert reference['protocol_sha256']==hashlib.sha256((PROFILE/'protocol.json').read_bytes()).hexdigest()
    assert protocol['license']['identifier']=='CC-BY-NC-3.0'
    assert not protocol['license']['raw_data_in_repository']
    assert not protocol['license']['package_assets_allowlist']
    assert sum(map(len,protocol['features'].values()))==55
    assert len(protocol['output_feature_order'])==66
    assert protocol['output_feature_order']==sorted(set(protocol['output_feature_order']))
    assert protocol['settings']['weightingNorm'] is None
    assert protocol['settings']['resampledPixelSpacing'] is None
    assert protocol['claims']['clinical_preset'] is False
    assert reference['geometry']['shape_xyz']==[204,201,60]
    assert reference['geometry']['affine_ras_mm'][0][0]<0
    for header in reference['source_headers'].values():
        assert header['xyzt_units']&7==2
        assert (header['scl_slope'],header['scl_intercept'])==(1.,0.)
    assert reference['rounding_observation']['rounding_applied'] is False
    assert reference['basic_shape']['original_shape_basic_VoxelCount']==125256
    assert reference['cases']['PAT1_original_3d_bin25']['intensity_roi_voxels']==125256
    assert reference['cases']['PAT1_resegmented_3d_bin25']['intensity_roi_voxels']==114596
    for case in reference['cases'].values():assert len(case['features'])==55
    manifest=json.loads((PROFILE/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/manifest['generator_path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['generator_sha256']
    for name,expected in manifest['files_sha256'].items():
        data=(PROFILE/name).read_bytes()
        assert b'\r\n' not in data
        assert hashlib.sha256(data).hexdigest()==expected


def test_fixed_block_count_matrices_satisfy_exact_hand_invariants():
    reference=json.loads((PROFILE/'reference.json').read_text(encoding='utf-8'))['matrix_case']
    assert reference['geometry']['shape_xyz']==[16,16,8]
    assert reference['roi_voxels']==2048
    for direction in reference['glcm']:
        offset=direction['offset_xyz']
        expected=2
        for size,step in zip((16,16,8),offset):expected*=size-abs(step)
        assert sum(entry[2] for entry in direction['matrix']['entries'])==expected
    for direction in reference['glrlm']:
        assert sum(length*count for _,length,count in direction['matrix']['entries'])==2048
    assert sum(size*count for _,size,count in reference['glszm']['entries'])==2048
    assert sum(count for _,_,count in reference['gldm']['entries'])==2048
    assert sum(row['count'] for row in reference['ngtdm'])==2048


def test_noncommercial_sources_never_download_inside_git_and_cached_hashes_are_checked(tmp_path):
    with pytest.raises(ValueError,match='outside every Git'):
        tool.download(ROOT/'artifacts/public-ct-forbidden',{'source':{'files':{}}})
    (tmp_path/'image.nii.gz').write_bytes(b'not the frozen input')
    protocol={'source':{'files':{'image.nii.gz':dict(bytes=123,sha256='0'*64,url='https://example.invalid/not-called')}}}
    with pytest.raises(ValueError,match='length/SHA256 mismatch'):
        tool.download(tmp_path,protocol)


def test_tight_bbox_supplement_preserves_full_roi_features_and_world_geometry():
    crop=json.loads((PROFILE/'bbox_crop_reference.json').read_text(encoding='utf-8'))
    full=json.loads((PROFILE/'reference.json').read_text(encoding='utf-8'))
    manifest=json.loads((PROFILE/'bbox_crop_manifest.json').read_text(encoding='utf-8'))
    assert manifest['reference_sha256']==hashlib.sha256((PROFILE/'bbox_crop_reference.json').read_bytes()).hexdigest()
    assert crop['full_reference_sha256']==hashlib.sha256((PROFILE/'reference.json').read_bytes()).hexdigest()
    assert crop['protocol_sha256']==hashlib.sha256((PROFILE/'protocol.json').read_bytes()).hexdigest()
    assert crop['roi_voxels']==125256
    assert crop['geometry']['shape_xyz']==[100,99,26]
    assert len(crop['features'])==55
    assert all(row['passed'] and row['absolute_error']==0 for row in crop['full_vs_crop'])
    for axis in 'XYZ':
        assert crop['basic_shape']['original_shape_basic_BoundingBoxMin'+axis+'_index']==0
        assert crop['basic_shape']['original_shape_basic_Centroid'+axis+'_mm']==pytest.approx(full['basic_shape']['original_shape_basic_Centroid'+axis+'_mm'],abs=1e-6,rel=0)
    assert crop['basic_shape']['original_shape_basic_Volume_mm3']==full['basic_shape']['original_shape_basic_Volume_mm3']
