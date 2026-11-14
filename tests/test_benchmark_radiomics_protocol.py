"""Pure corpus/protocol tests. No feature extraction or benchmark timing is run."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('benchmark_radiomics',ROOT/'tools/benchmark_radiomics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_predeclared_case_grid_and_input_generation_are_deterministic(tmp_path):
    plan,digest=module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json')
    first=module.prepare(plan,digest,tmp_path/'first');second=module.prepare(plan,digest,tmp_path/'second')
    assert first==second
    declared=module.cases(plan,first)
    assert len(declared)==58
    assert sum(case['level']=='L1' for case in declared)==35
    assert sum(case['level']=='L2' for case in declared)==5
    assert sum(case['level']=='L3' for case in declared)==6
    assert sum(case['level']=='L4' for case in declared)==12
    assert {case['roi_count'] for case in declared if case['level']=='L4'}=={100,500,1000}
    assert {case['threads'] for case in declared if case['level']=='L4'}=={1,4}
    assert len({case['id'] for case in declared})==58
    assert len(module.expected_columns(plan,2))==66
    assert len(module.expected_columns(plan,3))==66
    assert len(module.expected_columns(plan,3,'glcm'))==10
    assert len(module.expected_columns(plan,3,'shape_basic'))==11
    phantom=next(case for case in first['datasets'] if case['id']=='ibsi_digital_3d')
    assert phantom['roi_voxels']==74
    assert phantom['present_gray_values']==[1.,3.,4.,6.]
    assert phantom['files_sha256']['image.nii.gz']=='83773ac2a288aa93cf819a98eaf301c18d35d5876182ec9c18733184e2b7a83a'
    for record in first['datasets']:
        array,affine=module.read_nifti_fixture(tmp_path/'first'/record['directory']/'image.nii.gz')
        assert list(reversed(array.shape))==record['shape_xyz']
        np.testing.assert_array_equal(affine,record['affine_ras_mm'])
    with pytest.raises(FileExistsError):
        module.prepare(plan,digest,tmp_path/'first')


def test_shape_support_spec_is_separate_and_scalar_comparison_preserves_failures(tmp_path):
    affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[10.,-20.,30.]
    values=module.shape_basic_spec(np.array([[1,0],[0,1]],dtype=np.uint8),affine)
    assert values['original_shape_basic_VoxelCount']==2
    assert values['original_shape_basic_Area_mm2']==12
    assert [values['original_shape_basic_Centroid'+a+'_mm'] for a in 'XYZ']==[11.,-18.5,30.]
    plan,_=module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json')
    a=dict(sample_ids=['P'],feature_names=['original_firstorder_Mean','undefined'],values=[[3.,'NaN']])
    b=json.loads(json.dumps(a));module.write_json(tmp_path/'a.json',a);module.write_json(tmp_path/'b.json',b)
    assert module.compare_artifacts(tmp_path/'a.json',tmp_path/'b.json',plan)['passed']
    b['values'][0][1]=0.;module.write_json(tmp_path/'wrong.json',b)
    assert not module.compare_artifacts(tmp_path/'a.json',tmp_path/'wrong.json',plan)['passed']
    with pytest.raises(ValueError,match='approved SHA256'):
        module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json','0'*64)


def test_report_writer_works_on_supported_python_without_claiming_missing_evidence(tmp_path):
    report=dict(plan_sha256='a'*64,results=[],missing_evidence=['allocator counts unmeasured'])
    module.write_report_markdown(tmp_path/'report.md',report)
    assert 'allocator counts unmeasured' in (tmp_path/'report.md').read_text()
