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


def test_public_version_string_cannot_replace_frozen_source_and_compiled_hashes():
    expected={'glcm.py':'a'*64,'_cmatrices.cp39-win_amd64.pyd':'b'*64}
    module.validate_reference_files(dict(expected),expected,'c'*40,'c'*40)
    changed=dict(expected);changed['glcm.py']='d'*64
    with pytest.raises(RuntimeError,match='differs from the frozen'):
        module.validate_reference_files(changed,expected,'c'*40,'c'*40)
    with pytest.raises(RuntimeError,match='missing'):
        module.validate_reference_files(changed,None,'c'*40,'c'*40)
    with pytest.raises(RuntimeError,match='compiled-extension'):
        module.validate_reference_files({'glcm.py':'a'*64},{'glcm.py':'a'*64},'c'*40,'c'*40)


def test_actual_thread_and_lifetime_memory_probes_cannot_be_replaced_by_labels():
    for pools in (None,[],[{'user_api':'blas','num_threads':24}],[{'user_api':'unknown','num_threads':1}]):
        with pytest.raises(RuntimeError,match='matched-resource'):
            module.validate_thread_resources(pools,1,'python')
    module.validate_thread_resources([{'user_api':'blas','num_threads':1}],1,'python')
    with pytest.raises(RuntimeError,match='SimpleITK'):
        module.validate_thread_resources([{'user_api':'blas','num_threads':1}],4,'python')
    assert module.merge_peak_memory(100,200)==200
    assert module.merge_peak_memory(300,200)==300
    assert module.merge_peak_memory(None,200)==200


def test_comparison_file_failure_is_retained_and_cache_failure_blocks_overall_pass(tmp_path):
    plan,_=module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json')
    (tmp_path/'broken.json').write_text('{broken',encoding='utf-8')
    comparison=module.safe_compare_artifacts(tmp_path/'broken.json',tmp_path/'missing.json',plan)
    assert comparison['passed'] is False and comparison['error']
    primary=[{'valid_speed_comparison':True} for _ in range(58)]
    cache=dict(population_and_hit_observations=[dict(status='ok',correctness_checked=True) for _ in range(14)])
    records=[dict(status='ok') for _ in range(812)]
    assert module.overall_status(primary,cache,records)['correctness_status']=='passed_measured_applicable_cases'
    cache['population_and_hit_observations'][0]['status']='failed'
    states=module.overall_status(primary,cache,records)
    assert states['correctness_status'].startswith('failed_')
    assert states['operational_status'].startswith('failed_')


def test_native_receipt_requires_real_fields_and_both_artifact_hashes(tmp_path):
    with pytest.raises(ValueError,match='mandatory'):
        module.validate_build_manifest({},tmp_path/'none.whl',tmp_path)
    wheel=tmp_path/'wheel.whl';wheel.write_bytes(b'synthetic protocol fixture, not executable')
    cli=tmp_path/'cli';cli.write_bytes(b'synthetic cli fixture, not executable')
    build=dict(schema='radiomics.native-build.v1',source_tree_clean=True,source_revision='a'*40,source_tree='b'*40,
        cargo_lock_sha256='c'*64,core_implementation_sha256='d'*64,rustc_verbose='fixture rustc',rust_target='fixture target',
        profile=dict(name='release',opt_level=3,lto='thin',codegen_units=1),rust_flags=dict(RUSTFLAGS=None,CARGO_ENCODED_RUSTFLAGS=None),
        build_commands=[['cargo','build','--release']],python_version='fixture python',maturin_version='fixture maturin',
        wheel=dict(path=str(wheel),sha256=module.sha256_file(wheel)),cli=dict(path=str(cli),sha256=module.sha256_file(cli)))
    module.validate_build_manifest(build,wheel,tmp_path)
    cli.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cli artifact'):
        module.validate_build_manifest(build,wheel,tmp_path)


def test_cold_wall_excludes_before_after_file_audits_and_uses_transport_peak(tmp_path,monkeypatch):
    plan,digest=module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json')
    events=[];clock=iter([10.,12.])
    def audit(request):
        events.append('audit');return {'verified':True}
    def tick():
        value=next(clock);events.append('start' if value==10. else 'stop');return value
    class FakeProcess:
        pid=123;returncode=0
        def __init__(self,*args,stdout,**kwargs):
            events.append('spawn');stdout.write((json.dumps(dict(status='ok',elapsed_seconds=.5,peak_memory_bytes=100))+'\n').encode())
            stdout.write((json.dumps(dict(worker_peak_after_transport_bytes=200))+'\n').encode())
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(module,'audit_observation_inputs',audit)
    monkeypatch.setattr(module.time,'perf_counter',tick)
    monkeypatch.setattr(module.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(module,'finish_worker_transport',lambda result,*args:events.append('postflight_transport') or result)
    request=dict(plan=str(ROOT/'benchmarks/v2/radiomics-plan.json'),approved_plan_sha256=digest,
        case=dict(state='cold_process_total'),observation_directory=str(tmp_path/'observation'))
    result=module.launch_worker('not executed',request,tmp_path/'request.json',tmp_path/'stdout.json',tmp_path/'stderr.txt',2**31,1800)
    assert events==['audit','start','spawn','stop','audit','postflight_transport']
    assert result['elapsed_seconds']==2. and result['api_elapsed_seconds']==.5
    assert result['worker_peak_memory_bytes']==200 and result['peak_memory_bytes']==200


def test_os_worker_launch_failure_becomes_a_retained_observation(tmp_path,monkeypatch):
    _,digest=module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json')
    monkeypatch.setattr(module,'audit_observation_inputs',lambda request:{'verified':True})
    def fail(*args,**kwargs):
        raise OSError('Synthetic launch failure; no program was executed.')
    monkeypatch.setattr(module.subprocess,'Popen',fail)
    request=dict(plan=str(ROOT/'benchmarks/v2/radiomics-plan.json'),approved_plan_sha256=digest,
        case=dict(state='cold_process_total'),observation_directory=str(tmp_path/'observation'))
    result=module.launch_worker('not executed',request,tmp_path/'request.json',tmp_path/'stdout.json',tmp_path/'stderr.txt',2**31,1800)
    assert result['status']=='failed' and result['worker_started'] is False
    assert 'OwnedWorkerLaunchOrMonitorFailure' in result['error']


def test_ct_is_append_only_with_case_specific_binning_and_dynamic_complete_counts(tmp_path):
    plan,digest=module.load_plan(ROOT/'benchmarks/v2/radiomics-plan.json')
    supplement,ct_digest=module.load_ct_plan(ROOT/'benchmarks/v2/radiomics-ct-plan.json',digest)
    manifest=module.prepare(plan,digest,tmp_path/'base')
    original=module.cases(plan,manifest)
    manifest['ct_supplement_sha256']=ct_digest
    manifest['datasets'].append(dict(id='public_ct_pat1',input_sha256='a'*64,roi_voxels=125256,present_gray_values=[-500.,0.,400.]))
    declared=module.cases(plan,manifest,supplement)
    assert declared[:58]==original
    assert len(declared)==60
    assert declared[58]['native_extraction']['discretization']['width']==25.
    assert declared[58]['roi_count']==1 and declared[58]['preparation']==[]
    assert declared[59]['preparation']==[dict(operation='crop_voxels',margin_voxels=[0,0,0])]
    binding_path=ROOT/'benchmarks/v2/radiomics-dimension-binding.json'
    binding,binding_sha=module.load_dimension_binding(binding_path,digest,ct_digest,module.sha256_file(binding_path))
    bound=module.bind_inactive_axis(declared,binding)
    changed=[]
    for old,new in zip(declared,bound):
        if old!=new:
            changed.append(new['id'])
            assert {key for key in old if old[key]!=new[key]}=={'preparation','configuration_sha256'}
            assert old['preparation'][-1]['margin_voxels']==[1,1,1]
            assert new['preparation'][-1]['margin_voxels']==[1,1,0]
        else:
            assert old['configuration_sha256']==new['configuration_sha256']
    assert changed==['L3-synthetic_2d_g8-full-n8-j1-warm_uncached','L3-synthetic_2d_g8-full-n8-j1-cold_process_total']
    assert [case['id'] for case in declared]==[case['id'] for case in bound]
    with pytest.raises(ValueError,match='unapproved'):
        module.load_dimension_binding(binding_path,digest,ct_digest,'0'*64)
    cache=dict(population_and_hit_observations=[dict(status='ok',correctness_checked=True) for _ in range(14)])
    summaries=[dict(valid_speed_comparison=True) for _ in range(60)]
    records=[dict(status='ok') for _ in range(840)]
    assert module.overall_status(summaries,cache,records,expected_case_count=60)['correctness_status']=='passed_measured_applicable_cases'
    assert module.overall_status(summaries,cache,records[:-1],expected_case_count=60)['correctness_status'].startswith('failed_')
