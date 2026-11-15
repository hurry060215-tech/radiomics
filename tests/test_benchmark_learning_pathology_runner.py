"""Protocol/input tests only; no algorithm timings or Golden regeneration."""
import importlib.util
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import benchmark_learning_pathology as runner
import benchmark_lp_runtime as runtime


def test_input_corpus_is_deterministic_and_patient_folds_are_actual_sets(tmp_path):
    plan,digest=runner.load_plan(ROOT/'benchmarks/v2/learning-pathology-plan.json')
    import numpy as np
    if (np.__version__!=plan['environments']['common_versions']['numpy']
            or importlib.metadata.version('Pillow')!=plan['environments']['common_versions']['Pillow']):
        pytest.skip('Deterministic benchmark corpus acceptance requires the exact predeclared NumPy/Pillow generation environment.')
    first=runner.prepare_inputs(plan,digest,tmp_path/'a');second=runner.prepare_inputs(plan,digest,tmp_path/'b')
    assert first==second
    assert first['pathology']['images']==25 and first['pathology']['study_patients']==12
    split=runner.read_json(tmp_path/'a/pathology-patient-plan.json')
    labels={row['patient_id']:row['target'] for row in runner.read_json(tmp_path/'a/pathology-labels.json')['records']}
    for nested in split['outer_folds']:
        outer=nested['outer']
        assert set(outer['training_patients']).isdisjoint(outer['validation_patients'])
        for inner in nested['inner']:
            assert set(inner['training_patients'])|set(inner['validation_patients'])==set(outer['training_patients'])
            assert set(labels[p] for p in inner['training_patients'])=={0,1}
            assert set(labels[p] for p in inner['validation_patients'])=={0,1}
    declared=runner.case_declarations(plan,first,dict(dataset_sha256='a'*64))
    assert len(declared)==16
    assert len({row['id'] for row in declared})==16
    with pytest.raises(FileExistsError):runner.prepare_inputs(plan,digest,tmp_path/'a')


def test_input_generation_rejects_nonfrozen_versions_before_writing(tmp_path,monkeypatch):
    import numpy as np
    plan,digest=runner.load_plan(ROOT/'benchmarks/v2/learning-pathology-plan.json')
    monkeypatch.setattr(np,'__version__','deliberately-not-the-frozen-version')
    target=tmp_path/'not-generated'
    with pytest.raises(RuntimeError,match='declared NumPy/Pillow'):
        runner.prepare_inputs(plan,digest,target)
    assert not target.exists()


def test_typed_nonfinite_and_scalar_layers_are_preserved():
    plan,_=runner.load_plan(ROOT/'benchmarks/v2/learning-pathology-plan.json')
    assert runner.compare_values({'a':'NaN'},{'a':'NaN'},plan)==[]
    assert runner.compare_values({'a':'NaN'},{'a':0.},plan)
    assert runner.compare_values({'counts':[1]},{'counts':[2]},plan)
    assert runner.compare_values({'values':[1.]},{'values':[1.+1e-5]},plan)
    assert runner.compare_values({'probabilities':[.2]},{'probabilities':[.2+1e-7]},plan)==[]


def test_single_file_probe_does_not_hash_the_entire_environment(tmp_path):
    path=tmp_path/'module.py';path.write_text('version=1',encoding='utf-8')
    (tmp_path/'unrelated.py').write_text('unrelated',encoding='utf-8')
    result=runner.module_files(SimpleNamespace(__file__=str(path)))
    assert result['single_file']=='module.py'
    assert set(result['files_sha256'])=={'module.py'}


def test_untimed_stages_never_emit_algorithm_elapsed_values(monkeypatch):
    def forbidden():raise AssertionError('Untimed numerical validation used a performance clock')
    monkeypatch.setattr(runtime.time,'perf_counter',forbidden)
    stages=runtime.Stages(False)
    with stages('public_call'):pass
    assert stages.seconds=={}


def test_spawn_failure_stays_a_failed_observation(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise OSError('synthetic spawn failure')
    monkeypatch.setattr(runtime.subprocess,'Popen',fail)
    request=dict(plan_contents={'execution':{'thread_environment':{}}})
    result=runtime.launch('not executed',ROOT/'tools/benchmark_learning_pathology.py',request,
        tmp_path/'request.json',tmp_path/'stdout',tmp_path/'stderr',audit=lambda request:{},
        finalize=lambda value,request:value,budget=2**31,timeout=1800,measured=False)
    assert result['status']=='failed' and not result['worker_started']
    assert 'synthetic spawn failure' in result['error']
