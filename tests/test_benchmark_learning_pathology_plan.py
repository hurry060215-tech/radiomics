"""Predeclaration integrity only. No model/image feature computation or timing."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def unique_object(pairs):
    result={}
    for key,value in pairs:
        assert key not in result,'Duplicate JSON key: '+key
        result[key]=value
    return result


def load():
    return json.loads((ROOT/'benchmarks/v2/learning-pathology-plan.json').read_text(),object_pairs_hook=unique_object)


def test_frozen_sources_and_separate_suites_have_not_been_substituted():
    plan=load()
    assert plan['status']=='predeclared_pending_main_review_no_measurements'
    assert plan['suites']==['machine_learning_statistics','pathology']
    for name,digest in plan['source_files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert plan['learning']['metrics']['CI'] is None
    assert plan['learning']['candidate_lambda_order']==[.05,.1]
    assert plan['execution']['paired_repetitions']==7
    assert plan['execution']['memory_budget_bytes']==2*1024**3


def test_declared_pathology_grid_padding_and_study_population_are_exact():
    plan=load();generator=plan['pathology']['generator'];tiles=plan['pathology']['tiling']
    width,height=generator['width'],generator['height']
    pw,ph=tiles['patch_size_wh'];sx,sy=tiles['stride_xy']
    positions=[(x,y,min(pw,width-x),min(ph,height-y)) for y in range(0,height,sy) for x in range(0,width,sx)]
    assert len(positions)==tiles['tiles_per_slide']==24
    assert sum(w<pw or h<ph for _,_,w,h in positions)==tiles['padded_tiles_per_slide']==9
    assert positions[-1]==(120,72,8,24)
    assert len(positions)*generator['total_slides']==tiles['total_tiles']==600
    assert generator['study_patients']==plan['pathology_patient_model_case']['patient_rows']==12
    assert generator['total_slides']==25
    assert generator['total_patients']==13
    assert tiles['normal_tiles']+tiles['QC_sentinel_tiles']==600


def test_workload_rows_and_diagnostic_stage_reporting_are_not_pooled():
    plan=load()
    rows=len(plan['learning']['primary_cases'])+1+len(plan['pathology']['primary_cases'])+1+int(plan['pathology_full_scene_diagnostic']['enabled'])
    assert rows==plan['reporting']['case_count_before_state_expansion']==8
    assert rows*len(plan['execution']['states'])==plan['reporting']['state_expanded_case_count']==16
    assert rows*2*7*2==plan['reporting']['paired_worker_observations']==224
    assert plan['pathology_patient_model_case']['suite']=='machine_learning_statistics'
    assert plan['pathology_full_scene_diagnostic']['role'].startswith('Additional whole-scene diagnostic')
    stages=plan['pathology_full_scene_diagnostic']['stage_columns']
    assert len(stages)==len(set(stages))
    assert {'decode_tile_qc_color','id_alignment_aggregation','model_fit_tune','model_predict_evaluate','frontend_exports'}<=set(stages)
