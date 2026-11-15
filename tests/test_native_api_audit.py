"""Independent adversarial public-API checks, using frozen synthetic inputs.

Developed against fe2e5dc plus the integration owner's current API fixes. Run
against the matching rebuilt extension; no Golden expectations are regenerated.
To inspect an integration worktree explicitly, set PYTHONPATH to its src and run
pytest with -o pythonpath= so this test worktree does not shadow that snapshot.
"""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import ov_radiomics as rad

pytestmark=pytest.mark.skipif(importlib.util.find_spec('ov_radiomics._native') is None,reason='Native extension not built.')
GOLDEN=Path(__file__).resolve().parents[1]/'golden/python_golden_v1'


@pytest.fixture(scope='module')
def numeric_study():
    source=json.loads((GOLDEN/'model_inputs.json').read_text())['datasets']['binary']
    records=source['records'];names=source['feature_names'];classes=source['classes']
    patients={patient:i+1000 for i,patient in enumerate(dict.fromkeys(row['patient_id'] for row in records))}
    frame=pd.DataFrame([dict(ID=i,patient_id=patients[row['patient_id']],group=row['cohort'],
        target=None if row['target'] is None else classes[row['target']],
        **dict(zip(names,row['values']))) for i,row in enumerate(records)])
    result=rad.train_models(frame,'target',names,backend='rust',classes=classes,
        positive_label=classes[1],cv=3,inner_cv=2,threshold=.73)
    return frame,names,classes,result


def test_numeric_metadata_is_preserved_without_row_float_promotion(numeric_study):
    from ov_radiomics._native_common import study
    frame,names,classes,result=numeric_study
    training=frame.loc[frame.group=='train',['ID','patient_id',*names]].iloc[:4]
    converted,_=study(training,None,names,task='binary',classes=classes,default_cohort='external')
    assert [row['sample_id'] for row in converted['samples']]==[str(v) for v in training.ID]
    assert [row['patient_id'] for row in converted['samples']]==[str(v) for v in training.patient_id]
    with pytest.raises(ValueError):
        result.models['LR'].predict(training)


@pytest.mark.parametrize('cohort',['train','test'])
def test_new_patients_cannot_impersonate_saved_training_or_holdout_cohort(numeric_study,cohort):
    frame,names,_,result=numeric_study
    new=frame.loc[frame.group=='test',['ID','patient_id',*names]].iloc[:4].copy()
    new['ID']=['new-'+str(v) for v in new.ID]
    new['patient_id']=['new-'+str(v) for v in new.patient_id]
    with pytest.raises(ValueError):
        result.models['LR'].predict(new,cohort=cohort)
    predicted=result.models['LR'].predict(new,cohort='independent_external')
    assert set(predicted.dataframe.group)=={'independent_external'}
    assert set(predicted.dataframe.prediction_type)=={'external'}


def test_existing_patient_and_sample_identity_cannot_be_reassigned(numeric_study):
    frame,names,_,result=numeric_study
    heldout=frame.loc[frame.group=='test',['ID','patient_id',*names]].iloc[:4].copy()
    with pytest.raises(ValueError):
        result.models['LR'].predict(heldout,cohort='train')
    changed=frame.loc[frame.group=='train',['ID','patient_id',*names]].iloc[:1].copy()
    changed['patient_id']='unrelated_patient'
    with pytest.raises(ValueError):
        result.models['LR'].predict(changed,cohort='external')


def test_plan_default_grouping_and_saved_threshold_have_real_patient_scope(numeric_study):
    frame,_,classes,result=numeric_study
    plan=rad.SplitPlan.from_data(frame,'target',classes=classes,outer_cv=3,inner_cv=2)
    assert len(set(plan.to_dict()['sample_patients'].values()))==frame.patient_id.nunique()
    assert plan.to_dict()['sample_patients']==dict(zip(frame.ID.astype(str),frame.patient_id.astype(str)))
    report=rad.evaluate(result,ci_method=None)
    assert report.loc[report.status=='evaluated','threshold'].eq(.73).all()
    assert report.n.sum()==frame.patient_id.nunique()
    assert report.attrs['native_reports']


def test_patient_aggregation_preserves_oof_provenance_in_public_views(numeric_study):
    _,_,_,result=numeric_study
    raw=rad.PredictionTable(result.to_dict()['models'][0]['oof'])
    assert set(raw.dataframe.prediction_type)=={'oof'}
    grouped=raw.aggregate_patients()
    assert set(grouped.dataframe.prediction_type)=={'oof'}
    assert grouped.dataframe.fold.notna().all()
    assert grouped.dataframe.patient_id.is_unique


def test_series_target_order_and_declared_selector_task_are_checked():
    X=pd.DataFrame({'signal':[0.,1.,2.,3.,4.,5.],'noise':[2.,1.,4.,3.,1.,2.]},index=list('abcdef'))
    wrong_order=pd.Series([0,0,0,1,1,1],index=list('fedcba'))
    with pytest.raises(ValueError):
        rad.FeatureSelector(backend='rust',univariate=None,lasso=False).fit(X,wrong_order)
    for task in ('multiclass','unknown'):
        with pytest.raises(ValueError):
            rad.FeatureSelector(backend='rust',task=task,univariate=None,lasso=False).fit(X,wrong_order.reindex(X.index))


def test_binary_class_order_and_positive_label_cannot_disagree(numeric_study):
    frame,names,classes,result=numeric_study
    with pytest.raises(ValueError):
        rad.train_models(frame,'target',names,backend='rust',classes=classes,positive_label=classes[0],cv=3,inner_cv=2)
    # Integer class declarations and Series promoted by missing external outcomes
    # still describe the same two classes after explicit native conversion.
    numeric=frame.copy();numeric['target']=numeric.target.map({classes[0]:0,classes[1]:1})
    fitted=rad.train_models(numeric,'target',names,backend='rust',classes=[0,1],positive_label=1,cv=3,inner_cv=2)
    assert fitted.classes==['0','1']
    np.testing.assert_array_equal(fitted.predictions.probability,fitted.predictions['probability::1'])


def test_named_missing_outcome_is_an_error_not_an_unlabeled_evaluation(numeric_study):
    frame,_,_,result=numeric_study
    heldout=frame.loc[frame.group=='test']
    predicted=result.predict(heldout)
    with pytest.raises(ValueError):
        rad.evaluate(predicted,data=heldout,target='misspelled',threshold=.73,ci_method=None)
    report=rad.evaluate(predicted,data=heldout.drop(columns='target'),threshold=.73,ci_method=None)
    assert set(report.status)=={'prediction_only'}


def test_oof_fusion_does_not_read_heldout_patient_outcomes(numeric_study):
    frame,names,classes,first=numeric_study
    config=dict(backend='rust',classes=classes,positive_label=classes[1],
        split_plan=first.split_plan,cv=3,inner_cv=2,threshold=.73)
    second=rad.train_models(frame,'target',names[:2],**config)
    original=rad.fusion_data({'Full':first,'Subset':second})
    fitted=rad.fit_fusion(original,backend='rust')
    changed=frame.copy();row=changed.index[changed.group=='test'][0]
    changed.loc[row,'target']=classes[1] if changed.loc[row,'target']==classes[0] else classes[0]
    # The deliberately inconsistent held-out patient's labels should be rejected
    # only when that cohort's patient outcome is evaluated, never during OOF fit.
    first_changed=rad.train_models(changed,'target',names,**config)
    second_changed=rad.train_models(changed,'target',names[:2],**config)
    assert first.models['LR'].to_dict()==first_changed.models['LR'].to_dict()
    assert second.models['LR'].to_dict()==second_changed.models['LR'].to_dict()
    observed=rad.fusion_data({'Full':first_changed,'Subset':second_changed})
    pd.testing.assert_frame_equal(original.frame,observed.frame)
    assert fitted.model.to_dict()==rad.fit_fusion(observed,backend='rust').model.to_dict()


@pytest.mark.parametrize('mask',[np.asarray([[True,False],[False,True],[False,False]]),np.full((2,3),np.nan)])
def test_tissue_mask_axes_and_missing_values_cannot_be_silently_reinterpreted(mask):
    source=json.loads((GOLDEN/'pathology_inputs.json').read_text())
    image=np.asarray([[[255,0,0],[0,255,0],[0,0,255]],[[255,255,0],[0,255,255],[255,0,255]]],dtype=np.uint8)
    patch=next(rad.pathology.tile_image(image,patient_id='P',slide_id='S',source='independent_audit',patch_size=(3,2)))
    with pytest.raises(ValueError):
        rad.pathology.color_features(patch,config={'fixed_edges':source['fixed_edges']},tissue_mask=mask)


def test_native_model_refuses_implicit_replacement(numeric_study,tmp_path):
    _,_,_,result=numeric_study
    model=result.models['LR'];path=tmp_path/'model.rad'
    model.save(path);before=path.read_bytes()
    with pytest.raises((ValueError,FileExistsError)):
        model.save(path)
    assert path.read_bytes()==before


def test_native_extraction_refuses_implicit_output_replacement(tmp_path):
    samples=pd.DataFrame([dict(ID='audit',patient_id='P',image=str(GOLDEN.parent/'ibsi_digital_v1/image.nii.gz'),
        mask=str(GOLDEN.parent/'ibsi_digital_v1/mask.nii.gz'))])
    result=rad.extract(samples,backend='rust',bin_width=1)
    path=tmp_path/'features.csv';path.write_text('existing-user-result',encoding='utf-8')
    with pytest.raises((ValueError,FileExistsError)):
        result.save(tmp_path)
    assert path.read_text(encoding='utf-8')=='existing-user-result'


def test_clinical_patient_scope_rejects_empty_id_instead_of_counting_a_patient():
    frame=pd.DataFrame(dict(patient_id=['','P1','P2','P3','P4','P5'],
        age=[20.,40.,35.,50.,25.,45.],target=[0,1,0,1,0,1]))
    with pytest.raises(ValueError):
        rad.baseline_table(frame,'target',['age'],continuous=['age'],backend='rust')


def test_auto_missing_native_backend_never_runs_an_implicit_reference_algorithm(monkeypatch):
    from ov_radiomics import _backend,metrics
    def absent(name):
        assert name=='ov_radiomics._native'
        raise ModuleNotFoundError('not built',name='ov_radiomics._native')
    def forbidden():
        raise AssertionError('auto called an optional reference algorithm')
    monkeypatch.setattr(_backend,'import_module',absent)
    monkeypatch.setattr(metrics,'_reference',forbidden)
    assert _backend.resolve_backend('python') is None
    for operation in (
        lambda:metrics.binary_metrics([0,0,1,1],[.1,.4,.35,.8],backend='auto'),
        lambda:metrics.delong_test([0,0,1,1],[.1,.4,.35,.8],[.2,.3,.4,.7],backend='auto'),
        lambda:metrics.decision_curve([0,0,1,1],[.1,.4,.35,.8],thresholds=[.5],backend='auto'),
        lambda:metrics.hosmer_lemeshow([0,0,1,1],[.1,.4,.35,.8],backend='auto'),
    ):
        with pytest.raises(ImportError):
            operation()
    information=_backend.backend_info()
    assert information['rust_core_version'] is None
    assert information['metrics_backend']!='python'


@pytest.mark.parametrize('case',json.loads((GOLDEN/'cases.json').read_text())['cases'],ids=lambda case:case['id'])
@pytest.mark.parametrize('aggregation',['direction_mean','merge_matrices'])
def test_each_selected_family_matches_frozen_301_not_a_rerecorded_native_output(case,aggregation):
    from ov_radiomics._native_common import request
    reference=json.loads((GOLDEN/'reference.json').read_text())['features'][case['id']]
    shape=tuple(reversed(case['shape']))
    image=rad.image.from_array(np.asarray(case['values']).reshape(shape))
    mask=rad.mask.from_array(np.asarray(case['mask'],dtype=np.uint32).reshape(shape),geometry=image.geometry)
    config=dict(profile='python_golden_v1',aggregation=aggregation,gldm_alpha=0,voxel_array_shift=0,
        discretization=dict(method='fixed_count',count=case['binCount']) if 'binCount' in case
        else dict(method='fixed_width',width=case['binWidth']))
    arguments=dict(image=image.to_dict(),morphology_mask=mask.to_dict(),intensity_mask=None,roi_label=1,config=config)
    families=('firstorder','glcm','glrlm','glszm','gldm','ngtdm')
    merged={}
    for family in families:
        source=reference['merged_features'] if aggregation=='merge_matrices' and family in ('glcm','glrlm') else reference['features']
        expected=source[family]
        if expected['status']!='ok':
            with pytest.raises(ValueError):
                request('array_features',**arguments,families=[family])
            continue
        observed=request('array_features',**arguments,families=[family])
        assert set(observed)=={'original_'+family+'_'+key for key in expected['value']}
        for feature,value in expected['value'].items():
            key='original_'+family+'_'+feature
            actual=float(observed[key])
            if isinstance(value,str):
                assert np.isnan(actual) if value=='NaN' else actual==float(value)
            else:
                assert np.isfinite(actual)
                assert abs(actual-value)<=1e-10+1e-8*abs(value)
            assert key not in merged
            merged[key]=observed[key]
    if any(case['mask']):
        assert len(merged)==55
        # The selected family path and full six-family path must agree exactly,
        # including the position and string state of undefined scalar results.
        assert request('array_features',**arguments,families=list(families))==merged
