"""Public Python/native boundary checks, using independently frozen task inputs."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import resolve_backend, try_native

pytestmark=pytest.mark.skipif(try_native() is None,reason='Native extension not built.')
GOLDEN=Path(__file__).resolve().parents[1]/'golden/python_golden_v1'


def dataset(task):
    source=json.loads((GOLDEN/'model_inputs.json').read_text())['datasets'][task]
    rows=[]
    for row in source['records']:
        target=row['target'] if task=='regression' or row['target'] is None else source['classes'][row['target']]
        rows.append(dict(ID=row['sample_id'],patient_id=row['patient_id'],group=row['cohort'],
            center=row['center'],target=target,**dict(zip(source['feature_names'],row['values']))))
    return pd.DataFrame(rows),source['feature_names'],source['classes']


@pytest.mark.parametrize('task',['binary','multiclass','regression'])
def test_public_native_task_prediction_persistence_and_patient_evaluation(task,tmp_path):
    frame,features,classes=dataset(task)
    result=rad.train_models(frame,'target',features,backend='rust',task=task,classes=classes,
        positive_label=classes[-1] if classes else 1,cv=3,inner_cv=2,scale='standard')
    assert len(result.oof_predictions)==sum(frame.group=='train')
    assert len(result.predictions)==len(frame)
    assert result.oof_predictions.groupby('patient_id').fold.nunique().max()==1
    report=rad.evaluate(result,backend='rust',ci_method=None)
    assert set(report.group)==set(frame.group)
    assert report.n.sum()==frame.patient_id.nunique()
    assert set(report.prediction_unit)=={'patient'}
    assert set(report.status) <= {'evaluated','prediction_only'}
    assert (report.status=='evaluated').any()
    result.save(tmp_path/'result')
    bundle=rad.load_model(tmp_path/'result/model-000.rad')
    external=frame.loc[frame.group!='train'].drop(columns='target').copy()
    original=result.predict(external)
    # A different process and deliberately reordered columns load the .rad file.
    (tmp_path/'external.json').write_text(json.dumps(external[list(reversed(external.columns))].to_dict(orient='list')),encoding='utf-8')
    code="""
import json,pandas as pd,ov_radiomics as rad
from ov_radiomics._native_common import study
bundle=rad.load_model(__import__('sys').argv[1]);frame=pd.DataFrame(json.load(open(__import__('sys').argv[2],encoding='utf-8')))
converted,_=study(frame,None,FEATURES,task=TASK,classes=CLASSES)
print(json.dumps(bundle.predict_study(converted).to_dict(),sort_keys=True))
""".replace('FEATURES',repr(features)).replace('TASK',repr(task)).replace('CLASSES',repr(classes))
    observed=subprocess.run([sys.executable,'-c',code,str(tmp_path/'result/model-000.rad'),str(tmp_path/'external.json')],
        check=True,capture_output=True,text=True)
    restored=rad.PredictionTable(json.loads(observed.stdout))
    a,b=original.dataframe,restored.dataframe
    assert a.ID.tolist()==b.ID.tolist()
    columns=['prediction'] if task=='regression' else ['probability::'+c for c in classes]
    np.testing.assert_array_equal(a[columns],b[columns])
    with pytest.raises(FileExistsError):
        result.save(tmp_path/'result')
    assert bundle.manifest['task']==task
    assert 'prediction_type' in report
    if task=='multiclass':
        assert {'macro_f1','micro_f1','weighted_f1'} <= set(report)
        assert 'confusion' in report.attrs['native_reports'][0]['report']


def test_patient_plan_fusion_leakage_and_test_perturbation():
    frame,features,classes=dataset('binary')
    config=dict(backend='rust',classes=classes,positive_label=classes[-1],cv=3,inner_cv=2,scale='standard')
    first=rad.train_models(frame,'target',features[:2],**config)
    second=rad.train_models(frame,'target',features[2:],split_plan=first.split_plan,**config)
    parts={'Rad':first,'Clinical':second}
    oof=rad.fusion_data(parts)
    fused=rad.fit_fusion(oof,backend='rust')
    held=rad.fusion_data(parts,cohort='test')
    probabilities=fused.predict_proba(held)
    assert probabilities.between(0,1).all()
    assert len(rad.evaluate(fused,ci_method=None))==frame.group.nunique()
    with pytest.raises(ValueError,match='OOF'):
        rad.fit_fusion(held,backend='rust')
    changed=frame.copy();test=changed.group!='train'
    changed.loc[test,features]=9999.;changed.loc[test,'target']=classes[0]
    another=rad.train_models(changed,'target',features[:2],split_plan=first.split_plan,**config)
    assert first.models['LR'].to_dict()==another.models['LR'].to_dict()
    pd.testing.assert_frame_equal(first.oof_predictions,another.oof_predictions)


def test_native_import_and_training_never_load_optional_reference_packages():
    code="""
import importlib.abc,sys
class RejectReference(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'sklearn','statsmodels','scipy','radiomics','SimpleITK','joblib','ov','wsidata','seaborn'}:
            raise AssertionError('Native workflow imported optional package '+fullname)
sys.meta_path.insert(0,RejectReference())
import ov_radiomics as rad
import runpy
namespace=runpy.run_path(TESTFILE)
frame,features,classes=namespace['dataset']('binary')
result=rad.train_models(frame,'target',features,backend='rust',classes=classes,positive_label=classes[-1],cv=3,inner_cv=2)
rad.evaluate(result,backend='rust',ci_method=None)
rad.plot_roc([0,0,1,1],[.1,.4,.35,.8])
rad.plot_calibration([0,0,1,1],[.1,.4,.35,.8],n_bins=2)
rad.plot_confusion_matrix([0,0,1,1],[.1,.4,.35,.8])
print('native_only_passed')
""".replace('TESTFILE',repr(str(Path(__file__).resolve())))
    completed=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    assert 'native_only_passed' in completed.stdout


def test_native_selector_does_not_relax_all_missing_or_zero_feature_errors():
    data=pd.DataFrame(dict(a=[0.,1.,2.,3.],b=[0.,-1.,-2.,-3.],all_missing=[np.nan]*4))
    selector=rad.FeatureSelector(backend='rust',univariate=None,lasso=False)
    with pytest.raises(ValueError,match='missing'):
        selector.fit(data,[0,0,1,1])
    selector=rad.FeatureSelector(backend='rust',univariate=None,lasso=False,all_missing='drop').fit(data,[0,0,1,1])
    assert selector.selected_features_==['a']
    assert selector.transform(data[['b','a','all_missing']]).shape==(4,1)
    with pytest.raises(ValueError,match='Multiclass'):
        rad.FeatureSelector(backend='rust',task='multiclass',sparse_method='squared_lasso').configurations()


def test_native_evaluation_uses_saved_threshold_and_preserves_full_report():
    frame,features,classes=dataset('binary')
    result=rad.train_models(frame,'target',features,backend='rust',classes=classes,
        positive_label=classes[-1],cv=3,inner_cv=2,threshold=.73)
    report=rad.evaluate(result,ci_method=None)
    assert report.loc[report.status=='evaluated','threshold'].eq(.73).all()
    assert {'average_precision','log_loss_eps1e15'} <= set(report)
    assert report.attrs['native_reports'][0]['report']['confusion']['size']==2
    assert result.predict(frame.loc[frame.group!='train'].drop(columns='target')).to_dict()==result.models['LR'].predict(frame.loc[frame.group!='train'].drop(columns='target')).to_dict()
