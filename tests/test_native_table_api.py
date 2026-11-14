"""Native table readers and center metadata adapters; no optional Parquet engine."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import try_native

pytestmark=pytest.mark.skipif(try_native() is None,reason='Native extension not built.')


def test_study_csv_roles_and_patient_join_are_loaded_in_core(tmp_path):
    (tmp_path/'features.csv').write_text('sample_id,patient_id,cohort,signal\n001,P1,train,1\n002,P1,train,\n003,P2,test,3\n',encoding='utf-8')
    (tmp_path/'clinical.csv').write_text('patient_id,target,sex\nP2,positive,M\nP1,negative,F\n',encoding='utf-8')
    spec=dict(schema_version=1,table='features.csv',clinical='clinical.csv',clinical_key='patient',
        outcome_column='target',numeric_features=['signal'],categorical_features=['sex'],optional_metadata={})
    (tmp_path/'study.json').write_text(json.dumps(spec),encoding='utf-8')
    loaded=rad.read_study(tmp_path/'study.json')
    frame=loaded.dataframe
    assert frame.ID.tolist()==['001','002','003']
    assert frame.patient_id.tolist()==['P1','P1','P2']
    assert loaded.features==['signal','sex'] and loaded.categorical==['sex'] and loaded.target=='target'
    assert frame.target.tolist()==['negative','negative','positive']
    assert frame.sex.tolist()==['F','F','M']
    assert np.isnan(frame.signal.iloc[1])
    assert len(loaded.report['inputs'])==2
    assert Path.cwd()!=tmp_path


@pytest.mark.parametrize('extension',['csv','parquet'])
def test_feature_table_native_read_preserves_metadata_missing_units_and_order(extension,tmp_path):
    source=rad.FeatureTable(dict(samples=[dict(sample_id='001',patient_id='P1'),dict(sample_id='002',patient_id='P1')],
        data=dict(rows=2,cols=2,values=[1.,0.,3.,4.],missing=[False,True,False,False]),
        features=[dict(name='b',definition='b definition',unit='mm'),dict(name='a',definition='a definition',unit=None)],
        effective_parameters=dict(input='external synthetic matrix'),quality=[]))
    path=tmp_path/('features.'+extension);source.save(path)
    assert rad.read_feature_table(path).to_dict()==source.to_dict()
    assert rad.FeatureTable.read(path).to_dict()==source.to_dict()


def test_public_split_plan_center_defaults_match_native_study_metadata():
    from ov_radiomics._native_common import study
    data=pd.DataFrame([dict(ID=f'S{i}',patient_id=f'P{i}',group='train',target=i%2,center='A',value=float(i)) for i in range(16)])
    plan=rad.SplitPlan.from_data(data,'target',outer_cv=2,inner_cv=2)
    assert set(plan.to_dict()['sample_centers'].values())=={'A'}
    plan.validate_data(data,'target')
    result=rad.train_models(data,'target',['value'],split_plan=plan,cv=2,inner_cv=2,backend='rust')
    assert len(result.oof_predictions)==16


def test_explicit_center_holdout_ignores_test_labels_and_preserves_external_patients():
    data=pd.DataFrame([dict(ID=f'S{i}',patient_id=f'P{i}',target=i%3,
        center='A' if i<24 else 'B' if i<30 else 'C',group='unassigned' if i<30 else 'external_X') for i in range(36)])
    plan=rad.SplitPlan.center_holdout(data,'target',held_out_centers=['B'],task='multiclass',outer_cv=3,inner_cv=2)
    mapped=plan.apply_cohorts(data)
    assert set(mapped.loc[mapped.center=='B','group'])=={'test'}
    assert set(mapped.loc[mapped.center=='A','group'])=={'train'}
    assert set(mapped.loc[mapped.center=='C','group'])=={'external_X'}
    plan.validate_data(mapped,'target')
    changed=data.copy();changed.loc[changed.center!='A','target']=999
    assert plan.fingerprint==rad.SplitPlan.center_holdout(changed,'target',held_out_centers=['B'],task='multiclass',outer_cv=3,inner_cv=2).fingerprint
