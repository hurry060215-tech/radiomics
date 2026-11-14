"""Per-center evaluation uses identified native prediction tables and point metrics."""
import numpy as np
import pandas as pd
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import try_native

pytestmark=pytest.mark.skipif(try_native() is None,reason='Native extension not built.')


def test_center_holdout_reuses_complete_native_pipeline_without_test_truth():
    frame=pd.DataFrame([dict(ID=f'S{i}',patient_id=f'P{i}',target=i%2,
        center='A' if i<16 else 'B',signal=(i%2)+.1*(i%3),noise=float(i%7)) for i in range(24)])
    plan=rad.SplitPlan.center_holdout(frame,'target',held_out_centers='B',outer_cv=2,inner_cv=2)
    mapped=plan.apply_cohorts(frame)
    fitted=rad.train_models(mapped,'target',['signal','noise'],split_plan=plan,backend='rust')
    first=fitted.to_dict()
    changed=mapped.copy();changed.loc[changed.center=='B','signal']=1e6;changed.loc[changed.center=='B','target']=999
    another=rad.train_models(changed,'target',['signal','noise'],split_plan=plan,backend='rust')
    assert first['models'][0]['model']==another.to_dict()['models'][0]['model']
    pd.testing.assert_frame_equal(fitted.oof_predictions,another.oof_predictions)
    assert set(fitted.split_plan.to_dict()['cohorts']['test'])=={f'P{i}' for i in range(16,24)}
    report=rad.evaluate(fitted,by_center=True,ci_method=None)
    assert set(report.center)=={'A','B'}
    assert report.set_index('center').n.to_dict()=={'A':16,'B':8}
    assert report.set_index('center').group.to_dict()=={'A':'train','B':'test'}
    model=fitted.models['LR']
    external=mapped.loc[mapped.group=='test'].drop(columns=['target','patient_id'])
    with pytest.raises(ValueError,match='distinct sample/patient'):
        model.predict(external)


def test_fractional_bootstrap_or_bin_counts_are_errors_not_reinterpreted():
    with pytest.raises(ValueError,match='integer'):
        rad.binary_metrics([0,0,1,1],[.1,.3,.7,.9],backend='rust',ci_method='bootstrap',n_bootstrap=2.5)
    with pytest.raises(ValueError,match='integer'):
        rad.hosmer_lemeshow([0,0,1,1],[.1,.3,.7,.9],backend='rust',n_bins=3.5)
