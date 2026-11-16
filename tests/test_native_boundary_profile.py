"""Diagnostic protocol checks; no CT extraction or benchmark results are fabricated."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'tools/profile_native_boundary.py'
spec=importlib.util.spec_from_file_location('boundary_profile',PATH)
profile=importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_frozen_feature_set_and_tolerances_are_strict():
    ref={'features':{'a':10.,'b':'NaN'}}
    profile.check_features({'a':10.+1e-9,'b':'NaN'},ref)
    with pytest.raises(ValueError,match='scalar mismatch'):
        profile.check_features({'a':10.+1e-4,'b':'NaN'},ref)
    with pytest.raises(ValueError,match='identity differs'):
        profile.check_features({'a':10.},ref)
    profile.check_features({'a':10.,'voxel_count':12},
        {'features':{'a':10.},'basic_shape':{'voxel_count':12}})


def test_summary_retains_small_sample_uncertainty():
    result=profile.summary([1e6,2e6,3e6,4e6,5e6,6e6,100e6])
    assert result['n']==7 and result['median_ms']==4.
    assert result['iqr_ms']==3. and result['p95_ms']>70.


def test_trace_restores_public_request_and_records_real_operations():
    from types import SimpleNamespace
    import json
    class Object:
        def to_dict(self):return {'x':1}
    old=lambda operation,**kwargs:None
    common=SimpleNamespace(request=old,encode_json=json.dumps)
    types=SimpleNamespace(_NativeObject=Object)
    native=SimpleNamespace(request_json=lambda text:'{"feature":1.0}')
    original_copy=Object.to_dict
    with profile.request_trace(common,types,native) as (requests,copies):
        assert common.request('array_features',image={'x':1})=={'feature':1.}
        assert Object().to_dict()=={'x':1}
    assert common.request is old and Object.to_dict is original_copy
    assert requests[0]['operation']=='array_features'
    assert requests[0]['request_bytes']>0 and requests[0]['decode_ns']>=0
    assert copies[0]['nanoseconds']>=0
