"""Separate the frozen compatibility target from the mathematical mean."""
import math
import numpy as np
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import try_native


@pytest.mark.skipif(try_native() is None, reason='Native extension not built.')
def test_mean_cancellation_matches_stable_mathematical_target():
    # Input and PyRadiomics3.0.1 observation were frozen by the independent
    # scalar audit before this test: manifest a7ae74c4...9581a, report79e90b6c...
    # Exact input mean is0.5; the frozen public PyRadiomics observation is0.
    # The approved stable target deliberately differs from that Python value.
    # Frozen evidence and the independent math.fsum check remain unchanged.
    values=np.asarray([1e16,1.,-1e16,1.,1e16,1.,-1e16,1.],dtype=np.float64).reshape(2,4)
    image=rad.image.from_array(values,affine=np.eye(4),spatial_unit='millimeter')
    mask=rad.mask.from_array(np.ones_like(values,dtype=np.uint32),geometry=image.geometry)
    sample=rad.ImageSample(dict(sample_id='scalar-cancellation',patient_id='synthetic-scalar'),image,mask)
    config=dict(profile='python_golden_v1',discretization=dict(method='fixed_count',count=8),
                aggregation='direction_mean',gldm_alpha=0,voxel_array_shift=0.)
    result=sample.extract(config,families=['firstorder'])
    assert result.features.loc[0,'original_firstorder_Mean']==pytest.approx(0.5,abs=1e-10,rel=1e-8)
    print('stable native Mean:', result.features.loc[0,'original_firstorder_Mean'])
    print('frozen public PyRadiomics Mean observation:', 0.0)


def test_mean_cancellation_mathematical_reference_is_separate():
    values = [1e16, 1., -1e16, 1., 1e16, 1., -1e16, 1.]
    mathematical_mean = math.fsum(values) / len(values)
    assert mathematical_mean == 0.5
    print('independent math.fsum Mean:', mathematical_mean)
