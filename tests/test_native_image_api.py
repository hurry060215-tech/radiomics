import numpy as np
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import resolve_backend

pytestmark = pytest.mark.skipif(resolve_backend('auto') is None, reason='Native extension not built.')


def test_native_array_image_mask_geometry_preparation_and_extraction():
    affine = [[2,0,0,10],[0,3,0,-20],[0,0,4,30],[0,0,0,1]]
    image = rad.image.from_array([[1,1,2],[2,3,3]], affine=affine, spatial_unit='mm')
    mask = rad.mask.from_array(np.ones((2,3)), geometry=image.geometry)
    sample = rad.ImageSample(dict(sample_id='sample',patient_id='patient'), image, mask)
    config = dict(profile='python_golden_v1',discretization=dict(method='fixed_width',width=1),
        aggregation='direction_mean',gldm_alpha=0,voxel_array_shift=0)
    table = sample.extract(config)
    assert table.features.original_firstorder_Mean.item() == 2.
    assert table.features.original_shape_basic_Area_mm2.item() == 36.
    intensity = rad.mask.resegment(image, mask, lower=2, upper=3)
    assert sample.extract(config,intensity_mask=intensity).features.original_firstorder_Mean.item() == 2.5
    scaled, fitted = rad.image.normalize(image, method='zscore', statistics='roi', mask=mask, return_transform=True)
    assert fitted['statistical_voxels'] == 6
    np.testing.assert_allclose(scaled.array.mean(),0,atol=1e-15)
    resampled = rad.image.resample_spacing(image,[1,1,4],interpolation='nearest')
    assert resampled.geometry['shape'] == [6,6,1]
    matched = rad.mask.resample_like(mask,resampled)
    assert matched.geometry == resampled.geometry
    with pytest.raises(ValueError,match='shape/axes'):
        rad.mask.from_array(np.ones((3,2)),geometry=image.geometry)


def test_unknown_units_shear_and_complex_data_are_explicit():
    image = rad.image.from_array(np.arange(25).reshape(5,5))
    mask = rad.mask.from_array(np.pad(np.ones((1,1)),2),geometry=image.geometry)
    with pytest.raises(ValueError,match='Unknown spatial units'):
        rad.mask.peritumoral_ring(mask,outer_mm=3)
    with pytest.raises(ValueError,match='Scalar real'):
        rad.image.from_array(np.ones((2,2),dtype=complex))
    with pytest.raises(ValueError,match='without truncation'):
        rad.mask.from_array([[1.5]*5]*5,geometry=image.geometry)
