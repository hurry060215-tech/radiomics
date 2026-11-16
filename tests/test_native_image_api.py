import numpy as np
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import resolve_backend, try_native

pytestmark = pytest.mark.skipif(try_native() is None, reason='Native extension not built.')


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
    assert next(definition['unit'] for definition in table.definitions if definition['name']=='original_shape_basic_Area_mm2')=='mm2'
    family=sample.extract(config,families=['firstorder'])
    assert len(family.definitions)==16
    assert all(definition['name'].startswith('original_firstorder_') for definition in family.definitions)
    np.testing.assert_array_equal(family.features,table.features[family.features.columns])
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


@pytest.mark.parametrize('nonfinite',[False,True])
def test_sample_construction_matches_dict_inputs_and_detaches_source_buffers(nonfinite):
    values=np.asarray([[1.,2.],[3.,4.]])
    labels=np.asarray([[0,1],[1,1]],dtype=np.uint32)
    if nonfinite:
        values[0,0]=np.nan  # Permitted image state outside the selected ROI.
    image=rad.image.from_array(values,spatial_unit='mm')
    mask=rad.mask.from_array(labels,geometry=image.geometry)
    metadata=dict(sample_id='owned-sample',patient_id='synthetic')
    expected=rad.ImageSample(metadata,image.to_dict(),mask.to_dict()).to_dict()
    sample=rad.ImageSample(metadata,image,mask)
    assert sample.to_dict()==expected
    # Even later changes to the source wrappers cannot affect the new sample.
    image._payload['values'][1]=999.
    image._payload['geometry']['affine'][0][3]=10.
    mask._payload['labels'][1]=0
    metadata['sample_id']='changed'
    detached=sample.to_dict()
    detached['image']['values'][1]=-999.
    assert sample.to_dict()==expected


@pytest.mark.parametrize('override',['subclass','instance','class'])
def test_sample_preserves_custom_volume_dictionary_protocol(override,monkeypatch):
    image=rad.image.from_array([[1.,2.],[3.,4.]],spatial_unit='mm')
    mask=rad.mask.from_array(np.ones((2,2),dtype=np.uint32),geometry=image.geometry)
    original=rad.ImageVolume.to_dict
    def transformed(value):
        payload=original(value)
        payload['values']=[2*x for x in payload['values']]
        return payload
    if override=='subclass':
        class CustomImage(rad.ImageVolume):
            def to_dict(self):
                return transformed(self)
        image=CustomImage(image.to_dict())
    elif override=='instance':
        image.to_dict=lambda:transformed(image)
    else:
        monkeypatch.setattr(rad.ImageVolume,'to_dict',transformed)
    sample=rad.ImageSample(dict(sample_id='custom',patient_id='synthetic'),image,mask)
    np.testing.assert_array_equal(sample.image.array,[[2.,4.],[6.,8.]])


def test_sample_preserves_custom_mask_protocol_and_rejects_misaligned_geometry():
    image=rad.image.from_array([[1.,2.],[3.,4.]],spatial_unit='mm')
    mask=rad.mask.from_array([[0,1],[1,1]],geometry=image.geometry)
    class CustomMask(rad.MaskVolume):
        def to_dict(self):
            payload=super().to_dict()
            payload['labels'][0]=1
            return payload
    sample=rad.ImageSample(dict(sample_id='custom-mask',patient_id='synthetic'),image,CustomMask(mask.to_dict()))
    np.testing.assert_array_equal(sample.mask.array,np.ones((2,2),dtype=np.uint32))
    other=mask.to_dict()
    other['geometry']['affine'][0][3]=100.
    with pytest.raises(ValueError):
        rad.ImageSample(dict(sample_id='bad-grid',patient_id='synthetic'),image,rad.MaskVolume(other))
