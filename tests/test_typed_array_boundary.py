"""The private typed path retains the JSON contract and owned sample semantics."""
import gc
import json
from pathlib import Path

import numpy as np
import pytest

import ov_radiomics as rad
from ov_radiomics._backend import resolve_backend, try_native
from ov_radiomics._native_common import array_features_typed_or_json, request

pytestmark = pytest.mark.skipif(try_native() is None, reason='Native extension not built.')

CONFIG = dict(profile='python_golden_v1',
    discretization=dict(method='fixed_width', width=1),
    aggregation='direction_mean', gldm_alpha=0, voxel_array_shift=0)


def arrays():
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    labels = np.ones(values.shape, dtype=np.uint32)
    image = rad.image.from_array(values, spatial_unit='mm')
    return values, labels, image.geometry


def test_typed_entry_matches_legacy_json_exactly():
    values, labels, geometry = arrays()
    native = resolve_backend('rust')
    actual = json.loads(native.array_features_typed(values, labels, json.dumps(dict(
        image_geometry=geometry, mask_geometry=geometry, config=CONFIG,
        roi_label=1, families=None, intensity_mask=None))))
    expected = request('array_features',
        image=dict(geometry=geometry, values=values.ravel().tolist()),
        morphology_mask=dict(geometry=geometry, labels=labels.ravel().tolist()),
        intensity_mask=None, config=CONFIG, roi_label=1, families=None)
    assert actual == expected


@pytest.mark.parametrize('kind', ['float32', 'strided', 'fortran', 'mask64', 'big_endian'])
def test_rust_rejects_incompatible_buffers_and_dispatch_falls_back(kind):
    values, labels, geometry = arrays()
    if kind == 'float32':
        values = values.astype(np.float32)
    elif kind == 'strided':
        source = np.zeros((2, 3, 8), dtype=np.float64)
        source[..., ::2] = values
        values = source[..., ::2]
    elif kind == 'fortran':
        values = np.asfortranarray(values)
    elif kind == 'mask64':
        labels = labels.astype(np.uint64)
    else:
        values = values.astype('>f8')
    native = resolve_backend('rust')
    parameters = dict(image_geometry=geometry, mask_geometry=geometry,
        config=CONFIG, roi_label=1, families=['firstorder'], intensity_mask=None)
    with pytest.raises((TypeError, ValueError)):
        native.array_features_typed(values, labels, json.dumps(parameters))
    actual = array_features_typed_or_json(values, labels, **parameters)
    assert actual['original_firstorder_Mean'] == 11.5


def test_sample_arc_views_share_storage_are_immutable_and_keep_owner_alive():
    values, labels, _ = arrays()
    image = rad.image.from_array(values, spatial_unit='mm')
    mask = rad.mask.from_array(labels, geometry=image.geometry)
    sample = rad.ImageSample(dict(sample_id='arc', patient_id='p'), image, mask)
    assert sample._owned_arrays is not None
    first, second = sample.image.array, sample.image.array
    mask_view = sample.mask.array
    assert np.shares_memory(first, second)
    for view in (first, second, first.view(), np.asarray(first), mask_view):
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view.setflags(write=True)
    expected = values.copy()
    values.fill(-999)
    image._payload['values'][0] = -100
    snapshot = sample.to_dict()
    snapshot['image']['values'][0] = -200
    np.testing.assert_array_equal(first, expected)
    del sample, image, mask, second
    gc.collect()
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(mask_view, labels)


def test_sample_owned_extraction_and_explicit_dict_path_are_equal():
    values, labels, _ = arrays()
    image = rad.image.from_array(values, spatial_unit='mm')
    mask = rad.mask.from_array(labels, geometry=image.geometry)
    metadata = dict(sample_id='same', patient_id='p')
    owned = rad.ImageSample(metadata, image, mask)
    legacy = rad.ImageSample(metadata, image.to_dict(), mask.to_dict())
    assert owned.to_dict() == legacy.to_dict()
    assert owned.extract(CONFIG).to_dict() == legacy.extract(CONFIG).to_dict()


def test_diagnostic_observes_owned_path_without_disabling_serializer_guard(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    from profile_typed_boundary import trace_boundary
    import ov_radiomics._native_common as common
    import ov_radiomics._types as types
    native = resolve_backend('rust')
    original_owner = native.OwnedArraySample
    original_serializer = types.ImageVolume.to_dict
    with trace_boundary(common, types, native) as (stage, events, phases):
        with stage('validate_image'):
            image = rad.image.from_array(np.ones((2, 3, 4)), spatial_unit='mm')
        with stage('validate_mask'):
            mask = rad.mask.from_array(np.ones((2, 3, 4)), geometry=image.geometry)
        with stage('validate_image_sample'):
            sample = rad.ImageSample(dict(sample_id='trace', patient_id='p'), image, mask)
        assert types.ImageVolume.to_dict is original_serializer
        with stage('array_features'):
            sample.extract(CONFIG)
    assert native.OwnedArraySample is original_owner
    assert types.ImageVolume.to_dict is original_serializer
    assert any(event['kind']=='native_owned_constructor' for event in events)
    assert any(event['kind']=='native_owned_extract' for event in events)
