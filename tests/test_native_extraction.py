"""Image-to-feature acceptance that needs no Python imaging/model backend."""
import builtins
import json
from pathlib import Path
import struct
import subprocess
import sys

import pandas as pd
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import resolve_backend


def write_nifti(path, values, shape=(3, 2)):
    header = bytearray(352)
    struct.pack_into('<i', header, 0, 348)
    struct.pack_into('<8h', header, 40, len(shape), *shape, *([1] * (7-len(shape))))
    struct.pack_into('<hh', header, 70, 64, 64)
    struct.pack_into('<8f', header, 76, 1, 2, 3, 4, 1, 1, 1, 1)
    struct.pack_into('<ff', header, 108, 352, 1)
    header[123] = 2
    struct.pack_into('<h', header, 254, 1)
    for offset, value in ((280, 2), (300, 3), (320, 4)):
        struct.pack_into('<f', header, offset, value)
    header[344:348] = b'n+1\0'
    Path(path).write_bytes(header + struct.pack('<' + 'd'*len(values), *values))


@pytest.fixture
def samples(tmp_path):
    if resolve_backend('auto') is None:
        pytest.skip('Native extension is not built.')
    write_nifti(tmp_path/'image.nii', [1, 1, 2, 2, 3, 3])
    write_nifti(tmp_path/'mask.nii', [1]*6)
    (tmp_path/'cohort.csv').write_text('sample_id,patient_id,image,mask\nB,P1,image.nii,mask.nii\nA,P1,image.nii,mask.nii\n')
    return rad.read_cohort(tmp_path/'cohort.csv')


def test_package_import_does_not_import_reference_or_imaging_packages():
    code = ('import sys; import ov_radiomics as rad; '
        'assert not set(["sklearn","statsmodels","scipy","joblib","radiomics","SimpleITK"]) & set(sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_api_threads_and_cache_without_reference_imports(samples, tmp_path, monkeypatch):
    original_import = builtins.__import__
    forbidden = {'sklearn', 'statsmodels', 'scipy', 'joblib', 'radiomics', 'SimpleITK'}
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in forbidden:
            raise AssertionError('Native extraction attempted Python backend import: ' + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    first = rad.extract(samples, backend='rust', profile='python_golden_v1', bin_width=1,
                        n_jobs=1, cache=tmp_path/'cache')
    second = rad.extract(samples, backend='rust', profile='python_golden_v1', bin_width=1,
                         n_jobs=3, cache=tmp_path/'cache')
    pd.testing.assert_frame_equal(first.features, second.features)
    assert list(first.features.ID) == ['A', 'B']
    assert list(first.diagnostics.patient_id) == ['P1', 'P1']
    assert first.features.original_firstorder_Mean.tolist() == [2., 2.]
    assert first.config['implementation_sha256'] == second.config['implementation_sha256']


def test_failures_remain_rows_and_auto_does_not_change_algorithm(samples):
    changed = samples.copy()
    changed.loc[0, 'image'] = 'missing-image.nii'
    result = rad.extract(changed, backend='auto', bin_width=1, on_error='record')
    assert len(result.features) == 2
    assert not result.diagnostics.loc[result.diagnostics.ID == 'B', 'ok'].item()
    assert result.features.loc[result.features.ID == 'B', 'original_firstorder_Mean'].isna().all()
    with pytest.raises(RuntimeError, match='B:'):
        rad.extract(changed, backend='rust', bin_width=1)
    with pytest.raises(ValueError, match='explicit bin_width'):
        rad.extract(samples, backend='auto')
    with pytest.raises(ValueError, match='Unsupported native configuration'):
        rad.extract(samples, params={'imageType': {'Wavelet': {}}}, backend='auto')


def test_native_preparation_and_strict_configuration(samples):
    config = dict(extraction=dict(profile='python_golden_v1',
        discretization=dict(method='fixed_width', width=1), aggregation='direction_mean',
        gldm_alpha=0, voxel_array_shift=0), labels=[1],
        preparation=[dict(operation='resegment', lower=2, upper=3)])
    result = rad.extract(samples, params=config, backend='rust')
    assert result.features.original_firstorder_Mean.tolist() == [2.5, 2.5]
    assert result.features.original_shape_basic_VoxelCount.tolist() == [6, 6]
    config['unsupported_filter'] = 'LoG'
    with pytest.raises(ValueError, match='unknown field'):
        rad.extract(samples, params=config, backend='rust')


def test_per_sample_roi_labels_are_not_silently_ignored(samples, tmp_path):
    write_nifti(tmp_path/'mask.nii', [1, 1, 2, 2, 0, 0])
    changed = samples.copy()
    changed['roi_label'] = [2.0, None]
    result = rad.extract(changed, backend='rust', bin_width=1)
    assert result.features.set_index('ID').original_firstorder_Mean.to_dict() == {'A': 1., 'B': 2.}
    changed.loc[0, 'roi_label'] = 2.5
    with pytest.raises(ValueError, match='without truncation'):
        rad.extract(changed, backend='rust', bin_width=1)
