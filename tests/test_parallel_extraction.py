"""Direction scheduling preserves exact output and an explicit local CPU budget."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import ov_radiomics as rad
from ov_radiomics._backend import try_native
from ov_radiomics._native_common import array_features_typed_or_json
from test_native_extraction import write_nifti

pytestmark = pytest.mark.skipif(try_native() is None, reason='Native extension not built.')
THREADS = min(8, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
CONFIG = dict(profile='python_golden_v1',
    discretization=dict(method='fixed_width', width=1),
    aggregation='direction_mean', gldm_alpha=0, voxel_array_shift=0)


def sample():
    values = (np.arange(60, dtype=float).reshape(3, 4, 5) % 7)
    image = rad.image.from_array(values, spatial_unit='mm')
    mask = rad.mask.from_array(np.ones_like(values, dtype=np.uint32), geometry=image.geometry)
    return rad.ImageSample(dict(sample_id='deterministic', patient_id='p'), image, mask)


@pytest.mark.parametrize('aggregation', ['direction_mean', 'merge_matrices'])
def test_three_parallel_repeats_and_serial_are_array_equal(aggregation):
    config = {**CONFIG, 'aggregation': aggregation}
    current = sample()
    expected = current.extract(config, threads=1).features.drop(columns='ID').to_numpy()
    assert np.isfinite(expected).all()
    for _ in range(3):
        actual = current.extract(config, threads=THREADS).features.drop(columns='ID').to_numpy()
        assert np.array_equal(actual, expected)


def test_rayon_environment_does_not_change_local_pool_result():
    script = '''
import json
import numpy as np
import ov_radiomics as rad
values=np.arange(60,dtype=float).reshape(3,4,5)%7
image=rad.image.from_array(values,spatial_unit='mm')
mask=rad.mask.from_array(np.ones_like(values,dtype=np.uint32),geometry=image.geometry)
sample=rad.ImageSample(dict(sample_id='p',patient_id='p'),image,mask)
config=CONFIG_PLACEHOLDER
print(json.dumps(sample.extract(config,threads=THREADS_PLACEHOLDER).features.drop(columns='ID').to_numpy().tolist()))
'''.replace('CONFIG_PLACEHOLDER', repr(CONFIG)).replace('THREADS_PLACEHOLDER', str(THREADS))
    outputs = []
    for count in ('1', '8'):
        result = subprocess.run([sys.executable, '-c', script],
            env={**os.environ, 'RAYON_NUM_THREADS': count}, capture_output=True, text=True, check=True)
        outputs.append(np.asarray(json.loads(result.stdout)))
    assert np.array_equal(*outputs)


def test_threads_cross_both_typed_and_json_fallback_gateways():
    current = sample()
    image, mask = current.image, current.mask
    parameters = dict(image_geometry=image.geometry, mask_geometry=mask.geometry,
        config=CONFIG, threads=THREADS)
    typed = array_features_typed_or_json(image.array, mask.array, **parameters)
    fallback = array_features_typed_or_json(image.array.astype(np.float32), mask.array, **parameters)
    assert typed == fallback


def test_public_file_extract_passes_threads_and_preserves_result(tmp_path):
    values = (np.arange(60) % 7).tolist()
    write_nifti(tmp_path/'image.nii', values, shape=(5, 4, 3))
    write_nifti(tmp_path/'mask.nii', [1]*60, shape=(5, 4, 3))
    samples = pd.DataFrame([dict(ID='one', patient_id='p',
        image=str(tmp_path/'image.nii'), mask=str(tmp_path/'mask.nii'))])
    serial = rad.extract(samples, backend='rust', bin_width=1, threads=1)
    parallel = rad.extract(samples, backend='rust', bin_width=1, threads=THREADS)
    assert np.array_equal(serial.features.to_numpy(), parallel.features.to_numpy())
    if THREADS > 1:
        with pytest.raises(ValueError, match='budget'):
            rad.extract(samples, backend='rust', bin_width=1, n_jobs=10**6, threads=THREADS)


@pytest.mark.parametrize('threads', [0, 10**6])
def test_invalid_thread_budget_is_rejected(threads):
    with pytest.raises(ValueError, match='positive|budget'):
        sample().extract(CONFIG, threads=threads)
