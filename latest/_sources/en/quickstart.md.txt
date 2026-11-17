# Quickstart

[English home](index.md) · [简体中文](../quickstart.md)

Install a matching candidate wheel or build the maintained source as described
in [installation](installation.md). A source tag or Actions artifact does not
mean the package is available on PyPI. Check the actual runtime first:

```python
import ov_radiomics as rad

print(rad.__version__)
print(rad.backend_info())
```

Expect `metrics_backend='rust'` and `native_compatible=True` with the current
native build. If the extension is missing or incompatible with the source tree,
rebuild in the active environment with `python -m maturin develop --release`.

## Extract from a small, explicit array

This synthetic example demonstrates geometry and feature extraction. Its
numbers are software checks, not clinical evidence or recommended imaging settings.

```python
import numpy as np
import ov_radiomics as rad

image = rad.image.from_array(
    np.array([[1., 1., 2.], [2., 3., 3.]]),
    affine=[[2, 0, 0, 10], [0, 3, 0, -20], [0, 0, 4, 30], [0, 0, 0, 1]],
    spatial_unit="mm",
)
mask = rad.mask.from_array(np.ones((2, 3)), geometry=image.geometry)
sample = rad.ImageSample(
    {"sample_id": "sample-1", "patient_id": "patient-1"}, image, mask
)
config = {
    "profile": "python_golden_v1",
    "discretization": {"method": "fixed_width", "width": 1},
    "aggregation": "direction_mean",
    "gldm_alpha": 0,
    "voxel_array_shift": 0,
}
table = sample.extract(config, families=["firstorder"])
print(table.features)
assert table.features.original_firstorder_Mean.item() == 2.0
```

The image and mask share explicit geometry. The mean is 2.0 for these six
voxels. Do not infer alignment from array dimensions alone for real images.

## Run a complete file workflow

From a full source checkout, use a new output directory for each run:

```bash
python examples/native/01_extract.py --output artifacts/first-extraction
python examples/real/nifti_extract.py --help
python examples/real/train_table.py train --help
python examples/real/train_table.py predict --help
python examples/real/patch_features.py --help
```

Provide genuine sample and patient IDs, image/mask pairs, explicit column roles,
observed outcomes and predeclared cohorts. Fit preprocessing and feature
selection within patient-grouped training splits. Held-out and external cohorts
must reuse the fitted training state. Never derive outcomes from filenames or
use held-out outcomes to select a model or threshold.

Continue with the [tutorials](tutorials/index.md),
[input contracts](../reference/user-guide/input-contracts.md),
[file workflows](../reference/user-guide/file-workflows.md), and
[result interpretation](../reference/user-guide/results-and-errors.md).
