# radiomics

[![Tests](https://github.com/hurry060215-tech/radiomics/actions/workflows/tests.yml/badge.svg)](https://github.com/hurry060215-tech/radiomics/actions/workflows/tests.yml)
[![Documentation](https://github.com/hurry060215-tech/radiomics/actions/workflows/docs.yml/badge.svg)](https://github.com/hurry060215-tech/radiomics/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**An independent Rust engine for radiomics and lightweight pathology, with Python APIs and a Python-free CLI.**

[中文介绍](README.zh-CN.md) · [Installation](website/installation.md) · [Tutorials](website/tutorials/index.md) · [Your own data](docs/user-guide/file-workflows.md) · [API](docs/api.md) · [Technical report](docs/technical-report.md)

## What can I do with it?

| Starting point | Workflow | Outputs |
|---|---|---|
| Scalar NIfTI images, ROI masks and clinical tables | Native preparation/extraction → patient-grouped selection and modeling → evaluation | Feature tables, QC, shared splits, `.rad` models and external predictions |
| Ordinary patches, decoded images, existing masks or embeddings | QC/features → tile/slide/patient aggregation → native models | QC tables, aggregation, predictions and actual tile-score overlays |

Binary classification, multiclass classification and regression share the same core. Python converts arrays/tables and draws figures; the CLI does not invoke Python. OmicVerse and other projects can use ordinary imports without owning or being required by this package.

The current Original subset contains **55 named statistical/texture features plus 11 `shape_basic` descriptors**. See the [support matrix](docs/support-matrix.md) for exact boundaries. This is not a DICOM/RTSTRUCT reader, general WSI decoder, segmentation engine or deep-model trainer.

## Install

The distribution is **`ov-radiomics`** and the import is **`ov_radiomics`**, avoiding a collision with PyRadiomics. Version remains **0.1.0**. No PyPI or crates.io publication is implied.

```bash
git clone https://github.com/hurry060215-tech/radiomics.git
cd radiomics
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install .
python -c "import ov_radiomics as rad; print(rad.backend_info())"
```

Source builds require Rust **1.85+** and a platform C/C++ linker. A prebuilt Windows/Linux wheel from a successful [Actions run](https://github.com/hurry060215-tech/radiomics/actions/workflows/tests.yml) can be installed directly without Rust. Base native workflows need NumPy, pandas and matplotlib; they do not require sklearn, SciPy, statsmodels, PyRadiomics, SimpleITK, ov, WSIData, Torch or a GPU. Read the [installation guide](website/installation.md) for platform details, wheel downloads and the separately built CLI.

## Start with an executed tutorial

```bash
python examples/native/01_extract.py --output artifacts/first-extraction
python -m pip install jupyterlab ipykernel
jupyter lab
```

Open a notebook in `website/tutorials/` and run its cells in order. Existing outputs show intermediate objects and figures; a fresh run creates a new result directory.

| Tutorial | Data and lesson |
|---|---|
| [01 · NIfTI + ROI](website/tutorials/01_extract.ipynb) | Synthetic tiny images; inspect affine, parameters and exact 24-voxel ROI |
| [02 · Imaging + clinical binary](website/tutorials/02_image_clinical_binary.ipynb) | Synthetic repeated lesions; shared patient splits and OOF fusion |
| [03 · Multiclass + regression](website/tutorials/03_multiclass_regression.ipynb) | Explicit tasks/classes, native models and external prediction |
| [04 · Regions, Delta + habitats](website/tutorials/04_regions_delta_habitat.ipynb) | Strict keys, anisotropic rings, longitudinal changes and local KMeans |
| [05 · Patches → patients](website/tutorials/05_patch_patient.ipynb) | Synthetic RGB patches; QC, hierarchy and actual tile-feature maps |
| [06 · Embeddings + local maps](website/tutorials/06_embedding_overlay.ipynb) | Explicit tile IDs, shuffled rows and true tile-level probabilities |
| [07 · Real public CT](website/tutorials/07_public_ct.ipynb) | IBSI PAT1 CT/GTV, pinned downloads, ROI preview and native extraction |

The first six are reproducible synthetic demonstrations, not clinical cohorts. The seventh uses real public CT under **CC-BY-NC-3.0**; it has no clinical outcomes and does not invent a classification study. Raw CT stays outside Git. See [data sources and licenses](website/datasets.md).

## Use your own files

File-driven examples consume existing inputs and explicit column roles:

```bash
python examples/real/nifti_extract.py --help
python examples/real/train_table.py train --help
python examples/real/train_table.py predict --help
python examples/real/patch_features.py --help
```

Follow the [input contracts](docs/user-guide/input-contracts.md), [NIfTI/clinical workflow](docs/user-guide/file-workflows.md), [pathology/embedding guide](docs/user-guide/pathology.md) and [output/error guide](docs/user-guide/results-and-errors.md). These explain complete manifests/configurations, expected exports and how to replace the demonstration data. Outcomes and IDs stay outside the feature matrix; patients stay together through inner and outer fitting. A patient probability cannot be broadcast into a local tile prediction map.

Training, selection and clinical APIs retain optional legacy defaults: request **`backend="rust"`** explicitly. `auto` requires the native extension and never silently falls back to a reference algorithm. Native models use versioned typed `.rad` files, with saved preprocessing, feature/class order and threshold. Historical optional-backend examples remain in [legacy Python workflows](docs/legacy-python.md).

## Validation and measured limits

Python compatibility freezes public PyRadiomics3.0.1; IBSI is an independent definition/reference track. Correctness and deployment gates are separate from performance targets. The [technical report](docs/technical-report.md) links numerical fixtures, raw runs, environment identities and limitations.

- **Radiomics:** 854 archived observations. Five speed targets missed; real-CT feature extraction measured approximately **0.400×** reference speed, while complete CT input→CSV measured **1.225×**. Small synthetic speedups are not generalized to clinical workloads.
- **ML/Statistics and Pathology:** 224 separate archived observations with matching outputs; results are limited to their declared workloads and frozen references.
- **IBSI digital track:** 54/55 scoped definitions conformant, with raw Kurtosis explicitly nonconformant. This is not blanket IBSI certification or clinical validation.
- **Deployment:** Windows/Linux native-only wheel checks and a no-Python/no-network CLI container. [Evidence index](docs/delivery-index.md) keeps each result tied to its tested revision.

## Documentation and contributions

The documentation follows the task-oriented organization of [OmicVerse](https://github.com/omicverse/omicverse) and its [tutorial repository](https://github.com/omicverse/omicverse-tutorials), with independently authored examples and evidence. Build a local searchable preview:

```bash
python -m pip install -r website/requirements.txt
python tools/build_website_sources.py
python -m sphinx -W --keep-going -b html website website/_build/html
python -m http.server 8000 --directory website/_build/html
```

See [CONTRIBUTING](CONTRIBUTING.md) for checks and tutorial execution, [security/data handling](SECURITY.md) before reporting private data, and [GitHub issues](https://github.com/hurry060215-tech/radiomics/issues) for reproducible problems. HTML documentation and compiled development artifacts are uploaded by CI; no hosted website or formal registry release is assumed.
