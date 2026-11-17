# Tutorials and learning routes

[English home](../index.md) · [简体中文教程目录](../../tutorials/index.md)

**The linked notebook narratives are in Simplified Chinese.** Python code,
parameter names and output schemas use the package's public API. Run cells in
order with a current native wheel and the complete repository. Synthetic lessons
01–06 are executed in fresh kernels during documentation builds; actual outputs
are checked before they are rendered. Public downloads and private-input lessons
require their separate execution instructions.

| Lesson | Input and purpose | Interpretation limit |
|---|---|---|
| [01: NIfTI and ROI extraction](../../tutorials/01_extract.ipynb) | Fixed synthetic image/mask, geometry checks and feature tables | Software behavior only |
| [02: Image/clinical binary modeling](../../tutorials/02_image_clinical_binary.ipynb) | Patient-grouped training, OOF scores, fusion and saved models | Synthetic outcomes; no clinical performance claim |
| [03: Multiclass and regression](../../tutorials/03_multiclass_regression.ipynb) | Explicit task/class order and held-out evaluation | Synthetic outcomes; no automatic target guessing |
| [04: Regions, delta and habitat](../../tutorials/04_regions_delta_habitat.ipynb) | Explicit pairing and local clustering | Declared settings only; not automatic registration |
| [05: Patch to patient](../../tutorials/05_patch_patient.ipynb) | QC and tile/slide/patient hierarchy | Ordinary images; not a universal WSI reader |
| [06: Embedding and overlay](../../tutorials/06_embedding_overlay.ipynb) | Identity-aligned arrays and true tile coordinates | Row count alone cannot establish alignment |
| [07: Public CT](../../tutorials/07_public_ct.ipynb) | One real IBSI PAT1 CT/ROI pair | Explicit noncommercial license acceptance; no outcome model |
| [08: Public tabular modeling](../../tutorials/08_public_tabular.ipynb) | UCI WDBC measurements and genuine diagnosis labels | Source-record IDs do not prove patient independence |
| [09: Local image extraction](../../tutorials/09_local_image_features.ipynb) | Externally supplied images and masks | Code-only publication; keep private data outside Git |
| [10: Local modeling](../../tutorials/10_local_modeling.ipynb) | External feature tables and genuine outcomes | Requires explicit cohorts and patient identity |

Read [data provenance and licensing](../../datasets.md),
[local-data instructions](../../tutorials/local-data.md), and
[validation scope](../../validation.md) before interpreting results.
Public-data lessons are not downloaded or rerun as an implicit documentation
build side effect. Their published historical outputs remain separately identified.
