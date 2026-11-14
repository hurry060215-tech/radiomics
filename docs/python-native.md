# Native Python workflows

Install a compiled development wheel, then `import ov_radiomics as rad`.
The core requires no sklearn, SciPy, statsmodels, PyRadiomics, SimpleITK, joblib,
ov or WSIData. NumPy/pandas handle conversion; matplotlib draws the basic plots.
Seaborn and external/reference packages are explicit extras. A native request
does not call an external estimator. `auto` requires a verified native extension;
an unbuilt source checkout must choose `backend='python'` explicitly for the
optional reference path. `backend_info()` probes availability without running it.

## Complete executable examples

Each script requires a **new** output directory and uses original synthetic/CC0
acceptance data. Scores are software demonstrations, not clinical evidence.

| Tutorial | Run from the repository root | Main outputs |
|---|---|---|
| NIfTI + ROI | `python examples/native/01_extract.py --output artifacts/native-01` | Scalar 3D features, geometry/quality, optional CLI parity via `--cli` |
| Imaging + clinical binary | `python examples/native/02_image_clinical_binary.py --output artifacts/native-02` | NIfTI pairs, extracted table, shared patient splits, two pipelines, OOF fusion, `.rad`, unlabeled predictions, patient ROC |
| Multiclass and regression | `python examples/native/03_multiclass_regression.py --output artifacts/native-03` | Softmax/Ridge, patient bootstrap, separate external identities, saved models |
| Regions, modalities, Delta, habitat | `python examples/native/04_regions_delta_habitat.py --output artifacts/native-04` | Anisotropic mm ring, explicit key joins, paired Delta, existing labels, local fitted KMeans state |
| Patches to patients | `python examples/native/05_patch_patient.py --output artifacts/native-05` | RGB8 PNG decoding, streamed QC/colors, hierarchy tables, empty-slide status, patient model, real tile-feature overlay |
| External embeddings and maps | `python examples/native/06_embedding_overlay.py --output artifacts/native-06` | Reversed-row ID alignment, slide/patient summaries, separate tile-unit model, real local probabilities, rejected patient broadcasting |

Tutorials 02/03/05/06 consume the versioned synthetic JSON fixtures in this
repository, without importing their reference implementations. See
[strict WSIData adaptation](interop.md) for optional existing containers. The
external-array tutorial does not claim that a real WSIData installation was tested.

## Training and persistence

```python
selector = rad.FeatureSelector(task='multiclass', backend='rust',
    univariate='auto', sparse_method='ovr_l1_logistic', alpha=0.01)
result = rad.train_models(data, 'outcome', feature_names, task='multiclass',
    patient_id='patient_id', split_plan=plan, selector=selector, backend='rust')
metrics = rad.evaluate(result, prediction_unit='patient', backend='rust')
result.save('fresh-results')
bundle = rad.load_model('fresh-results/model-000.rad')
predictions = bundle.predict(external_without_outcome, cohort='external_B')
```

The compatibility defaults for training/selection/clinical functions retain the
legacy Python path; specify `backend='rust'`. Native supported models are LR
(binary/softmax), Linear and Ridge (regression), KNN, GaussianNB (classification),
and RandomForest. Parameters follow the documented native objectives; sklearn
`C` is not silently reinterpreted as a mean-loss lambda. Unknown models and
unsupported parameters fail. See [native model definitions](models-native.md).

An explicit binary class order defines negative then positive. Numeric labels
are normalized without changing string IDs; an explicitly supplied positive label
must agree with that order. Without an explicit order, the binary default positive
label is 1. Sample and patient IDs stay outside the matrix. Repeated lesions are
grouped through all inner/outer fits. Identified feature/target Series must have
matching indices. New patients cannot be marked as original train/test patients;
saved cohort identities cannot be relabeled for prediction.

Fixed sparse alpha may be fitted directly on isolated training data. An alpha
grid is evaluated inside whole patient-grouped native pipelines using the
declared task metric; it is not a hidden standalone pooled LassoCV. All-missing
columns default to errors, explicit deletion is available, and zero-feature
results never trigger threshold relaxation.

`evaluate(result)` infers the engine from an already typed result; an explicit
`auto` only accepts native results. Native evaluation defaults to patient means
(tiles first within slide, then equal slides), uses the saved binary threshold,
and labels apparent/OOF/holdout/external provenance. A standalone binary
PredictionTable needs an explicit threshold. The DataFrame includes macro/micro/
weighted summaries, with full confusion/per-class native reports in
`metrics.attrs['native_reports']`. `PredictionTable.aggregate_patients()` retains
OOF/training-source semantics and is idempotent for patient-unit predictions.

Portable supervised models are typed `.rad` JSON containers, never pickle or
joblib. Pipeline `bundle.predict(DataFrame)` accepts named reordered features
without an outcome. Fusion bundles use `predict_study` with exact named component
studies. Missing/unknown schemas, types, columns and nonconverged state fail.
Default outputs refuse replacement; `overwrite=True` is explicit.

## Image, research and pathology objects

`rad.image.from_array` uses NumPy `(y,x)` or `(z,y,x)` axes and a full affine;
unknown units never become assumed mm. `rad.mask` provides explicit label,
connectivity and distance operations. `ImageSample.extract(config)` extracts
already prepared arrays. Its optional `families` selects supported families and
skips unselected matrix work, using the same frozen engine. Batch `rad.extract`
uses a native ExtractConfig, with the same optional top-level `families` field.
`ExtractionResult.config['cache_summary']` reports actual cache attempts/hits/
misses/rejections/write failures separately from deterministic feature payloads.
Failures before a cache lookup do not count as a hit or miss.

`rad.research.concat` requires exact explicit keys and declared allowed modality/
region differences. `rad.research.delta` pairs patient/lesion/modality/region and
explicit timepoints; zero denominators default to errors. `rad.habitat.fit`
saves native scaling and local cluster centers in `radiomics.habitat.v1` JSON;
it is an unsupervised state, not a three-task `.rad` classifier. Failed convergence
and empty clusters are errors. Cluster labels do not identify shared tissue types.

`rad.pathology` reads ordinary RGB8 PNG/JPEG or supported single-page TIFF,
streams patches, returns raw QC and reasons, uses fixed sRGB/D65 histograms and
circular Hue statistics, and reuses native gray textures and mask components.
No files are deleted by QC. Explicit tissue masks must match patch axes and
contain finite Boolean/binary values. Padding never enters denominators. Instance
statistics remain patch local even in overlapping tiles. Hierarchical aggregation
retains all-excluded and zero-tile parents; it never emits fake all-zero features.
Local overlays accept actual tile features/scores/PredictionTables and return
owned blocks with NaN/transparent missing coverage and counts. Patient probabilities
cannot be broadcast into local maps. The original global canvas is exported.

## Clinical units

Native `baseline_table`, `multivariable_logistic` and `univariate_logistic` require
nonempty patient identities. Default patient inference rejects repeated patients;
`analysis_unit='sample'` is an explicit row-level scope, not evidence of patient
independence. Logistic uses complete cases and reports exclusions, unpenalized
OR/Wald intervals, separation/rank/convergence diagnostics. Baselines summarize
observed values without imputation and reject missing outcomes. `rad.icc(matrix,
kind='ICC(2,1)')` or `ICC(3,1)` requires a complete balanced subject×reader matrix.
