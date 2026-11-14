# radiomics

An independently maintained Python package with a Rust numerical core: medical images and ROI masks → radiomics models → clinical models → probability fusion and evaluation.

The canonical repository is [hurry060215-tech/radiomics](https://github.com/hurry060215-tech/radiomics). OmicVerse and other projects can consume the package through ordinary Python imports; this repository is developed and released independently.

Native development is tracked in the [implementation and acceptance ledger](docs/native-development.md).
The [Chinese technical report](docs/technical-report.md) describes architecture,
numerical validation and measured limitations. See the [support matrix](docs/support-matrix.md),
[six executable native tutorials](docs/python-native.md),
[Radiomics benchmark results](docs/radiomics-benchmark-v2-results.md) and
[independent deployment measurements](docs/deployment-benchmark-v2-results.md).
The sections below describe the original Python workflow. New native functionality
is accepted module by module; the development plan is not a claim of full support.

**Historical Python scope:** single-center binary classification with independent
rows and optional external estimators. Current native workflows cover three tasks
and grouped repeated samples. No private patient data or protected Onekey code is
included. Public numerical reference data retain their separately stated licenses.

## Install

From a checkout:

```bash
# Source builds require Rust >= 1.85 plus a platform C/C++ linker.
# A prebuilt wheel does not require Rust on the user's machine.
python -m pip install -e ".[image,test,reference]"
# Optional classifier backends:
python -m pip install -e ".[boost]"
```

The distribution is named **ov-radiomics** and the import is **ov_radiomics**, avoiding a collision with PyRadiomics's `radiomics` import. No PyPI release is implied by this repository.

The Rust core and Python bindings live in `crates/radiomics-core` and `crates/radiomics-python`. Maturin builds an ABI3 extension for CPython 3.9+; CI tests compiled wheels on Windows and Linux. Development wheels are available as CI artifacts, not a promised PyPI release.

## Native engine development

The shared Rust core now implements scalar NIfTI preparation, the explicitly
listed Original feature subset, grouped binary/multiclass/regression pipelines,
native models, clinical tests/inference, evaluation, lightweight patch features,
hierarchical aggregation and tile overlays. Python supplies array/table views and
plots. The independent CLI exposes extraction, training, prediction, evaluation
and fusion, plus pathology patch, aggregation and overlay commands. Keyed
multimodal/region concatenation, Delta and typed KMeans habitats use the same core.
See the [development/acceptance ledger](docs/native-development.md) for scope,
independent Golden evidence and outstanding release gates. This working build is
not the completed M0–M5 release.

```python
import ov_radiomics as rad
print(rad.backend_info())
metrics = rad.binary_metrics(y_true, y_score, backend="rust")
reference = rad.binary_metrics(y_true, y_score, backend="python")
```

`backend="auto"` selects only the compiled native implementation. An absent or
broken extension raises a clear error; choose `backend="python"` explicitly for
the optional reference implementation. `backend_info()` is a read-only probe and
reports `unavailable` when the extension is absent. There are no environment
switches or global backend options.

Use `backend="rust"` explicitly for the native training/selection/clinical APIs;
their compatibility defaults retain the historical optional Python backend.
Native models save as typed `.rad` files and accept unlabeled external cohorts.
Base installation does not require sklearn, SciPy, statsmodels, PyRadiomics,
joblib, seaborn, ov or WSIData. Install `[reference,image]` for legacy workflows;
`[plots]` adds optional seaborn plots. Basic ROC/calibration/confusion plots use
the native core and matplotlib alone. See [build instructions](docs/rust.md).

```python
result = rad.train_models(data, "outcome", feature_names, task="multiclass",
    patient_id="patient_id", split_plan=plan, backend="rust")
metrics = rad.evaluate(result, prediction_unit="patient", backend="rust")
result.save("new-native-results")
model = rad.load_model("new-native-results/model-000.rad")
```

Performance and correctness have separate acceptance in
[Benchmark v2](docs/benchmark-v2.md). No unmeasured extraction/end-to-end speedup
or blanket IBSI conformance is claimed.

The core package does not import SimpleITK/PyRadiomics until image functionality is requested. PyRadiomics has compiled dependencies; the initial local validation used Python 3.9, NumPy 1.23, sklearn 1.0.2 and PyRadiomics 3.1.0. Consult CI and `docs/validation.md` for actually tested combinations. Installation on other Python/NumPy versions is not guaranteed by the minimum dependency declarations.

## Four-step tutorials

1. [Images, extraction and radiomics modeling](examples/01_radiomics.ipynb)
2. [Clinical baseline and logistic statistics](examples/02_clinical_statistics.ipynb)
3. [Clinical model training](examples/03_clinical_model.ipynb)
4. [Fusion, evaluation and nomogram](examples/04_fusion_evaluation.ipynb)

Run the notebooks in order from `examples/`. They generate synthetic images and labels; all resulting scores are software demonstrations, not clinical findings. The same workflow is available as a script:

```bash
python examples/demo.py --output examples/output
```

The demo refuses to overwrite an existing image directory. Choose a fresh output directory for another run.

## Basic usage

```python
import pandas as pd
import ov_radiomics as rad

samples = rad.match_images("study", image_dir="images", mask_dir="masks")
qc = rad.check_images(samples, roi_label=1)
features = rad.extract(samples, params="radiomics.yaml", roi_label=1)

# ID strings must match the image filenames exactly, e.g. P001.nii.gz.
labels = pd.read_csv("labels.csv", dtype={"ID": str})
data = rad.join_data(features.features, labels)
data = rad.split_data(data, target="label", test_size=0.2, random_state=0)

selector = rad.FeatureSelector(
    missing="median", univariate="mannwhitney", p_threshold=0.05,
    corr_method="pearson", corr_threshold=0.9,
    lasso=True, alpha="cv", cv=5,
)
result = rad.train_models(
    data, target="label", features=features.features.columns.drop("ID"),
    selector=selector, models=["LR", "SVM", "RandomForest"], cv=5,
)
metrics = rad.evaluate(result, threshold=0.5)
result.save("results/rad")
```

A screen that retains no features raises a clear error. It does not silently relax the study's selection criteria. For exploratory examples, explicitly adjust `p_threshold`, disable `univariate`, or change `alpha`.

Clinical training uses the same `train_models()` function with `categorical` and `continuous` settings. Keep the exact same patient split for radiomics and clinical models.

```python
oof = rad.fusion_data({"Rad": rad_result, "Clinical": clinical_result}, model="LR")
fusion = rad.fit_fusion(oof)
test = rad.fusion_data({"Rad": rad_result, "Clinical": clinical_result}, model="LR", cohort="test")
test["Fusion"] = fusion.predict_proba(test)
rad.plot_roc(test.y_true, test[["Rad", "Clinical", "Fusion"]])
rad.plot_nomogram(fusion)
```

### OmicVerse entry point

This stays a separate repository and Python package. OmicVerse can import it directly when needed:

```python
import ov_radiomics as radiomics
samples = radiomics.match_images("study")
```

No runtime registration, monkey patching, or changes to OmicVerse are required. If OmicVerse chooses to expose this as `ov.radiomics`, it can use its own normal import mechanism. This repository does **not** claim that current OmicVerse already ships that entry point.

## Method boundaries

- `ID` is unique. Native pipelines preserve repeated scans/lesions with patient-grouped splits throughout nested fitting; the legacy Python workflow still requires independent rows and rejects repeated `patient_id` values.
- Reserve the test cohort before any learned preprocessing or feature screening. Every outer CV fold refits the complete pipeline. LASSO alpha tuning also refits preprocessing within its inner folds.
- AUC-based model comparisons report all folds; there is no search for a favorable test split.
- Fusion trains on training-set OOF probabilities. The held-out test cohort is never used to fit it. Training-cohort predictions are labeled **apparent** performance.
- OOF scores used to fit the fusion model are not themselves unbiased validation of that fusion model. Use the held-out cohort for its evaluation.
- A model chosen by mean CV AUC has selection uncertainty: do not interpret that selected CV score as an independent final validation estimate.
- Clinical baseline summaries report observed data and missing counts. They do not impute data or select features automatically. Select clinical variables on training data only; data-driven selection should itself be nested when estimating CV performance.
- Logistic inference uses complete cases and reports dropped counts. It is unpenalized and distinct from regularized predictive LR. The native public clinical API rejects repeated patients unless a row-level `analysis_unit='sample'` is explicitly requested. No automatic stepwise analysis is performed.
- ROC intervals and paired DeLong tests assume independent patients. Native evaluation aggregates predictions to patients before bootstrap by default; complex multicenter/repeated-measure inference is not implemented.
- Calibration plots do not recalibrate the model. H–L returns its group table and fails when the requested grouping is undefined; a large p-value is not proof of calibration.
- Binary/regression sparse selection uses squared-loss LASSO; multiclass uses OvR L1 Logistic and a union of selected features. Native alpha grids are tuned within patient-grouped whole pipelines. Legacy Python LASSO remains an explicit optional path. A squared-loss score is not a probability.
- Native models use documented mean-loss regularization/objectives and convergence checks. External Python models use their named backend parameters. Neither path claims to reproduce Onekey's encrypted model factory.

## Reference and validation

- [API and parameter reference](docs/api.md)
- [Scope and source-workflow mapping](docs/scope.md)
- [Validation evidence](docs/validation.md)
- [Provenance](docs/provenance.md)

```bash
python -m pytest -q
```

The original implementation in this repository is MIT licensed. External adapters
call separately licensed libraries. Native algorithms are independently written
against frozen public specifications/references; they do not incorporate encrypted
Onekey source. Third-party libraries and licensed reference data retain their own
licenses and attribution.
