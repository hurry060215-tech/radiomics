# Legacy optional Python workflows

Historical independent-row binary examples requiring explicit optional Python dependencies. For current patient-grouped native APIs use the [file guide](user-guide/file-workflows.md). The joblib persistence below applies to legacy models; native models use typed `.rad` files.

## Four-step tutorials

1. [Images, extraction and radiomics modeling](../examples/01_radiomics.ipynb)
2. [Clinical baseline and logistic statistics](../examples/02_clinical_statistics.ipynb)
3. [Clinical model training](../examples/03_clinical_model.ipynb)
4. [Fusion, evaluation and nomogram](../examples/04_fusion_evaluation.ipynb)

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

- [API and parameter reference](api.md)
- [Scope and source-workflow mapping](scope.md)
- [Validation evidence](validation.md)
- [Provenance](provenance.md)

```bash
python -m pytest -q
```

The original implementation in this repository is MIT licensed. External adapters
call separately licensed libraries. Native algorithms are independently written
against frozen public specifications/references; they do not incorporate encrypted
Onekey source. Third-party libraries and licensed reference data retain their own
licenses and attribution.
