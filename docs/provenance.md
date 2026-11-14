# Implementation provenance

This repository was written independently for the requested four-stage radiomics/clinical workflow. The supplied notebooks were read to identify user-facing steps and parameter requirements. No claim of a formal clean-room process is made: the implementer did read those notebooks.

No Onekey package source, notebooks, YAML presets, figures, manuscript text, encrypted bytecode, pretrained weights, or datasets are redistributed here. The local encrypted runtime was not decrypted. Only newly written modules, newly written documentation, and synthetic-data tutorials are included.

The legacy external/reference path uses separately installed SimpleITK,
PyRadiomics, SciPy, scikit-learn, statsmodels and joblib; optional adapters include
XGBoost, LightGBM and imbalanced-learn. The native core independently implements
the declared image/feature/model/statistics/pathology algorithms, using Rust
file-format, numerical and serialization dependencies. Python uses NumPy/pandas
for conversion and matplotlib/optional seaborn for plotting. Dependency licenses
remain separate. Golden fixtures identify their public source revisions, locked
environments and applicable data licenses; IBSI reference data are not silently
relicensed as project source.

The paired DeLong implementation uses empirical pairwise placement values, counting tied scores as one half. The two class-specific sample covariance contributions determine the variance of the paired AUC difference. It was written directly from this statistical construction; tests compare it to an independent explicit pairwise-matrix calculation. It is not copied from the source notebook's `delong` module.

The nomogram uses an additive rescaling of fitted logistic coefficients. A coefficient's minimum contribution over probability inputs [0,1] is moved into the baseline intercept; nonnegative remaining contributions are converted into points. The sum maps through the fitted logit exactly. A regression test checks this numerical identity.

Public algorithm references:

- [DeLong et al. (1988), comparing correlated ROC areas](https://pubmed.ncbi.nlm.nih.gov/3203132/)
- [PyRadiomics configuration](https://pyradiomics.readthedocs.io/en/latest/customization.html)
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: stacking and cross-validated predictions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

Synthetic image and clinical-data generators are provided in `examples/demo.py`. Generated AUCs or p-values are software validation artifacts and have no clinical interpretation.

The first Rust migration covered numerical metrics. The subsequent native engine
extends image extraction, modeling/statistics and lightweight pathology, tested
against separately frozen public Golden references and independent hand/IBSI
tracks. No protected source is ported. See [current acceptance status](native-development.md),
[Golden provenance](golden.md) and [Benchmark v2](benchmark-v2.md). Correctness,
standards compatibility and computational performance remain distinct claims.
