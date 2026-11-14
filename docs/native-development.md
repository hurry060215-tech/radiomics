# Independent native development and acceptance ledger

This work continues the personal repository `hurry060215-tech/radiomics` from
`2512e16979a544207f84c943e619ef33e02d3dc7`. The distribution/import/crate names and
version 0.1.0 are retained. No PyPI/crates.io release, upstream ov modification,
or automatic version change is authorized. This ledger distinguishes implemented
code from independently accepted capabilities; a pending row is not support.

## Architecture and ownership

One repository has a Rust core, thin PyO3 boundary, and a Python-independent CLI.
`crates/radiomics-core`, `crates/radiomics-python` and `crates/radiomics-cli` are
integrated. Public numerical contracts contain owned contiguous
buffers, dimensions, explicit missing flags and ordinary Rust types. Matrix and
image-library types stay private. The main worker owns contracts, dependency
versions, error types, serialization and public exports. Delegated workers use
separate branches/worktrees; changes are reviewed and integrated in dependency
order. Golden maintenance and final numerical acceptance are independent from
the feature/algorithm author.

| Contract | Definition and invariants | Current status |
|---|---|---|
| ImageSample | SampleMetadata, ImageVolume, MaskVolume; unique sample ID in a cohort, patient IDs may repeat; scalar 2D/3D; x-fastest buffers; native NIfTI world affine and declared units | Integrated with image/extraction core, CLI and Python; geometry and Golden checks pass |
| FeatureTable | Row-major NumericBuffer, sample metadata, ordered feature definitions, effective parameters and quality records; outcomes/IDs are separate metadata | Native extraction and CSV/Parquet I/O integrated and tested |
| TileTable | patient→slide→tile, level-0 upper-left x/y, extent distinct from patch dimensions, valid nonpadding rectangle, optional mpp, source/QC, optional named features | Integrated with patch tools, strict adapters, aggregation and overlays; empty-slide parent retained |
| SplitPlan | Cohorts, sample→patient and training class maps, actual outer/inner/final-inner patient sets, ordered classes, seed and versioned SHA256 ordering | Integrated with all nested training/selection and fusion; full patient sets checked |
| ModelBundle | Versioned manifest and typed fitted state with numerical arrays; JSON .rad envelope; no executable serialization | Pipeline/fusion typed states integrated; CLI and Python save/load tested |
| PredictionTable | Prediction unit, IDs, cohorts/folds, training source, ordered class probabilities or one regression value | Integrated with training/evaluation/overlays; wrong-level broadcasting and cohort overlap rejected |

`NumericBuffer` uses one missing flag per element, finite f64 storage and canonical
zero at missing positions. Extraction adapters must convert undefined results to
explicit missing positions and quality records; they must not silently turn them
into valid zeros. Image intensities can retain NaN outside the selected intensity
ROI; feature computation rejects nonfinite selected intensities. Geometry has a
full 4×4 affine in its declared native unit. Unknown units never become presumed
millimeters. Orthogonal-distance algorithms reject shear. Image/mask alignment
checks all outer voxel-grid corners to 1e-6 mm when units are known, including
singleton-axis spacing and direction.

`.rad` schema 1 is a JSON envelope containing `manifest` and a typed `fitted`
state. Its manifest records ordered features/classes, task, binary-only threshold,
package version, SplitPlan SHA256 and training configuration. State validation is
required before writing or after loading. Unknown schema, incompatible typed
state, nonconvergence, invalid buffers and missing named features are errors.
Input columns may be reordered by name. New outputs refuse replacement unless
explicitly requested. Python legacy joblib artifacts remain an external-backend
format and do not count as native model persistence.

SplitPlan generation uses `sha256-group-stratification-v1`, with input-row-order
independence and patient-level rare-class balancing. It supports patients with
multiple samples and even multiple observed classes while preserving the group.
Every training/validation fold must contain all classes; unsatisfiable assignments
error explicitly. Inner CV partitions only their outer training population.
Holdout/external labels never enter split generation. Fusion must compare complete
plans, not fold integers. Changing training labels invalidates the saved class
mapping; changing external labels cannot change its fingerprint.

## Feature disposition and dependencies

| Work item | This round | Acceptance dependency/status |
|---|---|---|
| NIfTI scalar 2D/3D, complete affine/forms/units; CSV cohort; CSV/Parquet tables | Native | Integrated; geometry fixtures and CSV/Parquet round-trip/three-task workflow checks pass |
| ROI bbox, voxel/mm margins, maximal section; spacing/reference-grid resampling | Native | Explicit nearest/linear, mask nearest-only; generic grid frozen separately from PyRadiomics ROI grid |
| Clipping, Z-score/MinMax, explicit statistic ROI, absolute resegmentation | Native | Integrated with separate morphology/intensity ROI and frozen numerical checks |
| Label selection/recode, explicit connectivity, largest component, dilation and anisotropic mm ring/organ intersection | Native | Implemented and tested, including shear/unknown-unit rejection |
| Original 16 first-order, shape_basic and 10/8/8/8/5 selected GLCM/GLRLM/GLSZM/GLDM/NGTDM features | Native, explicit subset only | Independent public 3.0.1 raw matrices and scalar/undefined-state acceptance pass; scoped IBSI matrix is separate |
| Batch parallelism, deterministic sorting, failures/export, content-addressed cache | Native | Outermost n_jobs only; input/config/implementation digests required |
| Numeric median/category mode, one-hot; Standard/MinMax/Robust; constant/variance/correlation/univariate/BH/sparse filters | Native | Core and Python fit states integrated; independent screening/preprocessing Golden and error checks pass |
| Binary/multiclass/regression: LR, OLS, Ridge, KNN; classification GaussianNB; CART RandomForest | Native | Implemented; independent objectives/KKT, convergence, model-quality and persistence checks pass |
| Binary/regression squared-loss LASSO; multiclass OvR L1 Logistic feature union | Native | Shared solver; category-renaming invariant and exact support checks pass |
| Grouped fixed holdout/nested CV/grid, center holdout, feature concatenation, weighted probability fusion and OOF Logistic fusion | Native | Grouped pipelines/fusion integrated with true patient sets and leakage checks; keyed feature concatenation remains pending |
| Baseline/test distributions, Fisher 2×2, unpenalized logistic OR/Wald CI and complete-case reporting | Native | Shared Logistic solver, LP separation/rank/score checks; frozen inference passes; Python enforces explicit patient/sample units |
| ICC(2,1), ICC(3,1) for complete balanced subject×reader matrices | Native | Typed result/CI and degenerate states match frozen reference; public Python API integrated |
| ROC/PR, calibration, confusion, patient bootstrap, macro/micro/weighted classification, MAE/RMSE/R² | Native | Full Rust evaluator/CDF/quantiles/bins/bootstrap integrated; exact legacy sample indices and independent evaluation checks pass |
| DeLong, DCA, H–L | Binary only | Existing Python/reference semantics retained; multiclass must use explicitly scoped binary comparisons |
| Multi-region/peritumoral/multimodal, keyed Delta, KMeans habitat, center-specific evaluation | Native | G pending; no inferred registration or batch correction |
| PNG/JPEG RGB8; single-page TIFF Gray8/Gray16/RGB8 | Native | Integrated with rejection of pyramidal/vendor/unsupported pixel types |
| Patch streaming/padding, raw QC/reasons, semantic coverage, RGB/HSV/Lab D65 features, shared grayscale textures | Native | Core and streaming Python iterator integrated; independent pixel/color/mask Golden pass; padding excluded |
| Existing semantic/instance mask objects; hierarchical mean/std/max/quartiles/area weighting | Native | Core and Python integrated; independent hierarchy/weight/missing-state checks pass; object counts stay patch local |
| Tile score rasterization, overlap means/coverage, transparent or NaN uncovered pixels, chunked output | Native | Owned block iterator and Python NPZ/manifest export integrated; exact global-coordinate coverage checks pass |
| External CSV/Parquet/NumPy features/embeddings and precomputed masks/registration/filter results | External input | Exact IDs/keys required; native model workflow reused |
| AnnData/WSIData | Optional Python adapter | Strict ID/geometry/generation-time history contract implemented and frozen tests pass; live third-party package installation/runtime checks pending |
| sklearn/statsmodels/PyRadiomics and existing extra Python models | Optional reference/external backend | Lazy reference paths; native wheel tested without these packages; they are never called by backend=rust |
| Plotting | Python | Native results/tables with units, provenance and valid coordinates |
| DICOM/RTSTRUCT, registration optimizer, N4, WSI/KFB, complex stains | Deferred | No placeholder implementation or support claim |
| Mesh surface/sphericity, LoG/Wavelet/LBP, local texture maps, SLIC/GPU habitats, nonstandard topology/fractal/ITH | Deferred | Existing externally generated features may be accepted as named external inputs |
| ComBat, complex survival, NRI/IDI, deep training, auto segmentation, MIL, GUI, paper generation | Deferred | No support claim |

Defaults remain Original only, no implicit resampling, explicit bin width/count,
distance one, symmetric GLCM, no distance weighting, direction averaging for
GLCM/GLRLM. Matrix merging is an explicitly different option. 2D and 3D are
separate modes; no 2.5D. PyRadiomics kurtosis and discretization definitions are
not relabeled IBSI. Derived standard deviation, if added, is sqrt(Variance) with
its own name. Deprecated GLDM metrics are excluded.

## Verification gates and current evidence

The baseline passed **37 Python and 6 Rust tests** on 2026-11-14, before changes.
Existing Python code still uses scientific external backends; the new core
contracts do not imply that the legacy Python training workflow is native.

The [Golden manifest](../golden/python_golden_v1/manifest.json) freezes initial
reference provenance, formula conventions, explicit parameters, failure behavior,
hashes and tolerances. [Golden rebuilding](golden.md) describes candidate review.
It is not a declaration that M0–M5 have passed. Remaining references must be
frozen before their algorithm implementation, including fixed three-task model
data/objectives, pathology specifications, licensed IBSI/CT data and performance
datasets. A module author cannot replace expected outputs to make a test pass.

Required numeric rule: abs(actual-reference) <= atol + rtol×abs(reference).
IDs/orders/classes/masks/bins/counts/split sets are exact; coordinates are within
1e-6 mm; deterministic features use atol1e-10/rtol1e-8; resampling/color use
atol1e-6/rtol1e-5; linear/logistic predictions use atol1e-6/rtol1e-5 plus
objective/gradient/KKT/convergence. NaN/Inf states and positions must match.

RandomForest: exact hand-tree tests; repeatability across thread counts and
save/load; one frozen dataset for each of three tasks, five seeds, 200 trees against matched reference
parameters. Per-dataset mean binary AUC loss <=0.03 and Brier increase <=0.03;
multiclass macro-F1 loss <=0.04; regression RMSE increase <=max(0.10×reference RMSE,
0.02×test-target SD). Do not require sklearn's tree sequence for the same seed.

The user-approved [Radiomics Benchmark v2](benchmark-v2.md) replaces the uniform
extraction speed gate. Release Gate covers applicable correctness, API compatibility,
numerical stability, independent operation and cross-platform installation.
Performance Target is tracked separately: L1 reports individual operators without
a uniform speed gate; L2 uses <=1.10× native/reference runtime as an engineering
regression threshold and >=1.5× acceleration as a target; L3 end-to-end targets
>=1.5×; L4 covers 100/500/1,000 ROI throughput/scaling; L5 covers deployment.
The <=1.20× memory ratio remains engineering regression protection. Correct slow
results require disclosure/optimization and do not automatically fail numerical
acceptance or prevent a properly scoped experimental release. Actual publication
still requires separate authorization.

Radiomics, Machine Learning & Statistics, and Pathology have separate suites.
Reports retain raw runs, CPU/toolchain/dependency/thread manifests, cold/warm/cache
states, median/IQR/p95 and sample size; at least seven alternating warm pairs are
required. No data/seed/tolerance changes are allowed after seeing results.
PyRadiomics compatibility and independent IBSI conformance have separate per-feature
statuses and evidence; pending IBSI checks never become assumed passes.
The [complete Radiomics measurement](radiomics-benchmark-v2-results.md) now
retains 840 paired backend observations and 14 cache observations. All applicable
numerical/resource checks passed. Performance targets are mixed: actual CT
prepared-array extraction is slower (0.400× speedup), while CT end-to-end is
1.225×, below the 1.5× target. Small synthetic end-to-end gains are not generalized
to CT. ML/Statistics and Pathology performance remain pending their own complete
measurement; L5 install/startup timing and allocator instrumentation are unmeasured.

| Milestone | Required completion | Status |
|---|---|---|
| M0 | Scope, six contracts, formulas/params, all required reference environments/data | Six contracts and independently frozen image/model/statistics/pathology/research references integrated; scoped IBSI and licensed external CT references verified; remaining suite measurements pending |
| M1 | Python-free NIfTI→preprocessing→Original features→CLI, exact matrix/geometry Golden | Golden, scoped IBSI, public CT 396 comparisons and 28 exact matrices passed; Python-free CLI verified; Radiomics performance measured with explicit misses |
| M2 | Three-task native selection/models/stats/CV/fusion/.rad and leakage audit | Core/CLI/Python integrated; independent three-task nested-fit/leakage/persistence checks pass; remaining full delivery/installation audit pending |
| M3 | Strictly keyed multi-region/modality/Delta/habitat/center workflows | Native keyed concat/Delta/habitat and frozen tests integrated; Python center holdout/per-center evaluation and CLI holdout added with focused validation; final scope audit pending |
| M4 | Independent patch and embedding pipelines, aggregation/maps without GPU/ov | Core/Python/CLI and executable tutorials pass; actual pinned AnnData/WSIData dense metadata paths verified; full Pathology performance pending |
| M5 | Python/CLI, Windows/Linux wheels/crate artifacts, six runnable tutorials, full installation/performance acceptance | Pending |

Final tutorials cover NIfTI/ROI extraction; imaging+clinical binary modeling;
multiclass/regression; peritumoral/multimodal/Delta/existing habitat masks;
patch→QC/features→patient modeling; and embeddings/WSIData→aligned aggregation→
prediction/maps. CLI exposes only genuinely implemented commands. Python native
tests must omit sklearn/statsmodels/PyRadiomics; CLI must run without Python;
Windows/Linux wheel installation and absence of ov/WSIData must be tested.

Historical local checkpoint on 2026-11-14: **121 Python tests**, **143 Rust core
tests plus 6 CLI integration tests**, and **25 installed-wheel native-only
tests** passed on Windows. The latter environment has no sklearn, SciPy,
statsmodels, PyRadiomics, SimpleITK, joblib, ov or WSIData. Python subprocess
tests additionally reject reference imports, and compare cross-process `.rad`
predictions exactly on unchanged f64 inputs. These counts describe the tested
checkpoint, not an all-platform release approval. Basic native plots use
matplotlib; seaborn remains an explicit optional extra. Performance reporting
utilities and a first diagnostic metric measurement are available; full
L1–L5/three-workflow evidence was pending at that historical checkpoint.

Later evidence is version scoped: immutable native v2 (source `48ecb310`) passed
64 installed-wheel native-only checks and the six executable tutorials. The
public CT report records 396/396 API/CLI comparisons plus exact matrix/geometry
checks; actual optional-container results are in
[optional-interop-acceptance.md](optional-interop-acceptance.md). CI run
[43602824324](https://github.com/hurry060215-tech/radiomics/actions/runs/43602824324)
passed all nine jobs at `9732b3a`, including Windows/Linux installation and an
Ubuntu container with neither Python nor network running extraction, training,
model reload, prediction and evaluation. Later API changes require their own
latest-head CI; the older green run is not a blanket approval of later commits.

The later immutable v3 snapshot (`45140bf`) was built with the same native
core identity and explicitly recorded release flags. Its installed Windows wheel
passed 71 native-only tests, including the added table/center APIs and identity
boundaries. The receipt and wheel/CLI are at
`E:/radiomics-build/snapshot-native-v3/`; this does not replace the immutable v2
software used for performance measurement.
