# Independent M0–M5 completion audit

This is historical version-scoped delivery evidence. Later
[native boundaries](native-boundaries.md), [resource](native-performance-diagnosis.md)
and [scalar accuracy](scalar-accuracy-audit.md) audits must be consulted for new
findings. The new high-dynamic-range scalar counterexample is an open
compatibility/stability limitation, not a passed item in this older audit.

The original `45140bff` audit below is a historical cutoff. The final closure
follow-up at the end supersedes its pending markers where later, separately
identified evidence closes them; it does not rewrite the original observations.

Read-only audit of root source `45140bff46b3ef1ae7eff9a170acb3a1116308b1` and
the version-scoped evidence available with it. The root worktree was clean when
read. This audit adds no algorithm, API, dependency, timing or success criterion.
It compares the original full development plan with the user-approved Benchmark
v2 amendment. Later fixes/measurements need their own evidence update.

**The implementation is substantial, but the entire M0–M5 contract cannot yet be
marked complete at this snapshot.** In particular, final source-distribution
test/tutorial assets, CLI center-specific evaluation, completed ML/Pathology
performance reports, L5 cost measurements and current-source CI remain open.
Performance misses are disclosed targets, not failed numerical definitions.

Status vocabulary: **I** = implemented source inspected; **V** = applicable
recorded verification exists for the stated scope/version; **P** = required work
or acceptance evidence pending; **U** = currently unsupported. I/V never mean
all possible data/configurations or later revisions have passed.

## Architecture and six contracts

| Requirement | Status | Evidence and limit |
|---|---|---|
| One independent repository; core/PyO3/CLI share Rust computation | I/V | Three `ov-radiomics-*` crates; native-only wheels, subprocess CLI tests and Python-free CI container. Existing package/version retained at 0.1.0. No ov source integration or formal publication was executed. |
| Public contiguous buffers, dimensions and explicit missing flags | I/V | `contracts.rs`, contract tests and native adversarial API audit. Matrix/image library types do not appear in public numerical contracts. |
| ImageSample | I/V | Separate sample/patient IDs, image/mask/metadata, scalar grid; repeated patient samples accepted. Geometry and public CT checks. |
| FeatureTable | I/V | Numeric matrix, ordered definitions, metadata, parameters and quality; CSV/Parquet round trips. IDs/outcomes excluded from model features. |
| TileTable | I/V | Patient/slide/tile hierarchy, level-0 geometry, valid pixels, optional MPP, QC and features; zero-tile parent preservation tested. |
| SplitPlan | I/V | Actual sample/patient/cohort/center/class mapping, outer/inner/final-inner sets and RNG identity. Full plan comparison for fusion. |
| ModelBundle | I/V | Schema-one typed `.rad` pipeline/fusion states; feature/class order, threshold/config and provenance. Unknown schema/type and nonconvergence errors; no pickle/joblib native format. |
| PredictionTable | I/V | Actual unit, IDs, probabilities/values, cohorts/folds/training source; hierarchy, uniqueness and overlap checks. |
| Native/reference dependency isolation | I/V | Base requires NumPy/pandas/matplotlib; reference/image/boost extras lazy. Explicit `auto` requires native; no silent external fallback. |

## Images, masks and explicit feature subset

| Requirement | Status | Evidence and limit |
|---|---|---|
| Cohort CSV and scalar NIfTI `.nii/.nii.gz` | I/V | Native NIfTI-1 2D/3D, relative CSV paths, per-sample labels. Unsupported 4D/RGB/complex inputs fail. NIfTI-2/Analyze are declared current format limits, not implied support. |
| Full affine/forms, spacing/unit/origin/direction | I/V | sform precedence/qform diagnostics, eight grid corners, unknown-unit and shear policies; geometry fixtures/public CT. Unknown units are not guessed as mm. |
| ROI bbox, voxel/mm margins, maximal ROI section | I/V core | Public Rust imaging functions and fixtures. Automatic ROI crop steps exist in CLI; do not infer every core operation has a separate Python helper/CLI option. |
| Specified spacing/reference-grid resampling | I/V | Shared linear/nearest image and nearest-only mask operations; explicit generic grid frozen independently of PyRadiomics ROI resampling. Default no resampling. |
| Clip range/percentile, Z-score/MinMax, resegmentation | I/V | Explicit statistics scope, separate morphology/intensity masks; CT resegmentation preserves original morphology. |
| Label selection/recode, connectivity/components, largest component, dilation, mm ring/organ | I/V core | Native functions/geometry tests. Python exposes the common workflow subset; core-only helpers must be described as such. No implicit registration/shear approximation. |
| Outermost parallelism, deterministic sorting/failures/cache/export | I/V | Content/config/implementation/payload hashes, moving-input checks, cache counters and corrupt-cache diagnostics. No silent sample dropping. |
| Original only; explicit bin width/count; 2D/3D, direction mean vs merge | I/V | Public 3.0.1 scalar/raw-matrix/boundary Golden, family-specific native checks; no 2.5D/filter or entire-family claim. |
| `shape_basic` | I/V | Count, physical voxel volume/area, bbox and world centroid; independent geometry arithmetic. Not mesh surface/sphericity or PyRadiomics shape family. |

The **55 supported first-order/texture features** are precisely the following;
their existence is corroborated by the frozen validation matrix and public CT
comparisons, not inferred from a family label:

| Family | Explicit supported names |
|---|---|
| First-order (16) | Mean, Median, Minimum, Maximum, 10Percentile, 90Percentile, Range, InterquartileRange, Variance, MeanAbsoluteDeviation, Skewness, Kurtosis, Energy, RootMeanSquared, Entropy, Uniformity |
| GLCM (10) | JointAverage, JointEnergy, JointEntropy, MaximumProbability, Contrast, Correlation, DifferenceAverage, Id, Idm, SumSquares |
| GLRLM (8) | ShortRunEmphasis, LongRunEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, RunLengthNonUniformity, RunLengthNonUniformityNormalized, RunPercentage, RunEntropy |
| GLSZM (8) | SmallAreaEmphasis, LargeAreaEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, SizeZoneNonUniformity, SizeZoneNonUniformityNormalized, ZonePercentage, ZoneEntropy |
| GLDM (8) | SmallDependenceEmphasis, LargeDependenceEmphasis, GrayLevelNonUniformity, DependenceNonUniformity, DependenceNonUniformityNormalized, GrayLevelVariance, DependenceVariance, DependenceEntropy |
| NGTDM (5) | Coarseness, Contrast, Busyness, Complexity, Strength |

Derived standard deviation is not claimed as a separate nondeprecated reference
feature. Deprecated GLDM normalized gray nonuniformity/DependencePercentage are
excluded, as planned.

## Tables, selection, models and research workflow

| Requirement | Status | Evidence and limit |
|---|---|---|
| CSV/Parquet, explicit roles/joins, external NumPy/array data | I/V | Native table I/O and StudyTableSpec tests; strict key/feature/type handling and precision-safe conversion. Base Parquet path does not require PyArrow. |
| Median/mode/one-hot and Standard/MinMax/Robust fitted state | I/V | Frozen train/evaluation table references, column reorder/unknown category/all-missing tests. No test-statistic fitting. |
| Variance/univariate/BH/correlation/sparse screening | I/V | Raw variance preserved before scaling, stable original-order correlation retention, zero-feature errors, task-aware tests. Native correlation is Pearson; legacy extra methods remain explicit Python options. |
| Binary/regression squared LASSO; multiclass OvR L1 union | I/V | Shared sparse solvers, KKT/support checks and class-renaming invariance. Alpha grids live inside grouped whole-pipeline CV; standalone legacy LassoCV is not silently emulated. |
| LR binary/softmax | I/V | Positive L2, unpenalized intercept, objective/gradient/convergence references. Unpenalized inference uses its dedicated diagnostic workflow. |
| Linear OLS / Ridge regression | I/V | Shared stable solver/SVD/rank checks, frozen predictions/objectives and persistence. |
| KNN classification/regression | I/V | Euclidean uniform/distance, saved training state; frozen three-task predictions. |
| GaussianNB binary/multiclass | I/V | Ordered classes, priors/smoothing and saved state; no regression claim. |
| CART RandomForest three tasks | I/V | Hand tie/split/leaf tests, independent tree seeds, repeat/thread/load checks; one fixed dataset per task, five seeds, 200 trees and unchanged per-task quality gates. |
| Fixed patient holdout, stratified grouped/nested CV, finite grid | I/V | Full per-fit audits; rare-class/missing-class/cohort-overlap errors and frozen inner/outer results. |
| Held-out features/labels cannot alter fits, selector, threshold or OOF | I/V | Independent core/Python/CLI perturbation tests, numeric-ID canonicalization and source provenance audit. Does not authorize tuning using test results. |
| Weighted probability averaging and OOF Logistic/softmax fusion | I/V | Explicit weights, complete shared SplitPlan sets, training-only meta fits. Apparent meta-training vs unbiased held-out evaluation distinguished. |
| Unlabeled external prediction and multiple disjoint cohorts | I/V | Save/load/predict across processes; strict IDs, original memberships, training overlap and external-cohort overlap rejected. |
| Probability calibration evaluation | I/V | Native calibration curves/bins/Brier and full evaluator. No fitted Platt/isotonic calibrator is exposed in the native Pipeline/ModelBundle; do not claim trainable calibration merely from a calibration curve. |
| Multiregion/peritumoral/multimodal concat | I/V | Shared extractor, explicit keys/prefixes/allowed metadata differences; caller-declared prior registration. Tutorial04 and independent frozen concat tests. |
| Delta difference/ratio/relative change | I/V | Fixed patient/lesion/modality/region/time keys, missing/zero/overflow policies and source pairing provenance. |
| KMeans habitat | I/V | Shared scaler/RNG, saved centers/config/IDs/report, exact ties and direct Euclidean reference; empty cluster/nonconvergence errors. Local cluster IDs have no cross-patient tissue identity. |
| Center holdout | I/V | Native SplitPlan center assignment, Python API plus CLI `train --hold-out-center`; external cohorts retained. |
| Center-specific evaluation | I/V Python; P CLI | `evaluate(by_center=True)` validates one known center per actual unit and uses native metrics. CLI EvaluateCommand currently has no `--by-center` flag/center partition orchestration. |

## Native statistics and evaluation

| Requirement | Status | Evidence and limit |
|---|---|---|
| Baselines, t/Welch/MW/KW/chi-square/Fisher2×2 | I/V | Shared Rust tests/CDFs, frozen observed-data references and explicit class/missingness semantics. |
| Univariate/multivariable unpenalized Logit, OR/Wald CI | I/V | Shared numerical solver, complete cases/n_dropped/indices, scaled rank, separation LP and convergence/score/information checks. Patient default rejects repeated patients; sample scope explicit. |
| ICC(2,1)/ICC(3,1) estimate/type/CI | I/V | Complete balanced matrix, reference CI/nonfinite/error cases; no unspecified ICC-threshold API. |
| ROC/PR/calibration/confusion; binary/multiclass summaries | I/V | Frozen curves/thresholds/bin membership, macro/micro/weighted measures and class ordering. |
| MAE/RMSE/R², patient bootstrap and RNG | I/V | Native distributions/quantiles and legacy fixed-index compatibility; separate recorded RNG for new components. Degenerate states retained. |
| DeLong, DCA, H–L | I/V binary only | No silent global multiclass reuse. |

## Lightweight pathology and optional interoperation

| Requirement | Status | Evidence and limit |
|---|---|---|
| RGB8 PNG/JPEG; single-page TIFF Gray8/16/RGB8 | I/V | Real native decoders/type/page rejection; no WSI/vendor decoding claim. |
| Streaming drop/pad tiling; QC raw scores/reasons | I/V | Fixed patches, valid-pixel/padding denominators, retained failures and no file deletion. |
| RGB/HSV/Lab D65 stats/fixed histograms/circular Hue | I/V | Independent small Python/hand references; hue angle not flattened into linear generic means. |
| Gray texture reuse, semantic/instance object statistics | I/V | Same texture/components engine; instance/semantic distinction, patch-local count/centroid/bbox/physical units/boundary checks. Overlap counts are not a whole-slide cell total. |
| ID-aligned embeddings; hierarchical statistics/weights | I/V | Exact IDs, area/physical-scale policies, equal-slide patient defaults, zero-tile/all-excluded states. No fake all-zero features. |
| True tile scores/features → block overlay/coverage | I/V | Global-index half-open sampling, overlap means, NaN/transparent holes, original canvas manifest; patient probabilities cannot be broadcast. |
| AnnData/WSIData optional adapter | I/V scoped | Actual installed dense AnnData/GeoDataFrame/WSIData/TableModel, H5AD, history/ID/center-coordinate and native aggregation smoke passed at pinned0f93. No dependency added to base. |
| MPP and optional-reader limits | I/V scoped | Actual predecoded reader defaults None despite requestedMPP1/20×; explicit cached.5 retained. Pinned upstream reader=None constructor fails; a real predecoded reader is required. No WSI reads/Torch/vendor module loaded. |
| Ordinary pathology CLI scene | I/V | `pathology patches/aggregate/overlay`, seven subprocess E2Es, native training/predict integration. Matrices/metadata may remain in memory; full-slide decoding is not claimed. |

## Interfaces, tutorials and deployment evidence

All six required executable tutorial files exist in `examples/native` and were
run against the immutable installed native-v2 wheel on Windows, with provenance
and result directories under `E:/radiomics-build/snapshot-native-v2/tutorial-*`
(tutorial01 has separate extraction evidence):

| Tutorial | Status | Specific boundary |
|---|---|---|
| 01 NIfTI/ROI extraction | I/V | Image-to-table and CLI parity. |
| 02 imaging+clinical binary | I/V | Shared patient splits, pipelines/fusion, saved models and unlabeled external prediction. |
| 03 multiclass/regression | I/V | Task metrics, bootstrap, identities and persistence. |
| 04 peritumoral/modalities/Delta/given habitats | I/V | Synthetic aligned inputs, explicit pair keys, existing labels and local KMeans. |
| 05 patches→QC/features→patient model | I/V | Ordinary PNG, hierarchy, empty parents, model and tile feature map. |
| 06 external embeddings→aggregation/model/maps | I/V array route | Reordered explicit IDs and genuine separate tile model. Actual WSIData metadata adapter has a separate smoke; do not call this tutorial itself a full vendor WSI workflow. |

| Delivery requirement | Status | Evidence and gap |
|---|---|---|
| Public Python convenience/native namespaces | I/V | Explicit native training/extraction/evaluation/image/mask/research/habitat/pathology/interop; saved .rad models and adversarial API tests. Old/reference choices remain explicit. |
| CLI check/extract/train/predict/evaluate/pathology | I/V | All requested command names present, plus fuse. Center-specific evaluation flag remains the noted gap. |
| Windows wheel/base native operation | V version scoped | Immutable v2 source48ecb310, 64 installed-wheel checks and tutorials; optional/base separate processes. Later Python API additions are not automatically that artifact. |
| Linux/Windows CI install + no-Python CLI container | V older code; P latest | Run43602824324 succeeded at9732b3a with nine jobs, including no-Python/no-network container workflows. Live audit found newer run43612073698 at4a658290 **in_progress**, not passed; no completed run for local45140b was returned. |
| Rust1.85/all targets and core/dependent .crate development artifacts | V snapshot scoped | Actual core verification/tests, dependent local-patch artifacts/tests; registry core is unpublished, so no independent crates.io install claim. |
| Python sdist build/test/tutorial assets | P complete delivery | Staged Rust fixture assets and derived-lock sdist→locked wheel passed on earlier snapshot. Current pyproject explicitly includes docs/tests, not examples/native or root tutorial fixtures; shipped Python tests reference root golden paths. Need final source package asset listing and actual unpacked tests/tutorials, not just a Cargo check. |
| Single support matrix/readme/current docs | P reconciliation | README still says pathology CLI/research pending despite implemented modules; old audit/ledger statements are historical. Native/external/experimental/deferred status must be consolidated around final evidence. |

## Golden, standards and performance policy

| Required evidence | Status | Scope |
|---|---|---|
| Public3.0.1 Python Golden, hand/boundary matrices and original API | V | Frozen parameters/source/dependencies/hashes and layered tolerances; independent adversarial leak/API audit. Original baseline37Python/6Rust remains historical, not current coverage count. |
| Separate3.1.0 difference report | V limited | Ten frozen feature cases, not presumed universal equivalence. |
| Independent IBSI digital track | V scoped | 55 selected features: 54 conformant in declared digital configurations, raw Kurtosis nonconformant; 110 comparisons across aggregation modes. No blanket CT/resampling/2D compliance. |
| Fixed public CT/ROI | V | Pinned/licensed external PAT1 source,396 API/CLI scalar/basic-geometry comparisons,28 exact matrices/NGTDM and buffer/affine checks. Explicit compatibility configuration, not IBSI CT A–E. NC data/derived provenance separate, raw images excluded from repository. |
| Radiomics L1–L4 measurements | V snapshot scoped | 60 declared cases/840 backend observations +14 cache observations, raw/spread/RSS/budget metadata. Real-CT L2 speedup0.400× and L3 1.225× disclose target misses; no favorable overall average. |
| Complete ML & Statistics / Pathology measurements | P report at cutoff | Separate frozen224-attempt plan and untimed validation artifacts exist; no completed result report was in inspected root docs. Warm metric-only benchmark cannot stand in for those complete workflows. |
| L5 deployment cost/startup measurements | P | Independent operation/install correctness has evidence; cold startup/install size/cost/call overhead comparison has not been completed as L5 measurements. |
| Memory and scaling | V declared Radiomics grid; P others |60 median RSS ratios<1.20 within2GiB; synthetic100/500/1000ROI and1/4workers disclosed. Not theoretical maximum throughput or all-image memory bounds; allocator diagnostics remain unmeasured. |

Under Benchmark v2, performance misses require disclosure/optimization, not
altered seeds/data/tolerances, failed algorithm labels or automatic prohibition
of an accurately scoped experimental build. Missing required measurement/report
work still prevents saying the entire agreed plan is finished. Correctness,
numerical stability, API/cohort provenance and applicable deployment gates remain
release requirements. PyPI/crates.io publication, version change and ov source
integration remain separately unauthorized, not work this audit asks to perform.

## Remaining necessary work, in dependency order

1. Close CLI center evaluation using the same native metrics/aggregation and
   explicit metadata validation; do not split an already aggregated multi-center
   patient score or relabel cohorts. Retain the original default behavior.
2. Complete final sdist Python test/tutorial/fixture assets and inspect the real
   archive. Keep NC CT assets outside the base package; make intentionally
   external-only tests explicit. Run enabled tests and the six tutorials from the
   unpacked final source artifact, and preserve root lock/fixture bytes.
3. Finish and independently review the predeclared ML/Pathology224 observations,
   retaining failures/raw data and separate suite reports. Disclose misses under
   v2. Do not replace them with metric microbenchmarks or Radiomics gains.
4. Supply L5 cost/startup/call-overhead evidence, with raw data/versions/scope,
   or leave that performance deliverable explicitly pending; do not call install
   correctness a latency measurement.
5. Reconcile README/support/status documentation and build final artifact
   provenance. Verify the final code-bearing revision's CI/installation results;
   an in-progress run or older green commit is not current acceptance. Pure
   documentation changes can cite demonstrated code equivalence rather than
   inventing another algorithm gate.

## Explicitly deferred items are not missing obligations this round

DICOM sequence/RTSTRUCT, registration optimizers, N4, general WSI/KFB/vendor
decoding, complex stain separation, mesh surface/sphericity, LoG/Wavelet/LBP,
local sliding texture maps, SLIC/GPU habitats, nonstandard topology/fractal/ITH,
ComBat, complex survival, NRI/IDI, deep training, automatic segmentation, MIL,
GUI and paper generation remain deferred. No placeholder or support claim is
required for them. Existing externally prepared results may enter supported
contracts; accepting them does not implement the deferred processing.

## Final closure follow-up: 2026-11-15

This follow-up reads existing source, receipts, raw observations and completed
CI results. It introduces no algorithm changes, new measurements or relaxed
acceptance criteria. Root source inspected for the final packaging notes was
`342d2179d7c5e8127ca697e0b92327b5eb5313dc`. The final code-bearing revision is
`76808bb468f44a586ebf12db5a9be175b40b010d`; the difference from that revision to
the inspected root was only `docs/native-package-assets.md`. Later report-only
commits must retain the artifact/version qualifications below.

| Original remaining item | Final status | Concrete closure and limits |
|---|---|---|
| 1. CLI center-specific evaluation | Closed, I/V | `evaluate --by-center` was merged at `76808bb`. It aligns the original prediction source, validates actual source-unit center metadata, then reuses existing evaluation/patient aggregation and native metrics for each center without renaming cohorts. Missing/unknown centers, mixed centers within an already aggregated patient/slide and invalid provenance are errors; a sample-level patient may genuinely span centers and is partitioned before aggregation. Three focused E2E cases, the full 16-test CLI suite, Clippy and 210 Python tests passed in Root's recorded checks. The independently inspected nine-job CI run at that code revision also succeeded. |
| 2. Final source-distribution assets and execution | Closed within recorded artifacts, V | Root added the six tutorials and licensed frozen fixtures to the source asset list. `E:/radiomics-build/source-delivery-v1/tutorial-evidence.json` records six exit-zero tutorials from the unpacked archive (archive SHA256 `0ec85f15506c8b0a79241644b1eaff35048f05a968f10cb64198aa5a7a24db6a`) using the separately installed immutable-v3 wheel. Final derived sdist receipts additionally record a locked Rust1.85 build and wheel installation, 206 enabled Python tests and all six tutorials; the final v4 native-only wheel has 71 passing installed tests. The same core `.crate` unpacked outside the repository at `D:/radiomics-final-build/core-source-v1` passed 150 tests under Rust1.85. This does not claim an independent crates.io registry installation: core remains unpublished. |
| 3. ML & Statistics / Pathology224 | Closed, measurements and tracked report integration complete | The predeclared 16 cases contain all 224 observations (seven pairs per case), every raw status is `ok`, and `complete_declared_grid=true`. Both suite statuses are `passed_recorded_cases`. The inspected report is `E:/radiomics-build/benchmark-v2/learning-pathology-full-v2-16case-1/report.json`, SHA256 `8d41bc67078339724be24476ce2e40ddeecd65a89c7b9f820eabe69d9652cdc5`; its sibling `raw-observations.jsonl` retains all observations. There are no runtime or memory target misses in this grid. Archive/report commits a6dc123 and 24dde78 integrate the complete measurements, separate result tables and final technical-report links. |
| 4. L5 startup/install/call overhead | Closed for the declared L5 measurement scope, V | `docs/deployment-benchmark-v2-results.md` and `benchmarks/results/deployment-native-45140bff/` retain seven fresh-process CLI starts, seven Python import/backend queries and one offline wheel-payload installation. CLI median is 3.611 ms and Python import/query median 45.052 ms, with their IQR/p95 and all raw values retained. The single payload installation took 0.762398 s; it is an n=1 payload observation, not full dependency or Python/venv installation cost. The endpoints differ, so no language speed ratio is asserted. The first slow CLI startup is retained. OS cache was not purged. Linux startup and complete dependency-install latency remain unmeasured scope limitations. |
| 5. Final code CI, artifact identity and documentation | Closed for recorded code, deployment and report evidence | Completed runs `43615312443` and `43615315617` succeeded at `76808bb`. The former has nine successful jobs, including Windows/Linux native-only checks and a no-Python/no-network CLI container. Immutable-v4 `build-manifest.json` identifies that revision and the unchanged core implementation SHA256 `a39c66ee16dcf04aefd79f2b3996e1be5666b2903b0f40e991aaf5097d54c2a8`. README links now lead to delivery/report documentation; the final report incorporates the completed224 archive and this superseding audit; historical pending markers above retain their original cutoff. |

The final source-package receipts are under
`.build-tools/final-native-source-v1/`. They record
`retained_nodes_unchanged=true`, `repository_lock_unchanged=true`, a passed locked
check and a passed wheel build. The derived wheel SHA256 is
`06dc498f26fcb274e81d02692c1a3740f3e0d9c0c233dc329f6c8b2ccd28d959`.
The original `final-core-crate-v2/preparation.json` remains **failed**: its
Cargo verification extraction was under the Root workspace, selecting the
workspace fixture context. It is not relabeled passed. The separately executed
outside-repository 150-test result closes standalone core-source acceptance;
both pieces of evidence and the path-context diagnosis remain available in
`docs/native-package-assets.md`. External NC public-CT assets and their derived
protocol remain outside base-package asset lists; their earlier independent CT
acceptance is separate from the 206 enabled package tests.

For the completed224 grid, the ten ML/statistics result rows have a maximum
median native/reference runtime ratio of `1.0193622612401778`; the six pathology
rows have a maximum of `0.34673004950728203`. The maximum median RSS ratio over
the 16 cases is `1.0599699021820919`, below `1.20`; the maximum measured worker
footer peak is `201.9375 MiB`. These are per-case results, not an average that
combines the three software suites. The binding identifies source `48ecb310`
and its immutable-v2 wheel, with core implementation SHA256 `a39c66ee…`.
The report records its source-text binding, raw-byte fixture hashes and the
explicitly approved CRLF/LF-only comparison for two reference scripts. Final-v4
installation/CLI evidence is separate; these timings are not silently rebadged
as measurements of every later Python/CLI source change.

Remaining measurement limitations are explicit: no multicore scaling for this
224 grid, no bootstrap/DeLong predictive-CI runtime case, and uncontrolled OS
disk cache/background load. Windows launcher supervision plus sampled/footer
memory accounting establishes the recorded observed resource result, **not
absolute instantaneous memory containment**. An identity-safe shared monitoring
helper is now implemented with four targeted tests, as described in
[its monitoring scope](owned-process-tree-monitoring.md). It has not been wired
into or used to resample this archived grid; no observation is discarded to hide
the original supervision limitation.

The numerical and support qualifications from the original audit continue to
apply: the independently declared IBSI digital track reports raw Kurtosis as
nonconformant rather than changing the Python-compatible definition; the PAT1
configuration is not a claim of IBSI CT A–E conformance; real-CT Radiomics L2/L3
performance target misses remain disclosed. Optional-container evidence covers
the pinned real AnnData/WSIData/GeoDataFrame metadata and predecoded-image path,
not vendor/WSI decoding or a tested full upstream Torch dependency installation.
Unknown physical scale is not inferred from reader defaults. No later report
may turn these scoped passes into universal clinical validity or an assertion
that all features in a family are implemented.

At this follow-up there is no remaining implementation request identified by
this audit within the agreed non-deferred scope. The final archive, technical
report and support-status reconciliation is complete: see the
[224-observation results](learning-pathology-benchmark-v2-results.md),
[technical report](technical-report.md), [support matrix](support-matrix.md)
and [artifact delivery index](delivery-index.md). This closes the development
delivery within its declared evidence, without converting performance misses
or unmeasured extensions into passes. The explicitly deferred feature list
above, formal registry publication, a version-number change and changes to ov
remain outside this round's authorization.
