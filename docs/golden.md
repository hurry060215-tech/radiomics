# Frozen reference: python_golden_v1

The initial Golden set is frozen before the native feature implementation. It is
maintained independently from the feature author and accepted by the main agent.
It covers ten original synthetic image/mask inputs, 55 explicitly named features,
their raw texture matrices, and the existing binary metric API at commit
`2512e16979a544207f84c943e619ef33e02d3dc7`. It is an initial acceptance set;
the manifest lists the remaining data, model, pathology, installation and
performance requirements. Passing these tests does not complete M0–M5.

## Benchmark v2 acceptance policy

Correctness, API contracts, stability, independent execution and cross-platform
installation form the Release Gate. Performance targets are reported separately.
A missed target requires an explicit limitation and optimization work; it does
not automatically invalidate a numerically correct algorithm or prohibit an
experimental release. Publishing and version changes still need separate user
authorization.

Radiomics, `ml_stats` and pathology have independent benchmark suites and reports;
they must not be combined into one total score. Radiomics reports L1 by feature
family without a uniform speedup gate; L2 measures the complete supported feature
set on fixed prepared image/ROI buffers with an engineering regression budget
of Rust time ≤1.10× its matched reference and a ≥1.5× speedup target; L3 measures
the full I/O, preparation, extraction and export API with a ≥1.5× target; L4
reports 100/500/1000-ROI throughput across image sizes, gray levels and thread
counts; L5 covers deployment. Matched peak memory ≤1.20× remains an engineering
protection budget and is disclosed when missed.

Reports retain seven warm repetitions in alternating order plus cold-start
measurements, raw times, median/IQR/p95, CPU, thread count, Rust flags, Python/C
library versions, cache status and single-thread versus scaling results. The
main project [Benchmark v2 specification](benchmark-v2.md) is authoritative.
Existing numerical fixture bytes, hashes and tolerances stay unchanged by this
benchmark-policy update.

`feature_validation_matrix.json` separates `pyradiomics_compatible`,
`ibsi_conformant` and `deviation_reason` for each of the 55 named features.
PyRadiomics compatibility is restricted to the frozen, independently executed
scope. All independent IBSI acceptance entries remain `null` pending the second
track; they are not inferred from Python parity. Raw versus excess kurtosis and
the fixed-width anchor differences are stated explicitly. Project `shape_basic`
descriptors do not claim a complete standard morphology family.

The reference interpreter really imports public **PyRadiomics v3.0.1**. It does
not reuse the installed v3.1.0 extension. Public source tag `v3.0.1` resolves to
`c428a771c8cf682c462162b6babc836150d5d9d6`; the manifest records the PyPI source
archive URL and SHA256, and `reference.json` records the imported Python files
and compiled-extension hashes. Platform-specific extension hashes are provenance,
not a requirement that Linux produce byte-identical binaries.

## What is frozen

- `golden/python_golden_v1/cases.json`: x-fastest input buffers, explicit masks,
  dimensions, and bin width/count. The synthetic inputs are original CC0 data.
- `hand_calculations.json`: an independent six-pixel hand calculation covering
  all 16 first-order features, horizontal GLCM/GLRLM counts, and full-neighborhood
  GLSZM/GLDM/NGTDM matrices. The reference cross-check passes before Rust is used.
- `reference.json`: 2D/3D feature outputs and warnings/errors, full raw matrices,
  explicit merged-direction GLCM/GLRLM outputs, and existing metric outputs from
  `backend="python"`. The metric source files are checked against `2512e16`.
  Sixteen stratified NumPy RandomState bootstrap index vectors are frozen, along
  with H–L membership counts and expected sums.
- `geometry.json`: SimpleITK 2.4.1 linear/nearest results on an explicit reference
  grid, a flipped 2D image and rotated 3D image with anisotropic spacing and nonzero
  origins, cropping, and half-index/boundary probes. NIfTI RAS and SimpleITK LPS
  conversions are explicit. The generic spacing grid preserves the lower voxel
  outer edge: `size=ceil(n*old/new)`,
  `origin_new=origin + direction @ ((new-old)/2)`. This is **not** PyRadiomics's
  ROI-based resampling grid. Extraction defaults to no resampling.
- `pyradiomics_310_differences.json`: a separate exact-value/status/warning
  comparison against the locally installed v3.1.0. It found zero differences
  on these ten synthetic inputs. This does not establish general equivalence.
- `manifest.json`: per-layer tolerances, source provenance, failure behavior,
  fixture hashes, and the remaining acceptance requirements.
- `binning_edges.json`: 15 additional actual PyRadiomics 3.0.1/NumPy 1.23.0
  decimal/nextafter/count/large-constant boundary cases, frozen before the native
  floating-edge repair. Finite positive gray labels must match exactly. At
  magnitude `1e16`, reference count binning errors and width binning produces
  nonpositive ROI labels. Native code explicitly rejects these configurations
  as invalid/unsupported under its positive-label contract; it must not export
  a successful all-zero ROI. Actual reference outputs remain in the fixture.
- `model_inputs.json` and `model_reference.json`: independent solver-level
  references for three fixed synthetic tasks, each with 66 patients and two
  lesions per patient. Forty-eight training patients, twelve test patients and
  six unlabeled external patients retain separate identities. Scaling fits only
  the 96 training rows. References include binary/multinomial LR, OLS/Ridge,
  uniform/distance KNN, Gaussian NB, squared-loss LASSO and multiclass OvR L1
  Logistic selection. Parameters, source revisions, predictions, objectives,
  gradients/KKT, convergence and class order are frozen. They validate numerical
  solvers; nested workflow leakage and model selection still require separate
  acceptance tests.
- `pathology_spec.py`, `pathology_inputs.json`, `pathology_reference.json` and
  `pathology_hand.json`: a small reviewed pure-Python specification, eight RGB8
  patch inputs, three tiling cases, four instance/semantic masks and independent
  hand calculations. QC includes only valid pixels, Laplacian uses interior
  valid pixels, and explicit thresholds decide exclusion. RGB/HSV/Lab_D65
  channel statistics and fixed histogram counts, circular Hue, padding-excluded
  annotation coverage, patch-local object geometry and scale conversion are
  frozen before the Rust pixel implementation. Gray8/Gray16 are read as raw
  image/texture values and require an explicit mapping for QC/colors.
- `forest_reference.json` and `forest_hand.json`: 200-tree forests on each of
  the same three fixed datasets, five fixed model seeds, explicit parameters,
  per-sample predictions and patient aggregates. Two lesion scores are averaged
  before the primary patient-level behavioral metrics. Five hand-derived CART
  cases freeze splits, leaf predictions and the tie rule (lowest original
  candidate feature index, then lowest threshold). Sklearn trees and native
  trees need not be identical. The user's per-dataset AUC/Brier, macro-F1 and
  RMSE limits apply to five-seed means independently, with native repeatability
  across thread counts and serialization checked separately. These small
  synthetic datasets test software behavior, not clinical model performance.
- `table_inputs.json`, `table_reference.json` and `table_hand.json`: six-row
  train-only numeric/category imputation, deterministic mode ties, full one-hot
  encoding with unknown-category errors, Standard/MinMax/Robust scaler states,
  low-variance and stable absolute-correlation filtering, and task-specific
  Mann–Whitney/Kruskal–Wallis/Pearson tests with optional BH adjustment. Hand
  medians, population variance, linear quantiles, exact rank-test probability,
  tie-corrected H and BH values independently check the reference. Entirely
  missing columns default to error; dropping them must be explicit. A screen
  retaining zero features errors instead of relaxing the threshold.
- `statistics_reference.json`: fixed SciPy normal/Student-t/chi-square/F tails
  and quantiles, t/Welch/chi-square/Fisher tests, nonfinite boundary states, and
  Statsmodels complete-case unpenalized binary Logistic inference. Per-variable
  and multivariable used IDs, dropped counts, score, observed information,
  covariance, coefficients, OR and normal-Wald CIs are frozen. A small optimizer
  gradient is not a separation diagnostic; rank deficiency, separation and
  nonconvergence remain explicit inference errors.
- `icc_reference.json`: actual public Pingouin 0.5.5 ICC(2,1) and ICC(3,1)
  outputs on balanced, biased-reader, low-agreement, perfect, constant and
  missing-rating matrices. Only temporary output-format rounding is disabled
  and then restored; no numerical reference is reconstructed from printed
  two-decimal intervals. A separate complete public dependency lock reproduces
  the isolated ICC environment. Hand ANOVA mean squares cross-check the point
  estimates. Negative estimates and bounds remain negative. Perfect agreement
  retains point estimate 1, undefined CI and infinite F as in the reference;
  constant data retains undefined statistics with diagnostics; missing data
  errors. Neither an omitted ICC type nor mixed/incomplete ratings is accepted.
- `separation_reference.json`: complete/quasi/overlap/rank-deficient hand data
  and independent SciPy/HiGHS LP certificates, including 1e±12 feature scaling.
  The augmented design is maxabs-scaled before SVD and LP. Certificate vectors
  are not unique, so feasibility and the optimal signed-margin sum are checked,
  without demanding a particular vector. A positive objective with a strictly
  nonnegative verified certificate detects separation; small positive or
  negative-margin certificates are indeterminate and reject inference.
  An optimal zero objective only means no direction was detected at the declared
  resolution; observed-information and convergence checks are still required.
  One optimal vector cannot classify complete versus quasi separation.
- `workflow_plans.json` and `workflow_reference.json`: explicit 3-outer/2-inner
  patient partitions on the original three-task data, with real sample/patient/
  center membership, every inner preprocessing/filter/model fit, selected grid
  candidate, OOF scores and final train/test/unlabeled-external predictions.
  Every inner fold refits median imputation, population scaling, variance and
  absolute-correlation filtering. Scores aggregate lesion predictions within
  patient first, then average fold-level scores equally. Differences ≤1e-12
  are explicit candidate-order ties. Test labels and features do not fit any
  training state, tuning choice, threshold or OOF prediction.
- `fusion_reference.json`: two fixed-penalty base components on exactly the
  same explicit patient plan. Sample OOF probabilities aggregate to 48 patients
  before fitting the two-column Logistic meta model; explicit equal-weight
  averaging uses the same patient unit. Base sample OOF and final predictions
  remain available for provenance. Meta training predictions are apparent,
  while held-out and unlabeled external patients retain their cohort. Neither
  matching fold integers alone nor broadcasting patient scores to tile maps
  is accepted as alignment.
- `evaluation_reference.json`: patient-level binary and multiclass curves,
  macro/micro/weighted summaries, calibration, regression and a 64-draw binary
  patient bootstrap. Public sklearn 1.0.2 conventions are explicit: ROC starts
  at `max(score)+1`; PR retains thresholds back to its first full-recall point;
  uniform calibration uses `linspace(0,1+1e-8)`, quantile calibration expands
  only the final edge, and `digitize(right=False)` defines membership. Stored
  memberships independently reproduce every reference bin mean. AP's step
  integral and trapezoidal PR AUC have separate names. Native bootstrap must
  preserve the frozen legacy binary sampling indices; new components use the
  documented native RNG scheme. Constant and single-observation R² behavior is
  recorded explicitly instead of guessed.
- `baseline_reference.json`: observed-data binary and multiclass baseline
  summaries with per-variable missing counts and observed denominators, sample
  SD and linear quantiles. Continuous defaults are MW/KW; categorical auto uses
  Fisher only for 2×2 with expected counts below five, otherwise chi-square
  without Yates. Constant/all-missing variables remain explicit undefined
  summaries. The binary fixture explicitly excludes its source's one missing
  outcome during fixture assembly; the public API rejects missing outcomes.
  Sample-level dependency warnings remain visible for these repeated-patient
  numerical examples.

The clinical Logistic fixture reuses lesion rows to validate numerical solvers;
its 94/92 complete cases are sample counts, not independent-patient counts.
Public clinical inference defaults to patient analysis and rejects repeated
patients, or requires explicit sample analysis with a dependence warning. This
scope clarification does not change the frozen numerical fixture bytes.

`Lab_D65` uses the sRGB transfer, the W3C D65 RGB-to-XYZ matrix and the CIELAB
piecewise formula with D65 white. It does not adapt to D50. The profile preserves
the theoretical identity `a=b=0` exactly when `R=G=B`; this prevents an arbitrary
floating-point sign at the fixed histogram's zero edge. Other colors are not
rounded by an epsilon rule. Missing Hue or insufficient Laplacian support stays
missing. Object coordinates use pixel edges at integers and centers at `x+.5,
y+.5`. Object MPP is the **patch-pixel** scale: multiply level-0 MPP by
`width0/patch_width` and `height0/patch_height`. Missing MPP produces pixel units
only. Patch counts remain local even when tiles overlap; this does not create a
deduplicated whole-slide cell count.

The model objectives use sample means, with unpenalized intercepts. L2 Logistic
uses `mean(NLL) + lambda/2 * ||W||²`, mapped to sklearn
`C=1/(n*lambda)`; Ridge uses `SSE/(2*n)+lambda/2*||w||²`, mapped to sklearn
`alpha=n*lambda`. LASSO uses `SSE/(2*n)+alpha*||w||₁`; OvR L1 Logistic uses
`mean(binary NLL)+alpha*||w||₁`, mapped to `C=1/(n*alpha)` separately for each
class and takes the union of selected feature indices. Integer multiclass
indices are labels, never a continuous regression response for sparse selection.
The softmax common-intercept shift is nonidentifiable, so predictions, objective,
gradient and centered intercepts define the comparison. OLS rejects deficient
design rank even though sklearn can return a minimum-norm solution. The fixture
records this deliberate native failure rule.

GLCM is symmetric, distance one, with no physical-distance weighting. Default
GLCM/GLRLM features average independent direction results. The additional
`no_weighting` results merge matrices first; these settings are not synonyms.
The raw matrix fixtures retain missing gray-level slots, actual labels, and
direction vectors in both NumPy array axes and x/y/z. An implementation may trim
trailing all-zero size columns under the manifest's explicit comparison rule.

## Degenerate inputs

Reference classes are invoked directly so their numeric boundary behavior can be
inspected independently from an extractor's ROI policy. Empty ROIs error. A
singleton returns NaN for GLCM/GLRLM and finite results for the other families;
NGTDM coarseness is `1e6`. PyRadiomics 3.0.1's C implementation retains isolated
centers in NGTDM counts with difference sum zero. Constant data with `binCount=4`
or `5` discretizes to gray level `3`, following NumPy histogram's expanded
constant range. Constant first-order skewness/kurtosis are zero in this frozen
reference, while entropy can be a tiny negative value because the reference adds
machine epsilon inside `log2`.

Undefined values remain JSON strings `NaN`, `Inf` or `-Inf` and must match by
location and state. No test may drop nonfinite values before comparison. Warning
strings are provenance; native implementations need equivalent state and
diagnostics, not identical Python warning wording.

## Rebuilding a review candidate

Use an isolated CPython 3.9.21 environment and the exact versions in
`golden/python_golden_v1/requirements-reference.txt`. Do not change a shared
analysis environment. Building PyRadiomics 3.0.1 requires NumPy already installed
and `--no-build-isolation`, because its legacy build metadata omits that build
dependency. Supply a platform C compiler. For example:

```text
python -m pip install numpy==1.23.0 setuptools==79.0.0 wheel==0.45.1
python -m pip install --no-build-isolation -r golden/python_golden_v1/requirements-reference.txt
git worktree add /fresh/baseline-worktree 2512e16
python tools/golden_generate_reference.py --baseline-source /fresh/baseline-worktree --output /fresh/reference-candidate
python tools/golden_geometry_reference.py --output /fresh/geometry-candidate.json
python -m pytest tests/test_golden_reference.py -q
```

Each generator refuses an existing destination. Compare the candidate with the
accepted files and send the numerical, configuration, source-version and failure
differences to the main agent. Feature authors cannot update expected values to
make their implementation pass. Test execution only consumes the accepted files
and validates their SHA256; it never regenerates them.

To repeat the local-version drift report, run
`tools/golden_compare_pyradiomics.py --output /fresh/pyradiomics-310-report.json`
using a separate environment containing v3.1.0.

## Independent IBSI checks

IBSI reference values are separate acceptance evidence. The
[official reference-data specification](https://ibsi.readthedocs.io/en/master/05_Reference_data_sets.html)
describes a 5×4×4 digital phantom with 2 mm isotropic voxels and missing gray
levels. Public CT/ROI and digital phantom data are not yet vendored: explicit
redistribution permission must be recorded first. Repository visibility alone
does not establish a redistribution license.

The [PyRadiomics difference documentation](https://pyradiomics.readthedocs.io/en/stable/faq.html)
explains differences in fixed-width discretization, resampling grid alignment,
and kurtosis. These are recorded definition differences. Acceptance must not
inflate tolerances to hide them or combine IBSI and PyRadiomics into one standard.
