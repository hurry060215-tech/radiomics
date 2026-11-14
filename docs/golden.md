# Frozen reference: python_golden_v1

The initial Golden set is frozen before the native feature implementation. It is
maintained independently from the feature author and accepted by the main agent.
It covers ten original synthetic image/mask inputs, 55 explicitly named features,
their raw texture matrices, and the existing binary metric API at commit
`2512e16979a544207f84c943e619ef33e02d3dc7`. It is an initial acceptance set;
the manifest lists the remaining data, model, pathology, installation and
performance requirements. Passing these tests does not complete M0–M5.

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
