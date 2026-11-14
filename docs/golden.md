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
