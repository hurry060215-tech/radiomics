# Native Original feature subset

The Rust module `ov_radiomics_core::features` computes a fixed subset of Original
features in scalar 2D and 3D images. It accepts the shared `ImageVolume`,
`MaskVolume` and `ExtractionConfig`; it does not call Python or PyRadiomics.
All calculations use the same x-fastest image buffers as the image module.

`extract(image, morphology_mask, intensity_mask, roi_label, config)` returns a
deterministically ordered `BTreeMap<String, f64>`. It validates both geometry and
ROI buffers before calculation. The optional intensity mask must be a subset of
the morphology ROI. Basic shape uses the morphology ROI, while intensity and
texture use the intensity ROI. Empty ROIs, nonfinite selected intensities,
misaligned grids and unsupported profiles are errors. Undefined numerical
features remain NaN; tabular callers must retain their positions as explicit
missing flags rather than silently replacing them with zero.

The profile is `python_golden_v1`. Binning is mandatory: choose either
`{"method":"fixed_width","width":...}` or
`{"method":"fixed_count","count":...}`. Width must be finite and positive;
count must be positive. Width bins are aligned to zero. Missing gray levels keep
their original discrete labels in feature equations. Fixed-count binning expands
a constant ROI's range by ±0.5, following NumPy histogram semantics, rather than
assigning a constant ROI an arbitrary gray label of one.

Floating-point bin edges follow the actual NumPy `arange`/`linspace` and
`digitize` operations frozen in the reference, including representable values
immediately adjacent to an edge. Applying the documentation's idealized floor
formula directly would produce different labels for some decimal widths. The
Rust code generates these edges implicitly and searches them without allocating
a potentially enormous edge array. Shared JSON parsing must enable
`serde_json/float_roundtrip` to preserve these exact input floats.
If a huge constant intensity cannot represent the histogram range, or binning
would produce a nonpositive/out-of-range gray label, the native API returns an
explicit input error; it does not accept an all-zero discretized ROI. The Golden
manifest retains the reference's original failures/nonpositive-label outputs
and documents this rejection boundary.

## Explicit supported features

All names are prefixed `original_` and their family name.

| Family | Supported suffixes |
| --- | --- |
| `firstorder` | Mean, Median, Minimum, Maximum, 10Percentile, 90Percentile, Range, InterquartileRange, Variance, MeanAbsoluteDeviation, Skewness, Kurtosis, Energy, RootMeanSquared, Entropy, Uniformity |
| `glcm` | JointAverage, JointEnergy, JointEntropy, MaximumProbability, Contrast, Correlation, DifferenceAverage, Id, Idm, SumSquares |
| `glrlm` | ShortRunEmphasis, LongRunEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, RunLengthNonUniformity, RunLengthNonUniformityNormalized, RunPercentage, RunEntropy |
| `glszm` | SmallAreaEmphasis, LargeAreaEmphasis, GrayLevelNonUniformity, GrayLevelNonUniformityNormalized, SizeZoneNonUniformity, SizeZoneNonUniformityNormalized, ZonePercentage, ZoneEntropy |
| `gldm` | SmallDependenceEmphasis, LargeDependenceEmphasis, GrayLevelNonUniformity, DependenceNonUniformity, DependenceNonUniformityNormalized, GrayLevelVariance, DependenceVariance, DependenceEntropy |
| `ngtdm` | Coarseness, Contrast, Busyness, Complexity, Strength |

`shape_basic` intentionally has its own family name: VoxelCount,
BoundingBoxMin{X,Y,Z}_index, BoundingBoxSize{X,Y,Z}_voxels,
Centroid{X,Y,Z}_mm and Volume_mm3 (3D) or Area_mm2 (2D). Known meter and micron
units are converted to millimeters. Unknown spatial units produce `_native`,
`_native2` and `_native3` suffixes, with no claim that these are millimeters.
The centroid is in the NIfTI world coordinate system. Bbox indices include voxel
centers from the inclusive minimum through minimum + size − 1.
Volumes use the absolute affine determinant; areas use the norm of the first
two affine columns' cross product. These formulas retain exact parallelepiped
geometry for sheared grids without approximating Euclidean distances.

## Matrix definitions and diagnostic access

Only distance one, symmetric GLCM, no distance weighting and no image filters
are included. `TextureAggregation::DirectionMean` computes GLCM/GLRLM features
in each nonempty direction then averages. `MergeMatrices` sums counts first;
these modes are different. A merged GLRLM's RunPercentage denominator includes
the number of valid directions. GLSZM uses 8-connectivity in 2D and
26-connectivity in 3D. GLDM uses the same full neighborhood, includes the center
in dependence size, and uses the explicitly configured integer `gldm_alpha`.

`discretize` exposes the positive gray label buffer and zero outside ROI.
`texture_matrices` exposes the same matrix implementation used by `extract`:

- GLCM is raw symmetric integer counts per direction, with both actual gray
  label axes. Direction offsets are x,y,z; order matches PyRadiomics' z,y,x
  angle generation after coordinate conversion.
- GLRLM, GLSZM and GLDM use compact explicit `row_values`, `column_values` and
  row-major `counts`. Missing gray-label slots and absent size columns can be
  reconstructed from these axis values without renumbering labels.
- NGTDM stores gray label, center count and sum of absolute neighbor differences.

The reference has several deliberate edge conventions. Entropy adds float64
epsilon inside log2. Kurtosis is the population standardized fourth moment
without subtraction of three. A constant first-order ROI has zero skewness and
kurtosis. Constant GLCM correlation is one. Homogeneous NGTDM coarseness is
1,000,000. PyRadiomics' C matrix code counts an isolated NGTDM center with zero
difference, and removes a GLRLM direction when every geometric line has at most
one ROI element. These executable source behaviors take precedence over
conflicting prose definitions.

## Validation and boundaries

The hand matrix and first-order tests were frozen independently in Golden
commit `5dbbee1` before this implementation. They validate actual counts rather
than only final scalars. Additional tests cover affine flips and nonzero origin,
unknown units, missing discrete gray labels, negative bin boundaries,
resegmentation-mask separation, invalid geometry and nonfinite ROI intensities.
Independent Golden tests also compare all 55 requested first-order/texture
features, merged GLCM/GLRLM, discrete labels and all five raw matrix families
against actual version-pinned PyRadiomics 3.0.1 outputs for nine nonempty cases
and explicit empty-ROI errors. Fifteen additional binning cases check decimal
width/count edges, nextafter inputs, and representability failures. Scalar
checks use the frozen tolerance; integer matrices, labels and floating bin edges
are exact. These fixtures do not establish IBSI conformance or any extraction
speed claim.

No mesh surface area, sphericity, filters, 2.5D aggregation or sliding-window
texture maps are implemented in this module. Scalar 2D and scalar 3D remain
separate extraction inputs; a singleton axis does not request 2.5D behavior.

Selective extraction uses `features::extract_families` with the same counting
and scalar functions. Valid selectors are `firstorder`, `shape_basic`, `glcm`,
`glrlm`, `glszm`, `gldm`, `ngtdm`; duplicates/empty lists error. Unselected texture
matrices are not constructed. `extract` remains the complete supported subset,
with exact same Golden results. Prepared Python samples accept
`sample.extract(config, families=['glcm'])`; batch ExtractConfig has an optional
top-level `families` array. This enables scoped operator measurements without
timing the whole engine and relabeling it a single family. Report actual public
conversion/binning/selected-matrix/scalar work, not invented internal timings.

Reference definitions: [PyRadiomics 3.0.1 source](https://github.com/AIM-Harvard/pyradiomics/tree/v3.0.1/radiomics),
[documented IBSI differences](https://pyradiomics.readthedocs.io/en/stable/faq.html).
