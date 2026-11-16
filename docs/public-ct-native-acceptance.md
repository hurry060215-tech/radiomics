# Independent public PAT1 CT native acceptance

以下路径为记录时的本机位置，复现时请替换为自己的工作目录。`$WORKDIR` 表示仓库外的构建、缓存和证据目录；源码相对路径以仓库根目录为基准。这里只替换维护文档的路径，不改写历史测量文件。

The frozen public CT validation passed for immutable native snapshot
`9732b3a4c9991343d9597bf27dcbe5f7be02b597`, core implementation SHA256
`a39c66ee16dcf04aefd79f2b3996e1be5666b2903b0f40e991aaf5097d54c2a8`.
The worker preparing and executing this audit did not implement the image/feature
engine. The main worker independently rebuilt and reviewed the public 3.0.1
reference before native execution. No algorithm, expected numerical value,
tolerance, dependency graph or immutable artifact was changed to obtain a pass.

## Inputs and scope

The source is the [IBSI PAT1 CT/GTV-1 NIfTI dataset](https://ibsi.readthedocs.io/en/master/05_Reference_data_sets.html),
pinned to `theibsi/data_sets@6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`.
The [dataset README](https://raw.githubusercontent.com/theibsi/data_sets/6da96021bc91faf4c0cb7fd7fa56a4225d2064a8/ibsi_1_ct_radiomics_phantom/README.md)
identifies its CC-BY-NC-3.0 license, Philippe Lambin/CancerData origin and Alex
Zwanenburg's NIfTI conversion. Raw image/ROI files stay in an external artifact
directory. New protocol/derived records retain their separate source-specific
notice and are excluded from the package-assets allowlist; no blanket MIT claim
is made over them.

This is an explicitly declared numerical compatibility configuration:
Original 3D, bin width 25 HU, no resampling/rounding/normalization, distance one,
symmetric GLCM, direction-mean aggregation and GLDM alpha zero. It is not an IBSI
CT configuration A–E or a clinical CT preset. Source values are already exactly
integral; lossless float64 conversion in the reference does not round them.

| Case | Morphology / intensity voxels | Processing | Features compared per backend |
|---|---|---|---|
| Original | 125,256 / 125,256 | Native NIfTI reading, unchanged ROI | 55 selected statistics/textures + 11 basic geometric quantities |
| Resegmented | 125,256 / 114,596 | Inclusive −500…400 HU intensity subset; original morphology retained | Same 66 |
| Tight bbox | 125,256 / 125,256 | xyz=(51,51,17), size=(100,99,26), margin zero; no clipping | Same 66, with bbox indices local to cropped grid |

The raw image/mask SHA256 values are respectively
`909129fdf99d5f5fd17bc568b1834d31edce8663ea6746c6020720bfc1d7bbcb`
and `0e9243e473b1255c60e6460df465d0b07b65436d6e0c691fe3d17bca0594ae78`.
The image is 204×201×60 with explicit millimeter units, approximately
0.977×0.977×3 mm spacing, nonzero origin and X/Y reversal in RAS. Full affine,
header/scaling, input digests and effective parameters remain in the frozen
[protocol](../golden/public_ct_v1/protocol.json) and
[reference](../golden/public_ct_v1/reference.json).

## Actual results

All **396/396** scalar/basic-geometry comparisons passed: three cases × 66
quantities × installed Python Rust API and independent Rust CLI. Every CLI/API
value was also exactly equal for the matching core. Sample/patient identities,
column order, native completion/failure status and implementation identity were
checked. Morphological volume stayed 358,681.451871528 mm³ through intensity
resegmentation; the tight crop preserved world centroid/volume while resetting
local bbox minima to zero.

The worst fraction of the predeclared allowed scalar error was approximately
0.00007738 for original mean absolute deviation: absolute error
1.2389023140713107e-10 against allowed error 1.6011437495721096e-6.
This is a comparison under the original `atol=1e-10, rtol=1e-8`, not a changed
tolerance or a speed/clinical result.

Native image/mask buffers for the complete source and fixed xyz=(100,100,26),
size=(16,16,8) subgrid matched the independent little-endian digests exactly.
The discretized uint32 gray buffer also matched exactly. Direction identity and
order, original gray/size/run labels and all nonzero integer counts matched for
**13 GLCM + 13 GLRLM + GLSZM + GLDM = 28 matrices**. NGTDM's eight gray labels and
voxel counts matched exactly, and all difference sums passed the original
deterministic tolerance with observed difference zero.

The Python image and mask geometry, Rust-probe source geometry and cropped
geometry all retained `millimeter` units. Their eight voxel outer-grid corners
had observed world-coordinate error zero against the reference, within the
predeclared absolute 1e-6 mm gate.

The native-only environment had NumPy 2/Pandas 2 and no reference packages.
All 25 installed package/PyO3 binary files were checked against the immutable
wheel manifest before execution. A meta-path guard rejected any attempted import
of PyRadiomics, SimpleITK, SciPy, sklearn, statsmodels, joblib, ov/OmicVerse,
WSIData or seaborn. None was loaded. CLI extraction succeeded with empty PATH.
This records a concrete Python-free invocation, not a claim that Windows had
never had Python installed or a substitute for separate Linux/container gates.

## Harness and provenance

`crates/radiomics-core/examples/public_ct_probe.rs` only calls the existing native
read/crop/discretize/texture-matrix APIs. It asserts the expected implementation
SHA before reading CT data and emits hashes, geometry and original matrices.
It adds no alternate algorithm, dependency or public export. The probe is an
acceptance example outside the explicit distributed-crate source include list.

The acceptance worktree started from the immutable source revision. Its Windows
Git checkout had translated Cargo.lock to CRLF; the working copy was restored to
the exact immutable revision's Git-blob LF bytes before building. That is the
same dependency graph and no committed lockfile diff. The resulting probe's
implementation SHA matched both the immutable CLI and wheel. No temporary
consumer project, altered dependency graph or new Cargo.lock was substituted.

The standalone runner [public_ct_native_runner.py](../tests/public_ct_native_runner.py)
never generates references and requires a fresh external output directory:

```powershell
& "$WORKDIR/native-env/Scripts/python.exe" tests/public_ct_native_runner.py `
    --data-dir $WORKDIR/public-ct-v1/source `
    --cli $WORKDIR/snapshot-native-v1/cli/radiomics.exe `
    --build-manifest $WORKDIR/snapshot-native-v1/build-manifest.json `
    --probe $WORKDIR/public-ct-v1/native-probe-v1.json `
    --output $WORKDIR/public-ct-v1/fresh-native-acceptance
```

Raw native extractions, CLI output/config/cohort tables and the complete receipt
are retained under `$WORKDIR/public-ct-v1/native-acceptance-v1`.
The probe output is `$WORKDIR/public-ct-v1/native-probe-v1.json`.
The checked-in [native acceptance report](../golden/public_ct_v1/native_acceptance_report.json)
retains per-feature errors, units/geometry, exact-count results, immutable file
identity, reference hashes and separate compatibility/IBSI fields. Basic voxel
geometry is identified as an independent geometric reference, rather than a
PyRadiomics mesh-shape feature.

The fixed public CT input requirement is now supported by actual native evidence
for this scope. These results do not establish general IBSI CT conformance,
clinical model performance, resampling/filter correctness beyond other tests,
unmeasured speed/memory targets, cross-platform installation or release approval.
