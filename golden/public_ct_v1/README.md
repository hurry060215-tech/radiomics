# Public CT numerical validation reference candidate

This pre-native candidate supplements synthetic and digital-phantom checks with
one fixed public CT/ROI. It does not replace the independent IBSI conformance
track and does not claim to implement IBSI CT configurations A–E. The chosen 3D,
no-resampling, 25-HU fixed-bin-width protocol is an explicit numerical test, not
a clinical CT preset. Native execution is forbidden until main-worker review of
this candidate; reference generation never imports the native package.

The [official IBSI reference-data description](https://ibsi.readthedocs.io/en/master/05_Reference_data_sets.html)
identifies PAT1/GTV-1 and the NIfTI conversion. Pinned source revision is
`6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`. See
[LICENSE_NOTICE.md](LICENSE_NOTICE.md) for CC-BY-NC-3.0, attribution, citations and
distribution boundaries. Raw data stays outside Git and the packaging allowlist.

Source image and mask are 204×201×60. Header units explicitly specify millimeters,
spacing is approximately 0.977×0.977×3 mm, axes X/Y are flipped in RAS, and origin
is nonzero. Image datatype is float32 with scaling 1/0; the mask is uint8 0/1.
Every loaded image voxel is already an exact integer, so the reference applies
no rounding. The image is losslessly cast to float64 before scalar extraction.

The first complete 3D ROI has 125,256 morphology/intensity voxels. The second
retains that morphology mask and explicitly restricts the intensity mask to
inclusive −500…400 HU, giving 114,596 intensity voxels. Each computes precisely
the declared 55 Original first-order/texture features. Independent `shape_basic`
reference values use voxel geometry and the original morphology mask.

Raw matrix acceptance uses the fixed source subgrid starting at xyz=(100,100,26),
size=(16,16,8), exclusive end=(116,116,34). These coordinates were fixed before
native execution. It contains 2,048 ROI voxels. Reference matrices preserve actual
gray and size/run labels and integer nonzero counts; they do not renumber missing
levels. The full/cropped image/mask/discretized buffers are represented by hashes,
not stored source voxel arrays.

All effective extraction settings, enabled feature names, geometry, header
observations, input hashes, public PyRadiomics 3.0.1 source/build fingerprints and
existing numerical tolerances are frozen in the protocol/reference. PyRadiomics
3.1.0 is not used. Matrix counts and buffer digests are exact; NGTDM difference
sums and deterministic features use `atol=1e-10, rtol=1e-8`; geometry uses absolute
1e-6 mm. No tolerance may be relaxed after seeing native output.

```powershell
E:/radiomics-build/golden/env-301/Scripts/python.exe tools/validate_public_ct.py `
    reference --data-dir E:/radiomics-build/public-ct-v1/source `
    --output E:/radiomics-build/public-ct-v1/fresh-reference-candidate.json
```

Existing candidate outputs are refused. Cached downloads are reused only after
exact source-byte length and SHA256 checks. License and original source README
are downloaded alongside the external NIfTI inputs, and input paths inside any
Git worktree are rejected. The reference environment is read, never installed
into or modified by this tool.

All frozen text files use explicit UTF-8/LF bytes on Windows and Linux. A
pre-native candidate correction replaced Windows newline translation and updated
only byte-hash/protocol-hash metadata; numerical values, settings and matrices
were unchanged. The profile-local attributes and byte writer preserve that
canonical representation through Git checkouts and independent rebuilding.
