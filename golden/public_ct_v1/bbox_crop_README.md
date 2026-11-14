# Independent tight-bounding-box reference supplement

This separate reference supplements the original frozen PAT1 protocol without
changing its values or tolerances. The source-specific CC-BY-NC-3.0 notice in
LICENSE_NOTICE.md also applies to this derived numerical record; no raw data is
stored or added to package assets.

Before CT native execution or benchmark timing, public PyRadiomics 3.0.1 was run
again after an explicit xyz start=(51,51,17), size=(100,99,26), margin-zero crop.
There is no clipping, normalization, resampling, rounding or resegmentation.
The native/world affine is updated in memory; no intermediate NIfTI is written.

All 55 scalar results are exactly equal to their full-source counterpart in the
frozen Python environment. The local bounding-box minima become zero; ROI voxel
count, dimensions, world centroid and physical volume retain their original
meaning. The supplement records the complete cropped affine, independent basic
shape values, image/mask input digests and per-feature absolute differences.

`bbox_crop_manifest.json` binds this supplement to the original protocol and
full reference SHA256. Rebuild into a new external candidate with:

```powershell
E:/radiomics-build/golden/env-301/Scripts/python.exe tests/public_ct_bbox_reference.py `
    --data-dir E:/radiomics-build/public-ct-v1/source --output E:/radiomics-build/public-ct-v1/fresh-bbox-candidate.json
```

This is a pre-native reference candidate requiring main-worker review. It is not
an IBSI CT conformance configuration or a performance result.
