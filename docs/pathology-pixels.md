# Native lightweight pathology pixels

`radiomics-core::pathology_pixels` processes ordinary decoded images and existing
masks. It calls the shared image-component, radiomics-feature and scalar-statistics
engines. It has no Python, OpenCV, GPU, WSIData or OmicVerse runtime dependency.
CLI/Python availability is established separately by integration tests.

## Streaming patches and coordinates

`tile_image(&RasterImage, &TileSpec, &SlideMetadata)` returns a `TileIterator`.
Each iterator item contains the same shared `TileRecord` contract plus one owned
`RasterImage`. The source image is borrowed; generated patches are not accumulated.
Image decoding is a separate operation and is not a promise of streaming WSI decode.

`TileSpec` specifies `[width,height]` patch size, `[x,y]` stride, explicit `drop`
or `pad` policy, and a scalar `pad_value`. RGB8 and Gray8 reject values greater
than 255; Gray16 preserves the full `u16` range. Drop yields only complete patches;
pad yields every positive-overlap patch starting at a stride multiple. Output is
deterministically ordered by y, then x. Stride can exceed patch size; uncovered
pixels then remain uncovered.

`SlideMetadata` retains patient ID, slide ID, source, level-0 origin and per-axis
source-pixel scale. `pixel_size0=[2,3]` means one decoded source pixel spans two
level-0 pixels horizontally and three vertically. Values are explicit, finite and
positive. The source's first pixel edge is `origin0`, with x right and y down.
MPP, when present, means **micrometers per level-0 pixel** and is never inferred
from DPI or a magnification string.

`x0/y0/width0/height0` are level-0 pixel-edge coordinates. Nominal width/height
include any padding; `valid_pixels=[left,top,width,height]` identifies actual
source pixels within the patch. All statistics exclude padding. Deterministic tile
IDs use a versioned SHA256 of patient/slide/source identity, explicit origin/scale,
local source position and patch dimensions, preventing collisions across decoded
regions or scale variants. IDs are stable for identical inputs and configuration.

## QC with explicit missing scores

`qc_patch(image, valid_pixels, config)` requires sRGB RGB8. Gray8/Gray16 are not
silently scaled or replicated into RGB. They remain usable through the raw-value
traditional-feature adapter.

`QcConfig::default()` computes scores without excluding any patch. Applications
may explicitly set maximum white, low-saturation or black-border fractions, minimum
tissue fraction, or minimum Laplacian variance. The operation returns raw scores,
denominators, tissue mask, ordered exclusion reasons and an `accepted` flag; it
does not delete files. Equality at a requested limit is accepted.

| Score | Frozen definition |
|---|---|
| `white_fraction` | Fraction of valid pixels whose three RGB channels are all at least `white_threshold`, in 0..255 units. |
| `low_saturation_fraction` | Fraction whose HSV S is at most the requested 0..1 threshold. |
| `black_border_fraction` | Among the union of valid-rectangle boundary bands, fraction whose maximum RGB channel is at most `black_threshold`. Corners are counted once; the denominator is boundary pixels. Band width must be a positive integer. |
| `tissue_fraction` | Fraction satisfying inclusive `S>=tissue_s_min` and `tissue_v_min<=V<=tissue_v_max`. This is a configurable pixel heuristic, not segmentation or tissue diagnosis. |
| `laplacian_variance` | Population variance of the cross-kernel Laplacian on floating BT.601 gray (`.299R+.587G+.114B`), using only centers with all four neighbors inside the valid rectangle. |

The Laplacian kernel is `[0,1,0; 1,-4,1; 0,1,0]`. There is no synthetic border
padding for this score. A valid region smaller than 3×3 has no eligible centers,
so variance is `None`; it is not zero. An explicit minimum then yields
`laplacian_variance_undefined`. `PatchQc` serializes this state as JSON null.
`apply_to_record` projects only finite present scores into `TileRecord.qc` while
retaining exclusion reasons. Export the complete `PatchQc` result as well to retain
undefined states and denominators; the finite-score projection alone is incomplete.

## Color features and circular Hue

`color_features(image, valid_pixels, &ColorConfig, optional_tissue_mask)` returns
RGB, HSV S/V and CIELAB_D65 channel statistics plus circular Hue statistics.
Each ordinary channel has mean, population standard deviation, min/max and linear
q25/q50/q75 quantiles. `ColorConfig.fixed_edges` requires externally fixed,
finite, strictly increasing histogram edges for exactly `R,G,B,S,V,H,L,a,b`.
Bins are left-closed/right-open except the final right endpoint. A value outside
the specified range is an error; edges are never fitted separately per patch.
Outputs include exact counts, denominator and normalized histogram probabilities.

RGB uses 0..255 units; HSV S/V uses 0..1 and Hue degrees in [0,360). Hue excludes
achromatic pixels (`S=0`) and uses a circular mean, mean resultant length R and
circular variance `1-R`. Hue has no arithmetic mean or quantiles. With no
chromatic pixels all circular statistics are missing and histogram probabilities
are missing; exact zero counts and the zero denominator remain. If `R<=1e-12`,
the mean direction is undefined while R/variance remain available. An empty
selected tissue mask returns `status="empty_statistical_mask"`, no channels and
no Hue, rather than invented zero features.

The color profile explicitly assumes sRGB, uses the published piecewise sRGB
transfer function and exact rational sRGB-to-XYZ D65 matrix, then computes CIELAB
relative to D65 `xy=(0.3127,0.3290)`. It does not perform D50 adaptation or ICC
profile conversion. For an exactly neutral RGB triplet (`r=g=b`), a=b=0 is
preserved exactly as a frozen mathematical identity at histogram boundaries.
Other colors are not rounded to force agreement.

## Existing mask statistics

`annotation_coverage` accepts a row-major u32 semantic mask, valid rectangle and
explicit class list, including background if requested. Counts/fractions use only
valid pixels; absent requested classes have zero count with the real denominator.

`object_features` requires explicit `MaskKind::Instance` or `MaskKind::Semantic`
and 4/8 connectivity. Instance mode interprets nonzero labels as existing IDs,
even if their visible pixels are disconnected. Semantic mode calls the shared
`imaging::connected_components` with separate source classes. Padding is ignored.
Objects have source label, area in pixels, centroid, bbox and valid-boundary flag.
Centroids average pixel centers `x+0.5,y+0.5` in the patch's pixel-edge system.
Bounding boxes use `[left,top,width,height]`; size distributions report the same
ordinary statistics as color channels. Empty foreground has an explicit status
and missing size distribution.

Physical area is present only when both explicit per-axis **patch-pixel** MPP
values are available. `patch_pixel_mpp(TileRecord)` derives them from level-0 MPP
times `(width0/patch_width,height0/patch_height)`. `object_features_for_tile` uses
this conversion. An absent axis leaves physical area unavailable; negative,
nonfinite or overflowing scales are errors.

All object counts have `count_scope="patch_local"`. When tiles overlap, the
output includes `patch_local_instance_counts_must_not_be_summed_across_overlapping_tiles`.
These counts do not establish a deduplicated whole-slide cell count without global
instance identities. This module also does not create local score maps by
broadcasting patient-level model probabilities.

## Traditional grayscale features

`texture_features` accepts explicit Gray8 or Gray16, valid rectangle, optional
tissue mask and the shared `ExtractionConfig`. It preserves raw gray values and
calls the sole `features::extract` implementation for first-order/texture outputs.
It does not reimplement discretization or texture matrices and excludes the
separately named `shape_basic` outputs from this adapter. RGB conversion must be
performed explicitly by the caller. Empty selected masks produce the shared
extractor's explicit error; no all-zero feature vector is generated.

## Validation and support boundary

The native tests consume independently frozen A fixtures directly:
`pathology_inputs.json`, `pathology_reference.json` and the hand-check cases in
`golden/python_golden_v1/`. They compare eight patches, three tilings and four
masks, including Hue wrap/undefined behavior, exact fixed-bin histogram counts,
padding, semantic diagonal connectivity, fragmented instance IDs, missing blur
variance, empty tissue masks and anisotropic physical scales. Additional tests
check invalid arguments, tile identity, preserved Gray16 values, shared texture
results and JSON missing-score export. Numerical tolerances remain those in the
Golden manifest; no expected results were generated from the Rust implementation.

No WSI/vendor decoder, stain separation, segmentation, deep-model training, MIL,
global overlapping-instance deduplication or inferred physical resolution is
provided. Performance is reported under the separate Pathology suite specified
in `docs/benchmark-v2.md`; passing numerical tests alone makes no speed claim.

Primary formula sources:
[W3C sRGB/D65 color conversion](https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code),
[OpenCV published HSV/gray formulas](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html),
and [OpenCV cross-kernel Laplacian](https://docs.opencv.org/4.13.0/d5/d0f/tutorial_py_gradients.html).
