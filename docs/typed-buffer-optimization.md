# Typed array boundary and owned sample buffers

This change reduces boundary overhead for array extraction. It preserves the
feature formulas, feature names, explicit nonfinite states and JSON contracts.
The original 854 benchmark v2 observations and their published L2/L3 results
remain unchanged. The timings below are a separate boundary diagnostic.

## Ownership and compatibility

`ImageSample` normalizes the image once into an internal immutable `Arc<[f64]>`.
The mask and geometry also remain in its immutable native owner. Extraction and
sample image/mask views retain that owner and share its buffers. Array views from
the sample are read-only and retain the owner after the Python sample is deleted.
Callers can use `.array.copy()` when they need an independently writable array.
`to_dict()` still materializes an independent dictionary, including independent
image, mask and geometry values. Explicit custom serializers are still honored.

The private native `array_features_typed` entry accepts three-dimensional
float64 images and uint32 masks in C layout. Rust rejects other dtypes, byte order
or layouts. The private Python dispatch falls back to the existing JSON request;
it does not repair layout before calling Rust. Construction explicitly includes
normalization copies inside its timing. The core borrows the normalized image
slice through `ImageView`; existing owned-vector public contracts remain valid.

## Matched five-stage diagnostic

The same fixed PAT1 input was used: shape `[100, 99, 26]` in x/y/z order,
125256 ROI voxels, full prepared RAS affine and the original CT66 configuration.
Each path had one warmup followed by seven measurements. Both used Python 3.9.21,
NumPy 1.23.0, pandas 1.3.5 and threadpoolctl 3.6.0 on the same Windows machine
with an Intel Core Ultra 9 285K. Numerical thread pools were limited to one.
Every output matched all 66 frozen reference values at the original tolerances.

The baseline used the unchanged current-main Python source at `20b7fc0` with its
existing compiled core. The candidate used the installed wheel built from this
change. The diagnostic script was committed at `e0b72a3`; the raw report's
`source_revision` identifies this script checkout, not the baseline installation.
The actual measured installations and binary hashes are separately recorded.

| Same-call exclusive phase | Current-main baseline median ms | Candidate median ms |
|---|---:|---:|
| validate_image | 53.9591 | 54.6290 |
| validate_mask | 26.2976 | 27.6975 |
| validate_image_sample | 69.3236 | 12.5009 |
| array_features | 147.0857 | 100.3762 |
| validate_feature_table | 0.2955 | 0.2567 |

Feature table construction is nested inside `sample.extract`. The raw report
retains inclusive clocks and the exclusive times above. Independent phase
medians must not be added to manufacture a particular call's duration.

For each complete instrumented call, the script records its actual Python
encoding, decoding and deepcopy clocks, divides their sum by that call's total
time, and then reports the median of those seven fractions:

- Current-main baseline: **43.4559%**.
- Candidate: **27.5599%**.
- Historical earlier-snapshot diagnostic: **52.59%**, retained as historical evidence.

Uninstrumented complete-array medians were 296.4614 ms and 194.5151 ms respectively,
a **1.5241x** speed ratio in this diagnostic. Instrumented medians were 297.2490 ms
and 195.9748 ms. This does not replace formal L2 394.163 ms or L3 183.807 ms, and
does not establish a new comparison with the reference backend.

## Reproduction and evidence

Run the same script separately with the baseline and candidate interpreters:

```text
python tools/profile_typed_boundary.py --fixtures <fixture-directory> --output <new-output-directory> --repeats 7 --warmups 1 --acknowledge-license
```

The output directory must be outside Git. The script writes no image pixels,
mask arrays or request buffers. It preserves raw rounds and real stage events,
including explicit normalization in the constructor. It never monkeypatches
`to_dict`, so serializer-identity checks keep the production fast path active.

The accepted run's external reports have these SHA256 identities:

- Baseline report: `49857ad07d550bee3a762c73fac5d27fca92d3a1d1bda00d811bb781efc6f24a`.
- Candidate report: `5b25a2a627455b47ff2c1001216af503b92aee49a5e227ba3a091ce0928c2039`.
- Candidate wheel: `720c1478d3a54e5e1e9bc8a98a509aca3d831f4e63e03ddcddae4528c2654ca8`.

The raw reports include platform, CPU, numerical dependency versions, native
binary SHA256, implementation identity, frozen reference SHA256 and all rounds.
They are diagnostic evidence, not a release or cross-platform support claim.
