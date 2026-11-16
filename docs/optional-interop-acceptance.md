# Actual optional AnnData/WSIData container acceptance

<!-- PR-2: replace recorded machine paths with reproducible work-directory placeholders. -->

Actual installed-container smoke validation passed on Windows, separately from
the existing source audit and duck-typed fixtures. No native algorithm, public
contract, package version, ov source, base dependency declaration or reference
environment was modified. The executable validator is
[validate_optional_interop.py](../tools/validate_optional_interop.py); it installs
nothing and refuses an existing evidence output directory.

## Environment and verified scope

The dedicated environment was created with
an existing isolated CPython 3.11.15 interpreter.
<!-- PR-2: document the dedicated environment location relative to the workspace. --> It does not inherit system site
packages. The relevant tested versions were:

| Component | Actual version/source |
|---|---|
| Python | 3.11.15 |
| AnnData | 0.12.6 |
| SpatialData | 0.7.3 |
| GeoPandas / Shapely | 1.1.1 / 2.1.2 |
| NumPy / Pandas / SciPy | 2.3.5 / 2.3.3 / 1.16.3 |
| OpenCV headless | 4.11.0.86 |
| WSIData | 0.12.0, exact Git revision `0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286` |
| ov-radiomics | Immutable native v2 wheel, version 0.1.0 |

The validator checked installed WSIData's PEP 610 `direct_url.json` commit and
all 25 native-package Python/PyO3 files against the immutable build manifest.
Core implementation identity was
`a39c66ee16dcf04aefd79f2b3996e1be5666b2903b0f40e991aaf5097d54c2a8`.
The optional stack was installed only in this environment; external wrapper libraries, the base
native environment, public 3.0.1 reference and benchmark environments were not
modified by the optional installation.

WSIData's [pinned metadata](https://github.com/rendeirolab/wsidata/blob/0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286/pyproject.toml)
requires Python >=3.11 and declares Torch/OpenSlide dependencies. For this bounded
metadata/container smoke it was installed from the exact revision with
`--no-deps`, after installing the necessary scientific/SpatialData stack.
Torch, torchvision, OpenSlide bindings, CUDA/CuCIM and vendor readers were absent.
The reviewed package lazily imports deep-model code, so actual container imports
and the following operations worked without it. This is not a claim that every
declared WSIData dependency is satisfied or that its full decoder/dataset/model
surface works in this reduced environment.

## Actual AnnData → native hierarchy

The positive test constructed a real `anndata._core.anndata.AnnData` with a dense
6×2 X, explicit shuffled `obs.tile_id`, independent original observation names,
ordered `var_names` and stored `uns`. Conversion through `from_anndata` matched
the independently frozen alignment exactly, including missing-value locations.
The resulting native slide/patient aggregation matched the frozen hierarchy
reference under its existing tolerance.

The actual object was saved to H5AD and loaded again as in-memory AnnData; its
conversion remained identical. Adapter calls did not mutate `uns`. Duplicate tile
IDs, missing tile IDs and actual SciPy CSR X all produced explicit errors. Sparse
input was not silently densified. Backed/lazy matrices were not validated by this
smoke and no support claim follows for them.

## Actual WSIData geometry, history and aggregation

The positive test used real `WSIData`, `GeoDataFrame`/Shapely polygons,
`ShapesModel`, `TileSpec`, `TableModel` and AnnData objects. It used the upstream
`SpatialDataImage2DReader` over an already decoded synthetic 20×24 RGB image.
No WSI file, vendor reader, foundation model or pretrained embedding was used.
Profiling observed **zero calls** to pixel-reader create/read/thumbnail methods.

The pinned [WSIData constructor](https://github.com/rendeirolab/wsidata/blob/0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286/wsidata/_model/core.py)
unconditionally accesses `reader.properties`; actual `reader=None` construction
fails even with cached slide metadata. This limitation was reproduced and
recorded. No `__new__` bypass or monkeypatch of upstream classes was used. The
real predecoded-image reader supplies metadata without WSI decoding, which is
the validated path here.

The geometry used three authoritative rectangles and deliberately center-valued
x/y columns. Native output used the polygons' left/top coordinates. Changing only
those nonauthoritative x/y columns did not move a tile or change the verified
result. The nonsquare height/width correctly became native level-0 width/height
24×20. SpatialData's actual Identity transformation API was exercised.

A native generation snapshot was created before the synthetic embedding table.
The table retained it as a JSON string in `uns`, with actual SpatialData
region/instance linkage. Shuffled observations aligned by explicit tile IDs and
produced a native patient e0 mean of 20.0. Further row permutation and H5AD
round-trip preserved alignment. Original WSI metadata and saved history were not
mutated by conversion.

The following negative cases produced the expected explicit errors:

- Missing historical generation manifest, without retroactively stamping one.
- Wrong embedding IDs with the same tile count.
- Shifted polygons with the same IDs/count, invalidating the saved geometry.
- Unknown MPP when physical-area aggregation was requested.

## Unknown physical scale remains unknown

The actual [predecoded-image reader](https://github.com/rendeirolab/wsidata/blob/0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286/wsidata/reader/spatialdata_image2d.py)
starts with `properties.mpp=None`. The test simultaneously supplied a requested
TileSpec mpp of 1.0 and magnification 20×; native tile MPP remained missing and
physical-area weighting was rejected. Neither value became a guessed physical
calibration. A second real container supplied an explicit cached slide MPP of
0.5; both native axes retained it and physical-area aggregation succeeded.

This validates this reader/revision's behavior. It does not establish how every
third-party backend treats a placeholder calibration such as 1.0. The adapter
cannot authenticate an arbitrary upstream scalar's provenance merely from its
numerical value; other reader metadata policies need separate evidence.

## Base operation without optional packages

A separate process in `E:\radiomics-build\native-env` checked that AnnData,
WSIData, SpatialData, GeoPandas and Shapely were absent. With imports of those
packages blocked, ordinary external arrays still aligned and aggregated through
Rust to the expected patient mean. Native snapshot v2 package-file identity
verification passed.

An earlier base attempt correctly rejected a v1 manifest against this environment
after its Python `_types.py` had been upgraded to v2. The old failed receipt was
preserved. The required remaining check used the matching v2 manifest and passed;
no expectation, algorithm or installed file was changed to hide the mismatch.

## Reproduction and retained evidence

Run with an already prepared optional environment and matching immutable wheel:

<!-- PR-2: restore the optional-interoperability command with explicit workspace and interpreter locations. -->

Raw evidence remains in the recorded local evidence directory:
<!-- PR-2: document the evidence directory relative to the workspace. -->
`install-stack.log`, `install-wsidata.log`, `requirements-optional.txt`,
`optional-v1/acceptance.json`, `base-v1/acceptance.json` (identity rejection),
`base-v2/acceptance.json`, and the actual H5AD round-trip files. The runtime
receipts retain class paths, package versions, Git installation identity, expected
error messages, snapshot identity, import probes and reader-call observations.

The optional runtime receipt SHA256 is
`355769b36736365677930319d7acb4fe336b28a1e4cc99e02edcc21694fce74f`;
the matching base-v2 receipt SHA256 is
`30473231d035307e526b7345edd441090942022163ebddb51cc4bdfa0c9cc462`.
The captured optional dependency freeze SHA256 is
`b97add415c8129aadacf24986c5e5f4f8487f45d3d176d5023726ae56514c7c3`.

This closes the actual-container smoke gap for the declared dense/metadata path.
It does not add WSI decoding, GPU/deep training, sparse/backed embedding support,
an upstream readerless constructor, cross-platform optional-package validation or
new base dependencies.
