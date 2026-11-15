# Explicit external embeddings

`ov_radiomics.interop.from_arrays` converts a dense real matrix and explicit row
IDs into the shared native `TileTable`. The caller supplies its patient/slide/tile
hierarchy, geometry and QC metadata. The adapter does not infer identity from row
count or positional agreement:

```python
import ov_radiomics.interop as interop

table = interop.from_arrays(
    embeddings,
    tile_ids=embedding_row_ids,
    tiles=tile_metadata,
    feature_names=embedding_feature_names,
    source="external_embeddings.npy",
)
```

`tile_metadata` is the public `TileTable` wrapper or its canonical dictionary.
The adapter calls the dedicated Rust alignment entry point, which validates
unique identities, exact ID sets, feature names, dimensions and missing buffers
and orders features into the existing tile order. The source table is unchanged.
Existing features cause an explicit conflict. Feature names can be supplied;
otherwise deterministic `embedding_0`, `embedding_1`, etc. are recorded.

The returned public `TileTable` retains provenance separately from the Rust data
contract. External associations are labeled `association="external/provided"`
and `cache_verified=false`. Input IDs, dimensions, selected feature order and
source are retained for an export manifest. This is not evidence that a third-party
WSI embedding cache was generated from the current geometry.

`from_anndata` accepts an existing AnnData-like table without importing or requiring
AnnData. It reads the specified `X`, uses `obs_names` as row identities by default,
or an explicitly selected `tile_id_column`, and uses `var_names` for feature order.
Original observation names remain in provenance. It does not guess that an arbitrary
observation index equals a tile index:

```python
table = interop.from_anndata(
    adata,
    tiles=tile_metadata,
    tile_id_column="tile_id",
    source="provided_features.h5ad",
)
```

Values must be dense, two-dimensional and real. Gray values and embeddings are
converted to float64 without learned normalization. NaNs become explicit missing
flags with canonical stored zero; infinities and complex arrays are errors,
including object arrays containing complex values with a zero imaginary component.
The adapter never casts complex arrays to real while discarding information.
String, datetime and structured matrices are rejected. Sparse/backed objects must
be explicitly converted by the caller when the dense allocation is appropriate.
Explicit integer identities are converted to decimal strings; string identities
retain leading zeros. Floating, missing and boolean IDs are rejected.

The module does not import wsidata, anndata or spatialdata at startup, and it
never calls a Python numerical backend for alignment. Native errors surface
without a fallback. Ten focused conversion tests check the boundary and delegation;
the real-extension integration test compares arrays and AnnData-like inputs against
the independently frozen Rust alignment reference. That integration test requires
the newly compiled binding; an old extension fails clearly instead of pretending
the native path was validated.

## Strict WSIData geometry and generation provenance

`snapshot_wsidata` reads an existing WSIData-like object's recorded geometry before
embedding generation. It returns a native manifest without modifying the WSI or
attaching proof to an existing cache. A caller saves this manifest with the table
generated from exactly that geometry, for example as a JSON string in
`generated_table.uns['radiomics_tile_manifest']`. A helper called after finding an
old cache cannot retroactively prove how that cache was generated.

```python
import json

manifest = interop.snapshot_wsidata(
    wsi,
    tile_key="tiles",
    source="sha256:the_explicit_recorded_input_digest",
    shape_to_level0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
)
# Retain this manifest with the embedding generated from this recorded tile grid.
# generated_table.uns["radiomics_tile_manifest"] = json.dumps(manifest)
```

`source` is an explicit stable identity, preferably an input-content digest;
the adapter does not infer immutable identity from a mutable filename. Recorded
WSIData scalar MPP, when available, supplies both level-0 axes. An explicitly
known anisotropic pair can instead be passed as `level0_mpp_xy`. Requested
`TileSpec.mpp`, image DPI and magnification are not used to guess missing level-0
MPP. The snapshot binds the recorded source identifier and metadata; it does not
claim to authenticate an upstream model's training or independently recompute
external embeddings.

`from_wsidata` reads the **specified** `wsi.tables[feature_key].X`. It requires
existing generation-time proof and cannot silently fall back to `from_arrays`:

```python
table = interop.from_wsidata(
    wsi,
    patient_id="P001",
    slide_id="globally_unique_slide_identifier",
    tile_key="tiles",
    feature_key="provided_backbone_tiles",
    source="sha256:the_same_recorded_input_digest",
    shape_to_level0=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
)
```

The Python boundary only obtains geometry, transformations, recorded table metadata
and arrays. Rust validates and interprets the normalized payload, constructs its
snapshot/digest, checks the historical association, builds `TileRecord` objects,
creates canonical IDs and calls the shared native alignment function. The core
requires the recorded SpatialData region/instance keys, observation region labels,
unique `obs_names`, and a bijection between authoritative shape `tile_id` and
table `obs.tile_id`. Matching row count or matching arbitrary DataFrame indices
does not prove this association. A multi-region table is not silently subsetted.

Raw shape IDs remain the authority during historical verification. After this
proof, Rust namespaces output IDs as compact UTF-8 JSON pairs
`[slide_id, raw_tile_id]`. There is no digit guessing or Unicode normalization.
Provenance retains the complete raw-to-canonical map and original embedding row
IDs in shape order. A slide ID must itself be globally unique; shared-contract
validation rejects a slide associated with different patients.

## Verified geometric definitions

Geometry comes from `wsi.shapes[tile_key].geometry`, not arbitrary x/y columns.
Valid two-dimensional Polygon objects without holes are required. Rust verifies
the actual closed polygon is an axis-aligned rectangle; it does not substitute
a bounding box for a rotated, nonrectangular or holed shape. Scale comes from
`wsi.attrs['tile_spec'][tile_key]` and an explicit shape-to-level-0 matrix.

With no explicit matrix, the optional SpatialData public transformation API is
queried and only an actual `Identity` transformation is accepted. A registered
display transform is not automatically a level-0 transform. All other transform
types require the caller's explicit matrix; Rust then requires positive diagonal
scale plus translation, rejecting shear, rotation, singularity and unknown axes.
This conservative default also rejects an arbitrary Affine object that merely
happens to encode identity until its intended mapping is supplied explicitly.

`TileSpec.width/height` are requested patch dimensions. Its declared
`base_width=int(width*base_downsample)` and corresponding height must match the
transformed geometry. `base_level` must be zero; operation level and downsample
are recorded read parameters. Continuous values are normalized to float64;
dimensions, levels and valid-pixel rectangles must be actual integers, not bools
or rounded floats. Integer coordinates that cannot be represented exactly in
float64 are rejected before conversion.

The nominal-extent consistency check uses a documented six-operation conservative
IEEE roundoff bound, capped at `1e-10` of the expected span when a discrepancy
exists. This only verifies metadata. It never rounds a transformed coordinate,
changes a raster boundary, or adds a membership epsilon. Actual coordinates remain
in the snapshot. Where the original coordinate magnitude makes the metadata
comparison unreliable, the adapter errors and asks for re-expressed coordinates.

Known slide dimensions can prove a full rectangle has no edge padding. Otherwise,
a recorded shape-row `valid_pixels=[left,top,width,height]` is required. If slide
dimensions are known, even an explicit valid rectangle must map inside the real
slide. An unsupported edge cannot be declared entirely valid to count padding as
tissue. Unrecorded physical resolution remains missing.

## Lossless snapshot protocol and cache guard

The schema is `radiomics.wsidata.tile_manifest.v1`. Bounds, transforms and MPP use
typed float64 normalization; dimensions and levels remain integers; unknown extra
TileSpec JSON values retain their types. Float canonicalization uses an exact
`float64_hex` representation equivalent to Python `float.hex`, including signed
zero and subnormals, with compact UTF-8 JSON and sorted keys. Both current and saved
digests and schema are validated.

Digest equality is also checked against recursive normalized snapshot structure,
types and values. This supplemental guard is necessary because an opaque extra
JSON object can deliberately look like a float tag. An independently frozen
collision example has the same v1 digest but different original metadata types;
the adapter rejects it. Existing normal snapshot and digest bytes were preserved.

The adapter never stamps missing provenance onto a pre-existing embedding table,
never trusts the old upstream cache merely because N or IDs match, and has no
trust-bypass option. Snapshot dictionaries or their JSON-string representation
can be read from the configured `manifest_key` without mutation. The resulting
wrapper carries verified generation-manifest association and the exact geometry
manifest into later export provenance.

## Source audit and validation scope

The schema was verified against local OmicVerse revision
`af354858bfb6cf7ef3f8418995fc066c84107bbb` and public WSIData revision
`0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286`. In particular:

- OmicVerse `_hest_fm.py` demonstrates x/y columns may be centers, while geometry
  stores the actual box. `_embed.py`'s existing cache key alone does not establish
  the geometry binding required here.
- [WSIData tile source](https://github.com/rendeirolab/wsidata/blob/0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286/wsidata/_model/tile.py),
  blob `4b6cb8a1e716b1c867301b8d973150523c8ac1ed`, defines requested/operation/base
  dimensions and level-0 shape bounds.
- [WSIData element source](https://github.com/rendeirolab/wsidata/blob/0f93c4d8c1255c1b8b7757b98e1e5cbdc13a6286/wsidata/io/_elems.py),
  blob `ece507e395d912e5575256ad42448de572570a65`, defines shape tile IDs and
  TableModel region/library/instance linkage.
- [SpatialData's public transformation API](https://spatialdata.scverse.org/en/stable/api/transformations_utils.html)
  is used only when optional WSIData transformation metadata is requested.

Four native tests compare exact frozen normalized snapshots/digests, raw row
association, canonical namespace and embeddings; verify scale/translation,
edge-validity, typed float cases and twenty invalid scenarios; and reject the
independently frozen float-tag collision. Python conversion tests inspect public
geometry behavior, optional dependency boundaries, missing-history failure and
unchanged caches. Real-extension integration tests call the same Rust builder and
alignment, including a stale cache with the same tile IDs. These use small
WSIData-like fixtures without a vendor decoder. No claim of vendor WSI decoding
or live foundation-model execution follows from those tests.
