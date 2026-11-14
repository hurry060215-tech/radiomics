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

Strict WSIData geometry/cache adaptation is a separate acceptance work package.
It must not silently fall back to this caller-provided external association path.
