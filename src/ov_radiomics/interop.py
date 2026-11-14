"""Thin external-array/container conversion into the shared native TileTable.

Optional AnnData/WSIData packages are not imported at module load. Identity alignment
and numerical computation belong to Rust. External arrays never claim an upstream
embedding cache has been verified.
"""
import copy
import json
from numbers import Integral

from ._backend import resolve_backend
from ._types import TileTable


def _identity_strings(values, label):
    """Convert explicit integer/string identities without lossy floats or guessed row positions."""
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, Integral) and not isinstance(value, bool):
            result.append(str(value))
        else:
            raise ValueError(f'{label} must contain explicit string or integer identities; '
                             'floating, missing and boolean IDs are unsupported.')
    return result


def _numeric_buffer(values):
    import numpy as np
    raw = values.to_numpy() if hasattr(values, 'to_numpy') else np.asarray(values)
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError('Embedding values must be a two-dimensional numeric matrix.')
    if raw.dtype.kind == 'c' or raw.dtype.kind == 'O' and any(
            isinstance(v, (complex, np.complexfloating)) for v in raw.flat):
        raise ValueError('Complex embeddings are unsupported; imaginary components cannot be discarded.')
    if raw.dtype.kind not in 'biufO' or raw.dtype.kind == 'O' and any(
            isinstance(v, (str, bytes)) for v in raw.flat):
        raise ValueError('Embedding values must be numeric, not strings, timestamps or structured data.')
    try:
        # pandas nullable values use an explicit NaN conversion; no learned filling occurs.
        matrix = values.to_numpy(dtype=np.float64, na_value=np.nan) if hasattr(values, 'to_numpy') else np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Embedding values cannot be converted to a finite-or-missing float64 matrix.') from exc
    matrix = np.asarray(matrix, dtype=np.float64)
    if np.isinf(matrix).any():
        raise ValueError('Embedding values contain infinity; only finite values or explicit missing values are supported.')
    missing = np.isnan(matrix)
    return dict(rows=int(matrix.shape[0]), cols=int(matrix.shape[1]),
                values=np.where(missing, 0., matrix).ravel(order='C').tolist(),
                missing=missing.ravel(order='C').tolist())


def _table_dict(tiles):
    if isinstance(tiles, TileTable):
        return tiles.to_dict()
    if isinstance(tiles, dict):
        return copy.deepcopy(tiles)
    raise TypeError('tiles must be a TileTable or its canonical native dictionary.')


def _from_matrix(values, tile_ids, tiles, feature_names, provenance):
    native = resolve_backend('rust')
    payload = _table_dict(tiles)
    buffer = _numeric_buffer(values)
    ids = _identity_strings(tile_ids, 'tile_ids')
    names = ([f'embedding_{i}' for i in range(buffer['cols'])] if feature_names is None
             else _identity_strings(feature_names, 'feature_names'))
    # The core owns duplicate checks, exact ID-set matching, buffer validation and row order.
    aligned = json.loads(native.align_tile_embeddings(
        json.dumps(payload, allow_nan=False), ids,
        json.dumps(buffer, allow_nan=False), names))
    metadata = dict(provenance, input_row_ids=ids, feature_names=names,
                    input_shape=[buffer['rows'], buffer['cols']],
                    missing_representation='explicit_numeric_buffer_flags')
    # This payload has just passed native alignment and validation; do not serialize
    # the entire embedding a second time through the external-dictionary constructor.
    return TileTable._from_native(aligned, provenance=metadata)


def from_arrays(values, *, tile_ids, tiles, feature_names=None, source=None):
    """Align external embeddings to an explicit tile hierarchy by IDs in Rust.

    ``values`` is a two-dimensional real numeric matrix, ``tile_ids`` identifies
    its rows, and ``tiles`` supplies the patient/slide/tile geometry and QC contract.
    NaNs remain missing. Complex values and infinities are errors. No positional
    association or upstream-cache verification is inferred. Existing features are
    an explicit native conflict; the input table is never modified.
    """
    return _from_matrix(values, tile_ids, tiles, feature_names,
                        dict(adapter='from_arrays', association='external/provided',
                             cache_verified=False, source=None if source is None else str(source)))


def from_anndata(adata, *, tiles, tile_id_column=None, feature_names=None, source=None):
    """Convert an existing AnnData-like table without importing or requiring AnnData.

    The default row identities are ``obs_names``. An explicitly selected obs
    column can instead supply tile identities, while original row names remain
    in provenance. ``var_names`` supplies feature order unless explicit names are
    provided. ``X`` must already be a dense real matrix; convert sparse/backed
    objects explicitly when their allocation is appropriate.
    """
    if not all(hasattr(adata, field) for field in ('X', 'obs_names', 'var_names')):
        raise TypeError('Expected AnnData-like X, obs_names and var_names attributes.')
    row_names = _identity_strings(adata.obs_names, 'AnnData obs_names')
    if tile_id_column is None:
        ids = row_names
    else:
        if not hasattr(adata, 'obs') or tile_id_column not in adata.obs:
            raise ValueError(f'AnnData obs lacks explicit tile ID column {tile_id_column!r}.')
        ids = _identity_strings(adata.obs[tile_id_column], 'AnnData tile ID column')
    names = adata.var_names if feature_names is None else feature_names
    return _from_matrix(adata.X, ids, tiles, names,
                        dict(adapter='from_anndata', association='external/provided',
                             cache_verified=False, source=None if source is None else str(source),
                             original_obs_names=row_names, tile_id_column=tile_id_column))
