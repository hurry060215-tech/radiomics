"""Thin external-array/container conversion into the shared native TileTable.

Optional AnnData/WSIData packages are not imported at module load. Identity alignment
and numerical computation belong to Rust. External arrays never claim an upstream
embedding cache has been verified.
"""
import copy
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

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


def _plain_metadata(value):
    """Convert array-library scalar/container representations, preserving metadata types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Metadata dictionaries require string keys.')
        return {key: _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if hasattr(value, 'tolist'):
        return _plain_metadata(value.tolist())
    if hasattr(value, 'item'):
        return _plain_metadata(value.item())
    raise TypeError('Metadata must consist of explicit JSON-compatible scalar/container values.')


def _shape_to_level0(shapes, explicit):
    if explicit is not None:
        return _plain_metadata(explicit)
    try:
        from spatialdata.transformations import Identity, get_transformation
    except ImportError as exc:
        raise ImportError('Reading implicit WSIData transforms needs optional spatialdata; '
                          'supply the explicit recorded shape_to_level0 matrix otherwise.') from exc
    transform = get_transformation(shapes, to_coordinate_system='global')
    if not isinstance(transform, Identity):
        raise ValueError('A non-identity registered transform is not automatically level-0 coordinates; '
                         'supply an explicit positive axis-aligned shape_to_level0 matrix.')
    return _plain_metadata(transform.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y')))


def _wsidata_geometry(wsi, *, tile_key, source, shape_to_level0, level0_mpp_xy):
    try:
        shapes = wsi.shapes[tile_key]
        spec = wsi.attrs['tile_spec'][tile_key]
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('WSIData must supply shapes[tile_key] and attrs["tile_spec"][tile_key].') from exc
    if 'tile_id' not in shapes:
        raise ValueError('WSIData shapes need the authoritative tile_id column; row positions are not identities.')
    ids = _identity_strings(shapes['tile_id'], 'Shape tile_id')
    geometries = list(shapes.geometry)
    if len(geometries) != len(ids):
        raise ValueError('Shape identity and geometry vectors differ.')
    valid_values = list(shapes['valid_pixels']) if 'valid_pixels' in shapes else [None] * len(ids)
    rows = []
    for raw_id, geometry, valid in zip(ids, geometries, valid_values):
        # Only public geometry structure is inspected; rectangle/scale arithmetic is Rust.
        if (getattr(geometry, 'geom_type', None) != 'Polygon'
                or not getattr(geometry, 'is_valid', False)
                or getattr(geometry, 'is_empty', True)
                or getattr(geometry, 'has_z', False)
                or len(geometry.interiors) != 0):
            raise ValueError('WSIData tiles must be valid 2D Polygon geometries without holes; '
                             'rotation/non-rectangles are checked by the native builder.')
        row = dict(tile_id=raw_id, geometry_corners=_plain_metadata(list(geometry.exterior.coords)))
        if valid is not None and not (isinstance(valid, Real) and math.isnan(valid)):
            row['valid_pixels'] = _plain_metadata(valid)
        rows.append(row)
    properties = getattr(wsi, 'properties', None)
    shape = getattr(properties, 'shape', None)
    slide_shape = None if shape is None else _plain_metadata(list(reversed(shape)))
    if level0_mpp_xy is None:
        recorded_mpp = getattr(properties, 'mpp', None)
        level0_mpp_xy = None if recorded_mpp is None else [recorded_mpp, recorded_mpp]
    if hasattr(spec, 'to_dict'):
        spec = spec.to_dict()
    return dict(tile_key=tile_key, shape_rows=rows, tile_spec=_plain_metadata(spec),
                shape_to_level0=_shape_to_level0(shapes, shape_to_level0),
                slide_shape_wh=slide_shape, level0_mpp_xy=_plain_metadata(level0_mpp_xy),
                source=source)


def snapshot_wsidata(wsi, *, source, tile_key='tiles', shape_to_level0=None, level0_mpp_xy=None):
    """Read-only generation-time geometry snapshot; call before computing embeddings.

    ``source`` is an explicit stable source identity, preferably an input-content
    digest. This helper returns a manifest and never attaches one to an existing
    table. Save it with the embedding generation provenance, for example as a JSON
    string in the generated AnnData ``uns['radiomics_tile_manifest']``.
    """
    payload = _wsidata_geometry(wsi, tile_key=tile_key, source=source,
                                shape_to_level0=shape_to_level0, level0_mpp_xy=level0_mpp_xy)
    return json.loads(resolve_backend('rust').wsidata_snapshot(json.dumps(payload, allow_nan=False)))


def from_wsidata(wsi, *, patient_id, slide_id, feature_key, source, tile_key='tiles',
                 shape_to_level0=None, level0_mpp_xy=None,
                 manifest_key='radiomics_tile_manifest'):
    """Strictly adapt an existing embedding table with historical geometry evidence.

    Actual polygons, recorded TileSpec and the specified table X/row identities
    are passed to Rust. Authoritative tile IDs, TableModel region/instance links
    and a valid generation-time geometry manifest are required. Missing provenance
    raises; this function never falls back to caller-provided arrays or silently
    stamps the current geometry onto an old cache. Canonical IDs are generated
    only by Rust as compact JSON pairs ``[slide_id, raw_tile_id]``.
    """
    geometry = _wsidata_geometry(wsi, tile_key=tile_key, source=source,
                                 shape_to_level0=shape_to_level0, level0_mpp_xy=level0_mpp_xy)
    try:
        table = wsi.tables[feature_key]
        link = table.uns['spatialdata_attrs']
        instance_key, region_key = link['instance_key'], link['region_key']
        region = link['region']
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError('Embedding table needs the recorded SpatialData TableModel region/instance linkage.') from exc
    if not isinstance(instance_key, str) or not isinstance(region_key, str):
        raise ValueError('TableModel instance_key and region_key must be explicit column names.')
    try:
        tile_ids = _identity_strings(table.obs[instance_key], 'Embedding obs tile_id')
        regions = _identity_strings(table.obs[region_key], 'Embedding observation region')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Embedding observations lack the declared TableModel linkage columns.') from exc
    saved = table.uns.get(manifest_key)
    if isinstance(saved, str):
        saved = json.loads(saved)
    payload = dict(geometry=geometry, saved_at_generation=_plain_metadata(saved),
                   table_region=_plain_metadata(region), table_region_key=region_key,
                   table_instance_key=instance_key, obs_regions=regions,
                   obs_tile_ids=tile_ids,
                   obs_row_ids=_identity_strings(table.obs_names, 'Embedding obs_names'),
                   patient_id=patient_id, slide_id=slide_id, feature_key=feature_key,
                   feature_names=_identity_strings(table.var_names, 'Embedding var_names'))
    result = json.loads(resolve_backend('rust').from_wsidata(
        json.dumps(payload, allow_nan=False), json.dumps(_numeric_buffer(table.X), allow_nan=False)))
    return TileTable._from_native(result['table'], provenance=result['provenance'])
