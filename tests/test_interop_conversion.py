"""Conversion/delegation tests. The independent Rust suite tests actual numerical alignment."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def adapter(monkeypatch):
    # Isolate conversion from the separately integrated public wrapper/PyO3 build.
    # The spy never implements sorting or identity-matching algorithms.
    class Table:
        def __init__(self, table, provenance=None):
            self.payload = copy.deepcopy(table)
            self.provenance = copy.deepcopy(provenance)

        def to_dict(self):
            return copy.deepcopy(self.payload)

    types_module = ModuleType('ov_radiomics._types')
    types_module.TileTable = Table
    monkeypatch.setitem(sys.modules, 'ov_radiomics._types', types_module)
    path = Path(__file__).resolve().parents[1] / 'src/ov_radiomics/interop.py'
    spec = importlib.util.spec_from_file_location('ov_radiomics._interop_conversion_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def align(table_json, ids, buffer_json, names):
        calls.append((json.loads(table_json), list(ids), json.loads(buffer_json), list(names)))
        result = json.loads(table_json)
        result['features'] = json.loads(buffer_json)
        result['feature_names'] = list(names)
        return json.dumps(result)

    monkeypatch.setattr(module, 'resolve_backend', lambda backend: SimpleNamespace(align_tile_embeddings=align))
    return module, calls


def test_arrays_preserve_input_order_missing_and_delegate_identity_once(adapter):
    module, calls = adapter
    table = dict(tiles=[dict(tile_id='A'), dict(tile_id='B')], features=None)
    output = module.from_arrays(np.array([[4., np.nan], [1., 2.]]),
                                tile_ids=['B', 'A'], tiles=table,
                                feature_names=['f1', 'f2'], source=Path('provided.npy'))
    assert len(calls) == 1
    sent, ids, buffer, names = calls[0]
    assert ids == ['B', 'A']
    assert names == ['f1', 'f2']
    assert buffer == dict(rows=2, cols=2, values=[4., 0., 1., 2.], missing=[False, True, False, False])
    assert table['features'] is None
    assert sent['tiles'][0]['tile_id'] == 'A'
    assert output.provenance['association'] == 'external/provided'
    assert output.provenance['cache_verified'] is False
    assert output.provenance['source'] == 'provided.npy'


@pytest.mark.parametrize('values', [np.array([[1+0j]]), np.array([[1+2j]], dtype=object),
                                    np.array([[np.complex128(1+0j)]], dtype=object),
                                    np.array([[np.inf]]), np.array([['1.2']]), np.array([1., 2.])])
def test_lossy_or_non_numeric_arrays_fail_before_native_conversion(adapter, values):
    module, calls = adapter
    with pytest.raises(ValueError):
        module.from_arrays(values, tile_ids=['A'], tiles={}, feature_names=['x'])
    assert not calls


def test_anndata_uses_explicit_authoritative_rows_without_importing_optional_packages(adapter):
    module, calls = adapter
    before = set(sys.modules)
    adata = SimpleNamespace(X=np.array([[2.], [1.]]), obs_names=['row-b', 'row-a'],
                            var_names=['fm'], obs={'tile_id': ['B', 'A']})
    output = module.from_anndata(adata, tiles={}, tile_id_column='tile_id')
    assert calls[0][1] == ['B', 'A']
    assert calls[0][2]['values'] == [2., 1.]
    assert output.provenance['original_obs_names'] == ['row-b', 'row-a']
    assert output.provenance['tile_id_column'] == 'tile_id'
    assert not ({'anndata', 'wsidata', 'spatialdata'} & (set(sys.modules)-before))


def test_native_identity_errors_surface_without_python_fallback(adapter, monkeypatch):
    module, calls = adapter
    def fail(*args):
        raise ValueError('Alignment: embedding and tile ID sets differ.')
    monkeypatch.setattr(module, 'resolve_backend', lambda backend: SimpleNamespace(align_tile_embeddings=fail))
    with pytest.raises(ValueError, match='ID sets differ'):
        module.from_arrays([[1.]], tile_ids=['wrong'], tiles={})
    assert not calls


def test_id_conversion_preserves_leading_zero_strings_and_rejects_float_guessing(adapter):
    module, _ = adapter
    assert module._identity_strings(['001', 1, np.int64(2)], 'IDs') == ['001', '1', '2']
    for ids in [[1.0], [None], [True]]:
        with pytest.raises(ValueError, match='identities'):
            module._identity_strings(ids, 'IDs')
