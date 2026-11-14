"""Real extension integration, required whenever a compiled native backend is present."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_external_arrays_and_anndata_rows_reach_the_same_rust_alignment():
    from ov_radiomics._backend import resolve_backend
    native = resolve_backend('auto')
    if native is None:
        pytest.skip('This source-only environment has no compiled extension.')
    assert hasattr(native, 'align_tile_embeddings'), 'Rebuild the extension with the native interop entry point.'
    from ov_radiomics import interop
    directory = Path(__file__).resolve().parents[1] / 'golden/python_golden_v1'
    inputs = json.loads((directory / 'pathology_aggregate_inputs.json').read_text())
    expected = json.loads((directory / 'pathology_aggregate_reference.json').read_text())
    table = dict(tiles=inputs['tiles'], features=None, feature_names=[],
                 slide_status={slide:'provided' for slide in inputs['slide_patients']},
                 slide_patients=inputs['slide_patients'])
    values = np.asarray(inputs['embedding_values'], dtype=float)
    arrays = interop.from_arrays(values, tile_ids=inputs['embedding_ids'], tiles=table,
                                 feature_names=inputs['feature_names'])
    adata = SimpleNamespace(X=values, obs_names=[f'row{i}' for i in range(len(values))],
                            var_names=inputs['feature_names'], obs={'tile_id':inputs['embedding_ids']})
    adapted = interop.from_anndata(adata, tile_id_column='tile_id', tiles=table)
    assert arrays.to_dict() == adapted.to_dict()
    buffer = arrays.to_dict()['features']
    rows = np.asarray(expected['alignment']['aligned_values'], dtype=float)
    np.testing.assert_array_equal(np.asarray(buffer['missing']).reshape(rows.shape), np.isnan(rows))
    np.testing.assert_array_equal(np.asarray(buffer['values']).reshape(rows.shape), np.nan_to_num(rows))
    assert arrays.provenance['association'] == 'external/provided'
    assert arrays.provenance['cache_verified'] is False
    assert table['features'] is None
