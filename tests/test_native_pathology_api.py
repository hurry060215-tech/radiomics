import json
from pathlib import Path
import numpy as np
import pytest
import ov_radiomics as rad
from ov_radiomics._backend import resolve_backend, try_native

pytestmark=pytest.mark.skipif(try_native() is None,reason='Native extension not built.')
ROOT=Path(__file__).resolve().parents[1]/'golden/python_golden_v1'


def compare(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:
            compare(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            compare(a,b)
    elif isinstance(expected,float):
        assert actual==pytest.approx(expected,abs=1e-6,rel=1e-5)
    else:
        assert actual==expected


def test_native_patch_qc_color_and_padding_use_independent_golden():
    source=json.loads((ROOT/'pathology_inputs.json').read_text())
    reference=json.loads((ROOT/'pathology_reference.json').read_text())
    for case in source['patches']:
        image=rad.pathology.from_array(np.asarray(case['pixels'],dtype=np.uint8).reshape(case['height'],case['width'],3))
        patch=next(rad.pathology.tile_image(image,patient_id='P',slide_id=case['id'],source='frozen_synthetic',
            patch_size=(case['width'],case['height'])))
        payload=patch.to_dict();payload['record']['valid_pixels']=case['valid_pixels']
        patch=rad.pathology.Patch(payload)
        qc=next(rad.pathology.qc_patches([patch],config=source['qc_config']))
        expected=reference['patches'][case['id']]
        compare(qc.qc,expected['qc'])
        color=rad.pathology.color_features(qc,config={'fixed_edges':source['fixed_edges']})
        compare(color,expected['colors'])
        tissue=rad.pathology.color_features(qc,config={'fixed_edges':source['fixed_edges']},tissue_mask=qc.qc['tissue_mask'])
        compare(tissue,expected['colors_tissue_only'])
    padded=list(rad.pathology.tile_image(np.ones((2,3,3),dtype=np.uint8)*255,patient_id='P',slide_id='S',
        source='synthetic',patch_size=(2,2),stride=(2,2),edge='pad'))
    assert len(padded)==2
    assert padded[-1].record['valid_pixels']==[0,0,1,2]
    assert next(rad.pathology.qc_patches([padded[-1]])).qc['valid_pixels']==2
    assert next(rad.pathology.qc_patches([padded[-1]])).qc['scores']['white_fraction']==1.


def test_external_embeddings_aggregate_and_overlay_blocks_preserve_identity_and_uncovered_nan(tmp_path):
    inputs=json.loads((ROOT/'pathology_aggregate_inputs.json').read_text())
    reference=json.loads((ROOT/'pathology_aggregate_reference.json').read_text())
    base=dict(tiles=inputs['tiles'],features=None,feature_names=[],slide_patients=inputs['slide_patients'],
        slide_status={slide:'provided' for slide in inputs['slide_patients']})
    table=rad.interop.from_arrays(np.asarray(inputs['embedding_values'],dtype=float),tile_ids=inputs['embedding_ids'],
        tiles=base,feature_names=inputs['feature_names'])
    for variant in inputs['aggregation_variants']:
        report=rad.pathology.aggregate(table,**{k:v for k,v in variant.items() if k!='id'})
        expected=reference['aggregation'][variant['id']]
        compare(report.report['slides'],expected['slides'])
        compare(report.report['patients'],expected['patients'])
        assert report.patient_features.metadata.patient_id.tolist()==[row['patient_id'] for row in report.report['patients']]
    overlay=inputs['overlay']
    sub=dict(base,tiles=[tile for tile in base['tiles'] if tile['slide_id']==overlay['slide_id']],
        slide_patients={'A':base['slide_patients']['A']},slide_status={'A':'provided'})
    stream=rad.pathology.rasterize_predictions(sub,tile_ids=overlay['score_ids'],scores=overlay['scores'],
        canvas=overlay['canvas'],block_size=overlay['block_size'])
    manifest=stream.save(tmp_path/'overlay')
    assert manifest['canvas']==overlay['canvas']
    assert len(manifest['blocks'])>1
    values=np.full((overlay['canvas']['height'],overlay['canvas']['width']),np.nan)
    coverage=np.zeros_like(values,dtype=np.uint64)
    for block in manifest['blocks']:
        data=np.load(tmp_path/'overlay'/block['path'])
        y,x=block['y'],block['x'];h,w=block['height'],block['width']
        values[y:y+h,x:x+w]=data['values'];coverage[y:y+h,x:x+w]=data['coverage']
        np.testing.assert_array_equal(data['alpha']==0,data['missing'])
    expected=reference['overlay']['whole']
    np.testing.assert_allclose(values.ravel(),np.asarray(expected['values'],dtype=float),equal_nan=True)
    np.testing.assert_array_equal(coverage.ravel(),expected['coverage'])
    with pytest.raises(ValueError,match='tile-level'):
        rad.pathology.rasterize_predictions(sub,tile_ids=overlay['score_ids'],scores=overlay['scores'],
            canvas=overlay['canvas'],prediction_unit='patient')


def test_semantic_objects_are_patch_local_with_explicit_physical_units():
    mask=np.array([[1,0],[0,1]],dtype=np.uint16)
    four=rad.pathology.object_features(mask,kind='semantic',connectivity=4,overlapping_tiles=True)
    eight=rad.pathology.object_features(mask,kind='semantic',connectivity=8,patch_mpp_um=(.5,2.))
    assert four['object_count']==2 and eight['object_count']==1
    assert four['count_scope']=='patch_local'
    assert four['objects'][0]['area_um2'] is None
    assert eight['objects'][0]['area_um2']==2.
    with pytest.raises(ValueError,match='without truncation'):
        rad.pathology.object_features(mask.astype(float)+.5,kind='instance',connectivity=4)
