"""Lightweight decoded patches, native QC/features, hierarchy and tile overlays.

Coordinates are level-0 pixel edges; mpp is optional and never inferred from DPI
or magnification. Native iterators retain one decoded image and one current patch
or overlay block. No optional WSI/container/reference package is imported here.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from ._backend import resolve_backend
from ._native_common import clean_json
from ._types import TileTable, FeatureTable, PredictionTable


def _dumps(value):
    return json.dumps(clean_json(value),allow_nan=False,ensure_ascii=False,separators=(',',':'))


def _call(operation,**kwargs):
    return json.loads(resolve_backend('rust').pathology_json(_dumps(dict(operation=operation,**kwargs))))


class RasterImage:
    def __init__(self,value):
        self._payload=_call('validate_image',image=value)

    @classmethod
    def _from_native(cls,value):
        instance=cls.__new__(cls);instance._payload=value;return instance

    def to_dict(self):
        return deepcopy(self._payload)

    @property
    def array(self):
        kind,values=next(iter(self._payload['pixels'].items()))
        shape=(self._payload['height'],self._payload['width'])+((3,) if kind=='Rgb8' else ())
        return np.asarray(values,dtype=np.uint16 if kind=='Gray16' else np.uint8).reshape(shape)


def from_array(array):
    values=np.asarray(array)
    if values.ndim==3 and values.shape[2]==3 and values.dtype==np.uint8:
        kind='Rgb8'
    elif values.ndim==2 and values.dtype in (np.dtype('uint8'),np.dtype('uint16')):
        kind='Gray8' if values.dtype==np.uint8 else 'Gray16'
    else:
        raise ValueError('Decoded pathology input must be RGB8 or 2D Gray8/Gray16; no implicit color or bit-depth conversion.')
    return RasterImage(dict(width=values.shape[1],height=values.shape[0],pixels={kind:values.ravel(order='C').tolist()}))


def read_image(path):
    return RasterImage._from_native(_call('read_image',path=str(path)))


class Patch:
    def __init__(self,payload,qc=None):
        self._payload=deepcopy(payload);self._qc=deepcopy(qc)

    def to_dict(self):
        return deepcopy(self._payload)

    @property
    def record(self):
        return deepcopy(self._payload['record'])

    @property
    def image(self):
        return RasterImage._from_native(deepcopy(self._payload['image']))

    @property
    def qc(self):
        return deepcopy(self._qc)


class PatchStream:
    def __init__(self,iterator):
        self._iterator=iterator

    def __iter__(self):
        return self

    def __next__(self):
        return Patch(json.loads(next(self._iterator)))


def tile_image(image,*,patient_id,slide_id,source=None,patch_size=(256,256),stride=None,
               edge='drop',pad_value=0,origin0=(0.,0.),pixel_size0=(1.,1.),
               mpp_x_um=None,mpp_y_um=None):
    """Stream patches; sizes/stride are (width,height) in decoded patch pixels."""
    if isinstance(image,(str,Path)):
        source=str(image) if source is None else source
        image=read_image(image)
    if not isinstance(image,RasterImage):
        image=from_array(image)
    if source is None:
        raise ValueError('Provide explicit provenance source for an already decoded image.')
    payload=dict(image=image.to_dict(),spec=dict(patch_size=list(patch_size),stride=list(stride or patch_size),edge=edge,pad_value=pad_value),
        metadata=dict(patient_id=patient_id,slide_id=slide_id,source=source,origin0=list(origin0),pixel_size0=list(pixel_size0),
            mpp_x_um=mpp_x_um,mpp_y_um=mpp_y_um))
    return PatchStream(resolve_backend('rust').pathology_tiles(_dumps(payload)))


_QC_DEFAULTS=dict(white_threshold=240.,low_saturation_threshold=.1,black_threshold=15.,border_band_px=1,
    tissue_s_min=.1,tissue_v_min=.05,tissue_v_max=.95,max_white_fraction=None,max_low_saturation_fraction=None,
    max_black_border_fraction=None,min_tissue_fraction=None,min_laplacian_variance=None)


def qc_patches(patches,*,config=None):
    """Yield every patch with raw QC and exclusion reasons; never delete files."""
    effective={**_QC_DEFAULTS,**(config or {})}
    for patch in patches:
        output=_call('qc',patch=patch.to_dict(),config=effective)
        yield Patch(dict(record=output['record'],image=patch.to_dict()['image']),output['qc'])


def color_features(patch,*,config,tissue_mask=None):
    """sRGB/D65 fixed-bin statistics. Hue is circular; no linear mean of angles."""
    return _call('color',patch=patch.to_dict(),config=config,
        tissue_mask=None if tissue_mask is None else np.asarray(tissue_mask,dtype=bool).ravel().tolist())


def texture_features(patch,*,config,tissue_mask=None):
    """Explicit Gray8/Gray16; reuse the same radiomics feature engine."""
    values=_call('texture',patch=patch.to_dict(),config=config,
        tissue_mask=None if tissue_mask is None else np.asarray(tissue_mask,dtype=bool).ravel().tolist())
    return {name:float(value) for name,value in values.items()}


def _mask(labels):
    array=np.asarray(labels)
    if array.ndim!=2 or array.dtype.kind not in 'iu' or (array<0).any() or (array>2**32-1).any():
        raise ValueError('Masks must be 2D nonnegative integer labels in uint32 range, without truncation.')
    return array


def annotation_coverage(labels,*,classes,valid_pixels=None):
    array=_mask(labels);height,width=array.shape
    return _call('annotation',width=width,height=height,labels=array.ravel().tolist(),
        rect=list(valid_pixels or (0,0,width,height)),classes=list(classes))


def object_features(labels,*,kind,connectivity,valid_pixels=None,patch_mpp_um=None,
                    overlapping_tiles=False,tile=None):
    """Patch-local instance/semantic statistics. Overlap counts are never summed."""
    array=_mask(labels);height,width=array.shape
    connectivity={4:'four',8:'eight'}.get(connectivity,connectivity)
    if tile is not None:
        if valid_pixels is not None or patch_mpp_um is not None:
            raise ValueError('TileRecord supplies valid pixels and physical scale; do not supply conflicting overrides.')
        record=tile.record if isinstance(tile,Patch) else tile
        if [width,height]!=[record['patch_width'],record['patch_height']]:
            raise ValueError('Mask dimensions differ from TileRecord.')
        return _call('objects_for_tile',labels=array.ravel().tolist(),record=record,kind=kind,
            connectivity=connectivity,overlapping_tiles=overlapping_tiles)
    return _call('objects',width=width,height=height,labels=array.ravel().tolist(),
        rect=list(valid_pixels or (0,0,width,height)),kind=kind,connectivity=connectivity,
        patch_mpp_um=patch_mpp_um,overlapping_tiles=overlapping_tiles)


@dataclass
class AggregationResult:
    report: dict
    slide_features: FeatureTable
    patient_features: FeatureTable

    @property
    def slides(self):
        return pd.DataFrame(self.report['slides'])

    @property
    def patients(self):
        return pd.DataFrame(self.report['patients'])


def aggregate(table,*,tile_weight='equal',patient_weight='equal',patient_stats=('mean','std','max','q25','q50','q75')):
    """Six tile→slide summaries, then declared slide→patient summaries; equal slides by default."""
    payload=_call('aggregate',table=table.to_dict() if isinstance(table,TileTable) else table,
        config=dict(tile_weight=tile_weight,patient_weight=patient_weight,patient_stats=list(patient_stats)))
    return AggregationResult(payload['report'],FeatureTable._from_native(payload['slide_features']),FeatureTable._from_native(payload['patient_features']))


class OverlayStream:
    def __init__(self,iterator,canvas,provenance):
        self._iterator=iterator;self._canvas=deepcopy(canvas);self._provenance=deepcopy(provenance)

    @property
    def canvas(self):
        return deepcopy(self._canvas)

    def __iter__(self):
        return self

    def __next__(self):
        block=json.loads(next(self._iterator));shape=(block['height'],block['width'])
        block['values']=np.asarray([float(v) for v in block['values']]).reshape(shape)
        for key,dtype in [('coverage',np.uint64),('missing',bool),('alpha',np.uint8)]:
            block[key]=np.asarray(block[key],dtype=dtype).reshape(shape)
        return block

    def save(self,directory,*,overwrite=False):
        """Save one compressed NumPy block at a time and original global canvas metadata."""
        path=Path(directory)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError('Overlay output is nonempty; choose a new directory or pass overwrite=True.')
        path.mkdir(parents=True,exist_ok=True);blocks=[]
        for block in self:
            name=f"block-{block['y']:012}-{block['x']:012}.npz"
            np.savez_compressed(path/name,**{k:block[k] for k in ('values','coverage','missing','alpha')})
            blocks.append({k:v for k,v in block.items() if k not in ('values','coverage','missing','alpha')}|dict(path=name))
        manifest=dict(schema='radiomics.overlay.v1',coordinate_space='level0_pixel_edges',canvas=self.canvas,blocks=blocks,
            algorithm='native global-index pixel-center sampling; half-open rectangles; overlap mean',provenance=self._provenance)
        (path/'manifest.json').write_text(_dumps(manifest),encoding='utf-8')
        return manifest


def rasterize_predictions(table,predictions=None,*,slide_id=None,class_name=None,canvas,block_size=(512,512),
                          feature=None,tile_ids=None,scores=None,prediction_unit='tile',clip=False):
    """Actual tile scores/features only; patient probability broadcasting is an error."""
    table=table.to_dict() if isinstance(table,TileTable) else table
    common=dict(table=table,canvas=canvas,block_size=list(block_size))
    if predictions is not None:
        if feature is not None or scores is not None or tile_ids is not None:
            raise ValueError('Select one overlay source: predictions, feature, or explicit tile scores.')
        payload=dict(operation='predictions',predictions=predictions.to_dict() if isinstance(predictions,PredictionTable) else predictions,
            slide_id=slide_id,**{'class':class_name},clip=clip,**common)
    elif feature is not None:
        if scores is not None or tile_ids is not None:
            raise ValueError('Feature and explicit tile scores cannot be mixed.')
        payload=dict(operation='features',slide_id=slide_id,feature=feature,clip=clip,**common)
    else:
        if tile_ids is None or scores is None:
            raise ValueError('Supply explicit tile IDs and real tile-level scores.')
        payload=dict(operation='scores',ids=list(tile_ids),scores=list(scores),
            config=dict(prediction_unit=prediction_unit,clip=clip),**common)
    serialized=_dumps(payload)
    provenance=dict(input_configuration_sha256=hashlib.sha256(serialized.encode('utf-8')).hexdigest(),
        source=payload['operation'],slide_id=slide_id,feature=feature,class_name=class_name,
        package_version=resolve_backend('rust').CORE_VERSION)
    return OverlayStream(resolve_backend('rust').pathology_overlay(serialized),canvas,provenance)
