"""Frozen independent schema/reference for strict optional WSIData identity adaptation.

Golden A reviews and executes this before the Rust normalized-payload builder.
Runtime Python only reads actual geometry APIs/transformations/X and converts buffers.
"""
import hashlib
import json
import math
import sys


SPAN_OPERATIONS=6
MAX_RELATIVE_SPAN_ROUNDOFF=1e-10


def real(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def f64(value):
    if not real(value):
        raise ValueError('finite real float64 metadata required')
    converted=float(value)
    if isinstance(value,int) and int(converted)!=value:
        raise ValueError('integer coordinate/scale is not exactly representable as float64')
    return converted


def nominal_span_check(low,high,scale,translation,actual,expected):
    """Six operations: two multiplies, two adds, endpoint subtraction, comparison subtraction.

    sys.float_info.epsilon is conservative (twice the IEEE unit roundoff).
    Scaling the positive magnitude sum prevents overflow in the bound itself.
    This check never changes any coordinate or downstream membership rule.
    """
    expected=f64(expected)
    difference=abs(actual-expected)
    if difference==0:
        return
    epsilon=sys.float_info.epsilon
    gamma=SPAN_OPERATIONS*epsilon/(1-SPAN_OPERATIONS*epsilon)
    terms=[abs(scale*low),abs(scale*high),abs(translation),abs(translation),abs(expected)]
    magnitude=max(terms)
    bound=(gamma*magnitude)*sum(value/magnitude for value in terms)
    if not math.isfinite(bound) or bound>MAX_RELATIVE_SPAN_ROUNDOFF*expected:
        raise ValueError('nominal extent cannot be reliably verified at this coordinate magnitude; re-express coordinates')
    if difference>bound:
        raise ValueError('geometry and TileSpec level0 extent disagree beyond declared roundoff')


def positive(value):
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value) and value > 0


def unique_strings(ids):
    if any(not isinstance(i,str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('authoritative IDs must be unique nonempty strings')


def rectangle_bounds(corners):
    if len(corners) != 5 or corners[0] != corners[-1] or any(len(p)!=2 or any(not real(v) for v in p) for p in corners):
        raise ValueError('closed finite rectangular polygon required')
    xs,ys=zip(*corners[:-1])
    lowx,lowy,highx,highy=min(xs),min(ys),max(xs),max(ys)
    if lowx == highx or lowy == highy or set(map(tuple,corners[:-1])) != {(lowx,lowy),(lowx,highy),(highx,lowy),(highx,highy)}:
        raise ValueError('geometry must be an axis-aligned rectangle, not its bounding approximation')
    if any((a[0]==b[0]) == (a[1]==b[1]) for a,b in zip(corners,corners[1:])):
        raise ValueError('rectangle edges must be axis aligned and nondegenerate')
    return [lowx,lowy,highx,highy]


def level0_rows(shape_rows,tile_spec,shape_to_level0,slide_shape_wh):
    required=['width','height','stride_width','stride_height','base_downsample','base_level','ops_level','ops_downsample']
    if any(k not in tile_spec for k in required):
        raise ValueError('incomplete recorded TileSpec')
    if any(not isinstance(tile_spec[k],int) or isinstance(tile_spec[k],bool) or tile_spec[k]<=0 for k in ['width','height','stride_width','stride_height']):
        raise ValueError('TileSpec dimensions must be positive integers')
    if type(tile_spec['base_level']) is not int or tile_spec['base_level'] != 0 or not positive(tile_spec['base_downsample']) or not positive(tile_spec['ops_downsample']) or type(tile_spec['ops_level']) is not int or tile_spec['ops_level']<0:
        raise ValueError('unproven level0 scale or read level')
    matrix=shape_to_level0
    if len(matrix)!=3 or any(len(row)!=3 or any(not real(v) for v in row) for row in matrix) or matrix[2]!=[0,0,1] or matrix[0][1]!=0 or matrix[1][0]!=0 or not positive(matrix[0][0]) or not positive(matrix[1][1]):
        raise ValueError('explicit finite positive axis-aligned transform to level0 required')
    matrix=[[f64(value) for value in row] for row in matrix]
    unique_strings([r['tile_id'] for r in shape_rows])
    if slide_shape_wh is not None and (len(slide_shape_wh)!=2 or any(type(v) is not int or v<=0 for v in slide_shape_wh)):
        raise ValueError('recorded slide shape must be two positive integer dimensions')
    width0=int(tile_spec['width']*tile_spec['base_downsample'])
    height0=int(tile_spec['height']*tile_spec['base_downsample'])
    if min(width0,height0)<=0:
        raise ValueError('degenerate TileSpec base extent')
    result=[]
    for row in shape_rows:
        x0,y0,x1,y1=map(f64,rectangle_bounds(row['geometry_corners']))
        bounds=[matrix[0][0]*x0+matrix[0][2],matrix[1][1]*y0+matrix[1][2],matrix[0][0]*x1+matrix[0][2],matrix[1][1]*y1+matrix[1][2]]
        if any(not math.isfinite(v) or v<0 for v in bounds) or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
            raise ValueError('level0 rectangle is nonfinite, negative or numerically degenerate')
        nominal_span_check(x0,x1,matrix[0][0],matrix[0][2],bounds[2]-bounds[0],width0)
        nominal_span_check(y0,y1,matrix[1][1],matrix[1][2],bounds[3]-bounds[1],height0)
        valid=row.get('valid_pixels')
        if valid is None:
            if slide_shape_wh is None or bounds[2]>slide_shape_wh[0] or bounds[3]>slide_shape_wh[1]:
                raise ValueError('edge-padding validity unknown: explicit valid_pixels required')
            valid=[0,0,tile_spec['width'],tile_spec['height']]
        left,top,w,h=valid
        if any(type(v) is not int for v in valid) or min(w,h)<=0 or min(left,top)<0 or left+w>tile_spec['width'] or top+h>tile_spec['height']:
            raise ValueError('invalid explicit valid rectangle')
        if slide_shape_wh is not None:
            sx=(bounds[2]-bounds[0])/tile_spec['width'];sy=(bounds[3]-bounds[1])/tile_spec['height']
            valid_bounds=[bounds[0]+left*sx,bounds[1]+top*sy,bounds[0]+(left+w)*sx,bounds[1]+(top+h)*sy]
            if valid_bounds[2]>slide_shape_wh[0] or valid_bounds[3]>slide_shape_wh[1]:
                raise ValueError('declared valid pixels include padding outside the recorded slide')
        result.append(dict(tile_id=row['tile_id'],bounds0=bounds,patch_size=[tile_spec['width'],tile_spec['height']],valid_pixels=valid))
    return result


def _canonical(value):
    # Float hex records exact IEEE values independently of JSON decimal formatting/version.
    if isinstance(value,float):
        if not math.isfinite(value):
            raise ValueError('nonfinite snapshot metadata')
        return {'float64_hex':value.hex()}
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):
            raise ValueError('snapshot metadata keys must be strings')
        return {str(k):_canonical(v) for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value,(str,int,bool)):
        return value
    raise ValueError('unsupported metadata type in snapshot')


def snapshot(tile_key,shape_rows,tile_spec,shape_to_level0,slide_shape_wh,level0_mpp_xy,source):
    """Read-only helper called before embedding generation; it does not attach or bless old cache."""
    if not isinstance(tile_key,str) or not tile_key.strip() or not isinstance(source,str) or not source.strip():
        raise ValueError('tile key and recorded immutable source identity must be nonempty strings')
    rows=level0_rows(shape_rows,tile_spec,shape_to_level0,slide_shape_wh)
    if level0_mpp_xy is not None and (len(level0_mpp_xy)!=2 or any(not positive(v) for v in level0_mpp_xy)):
        raise ValueError('recorded level0 MPP must be positive finite')
    normalized_spec=dict(tile_spec)
    for key in ['base_downsample','ops_downsample','mpp']:
        if normalized_spec.get(key) is not None:
            if not positive(normalized_spec[key]):
                raise ValueError('recorded downsample/MPP values must be positive finite')
            normalized_spec[key]=f64(normalized_spec[key])
    payload=dict(schema='radiomics.wsidata.tile_manifest.v1',tile_key=tile_key,tiles=sorted(rows,key=lambda r:r['tile_id']),
                 tile_spec=normalized_spec,shape_to_level0=[[f64(value) for value in row] for row in shape_to_level0],slide_shape_wh=slide_shape_wh,
                 level0_mpp_xy=None if level0_mpp_xy is None else [f64(value) for value in level0_mpp_xy],source=source,
                 nominal_span_validation=dict(operations=SPAN_OPERATIONS,epsilon=sys.float_info.epsilon,max_relative_roundoff=MAX_RELATIVE_SPAN_ROUNDOFF,coordinate_policy='actual coordinates retained unchanged; no raster tolerance'))
    serialized=json.dumps(_canonical(payload),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')
    return dict(snapshot=payload,sha256=hashlib.sha256(serialized).hexdigest())


def validate_embedding_link(current, saved_at_generation, table_region, tile_key, shape_ids, obs_tile_ids, obs_row_ids):
    unique_strings(shape_ids)
    unique_strings(obs_tile_ids)
    unique_strings(obs_row_ids)
    if table_region != tile_key or len(obs_row_ids)!=len(obs_tile_ids) or set(shape_ids)!=set(obs_tile_ids):
        raise ValueError('authoritative table tile IDs and shape IDs must form a region-linked bijection')
    if saved_at_generation is None:
        raise ValueError('embedding lacks generation-time tile geometry/TileSpec manifest')
    # Stored digest must agree with stored payload, and current normalized snapshot must agree.
    saved=saved_at_generation
    if current['snapshot'].get('schema')!='radiomics.wsidata.tile_manifest.v1' or saved['snapshot'].get('schema')!='radiomics.wsidata.tile_manifest.v1':
        raise ValueError('unknown WSIData tile manifest schema')
    current_bytes=json.dumps(_canonical(current['snapshot']),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')
    if hashlib.sha256(current_bytes).hexdigest()!=current['sha256']:
        raise ValueError('current normalized tile snapshot digest is invalid')
    serialized=json.dumps(_canonical(saved['snapshot']),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')
    if hashlib.sha256(serialized).hexdigest()!=saved['sha256'] or current['sha256']!=saved['sha256']:
        raise ValueError('embedding geometry or TileSpec snapshot differs')
    return [obs_tile_ids.index(tile_id) for tile_id in shape_ids]
