"""Frozen independent I specification. This is a reference, never a runtime backend.

Independently reviewed and executed by Golden A before the native implementation.
Coordinates: level0 pixel edges; raster samples cell centers in valid half-open rectangles.
"""
import math
import statistics

STATS = ['mean', 'std', 'max', 'q25', 'q50', 'q75']


def unique_ids(ids):
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('nonempty unique string IDs required')


def align_embeddings(tile_ids, embedding_ids, values):
    unique_ids(tile_ids)
    unique_ids(embedding_ids)
    if len(values) != len(embedding_ids) or set(tile_ids) != set(embedding_ids):
        raise ValueError('embedding/tile ID sets differ')
    widths = {len(row) for row in values}
    if len(widths) != 1 or not widths or min(widths) == 0:
        raise ValueError('rectangular nonempty embedding features required')
    if any(v is not None and not math.isfinite(v) for row in values for v in row):
        raise ValueError('embedding values must be finite or explicitly missing')
    mapping = dict(zip(embedding_ids, values))
    return [mapping[i][:] for i in tile_ids]


def linear_quantile(values, q):
    values = sorted(values)
    p = q * (len(values)-1)
    low, high = math.floor(p), math.ceil(p)
    return values[low] if low == high else values[low]*(high-p)+values[high]*(p-low)


def summary(values, weights=None, ids=None):
    ids=['row'+str(i).zfill(12) for i in range(len(values))] if ids is None else ids
    if len(ids)!=len(values):
        raise ValueError('summary IDs must match values')
    unique_ids(ids)
    if weights is not None and (len(weights) != len(values) or any(not math.isfinite(w) or w < 0 for w in weights)):
        raise ValueError('weights must match values and be finite nonnegative')
    if any(v is not None and not math.isfinite(v) for v in values):
        raise ValueError('values must be finite or explicitly missing')
    pairs = [(v, 1. if weights is None else weights[i],ids[i]) for i, v in enumerate(values)
             if v is not None and (weights is None or weights[i] > 0)]
    if not pairs:
        return dict(values={s: None for s in STATS}, count=0, weight_sum=0., status='empty')
    total = math.fsum(w for _, w, _ in pairs)
    if weights is None:
        x = [v for v, _, _ in pairs]
        mean = statistics.mean(x)
        variance = statistics.pvariance(x)
        quantile = lambda q: linear_quantile(x, q)
    else:
        mean = math.fsum(v*w for v, w, _ in pairs)/total
        variance = math.fsum(w*(v-mean)**2 for v, w, _ in pairs)/total
        ordered = sorted(pairs,key=lambda pair:(pair[0],pair[2]))
        cdf_total=0.
        for _,weight,_ in ordered:
            cdf_total+=weight
        def quantile(q):
            threshold, cumulative = q*cdf_total, 0.
            for v, w, _ in ordered:
                cumulative += w
                if cumulative >= threshold:
                    return v
            return ordered[-1][0]
    result = dict(mean=mean, std=math.sqrt(variance), max=max(v for v, _, _ in pairs),
                  q25=quantile(.25), q50=quantile(.5), q75=quantile(.75))
    return dict(values=result, count=len(pairs), weight_sum=total, status='ok')


def validate_tile(tile):
    for key in ['tile_id', 'slide_id', 'patient_id']:
        if not isinstance(tile[key], str) or not tile[key].strip():
            raise ValueError('missing hierarchy identity')
    for key in ['x0', 'y0', 'width0', 'height0']:
        if not math.isfinite(tile[key]) or tile[key] < 0 or key in ['width0', 'height0'] and tile[key] == 0:
            raise ValueError('invalid level0 tile geometry')
    pw, ph = tile['patch_width'], tile['patch_height']
    left, top, w, h = tile['valid_pixels']
    if min(pw, ph, w, h) <= 0 or min(left, top) < 0 or left+w > pw or top+h > ph:
        raise ValueError('invalid patch/valid-pixel rectangle')
    for key in ['mpp_x_um', 'mpp_y_um']:
        if tile.get(key) is not None and (not math.isfinite(tile[key]) or tile[key] <= 0):
            raise ValueError('invalid recorded level0 MPP')


def valid_bounds(tile):
    validate_tile(tile)
    left, top, w, h = tile['valid_pixels']
    sx, sy = tile['width0']/tile['patch_width'], tile['height0']/tile['patch_height']
    bounds=[tile['x0']+left*sx, tile['y0']+top*sy,
            tile['x0']+(left+w)*sx, tile['y0']+(top+h)*sy]
    if any(not math.isfinite(v) for v in bounds) or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
        raise ValueError('valid rectangle coordinates are not representable')
    return bounds


def area_weight(tile, mode):
    validate_tile(tile)
    if mode == 'equal':
        return 1.
    sx=tile['width0']/tile['patch_width']
    sy=tile['height0']/tile['patch_height']
    area=(sx*tile['valid_pixels'][2])*(sy*tile['valid_pixels'][3])
    if not math.isfinite(area) or area<=0:
        raise ValueError('valid patch area is not representable')
    if mode == 'level0_area':
        return area
    if mode == 'physical_area':
        if tile.get('mpp_x_um') is None or tile.get('mpp_y_um') is None:
            raise ValueError('physical area weighting requires both recorded MPP axes')
        physical=area*tile['mpp_x_um']*tile['mpp_y_um']
        if not math.isfinite(physical) or physical<=0:
            raise ValueError('physical patch area is not representable')
        return physical
    raise ValueError('unknown weight mode')


def aggregate(tiles, values, feature_names, slide_patients, tile_weight='equal', patient_weight='equal', patient_stats=STATS):
    unique_ids([t['tile_id'] for t in tiles])
    unique_ids(feature_names)
    if patient_weight not in ['equal', 'physical_area'] or not set(patient_stats).issubset(STATS) or not patient_stats or len(set(patient_stats)) != len(patient_stats):
        raise ValueError('patient weights must be equal or physical area; statistics explicit')
    if len(tiles) != len(values) or any(len(v) != len(feature_names) for v in values):
        raise ValueError('tile feature dimensions differ')
    for tile in tiles:
        validate_tile(tile)
        if slide_patients.get(tile['slide_id']) != tile['patient_id']:
            raise ValueError('slide-patient hierarchy mismatch')
    if any(not s.strip() or not p.strip() for s, p in slide_patients.items()):
        raise ValueError('empty slide/patient mapping')
    slides = []
    for slide_id, patient_id in sorted(slide_patients.items()):
        positions = sorted([i for i, t in enumerate(tiles) if t['slide_id'] == slide_id and not t.get('exclusion_reasons')], key=lambda i: tiles[i]['tile_id'])
        slide = dict(slide_id=slide_id, patient_id=patient_id,
                     status='ok' if positions else 'no_accepted_tiles', accepted_tiles=len(positions),
                     physical_patch_area_sum=None, features={}, observed={})
        if patient_weight == 'physical_area':
            slide['physical_patch_area_sum'] = math.fsum(area_weight(tiles[i], 'physical_area') for i in positions)
        weights = None if tile_weight == 'equal' else [area_weight(tiles[i], tile_weight) for i in positions]
        for j, name in enumerate(feature_names):
            summary_result = summary([values[i][j] for i in positions], weights,[tiles[i]['tile_id'] for i in positions])
            for statistic in STATS:
                column = f'slide_tile_{statistic}({name})'
                slide['features'][column] = summary_result['values'][statistic]
                slide['observed'][column] = dict(count=summary_result['count'], weight_sum=summary_result['weight_sum'])
        slides.append(slide)
    patients = []
    columns = [f'slide_tile_{s}({f})' for f in feature_names for s in STATS]
    for patient_id in sorted(set(slide_patients.values())):
        accepted = [s for s in slides if s['patient_id'] == patient_id and s['status'] == 'ok']
        patient = dict(patient_id=patient_id, status='ok' if accepted else 'no_accepted_slides',
                       accepted_slides=len(accepted), features={}, observed={})
        weights = None if patient_weight == 'equal' else [s['physical_patch_area_sum'] for s in accepted]
        for column in columns:
            result = summary([s['features'][column] for s in accepted], weights,[s['slide_id'] for s in accepted])
            for statistic in patient_stats:
                key = f'patient_{statistic}({column})'
                patient['features'][key] = result['values'][statistic]
                patient['observed'][key] = dict(count=result['count'], weight_sum=result['weight_sum'])
        patients.append(patient)
    return dict(slides=slides, patients=patients, tile_weight=tile_weight, patient_weight=patient_weight,
                slide_feature_names=columns,
                patient_feature_names=[f'patient_{s}({column})' for column in columns for s in patient_stats],
                weighted_quantile='inverted_cdf' if tile_weight != 'equal' or patient_weight != 'equal' else None,
                weight_scope='sum_of_valid_patch_contributions_not_deduplicated_slide_area')


def rasterize(tiles, score_ids, scores, canvas, *, prediction_unit='tile', clip=False):
    if prediction_unit != 'tile':
        raise ValueError('local overlays require actual tile-level scores')
    unique_ids([t['tile_id'] for t in tiles])
    unique_ids(score_ids)
    eligible = [t for t in tiles if not t.get('exclusion_reasons')]
    if len(score_ids) != len(scores) or set(score_ids) != {t['tile_id'] for t in eligible}:
        raise ValueError('scores must align exactly by accepted tile IDs')
    if any(not math.isfinite(v) for v in scores):
        raise ValueError('tile scores must be finite')
    if len({t['slide_id'] for t in tiles}) > 1:
        raise ValueError('one slide per overlay')
    width, height = canvas['width'], canvas['height']
    ox, oy = canvas['origin0']
    sx, sy = canvas['pixel_size0']
    if min(width, height) <= 0 or any(not math.isfinite(x) or x <= 0 for x in [sx,sy]) or any(not math.isfinite(x) or x < 0 for x in [ox,oy]):
        raise ValueError('invalid canvas')
    bounds = {}
    for tile in tiles:
        box = valid_bounds(tile)
        if not clip and (box[0] < ox or box[1] < oy or box[2] > ox+width*sx or box[3] > oy+height*sy):
            raise ValueError('tile valid rectangle outside canvas; request explicit clipping')
        bounds[tile['tile_id']] = box
    lookup = dict(zip(score_ids,scores))
    values, coverage = [], []
    for y in range(height):
        for x in range(width):
            px, py = ox+(x+.5)*sx, oy+(y+.5)*sy
            found = [lookup[t['tile_id']] for t in sorted(eligible,key=lambda t:t['tile_id'])
                     if bounds[t['tile_id']][0] <= px < bounds[t['tile_id']][2]
                     and bounds[t['tile_id']][1] <= py < bounds[t['tile_id']][3]]
            coverage.append(len(found))
            values.append(statistics.mean(found) if found else 'NaN')
    return dict(values=values, coverage=coverage, alpha=[255 if n else 0 for n in coverage],
                origin0=canvas['origin0'], pixel_size0=canvas['pixel_size0'],width=width,height=height)


def raster_blocks(tiles, score_ids, scores, canvas, block_size, **kwargs):
    bw,bh=block_size
    if min(bw,bh)<=0:
        raise ValueError('positive block dimensions required')
    blocks=[]
    # Validate the entire requested canvas first; per-block clipping only restricts output.
    whole=rasterize(tiles,score_ids,scores,canvas,**kwargs)
    for y in range(0,canvas['height'],bh):
        for x in range(0,canvas['width'],bw):
            sub=dict(canvas,width=min(bw,canvas['width']-x),height=min(bh,canvas['height']-y),
                origin0=[canvas['origin0'][0]+x*canvas['pixel_size0'][0],canvas['origin0'][1]+y*canvas['pixel_size0'][1]])
            # A reference oracle can slice the whole result. Native blocks must
            # be lazy and evaluate exactly the original global pixel centers.
            positions=[(y+dy)*canvas['width']+x+dx for dy in range(sub['height']) for dx in range(sub['width'])]
            result=dict(sub,values=[whole['values'][i] for i in positions],coverage=[whole['coverage'][i] for i in positions],alpha=[whole['alpha'][i] for i in positions])
            blocks.append(dict(x=x,y=y,**result))
    return blocks


def global_objects(tile, local_objects):
    validate_tile(tile)
    sx,sy=tile['width0']/tile['patch_width'],tile['height0']/tile['patch_height']
    rows=[]
    for local in local_objects:
        x,y=local['centroid_pixel_centers']
        bx,by,w,h=local['bbox_xywh']
        rows.append(dict(local=local,centroid0=[tile['x0']+x*sx,tile['y0']+y*sy],
             bbox0=[tile['x0']+bx*sx,tile['y0']+by*sy,w*sx,h*sy]))
    return dict(local_coordinate_space='patch_pixel_edges',coordinate_space='level0_pixel_edges',objects=rows)
