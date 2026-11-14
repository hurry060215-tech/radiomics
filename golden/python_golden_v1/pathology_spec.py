"""Frozen python_golden_v1 pathology specification; NOT a runtime implementation.

Pure Python with no Rust outputs. Independently reviewed and executed by Golden work package A.
RGB8 channels use 0..255, S/V 0..1, Hue degrees, Lab uses D65 without adaptation.
All calculations exclude padding through the supplied [left,top,width,height] rectangle.
"""
import colorsys
import math
import statistics


def quantile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] * (hi - p) + values[hi] * (p - lo) if lo != hi else values[lo]


def channel_stats(values):
    if not values:
        return None
    return dict(mean=statistics.mean(values), std=statistics.pstdev(values),
                minimum=min(values), maximum=max(values),
                q25=quantile(values, .25), q50=quantile(values, .5), q75=quantile(values, .75))


def histogram(values, edges):
    if len(edges) < 2 or any(not math.isfinite(e) for e in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError('fixed histogram edges must increase strictly')
    counts = [0] * (len(edges) - 1)
    for value in values:
        if not edges[0] <= value <= edges[-1]:
            raise ValueError('value outside fixed histogram edges')
        i = len(counts) - 1 if value == edges[-1] else next(i for i in range(len(counts)) if edges[i] <= value < edges[i+1])
        counts[i] += 1
    return dict(edges=edges, counts=counts, denominator=len(values),
                probabilities=[c / len(values) for c in counts] if values else None)


def rgb_to_lab_d65(rgb):
    v = [c / 255 for c in rgb]
    linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in v]
    matrix = [[506752 / 1228815, 87881 / 245763, 12673 / 70218],
              [87098 / 409605, 175762 / 245763, 12673 / 175545],
              [7918 / 409605, 87881 / 737289, 1001167 / 1053270]]
    xyz = [sum(a * b for a, b in zip(row, linear)) for row in matrix]
    white = [.3127 / .329, 1, (1 - .3127 - .329) / .329]
    ratio = [x / w for x, w in zip(xyz, white)]
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = [x ** (1 / 3) if x > epsilon else (kappa * x + 16) / 116 for x in ratio]
    lightness = 116 * f[1] - 16
    # Profile definition: a neutral sRGB triplet is theoretically achromatic.
    # Preserve that identity exactly at fixed histogram's a/b=0 boundary.
    # This is not epsilon rounding of arbitrary chromatic colors.
    if rgb[0] == rgb[1] == rgb[2]:
        return [lightness, 0., 0.]
    return [lightness, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])]


def valid_indices(width, height, rect):
    left, top, w, h = rect
    if min(w, h) <= 0 or min(left, top) < 0 or left + w > width or top + h > height:
        raise ValueError('invalid valid-pixel rectangle')
    return [(x, y, y * width + x) for y in range(top, top + h) for x in range(left, left + w)]


def colors(width, height, rgb_pixels, valid_rect, fixed_edges, tissue_mask=None):
    if len(rgb_pixels) != width * height:
        raise ValueError('RGB buffer shape mismatch')
    if tissue_mask is not None and len(tissue_mask) != width * height:
        raise ValueError('tissue mask shape mismatch')
    selected = [rgb_pixels[i] for _, _, i in valid_indices(width, height, valid_rect)
                if tissue_mask is None or tissue_mask[i]]
    if not selected:
        return dict(status='empty_statistical_mask', channels={}, hue=None)
    channels = {name: [] for name in ['R', 'G', 'B', 'S', 'V', 'L', 'a', 'b']}
    hues = []
    for pixel in selected:
        for name, value in zip(['R', 'G', 'B'], pixel):
            channels[name].append(value)
        h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in pixel))
        channels['S'].append(s)
        channels['V'].append(v)
        if s > 0:
            hues.append(h * 360)
        for name, value in zip(['L', 'a', 'b'], rgb_to_lab_d65(pixel)):
            channels[name].append(value)
    hue = dict(count=len(hues), mean_degrees=None, resultant_length=None, circular_variance=None,
               histogram=histogram(hues, fixed_edges['H']))
    if hues:
        cosine = statistics.mean(math.cos(math.radians(h)) for h in hues)
        sine = statistics.mean(math.sin(math.radians(h)) for h in hues)
        r = min(1., math.hypot(cosine, sine))
        hue.update(resultant_length=r, circular_variance=1-r)
        if r > 1e-12:
            hue['mean_degrees'] = math.degrees(math.atan2(sine, cosine)) % 360
            if hue['mean_degrees'] >= 360:
                hue['mean_degrees'] = 0.
    return dict(status='ok', channels={name:dict(statistics=channel_stats(values),
                histogram=histogram(values, fixed_edges[name])) for name, values in channels.items()}, hue=hue)


def qc(width, height, rgb_pixels, rect, cfg):
    if len(rgb_pixels) != width * height:
        raise ValueError('RGB buffer shape mismatch')
    if not isinstance(cfg['border_band_px'], int) or cfg['border_band_px'] <= 0:
        raise ValueError('border_band_px must be a positive integer')
    for key in ['white_threshold', 'black_threshold']:
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 255:
            raise ValueError('RGB threshold outside 0..255')
    for key in ['low_saturation_threshold', 'tissue_s_min', 'tissue_v_min', 'tissue_v_max']:
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError('HSV threshold outside 0..1')
    if cfg['tissue_v_min'] > cfg['tissue_v_max']:
        raise ValueError('tissue V bounds are reversed')
    for key in ['max_white_fraction', 'max_low_saturation_fraction', 'max_black_border_fraction', 'min_tissue_fraction']:
        limit = cfg.get(key)
        if limit is not None and (not math.isfinite(limit) or not 0 <= limit <= 1):
            raise ValueError('fraction QC limit outside 0..1')
    limit = cfg.get('min_laplacian_variance')
    if limit is not None and (not math.isfinite(limit) or limit < 0):
        raise ValueError('minimum Laplacian variance must be finite and nonnegative')
    selected = valid_indices(width, height, rect)
    left, top, w, h = rect
    tissue_mask = [False] * (width * height)
    white = low_sat = black_border = border_count = tissue_count = 0
    for x, y, i in selected:
        pixel = rgb_pixels[i]
        _, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in pixel))
        white += min(pixel) >= cfg['white_threshold']
        low_sat += s <= cfg['low_saturation_threshold']
        border = min(x-left, y-top, left+w-1-x, top+h-1-y) < cfg['border_band_px']
        border_count += border
        black_border += border and max(pixel) <= cfg['black_threshold']
        tissue = s >= cfg['tissue_s_min'] and cfg['tissue_v_min'] <= v <= cfg['tissue_v_max']
        tissue_mask[i] = tissue
        tissue_count += tissue
    gray = [.299*r + .587*g + .114*b for r, g, b in rgb_pixels]
    laplacian = [gray[i-1] + gray[i+1] + gray[i-width] + gray[i+width] - 4*gray[i]
                 for x, y, i in selected if left < x < left+w-1 and top < y < top+h-1]
    scores = dict(white_fraction=white / len(selected), low_saturation_fraction=low_sat / len(selected),
                  black_border_fraction=black_border / border_count,
                  tissue_fraction=tissue_count / len(selected),
                  laplacian_variance=statistics.pvariance(laplacian) if laplacian else None)
    reasons = []
    for name in ['white_fraction', 'low_saturation_fraction', 'black_border_fraction']:
        limit = cfg.get('max_' + name)
        if limit is not None and scores[name] > limit:
            reasons.append(name + '_above_maximum')
    for name in ['tissue_fraction', 'laplacian_variance']:
        limit = cfg.get('min_' + name)
        if limit is not None:
            if scores[name] is None:
                reasons.append(name + '_undefined')
            elif scores[name] < limit:
                reasons.append(name + '_below_minimum')
    return dict(scores=scores, valid_pixels=len(selected), border_pixels=border_count,
                laplacian_pixels=len(laplacian), tissue_mask=tissue_mask,
                exclusion_reasons=reasons, accepted=not reasons)


def tile_positions(width, height, patch_size, stride, edge, downsample=(1., 1.)):
    pw, ph = patch_size
    sx, sy = stride
    if min(width, height, pw, ph, sx, sy) <= 0 or edge not in ['drop', 'pad']:
        raise ValueError('invalid tiling configuration')
    if len(downsample) != 2 or any(not math.isfinite(v) or v <= 0 for v in downsample):
        raise ValueError('downsample must contain two positive finite scales')
    positions = []
    for y in range(0, height, sy):
        for x in range(0, width, sx):
            valid_w, valid_h = min(pw, width-x), min(ph, height-y)
            if edge == 'drop' and (valid_w < pw or valid_h < ph):
                continue
            positions.append(dict(x0=x*downsample[0], y0=y*downsample[1], width0=pw*downsample[0],
                 height0=ph*downsample[1], patch_width=pw, patch_height=ph, valid_pixels=[0, 0, valid_w, valid_h]))
    return positions


def annotation_coverage(width, height, labels, rect, classes):
    if len(labels) != width * height or len(set(classes)) != len(classes):
        raise ValueError('label shape mismatch or duplicate requested classes')
    selected = [labels[i] for _, _, i in valid_indices(width, height, rect)]
    return dict(denominator=len(selected), counts={str(c):selected.count(c) for c in classes},
                fractions={str(c):selected.count(c) / len(selected) for c in classes})


def objects(width, height, labels, rect, mask_kind, connectivity, mpp=None, overlapping_tiles=False):
    """Patch-local objects. Mask labels are u32; padding ignored; no global counts inferred."""
    if mask_kind not in ['instance', 'semantic'] or connectivity not in [4, 8]:
        raise ValueError('explicit mask kind and 4/8 connectivity required')
    if mpp is not None and (len(mpp) != 2 or any(not math.isfinite(v) or v <= 0 for v in mpp)):
        raise ValueError('patch-pixel MPP must contain two positive finite scales')
    valid = {i for _, _, i in valid_indices(width, height, rect)}
    left, top, w, h = rect
    if len(labels) != width * height:
        raise ValueError('label shape mismatch')
    groups = []
    if mask_kind == 'instance':
        for label in sorted({labels[i] for i in valid} - {0}):
            groups.append((label, [i for i in sorted(valid) if labels[i] == label]))
    else:
        visited = set()
        for seed in sorted(valid):
            if labels[seed] == 0 or seed in visited:
                continue
            stack, region = [seed], []
            visited.add(seed)
            while stack:
                i = stack.pop()
                region.append(i)
                x, y = i % width, i // width
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == dy == 0 or connectivity == 4 and abs(dx)+abs(dy) != 1:
                            continue
                        nx, ny = x+dx, y+dy
                        if not (0 <= nx < width and 0 <= ny < height):
                            continue
                        ni = ny*width+nx
                        if ni in valid and ni not in visited and labels[ni] == labels[seed]:
                            stack.append(ni)
                            visited.add(ni)
            groups.append((labels[seed], sorted(region)))
    rows = []
    for object_number, (source_label, members) in enumerate(groups, start=1):
        xs, ys = [i % width for i in members], [i // width for i in members]
        rows.append(dict(object_id=source_label if mask_kind == 'instance' else object_number,
            source_label=source_label, area_px=len(members),
            centroid_pixel_centers=[statistics.mean(x+.5 for x in xs), statistics.mean(y+.5 for y in ys)],
            bbox_xywh=[min(xs), min(ys), max(xs)-min(xs)+1, max(ys)-min(ys)+1],
            touches_valid_boundary=any(x in [left,left+w-1] or y in [top,top+h-1] for x,y in zip(xs,ys)),
            area_um2=len(members)*mpp[0]*mpp[1] if mpp is not None else None))
    warnings = ['patch_local_instance_counts_must_not_be_summed_across_overlapping_tiles'] if overlapping_tiles else []
    return dict(status='ok' if rows else 'empty_foreground', mask_kind=mask_kind,
                object_count=len(rows), objects=rows, size_distribution=channel_stats([r['area_px'] for r in rows]),
                warnings=warnings, count_scope='patch_local')
