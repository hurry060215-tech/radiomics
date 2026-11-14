"""Optional SimpleITK / PyRadiomics adapters; no image dependencies at import time."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from ._validation import unique_ids


def _sitk():
    try:
        import SimpleITK
        return SimpleITK
    except ImportError as exc:
        raise ImportError('Install ov-radiomics[image] to read medical images.') from exc


def _sample_roi_label(row, default):
    value = row.get('roi_label')
    if value is None or pd.isna(value):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('roi_label must be an integer.') from exc
    if isinstance(value, (float, np.floating)) and value != parsed:
        raise ValueError('roi_label must be an integer, without truncation.')
    return parsed


def match_images(root, image_dir="images", mask_dir="masks"):
    """Pair .nii/.nii.gz files by exact filename. IDs retain the filename."""
    root = Path(root)
    def files(directory):
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        return {p.name: p.resolve() for p in folder.iterdir()
                if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))}
    images, masks = files(image_dir), files(mask_dir)
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Image/mask names differ or are empty. Missing masks: {sorted(images.keys()-masks.keys())}; missing images: {sorted(masks.keys()-images.keys())}")
    return pd.DataFrame([{'ID': key, 'image': str(images[key]), 'mask': str(masks[key])}
                         for key in sorted(images)])


def check_images(samples, roi_label=1, geometry_tolerance=1e-6):
    """Read each pair and report geometry mismatch and absent ROI without modifying files."""
    unique_ids(samples)
    sitk = _sitk()
    rows = []
    for row in samples.to_dict('records'):
        result = {'ID': row['ID'], 'ok': False, 'error': ''}
        try:
            image, mask = sitk.ReadImage(str(row['image'])), sitk.ReadImage(str(row['mask']))
            if image.GetDimension() != mask.GetDimension():
                raise ValueError('Image/mask dimensions differ.')
            mismatch = []
            for name in ('Size', 'Spacing', 'Origin', 'Direction'):
                a, b = getattr(image, 'Get'+name)(), getattr(mask, 'Get'+name)()
                equal = a == b if name == 'Size' else np.allclose(a, b, atol=geometry_tolerance, rtol=0)
                if not equal:
                    mismatch.append(name)
            count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask) == _sample_roi_label(row, roi_label)))
            result.update(roi_voxels=count, geometry_mismatch=','.join(mismatch))
            if count == 0:
                raise ValueError('Requested ROI is empty or absent.')
            if mismatch:
                raise ValueError('Geometry mismatch: ' + ', '.join(mismatch))
            result['ok'] = True
        except Exception as exc:
            result['error'] = str(exc)
        rows.append(result)
    return pd.DataFrame(rows)


@dataclass
class ExtractionResult:
    features: pd.DataFrame
    diagnostics: pd.DataFrame
    config: dict

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(path / 'features.csv', index=False)
        self.diagnostics.to_csv(path / 'diagnostics.csv', index=False)
        (path / 'config.json').write_text(json.dumps(self.config, indent=2, default=str), encoding='utf-8')


def extract(samples, params=None, roi_label=1, correct_mask=False, n_jobs=1, on_error="raise",
            backend='python', profile=None, bin_width=None, bin_count=None, cache=None):
    """Extract ROI features. params is a PyRadiomics YAML path or parameter dictionary.

    Explicit correct_mask overrides the backend setting. No implicit disk cache.
    on_error='record' keeps failed IDs in diagnostics and features (NaN values).
    """
    if backend in ('rust', 'auto'):
        return _extract_native(samples, params, roi_label, correct_mask, n_jobs, on_error,
                               profile, bin_width, bin_count, cache)
    if backend != 'python':
        raise ValueError('backend must be python, rust, or auto.')
    if profile is not None or bin_width is not None or bin_count is not None or cache is not None:
        raise ValueError('profile/bin_width/bin_count/cache are native options; use backend="rust" or configure the explicit Python params.')
    from joblib import Parallel, delayed
    unique_ids(samples)
    if on_error not in ('raise', 'record') or samples.empty:
        raise ValueError("Use nonempty samples and on_error='raise' or 'record'.")
    try:
        import radiomics
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise ImportError('Install ov-radiomics[image] for extraction.') from exc
    import copy
    if isinstance(params, (str, Path)):
        import yaml
        config = yaml.safe_load(Path(params).read_text(encoding='utf-8'))
    else:
        config = copy.deepcopy(params)
    template = RadiomicsFeatureExtractor(config, correctMask=correct_mask) if config is not None else RadiomicsFeatureExtractor(correctMask=correct_mask)
    effective = {'setting': template.settings, 'imageType': template.enabledImagetypes,
                 'featureClass': template.enabledFeatures}

    def one(row):
        features, diagnostic = {'ID': row['ID']}, {'ID': row['ID'], 'ok': False}
        try:
            qc = check_images(pd.DataFrame([row]), roi_label=roi_label).iloc[0]
            if not qc['ok'] and not (correct_mask and str(qc['error']).startswith('Geometry mismatch:')):
                raise ValueError(qc['error'])
            extractor = RadiomicsFeatureExtractor(copy.deepcopy(effective))
            output = extractor.execute(str(row['image']), str(row['mask']), label=_sample_roi_label(row, roi_label))
            for key, value in output.items():
                if key.startswith('diagnostics_'):
                    diagnostic[key] = str(value)
                else:
                    features[key] = float(value)
            diagnostic.update(ok=True, error='')
        except Exception as exc:
            if on_error == 'raise':
                raise RuntimeError(f"Extraction failed for {row['ID']}: {exc}") from exc
            diagnostic['error'] = str(exc)
        return features, diagnostic
    results = Parallel(n_jobs=n_jobs)(delayed(one)(row) for row in samples.to_dict('records'))
    config_out = {'effective': effective, 'pyradiomics_version': radiomics.__version__,
                  'simpleitk_version': _sitk().Version_VersionString(), 'roi_label': roi_label,
                  'samples': samples.to_dict('records')}
    config_out['parameter_sha256'] = hashlib.sha256(json.dumps(effective, sort_keys=True, default=str).encode()).hexdigest()
    return ExtractionResult(pd.DataFrame([r[0] for r in results]), pd.DataFrame([r[1] for r in results]), config_out)


def read_cohort(path):
    """Read native cohort CSV: sample_id/patient_id/image/mask are required.

    Paths are relative to the CSV. Output ID is the sample identifier; repeated
    patient_id is preserved. Optional metadata stays outside numerical features.
    """
    from ._backend import resolve_backend
    native = resolve_backend('rust')
    records = json.loads(native.read_cohort(str(Path(path).resolve())))
    return pd.DataFrame([dict(ID=row['metadata']['sample_id'],
        **{k: v for k, v in row['metadata'].items() if k != 'sample_id'},
        image=row['image'], mask=row['mask'], organ_mask=row['organ_mask'], roi_label=row.get('roi_label')) for row in records])


def _extract_native(samples, params, roi_label, correct_mask, n_jobs, on_error,
                    profile, bin_width, bin_count, cache):
    from ._backend import resolve_backend
    import copy
    unique_ids(samples)
    if params is not None and (bin_width is not None or bin_count is not None):
        raise ValueError('Choose a complete native params configuration or bin_width/bin_count keywords.')
    if samples.empty or on_error not in ('raise', 'record'):
        raise ValueError('Use nonempty samples and on_error=raise or record.')
    if correct_mask:
        raise ValueError('Native extraction requires explicit image/mask alignment; resample the mask explicitly.')
    if isinstance(params, (str, Path)):
        config = json.loads(Path(params).read_text(encoding='utf-8'))
    elif params is None:
        if (bin_width is None) == (bin_count is None):
            raise ValueError('Native extraction requires exactly one explicit bin_width or bin_count.')
        discretization = {'method': 'fixed_width', 'width': bin_width} if bin_width is not None else {'method': 'fixed_count', 'count': bin_count}
        config = dict(extraction=dict(profile=profile or 'python_golden_v1',
            discretization=discretization, aggregation='direction_mean',
            gldm_alpha=0, voxel_array_shift=0.0), labels=[roi_label], preparation=[])
    else:
        if bin_width is not None or bin_count is not None:
            raise ValueError('Choose a complete native params configuration or bin_width/bin_count keywords.')
        config = copy.deepcopy(params)
    if not isinstance(config, dict) or not isinstance(config.get('extraction'), dict):
        raise ValueError('Unsupported native configuration. Supply ExtractConfig JSON; use backend="python" explicitly for external PyRadiomics settings.')
    if profile is not None and config['extraction'].get('profile') != profile:
        raise ValueError('profile conflicts with the supplied native configuration.')
    rows = []
    for row in samples.to_dict('records'):
        def optional(key):
            value = row.get(key)
            return None if value is None or pd.isna(value) else str(value)
        if pd.isna(row.get('patient_id', row['ID'])):
            raise ValueError('patient_id must be complete when supplied.')
        metadata = dict(sample_id=str(row['ID']), patient_id=str(row.get('patient_id', row['ID'])),
            **{k: optional(k) for k in ('slide_id', 'lesion_id', 'modality', 'region', 'timepoint', 'center')})
        label = _sample_roi_label(row, None)
        rows.append(dict(metadata=metadata, image=str(Path(row['image']).resolve()),
            mask=str(Path(row['mask']).resolve()), organ_mask=optional('organ_mask'), roi_label=label))
    native = resolve_backend('rust')
    batch = json.loads(native.extract_images(json.dumps(rows, allow_nan=False),
        json.dumps(config, allow_nan=False), n_jobs, str(Path(cache).resolve()) if cache is not None else None))
    if batch['failures'] and on_error == 'raise':
        failures = '; '.join(f"{f['metadata']['sample_id']}: {f['error']['message']}" for f in batch['failures'])
        raise RuntimeError('Native extraction failed: ' + failures)
    feature_rows = []
    diagnostic_rows = []
    for row in batch['records']:
        identifier = row['metadata']['sample_id']
        feature_rows.append(dict(ID=identifier, **{k: np.nan if v is None else v for k, v in row['features'].items()}))
        diagnostic_rows.append(dict(ID=identifier, patient_id=row['metadata']['patient_id'],
            ok=True, error='', cache_key=row['cache_key'], quality=json.dumps(row['quality'])))
    for row in batch['failures']:
        identifier = row['metadata']['sample_id']
        feature_rows.append(dict(ID=identifier))
        diagnostic_rows.append(dict(ID=identifier, patient_id=row['metadata']['patient_id'],
            ok=False, error=row['error']['message'], error_code=row['error']['code']))
    return ExtractionResult(pd.DataFrame(feature_rows).sort_values('ID').reset_index(drop=True),
        pd.DataFrame(diagnostic_rows).sort_values('ID').reset_index(drop=True), batch)
