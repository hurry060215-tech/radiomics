"""Array/table conversion and versioned calls into the shared native core."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from ._backend import resolve_backend
from .split import _label


def clean_json(value):
    if value is None or type(value) in (str,int,bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else ('NaN' if math.isnan(value) else ('Inf' if value>0 else '-Inf'))
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if np.isnan(value) else ('Inf' if value > 0 else '-Inf')
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def encode_json(value):
    """Fast finite typed-buffer path; explicit nonfinite image states on demand."""
    def convert(item):
        if isinstance(item,np.generic):
            return item.item()
        if isinstance(item,np.ndarray):
            return item.tolist()
        if isinstance(item,Path):
            return str(item)
        raise TypeError('Unsupported native JSON conversion type: '+type(item).__name__)
    try:
        return json.dumps(value,allow_nan=False,ensure_ascii=False,separators=(',',':'),default=convert)
    except ValueError as error:
        if 'Out of range float values' not in str(error):
            raise
        return json.dumps(clean_json(value),allow_nan=False,ensure_ascii=False,separators=(',',':'))


def request(operation, **values):
    native = resolve_backend('rust')
    if not hasattr(native, 'request_json'):
        raise ImportError('The native extension predates this API; rebuild the current checkout.')
    envelope = dict(schema_version=1, request=dict(operation=operation, **values))
    # Nonfinite image/statistic states have explicit string encodings in Rust.
    # Numeric feature buffers below instead use finite zero + missing flags.
    return json.loads(native.request_json(encode_json(envelope)))


def numeric_buffer(values):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError('Complex numerical values cannot be discarded or treated as real features.')
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError('A numerical feature matrix must be explicitly two-dimensional.')
    if np.isinf(matrix).any():
        raise ValueError('Numeric feature values contain infinity.')
    missing = np.isnan(matrix)
    return dict(rows=matrix.shape[0], cols=matrix.shape[1],
        values=np.where(missing, 0., matrix).ravel(order='C').tolist(),
        missing=missing.ravel(order='C').tolist())


def buffer_array(buffer):
    values = np.asarray(buffer['values'], dtype=float).reshape(buffer['rows'], buffer['cols'])
    values[np.asarray(buffer['missing'], dtype=bool).reshape(values.shape)] = np.nan
    return values


def string_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError('Observed numeric labels/metadata must be finite.')
    return _label(value)


def tabular(data, features=None, categorical=None):
    frame = data.copy(deep=False) if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    names = [str(c) for c in frame.columns]
    if len(names) != len(set(names)):
        raise ValueError('Feature names must be unique.')
    frame.columns = names
    features = names if features is None else [str(c) for c in features]
    categorical = [] if categorical is None else [str(c) for c in categorical]
    if len(features) != len(set(features)) or len(categorical) != len(set(categorical)) or not set(categorical) <= set(features):
        raise ValueError('Feature and categorical names must be unique and categorical columns must be features.')
    if not set(features) <= set(frame.columns):
        raise ValueError('Prediction/training data are missing named feature columns.')
    numeric = [c for c in features if c not in categorical]
    if numeric:
        if any(pd.api.types.is_complex_dtype(frame[column].dtype) for column in numeric):
            raise ValueError('Complex numerical feature columns are unsupported.')
        try:
            values = frame[numeric].to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError('Numeric features cannot be converted to float64; declare category columns explicitly.') from exc
    else:
        values = np.empty((len(frame), 0), dtype=np.float64)
    categories = [[string_value(v) for v in row] for row in frame[categorical].itertuples(index=False, name=None)]
    return dict(rows=len(frame), numeric_names=numeric, numeric=numeric_buffer(values),
        categorical_names=categorical, categorical_values=[v for row in categories for v in row])


def model_target(y, task, classes=None, positive_label=1):
    if task not in ('binary','multiclass','regression'):
        raise ValueError('task must be binary, multiclass, or regression.')
    series = pd.Series(y)
    if series.isna().any():
        raise ValueError('Training outcomes must be observed.')
    if task == 'regression':
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Regression outcomes must be finite.')
        return dict(task='regression', values=values.tolist()), []
    labels = [string_value(value) for value in series]
    if classes is None:
        classes = sorted(set(labels))
        if task == 'binary':
            positive = string_value(positive_label)
            if len(classes) != 2 or positive not in classes:
                raise ValueError('Binary outcomes need two classes including positive_label, or an explicit class order.')
            classes = [c for c in classes if c != positive] + [positive]
    classes = [string_value(c) for c in classes]
    if task=='binary' and len(classes)!=2 or task=='multiclass' and len(classes)<3:
        raise ValueError('Declared binary/multiclass task disagrees with the observed class count.')
    mapping = {c: i for i, c in enumerate(classes)}
    if any(label not in mapping for label in labels):
        raise ValueError('Observed outcome uses an unknown training class.')
    return dict(task='classification', classes=classes, labels=[mapping[label] for label in labels]), classes


def study(data, target, features, *, task, classes, categorical=None, split='group',
          id_col='ID', patient_id=None, fit_unit='sample', default_cohort=None):
    frame = data.copy(deep=False)
    if target is not None and target not in frame:
        raise ValueError('The explicitly named outcome column is missing: '+str(target))
    if id_col not in frame or frame[id_col].isna().any() or frame[id_col].duplicated().any():
        raise ValueError('Study data require unique nonmissing sample IDs.')
    group = patient_id or ('patient_id' if 'patient_id' in frame else id_col)
    if group not in frame or frame[group].isna().any():
        raise ValueError('Patient identities must be complete.')
    reserved = {id_col, group, split, target, 'slide_id', 'tile_id', 'lesion_id'}
    if set(features) & reserved:
        raise ValueError('IDs, cohorts and outcomes must remain separate from numerical/categorical model features.')
    if split not in frame and default_cohort is None:
        raise ValueError('Supply explicit cohort assignments before native grouped training.')
    cohorts = frame[split].tolist() if split in frame else [default_cohort]*len(frame)
    if any(pd.isna(c) or not str(c).strip() for c in cohorts):
        raise ValueError('Cohort assignments must be complete.')
    samples = []
    metadata_keys=('slide_id','lesion_id','modality','region','timepoint','center')
    columns=list(dict.fromkeys([id_col,group]+[key for key in metadata_keys if key in frame]))
    for values in frame[columns].itertuples(index=False,name=None):
        row=dict(zip(columns,values))
        optional={key:string_value(row.get(key)) for key in metadata_keys}
        samples.append(dict(sample_id=string_value(row[id_col]),patient_id=string_value(row[group]),**optional))
    if target is None or target not in frame:
        outcome = dict(task='unlabeled')
    elif task == 'regression':
        values = pd.to_numeric(frame[target], errors='raise').to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError('Regression outcomes contain infinity.')
        outcome = dict(task='regression', name=target, values=[None if np.isnan(v) else float(v) for v in values])
    else:
        outcome = dict(task='classification', name=target, values=[string_value(v) for v in frame[target]])
    return dict(samples=samples, cohorts=[str(c) for c in cohorts],
        features=tabular(frame, features, categorical), outcome=outcome), group


def prediction_frame(table, model='Model', truth=None):
    rows = []
    classes = table['classes']
    for row in table['rows']:
        source = row['training_source']
        kind = ('oof' if source.startswith(('oof:', 'oof_fixed_weight:')) else
            'apparent_meta_training' if source.startswith('apparent_meta_training') else
            'apparent' if row['cohort'] == 'train' else 'holdout' if row['cohort'] == 'test' else 'external')
        result = dict(ID=row['sample_id'], patient_id=row['patient_id'], slide_id=row['slide_id'],
            tile_id=row['tile_id'], model=model, group=row['cohort'], fold=row['fold'],
            prediction_unit=table['unit'], prediction_type=kind, training_source=source)
        if table['task'] == 'regression':
            result['prediction'] = row['value']
        else:
            result.update({'probability::'+c: p for c, p in zip(classes, row['probabilities'])})
            if table['task'] == 'binary':
                result['probability'] = row['probabilities'][1]
        if truth is not None:
            label = truth.get(row['sample_id'])
            result['target_value'] = label
            result['y_true'] = (label if table['task'] == 'regression' else classes.index(label)
                if label in classes else np.nan)
        rows.append(result)
    return pd.DataFrame(rows)
