"""Owned Python views of canonical native data objects; numerical work stays Rust."""
from copy import deepcopy
import json
from ._backend import resolve_backend


class TileTable:
    """Validated patient→slide→tile metadata and optional numeric features."""
    def __init__(self, table, provenance=None):
        if isinstance(table, TileTable):
            provenance = table.provenance if provenance is None else provenance
            table = table.to_dict()
        native = resolve_backend('rust')
        if not hasattr(native, 'validate_tile_table'):
            raise ImportError('Rebuild the current native extension for TileTable support.')
        self._table = json.loads(native.validate_tile_table(json.dumps(table, allow_nan=False)))
        self._provenance = deepcopy(provenance or {})

    def to_dict(self):
        return deepcopy(self._table)

    @classmethod
    def _from_native(cls, table, provenance=None):
        instance = cls.__new__(cls)
        instance._table = table
        instance._provenance = deepcopy(provenance or {})
        return instance

    @property
    def table(self):
        return self.to_dict()

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    @property
    def tiles(self):
        return deepcopy(self._table['tiles'])

    @property
    def feature_names(self):
        return list(self._table['feature_names'])

    @property
    def slide_status(self):
        return dict(self._table['slide_status'])

    @property
    def slide_patients(self):
        return dict(self._table['slide_patients'])

    @property
    def features(self):
        import numpy as np
        data = self._table['features']
        if data is None:
            return None
        shape = (data['rows'], data['cols'])
        values = np.asarray(data['values'], dtype=float).reshape(shape)
        values[np.asarray(data['missing'], dtype=bool).reshape(shape)] = np.nan
        return values

    def __getitem__(self, key):
        return deepcopy(self._table[key])

    def __len__(self):
        return len(self._table['tiles'])


class _NativeObject:
    _operation = None
    _argument = None

    def __init__(self, value):
        from ._native_common import request
        self._payload = request(self._operation, **{self._argument: value})

    def to_dict(self):
        return deepcopy(self._payload)

    @classmethod
    def _from_native(cls, payload):
        instance = cls.__new__(cls)
        instance._payload = payload
        return instance


class ImageVolume(_NativeObject):
    """Scalar 2D/3D x-fastest buffer with complete native-unit RAS geometry."""
    _operation, _argument = 'validate_image', 'image'

    @property
    def geometry(self):
        return deepcopy(self._payload['geometry'])

    @property
    def array(self):
        import numpy as np
        geometry = self._payload['geometry']
        shape = tuple(reversed(geometry['shape'][:geometry['ndim']]))
        return np.asarray(self._payload['values'], dtype=float).reshape(shape)

    @property
    def affine(self):
        import numpy as np
        return np.asarray(self._payload['geometry']['affine'], dtype=float)


class MaskVolume(ImageVolume):
    _operation, _argument = 'validate_mask', 'mask'

    @property
    def array(self):
        import numpy as np
        geometry = self._payload['geometry']
        shape = tuple(reversed(geometry['shape'][:geometry['ndim']]))
        return np.asarray(self._payload['labels'], dtype=np.uint32).reshape(shape)


class ImageSample(_NativeObject):
    _operation, _argument = 'validate_image_sample', 'sample'

    def __init__(self, metadata, image=None, mask=None):
        if image is None and mask is None:
            payload = metadata
        else:
            payload = dict(metadata=metadata,
                image=image.to_dict() if hasattr(image, 'to_dict') else image,
                mask=mask.to_dict() if hasattr(mask, 'to_dict') else mask)
        super().__init__(payload)

    @property
    def metadata(self):
        return deepcopy(self._payload['metadata'])

    @property
    def image(self):
        return ImageVolume(self._payload['image'])

    @property
    def mask(self):
        return MaskVolume(self._payload['mask'])

    def extract(self, config, *, roi_label=1, intensity_mask=None, families=None):
        """Extract an explicitly prepared array sample using the shared native engine."""
        import numpy as np
        from ._native_common import request
        output = request('array_features', image=self._payload['image'],
            morphology_mask=self._payload['mask'],
            intensity_mask=None if intensity_mask is None else intensity_mask.to_dict(),
            roi_label=roi_label, config=config, families=families)
        names = list(output)
        values = np.asarray([output[name] for name in names], dtype=float)
        missing = ~np.isfinite(values)
        quality = [dict(code='undefined_feature', message=f'{name}: {output[name]}',
            severity='warning', sample_id=self._payload['metadata']['sample_id'])
            for name, invalid in zip(names, missing) if invalid]
        table = dict(data=dict(rows=1, cols=len(names), values=np.where(missing, 0., values).tolist(),
            missing=missing.tolist()), samples=[self.metadata],
            features=[dict(name=name, definition='python_golden_v1 explicit native Original subset', unit=None) for name in names],
            effective_parameters=dict(extraction=config, roi_label=roi_label, families=families), quality=quality)
        return FeatureTable(table)


class FeatureTable(_NativeObject):
    _operation, _argument = 'validate_feature_table', 'table'

    @property
    def features(self):
        import pandas as pd
        from ._native_common import buffer_array
        frame = pd.DataFrame(buffer_array(self._payload['data']),
            columns=[f['name'] for f in self._payload['features']])
        frame.insert(0, 'ID', [s['sample_id'] for s in self._payload['samples']])
        return frame

    @property
    def metadata(self):
        import pandas as pd
        return pd.DataFrame(deepcopy(self._payload['samples']))

    @property
    def definitions(self):
        return deepcopy(self._payload['features'])

    @property
    def effective_parameters(self):
        return deepcopy(self._payload['effective_parameters'])

    def save(self, path, *, overwrite=False, null_value=''):
        from ._native_common import request
        return request('write_feature_table', table=self.to_dict(), path=str(path),
            overwrite=overwrite, null_value=null_value)


class PredictionTable(_NativeObject):
    _operation, _argument = 'validate_prediction_table', 'table'

    @property
    def dataframe(self):
        from ._native_common import prediction_frame
        return prediction_frame(self._payload)

    @property
    def prediction_unit(self):
        return self._payload['unit']

    def aggregate_patients(self):
        """Native equal-sample mean; tiles first average within slide, then equal slides."""
        from ._native_common import request
        return PredictionTable._from_native(request('aggregate_patient_predictions',predictions=self.to_dict()))


class ModelBundle:
    """Portable typed native `.rad` model; loading never executes serialized code."""
    def __init__(self, value):
        from ._native_common import request
        kind = value.get('manifest', {}).get('training_configuration', {}).get('model_kind')
        if kind not in ('pipeline', 'fusion'):
            raise ValueError('Unknown native model type; expected a typed pipeline or fusion .rad container.')
        self._kind = kind
        self._payload = request('validate_' + kind, model=value)

    @classmethod
    def load(cls, path):
        from pathlib import Path
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return deepcopy(self._payload)

    @property
    def manifest(self):
        return deepcopy(self._payload['manifest'])

    @property
    def classes_(self):
        import numpy as np
        return np.asarray(self._payload['manifest']['classes'])

    def save(self, path, *, overwrite=False):
        from ._native_common import request
        return request('save_' + self._kind, model=self.to_dict(), path=str(path), overwrite=overwrite)

    def predict_study(self, study):
        """Predict a canonical StudyData (or named fusion studies) without outcomes."""
        from ._native_common import request
        key = 'studies' if self._kind == 'fusion' else 'study'
        return PredictionTable(request('predict_' + self._kind, model=self.to_dict(), **{key: study}))

    def predict(self,data,*,id_col='ID',patient_id=None,split='group',cohort='external'):
        """Predict an identified DataFrame with no outcome column or refitting."""
        from ._native_common import study
        if self._kind!='pipeline':
            raise ValueError('Fusion loading requires named component studies through predict_study; retain each component identity.')
        fitted=self._payload['fitted']['preprocessing']
        numeric=[state['name'] for state in fitted['numeric']]
        categorical=[state['name'] for state in fitted['categorical']]
        converted,_=study(data,None,numeric+categorical,task=self.manifest['task'],classes=self.manifest['classes'],
            categorical=categorical,split=split,id_col=id_col,patient_id=patient_id,default_cohort=cohort)
        return self.predict_study(converted)

    def predict_proba(self, data):
        from ._native_common import request, tabular, buffer_array
        if self._kind != 'pipeline':
            raise ValueError('Fusion probability prediction requires named component studies with identity metadata.')
        fitted = self._payload['fitted']['preprocessing']
        numeric = [s['name'] for s in fitted['numeric']]
        categorical = [s['name'] for s in fitted['categorical']]
        result = request('predict_matrix', model=self.to_dict(),
            features=tabular(data, numeric+categorical, categorical))
        if result['task'] != 'classification':
            raise ValueError('This model predicts regression values, not probabilities.')
        return buffer_array(result['probabilities'])


def load_model(path):
    return ModelBundle.load(path)
