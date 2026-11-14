"""Native Lloyd KMeans with typed train-fitted scaling and local cluster labels."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ._native_common import request,numeric_buffer,buffer_array,string_value


def _matrix(data,ids=None,feature_names=None):
    if isinstance(data,pd.DataFrame):
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in data.dtypes):
            raise ValueError('Habitat features must be real; complex components cannot be discarded.')
        ids=list(data.index) if ids is None else ids
        feature_names=list(data.columns) if feature_names is None else feature_names
        values=data[list(feature_names)].to_numpy(dtype=float,na_value=np.nan)
    else:
        if ids is None or feature_names is None:
            raise ValueError('Array habitats require explicit observation IDs and feature names.')
        values=np.asarray(data)
    return numeric_buffer(values),[string_value(value) for value in ids],[str(value) for value in feature_names]


class HabitatModel:
    def __init__(self,state):
        self._state=request('habitat_validate',state=state)

    def to_dict(self):
        return deepcopy(self._state)

    @property
    def report(self):
        return deepcopy(self._state['report'])

    @property
    def centers_(self):
        return pd.DataFrame(buffer_array(request('habitat_centers',state=self._state)),columns=self._state['feature_names'])

    def predict(self,data,*,ids=None,feature_names=None):
        matrix,ids,names=_matrix(data,ids,feature_names)
        result=request('habitat_predict',state=self._state,data=matrix,ids=ids,feature_names=names)
        distances=buffer_array(result['distances'])
        frame=pd.DataFrame(dict(ID=result['ids'],cluster=result['labels']))
        for i in range(distances.shape[1]):
            frame[f'distance_to_cluster_{i}']=distances[:,i]
        frame.attrs.update(cluster_scope='Local mathematical cluster labels; not shared histologic tissue identities.',
            distance_space='Euclidean in saved training-scaled feature space')
        return frame

    def save(self,path,*,overwrite=False):
        state=request('habitat_validate',state=self._state)
        with Path(path).open('w' if overwrite else 'x',encoding='utf-8') as handle:
            json.dump(dict(schema='radiomics.habitat.v1',state=state),handle,ensure_ascii=False,allow_nan=False)

    @classmethod
    def load(cls,path):
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        if set(value)!={'schema','state'} or value['schema']!='radiomics.habitat.v1':
            raise ValueError('Unknown typed habitat container schema.')
        return cls(value['state'])


def fit(data,*,ids=None,feature_names=None,clusters=2,scaling='standard',initial_centers=None,
        random_state=0,tolerance=1e-12,max_iterations=300):
    """Fit supplied training observations only. Empty/nonconverged clusters error."""
    matrix,ids,names=_matrix(data,ids,feature_names)
    initialization=dict(method='kmeans_plus_plus') if initial_centers is None else dict(method='explicit',centers_raw=numeric_buffer(initial_centers))
    state=request('habitat_fit',data=matrix,ids=ids,feature_names=names,
        config=dict(clusters=clusters,scaling=scaling,initialization=initialization,seed=random_state,
            tolerance=tolerance,max_iterations=max_iterations))
    return HabitatModel(state)
