"""Native selector configuration and thin fit/transform views."""
from copy import deepcopy
import numpy as np
import pandas as pd
from ._native_common import request, tabular, model_target, buffer_array


class NativeFeatureSelector:
    """Train-fitted native screening; alpha grids are tuned with whole pipelines.

    A fixed alpha can be fitted directly. Pass an explicit ``alphas`` grid to
    ``train_models`` for patient-grouped joint pipeline selection. Standalone
    legacy LassoCV remains available with the explicit Python backend.
    """
    def __init__(self, scale='standard', missing='median', univariate='auto',
                 p_threshold=.05, p_adjust=None, corr_method='pearson', corr_threshold=.9,
                 lasso=True, alpha=.01, cv=5, coef_threshold=1e-6, random_state=0,
                 alphas=None, task='binary', sparse_method=None, variance_threshold=0.,
                 all_missing='error', backend='rust'):
        for key, value in locals().copy().items():
            if key != 'self':
                setattr(self, key, value)

    def get_params(self, deep=True):
        names = ('scale','missing','univariate','p_threshold','p_adjust','corr_method',
            'corr_threshold','lasso','alpha','cv','coef_threshold','random_state','alphas',
            'task','sparse_method','variance_threshold','all_missing','backend')
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError('Unknown selector parameter: '+key)
            setattr(self, key, value)
        return self

    def configurations(self, task=None):
        task = task or self.task
        if self.backend not in ('rust','auto') or self.missing != 'median':
            raise ValueError('Native selection uses median imputation; all_missing=drop explicitly deletes wholly missing columns. Use backend=python for legacy missing=drop semantics.')
        if self.corr_method != 'pearson' or self.p_adjust not in (None,'fdr_bh','bh'):
            raise ValueError('Native correlation is Pearson and optional adjustment is BH; choose an explicit external backend for other methods.')
        method = {None:'none','mannwhitney':'mann_whitney','kruskal':'kruskal_wallis'}.get(self.univariate,self.univariate)
        preprocessing = dict(scaling=self.scale or 'none', all_missing=self.all_missing)
        sparse = self.sparse_method
        if sparse is None and self.lasso:
            sparse = 'ovr_l1_logistic' if task == 'multiclass' else 'squared_lasso'
        sparse = {'lasso':'squared_lasso','l1_logistic':'binary_l1_logistic'}.get(sparse,sparse)
        if task == 'multiclass' and sparse == 'squared_lasso':
            raise ValueError('Multiclass labels cannot be treated as a continuous squared-loss LASSO target.')
        if sparse is None:
            alphas = [None]
        elif self.alphas is not None:
            alphas = list(self.alphas)
        elif self.alpha == 'cv':
            alphas = np.logspace(-4, -.5, 16).tolist()
        else:
            alphas = [float(self.alpha)]
        if not alphas:
            raise ValueError('Sparse alpha grid is empty.')
        configurations = []
        for alpha in alphas:
            state = None if sparse is None else dict(method=sparse, alpha=alpha,
                tolerance=1e-8, max_iterations=10000)
            configurations.append(dict(variance_threshold=self.variance_threshold,
                corr_threshold=self.corr_threshold, univariate=method,
                p_threshold=self.p_threshold, adjust_bh=self.p_adjust is not None,
                sparse=state, coefficient_threshold=self.coef_threshold))
        return preprocessing, configurations

    def fit(self, X, y, **kwargs):
        if kwargs:
            raise ValueError('Use train_models with SplitPlan for grouped alpha tuning; direct fixed-alpha fit takes X and y only.')
        preprocessing, configurations = self.configurations()
        if len(configurations) != 1:
            raise ValueError('An alpha grid must be fitted inside patient-grouped train_models, not on a standalone pooled selector.')
        frame = X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        target, self.classes_ = model_target(y,self.task)
        self._state = request('selector_fit',data=tabular(frame),target=target,
            preprocessing=preprocessing,selection=configurations[0])
        self._adopt(self._state)
        return self

    def _adopt(self, state):
        self._state = deepcopy(state)
        selector = state['selection']
        self.feature_names_in_ = list(selector['feature_names'])
        self.selected_features_ = [self.feature_names_in_[i] for i in selector['selected']]
        self.univariate_results_ = pd.DataFrame([dict(feature=self.feature_names_in_[r['feature_index']],
            p_value=r['p_value'],adjusted_p_value=r['adjusted_p_value']) for r in selector['univariate_results']])
        self.alpha_ = None if selector['sparse'] is None else selector['sparse']['spec']['alpha']

    def transform(self, X):
        if not hasattr(self,'_state'):
            raise ValueError('The selector is not fitted.')
        frame = X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        output = request('selector_transform',state=self._state,data=tabular(frame))
        return pd.DataFrame(buffer_array(output),index=frame.index,columns=self.selected_features_)

    def fit_transform(self, X, y, **kwargs):
        return self.fit(X,y,**kwargs).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_,dtype=object)
