"""Independent native radiomics with optional reference backends and Python plots."""
from importlib import import_module
from ._backend import backend_info
from .split import SplitPlan

__version__ = '0.1.0'

# Old public symbols still resolve to the same function/class; reference packages
# are imported only when their functionality is requested.
_EXPORTS = {
    'image': ('match_images', 'check_images', 'extract', 'read_cohort', 'ExtractionResult'),
    'selection': ('FeatureSelector',),
    'clinical': ('baseline_table', 'univariate_logistic', 'multivariable_logistic'),
    'modeling': ('split_data', 'make_classifier', 'train_models', 'evaluate', 'fusion_data',
                 'fit_fusion', 'TrainingResult', 'FusionResult'),
    'metrics': ('binary_metrics', 'delong_test', 'decision_curve', 'hosmer_lemeshow'),
    '_validation': ('join_data',),
    'plotting': ('plot_roc', 'plot_calibration', 'plot_dca', 'plot_confusion_matrix',
        'plot_prediction_scores', 'plot_correlation', 'plot_lasso_path', 'plot_lasso_cv',
        'plot_feature_weights', 'plot_cv_results', 'plot_metrics', 'plot_feature_summary',
        'plot_feature_pvalues', 'plot_embedding', 'plot_learning_curve', 'plot_nomogram'),
}
_LOOKUP = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = ['backend_info', 'SplitPlan', *_LOOKUP]


def __getattr__(name):
    module = _LOOKUP.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    try:
        value = getattr(import_module('.' + module, __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name.split('.')[0] in {'sklearn', 'statsmodels', 'scipy', 'joblib'}:
            raise ImportError(f'{name} requires the optional Python reference backend; '
                'install ov-radiomics[reference].') from exc
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
