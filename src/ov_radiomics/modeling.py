"""Public training routes; reference estimators are loaded only on request."""
from importlib import import_module
from ._backend import resolve_backend


def _implementation(backend):
    if backend == 'python':
        return import_module('._reference_modeling', __package__)
    # auto selects verified native capabilities, without an estimator fallback.
    resolve_backend('rust' if backend == 'auto' else backend)
    return import_module('._native_learning', __package__)


def train_models(*args, backend='python', **kwargs):
    """Fit with an explicit backend. Native fitting uses patient-grouped SplitPlan.

    rust/auto implement three tasks with typed .rad models. The compatibility
    default preserves the original Python workflow.
    """
    return _implementation(backend).train_models(*args, **kwargs)


def evaluate(predictions, *args, backend=None, **kwargs):
    from ._native_learning import NativeTrainingResult, NativeFusionResult
    from ._types import PredictionTable
    native = isinstance(predictions, (NativeTrainingResult, NativeFusionResult, PredictionTable))
    backend=('rust' if native else 'python') if backend is None else backend
    if native:
        if backend == 'python':
            raise ValueError('Native results are evaluated by the native core; use a reference prediction table for the Python reference.')
        return _implementation(backend).evaluate(predictions, *args, **kwargs)
    if backend in ('rust','auto'):
        raise TypeError('Native evaluation requires an identified PredictionTable or native TrainingResult, not an unverified score frame.')
    return _implementation('python').evaluate(predictions, *args, backend=backend, **kwargs)


def fusion_data(results, *args, backend=None, **kwargs):
    from ._native_learning import NativeTrainingResult
    native = bool(results) and all(isinstance(r, NativeTrainingResult) for r in results.values())
    if bool(results) and not native and any(isinstance(r,NativeTrainingResult) for r in results.values()):
        raise TypeError('Fusion cannot mix native and external/reference training results.')
    chosen=('rust' if native else 'python') if backend is None else backend
    if native and chosen=='python' or not native and chosen in ('auto','rust'):
        raise TypeError('Fusion backend and typed results differ; mixed/unverified tables cannot become native provenance.')
    return _implementation(chosen).fusion_data(results, *args, **kwargs)


def fit_fusion(inputs, *args, backend=None, **kwargs):
    from ._native_learning import NativeFusionData, NativeTrainingResult
    native = isinstance(inputs, NativeFusionData) or isinstance(inputs, dict) and bool(inputs) and all(isinstance(r, NativeTrainingResult) for r in inputs.values())
    if isinstance(inputs,dict) and not native and any(isinstance(r,NativeTrainingResult) for r in inputs.values()):
        raise TypeError('Fusion cannot mix native and external/reference training results.')
    chosen=('rust' if native else 'python') if backend is None else backend
    if native and chosen=='python' or not native and chosen in ('auto','rust'):
        raise TypeError('Fusion backend and typed results differ; mixed/unverified tables cannot become native provenance.')
    return _implementation(chosen).fit_fusion(inputs, *args, **kwargs)


def split_data(*args, backend='python', **kwargs):
    """Legacy random holdout; native workflows use an explicit SplitPlan."""
    if backend != 'python':
        raise ValueError('Use SplitPlan.from_data with explicit patient-level cohorts for native training.')
    return _implementation(backend).split_data(*args, **kwargs)


def make_classifier(*args, backend='python', **kwargs):
    """Optional external estimator. Native model specs go to train_models."""
    if backend != 'python':
        raise ValueError('Native models use train_models(model_params=...) and typed ModelBundle states.')
    return _implementation(backend).make_classifier(*args, **kwargs)


def __getattr__(name):
    # Preserve old imports and joblib paths without eager reference dependencies.
    if name in ('TrainingResult', 'FusionResult'):
        return getattr(_implementation('python'), name)
    raise AttributeError(name)
