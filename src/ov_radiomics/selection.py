"""Selector construction with lazy, explicit reference dependencies."""


class FeatureSelector:
    """Build a train-fitted selector for the declared backend and task.

    Native alpha grids are evaluated inside whole patient-grouped pipelines;
    fixed-alpha direct fit is available for isolated training data.
    """
    def __new__(cls, *args, backend='python', **kwargs):
        if backend == 'python':
            from ._reference_selection import FeatureSelector as ReferenceSelector
            return ReferenceSelector(*args, **kwargs)
        if backend not in ('auto', 'rust'):
            raise ValueError('backend must be python, rust, or auto.')
        from ._native_selection import NativeFeatureSelector
        return NativeFeatureSelector(*args, backend=backend, **kwargs)


def __getattr__(name):
    if name == '_Filter':
        from ._reference_selection import _Filter
        return _Filter
    raise AttributeError(name)
