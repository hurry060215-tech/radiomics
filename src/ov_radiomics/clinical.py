"""Clinical API with lazy reference packages and explicit native analysis units."""
from importlib import import_module
from ._backend import resolve_backend


def _call(name,args,kwargs,backend):
    if backend == 'python':
        module=import_module('._reference_clinical',__package__)
    else:
        resolve_backend('rust' if backend=='auto' else backend)
        module=import_module('._native_clinical',__package__)
    return getattr(module,name)(*args,**kwargs)


def baseline_table(*args,backend='python',**kwargs):
    return _call('baseline_table',args,kwargs,backend)


def multivariable_logistic(*args,backend='python',**kwargs):
    return _call('multivariable_logistic',args,kwargs,backend)


def univariate_logistic(*args,backend='python',**kwargs):
    return _call('univariate_logistic',args,kwargs,backend)


def icc(*args,backend='rust',**kwargs):
    if backend=='python':
        raise ValueError('ICC is a typed native API; its independent Python reference is an acceptance fixture.')
    return _call('icc',args,kwargs,backend)


def __getattr__(name):
    if name=='LogisticAnalysis':
        return getattr(import_module('._reference_clinical',__package__),name)
    raise AttributeError(name)
