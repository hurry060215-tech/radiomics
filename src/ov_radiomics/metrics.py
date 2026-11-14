"""Native binary evaluation with a lazy, explicit Python Golden reference path."""
import importlib
import numpy as np
import pandas as pd
from ._backend import resolve_backend
from ._validation import probabilities
from ._native_common import request


def _reference():
    try:
        return importlib.import_module('ov_radiomics._reference_metrics')
    except ModuleNotFoundError as exc:
        raise ImportError('Install ov-radiomics[reference] for the explicit Python reference backend.') from exc


def roc_auc_score(*args, **kwargs):
    # Kept as a lazy compatibility symbol; native calculations never call it.
    return _reference().roc_auc_score(*args, **kwargs)


def binary_metrics(y_true, y_score, threshold=0.5, positive_label=1, ci_method='delong',
                   confidence=0.95, n_bootstrap=2000, random_state=0, backend='auto'):
    if resolve_backend(backend) is None:
        return _reference().binary_metrics(y_true, y_score, threshold, positive_label,
            ci_method, confidence, n_bootstrap, random_state, backend='python')
    y, p = probabilities(y_true, y_score, positive_label)
    if not 0 <= threshold <= 1 or not 0 < confidence < 1:
        raise ValueError('Invalid threshold or confidence.')
    inputs = dict(y=y.tolist(), scores=p.tolist())
    summary = request('binary_metrics', **inputs, threshold=float(threshold))
    if ci_method == 'delong':
        interval = request('delong_interval', **inputs, confidence=confidence)
        low, high = interval['low'], interval['high']
    elif ci_method == 'bootstrap':
        if n_bootstrap < 2:
            raise ValueError('n_bootstrap must be at least 2.')
        if random_state is None or not isinstance(random_state, (int, np.integer)) or not 0 <= int(random_state) <= 2**32-1:
            raise ValueError('Native legacy bootstrap requires an explicit uint32 random_state.')
        interval = request('bootstrap_binary', **inputs, replicates=int(n_bootstrap),
            seed=int(random_state), confidence=confidence)
        low, high = interval['low'], interval['high']
    elif ci_method is None:
        low, high = np.nan, np.nan
    else:
        raise ValueError('ci_method must be delong, bootstrap, or None.')
    output = {'n': summary['n'], 'positive_n': summary['positive_n'],
        'accuracy': summary['accuracy'], 'auc': summary['auc'], 'ci_low': low, 'ci_high': high}
    for name in ('sensitivity', 'specificity', 'ppv', 'npv', 'precision', 'recall', 'f1', 'brier', 'threshold'):
        output[name] = float(summary[name])
    return pd.Series(output)


def delong_test(y_true, score_a, score_b, positive_label=1, backend='auto'):
    if resolve_backend(backend) is None:
        return _reference().delong_test(y_true, score_a, score_b, positive_label, backend='python')
    y, a = probabilities(y_true, score_a, positive_label)
    _, b = probabilities(y_true, score_b, positive_label)
    value = request('delong_test', y=y.tolist(), a=a.tolist(), b=b.tolist())
    return pd.Series({name: float(value[name]) for name in ('auc_a', 'auc_b', 'difference', 'z', 'p_value')})


def decision_curve(y_true, scores, thresholds=None, positive_label=1, backend='auto'):
    if resolve_backend(backend) is None:
        return _reference().decision_curve(y_true, scores, thresholds, positive_label, backend='python')
    if not isinstance(scores, (dict, pd.DataFrame)):
        scores = {'Model': scores}
    thresholds = np.asarray(np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all() or ((thresholds <= 0) | (thresholds >= 1)).any():
        raise ValueError('DCA thresholds must lie strictly between 0 and 1.')
    names = list(scores)
    if not names or set(names) & {'Treat all', 'Treat none'}:
        raise ValueError('Supply models with non-reserved names.')
    rows = []
    for name in names:
        y, p = probabilities(y_true, scores[name], positive_label)
        value = request('decision_curve', y=y.tolist(), scores=p.tolist(), thresholds=thresholds.tolist())
        rows.extend((name, t, benefit) for t, benefit in zip(thresholds, value['model_net_benefit']))
    for t, all_value, none_value in zip(thresholds, value['treat_all'], value['treat_none']):
        rows.extend([('Treat all', t, all_value), ('Treat none', t, none_value)])
    return pd.DataFrame(rows, columns=['model', 'threshold', 'net_benefit'])


def hosmer_lemeshow(y_true, y_score, n_bins=10, positive_label=1, backend='auto'):
    if resolve_backend(backend) is None:
        return _reference().hosmer_lemeshow(y_true, y_score, n_bins, positive_label, backend='python')
    y, p = probabilities(y_true, y_score, positive_label)
    if n_bins < 3:
        raise ValueError('At least three requested bins are required.')
    value = request('hosmer_lemeshow', y=y.tolist(), scores=p.tolist(), n_bins=int(n_bins))
    # Rust owns quantile edges and every numerical bin assignment. pandas only
    # formats the legacy categorical interval labels for display compatibility.
    from pandas.core.reshape.tile import _format_labels
    labels = _format_labels(np.asarray(value['quantile_edges']), precision=3, right=True, include_lowest=True)
    observed_bins = sorted(set(value['bin_membership']))
    index = pd.CategoricalIndex(pd.Categorical(labels.take(observed_bins), categories=labels, ordered=True), name='bin')
    table = pd.DataFrame({'n': [row['n'] for row in value['table']],
        'observed': np.asarray([int(row['observed']) for row in value['table']], dtype=y.dtype),
        'expected': [row['expected'] for row in value['table']]}, index=index)
    return dict(statistic=value['statistic'], df=value['df'], p_value=value['p_value'], table=table)
