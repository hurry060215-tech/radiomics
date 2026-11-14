"""Observed-data clinical views; inference and summaries run in Rust."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ._native_common import request, tabular, string_value, numeric_buffer


def _analysis_rows(data, analysis_unit, patient_id, id_col):
    if analysis_unit not in ('patient', 'sample'):
        raise ValueError('analysis_unit must be patient or sample.')
    column = patient_id or ('patient_id' if 'patient_id' in data else id_col)
    if column not in data or data[column].isna().any():
        raise ValueError('Clinical analysis requires complete patient identities; specify patient_id or id_col.')
    if analysis_unit == 'patient' and data[column].astype(str).duplicated().any():
        raise ValueError('Repeated patients require prior patient-level aggregation; choose analysis_unit=sample explicitly only for a row-level descriptive/inference analysis.')


def _feature_roles(data, target, features, categorical, split=None):
    features = list(features)
    categorical = list(categorical or [])
    if not features or len(set(features)) != len(features) or target in features or split in features:
        raise ValueError('Select unique features separately from outcome and cohort columns.')
    if not set(categorical) <= set(features):
        raise ValueError('Categorical columns must be selected features.')
    return features, categorical


@dataclass
class NativeLogisticAnalysis:
    table: pd.DataFrame
    model: dict
    n_used: int
    n_dropped: int
    reference: dict
    analysis_unit: str


def multivariable_logistic(data, target, features, categorical=None, reference=None,
                          confidence=.95, *, positive_label=1, analysis_unit='patient',
                          patient_id=None, id_col='ID', tolerance=1e-12, max_iterations=200):
    _analysis_rows(data, analysis_unit, patient_id, id_col)
    features, categorical = _feature_roles(data, target, features, categorical)
    reference = {str(k):string_value(v) for k,v in (reference or {}).items()}
    if not set(reference) <= set(categorical):
        raise ValueError('Reference categories must refer to selected categorical features.')
    # Missing outcomes are passed through, so complete-case counts and indices
    # remain those of the original input rather than an already filtered frame.
    observed = [string_value(v) for v in data[target]]
    classes = sorted({v for v in observed if v is not None})
    positive = string_value(positive_label)
    if len(classes) != 2 or positive not in classes:
        raise ValueError('Clinical Logistic requires exactly two observed classes and explicit positive_label.')
    y = [None if v is None else int(v == positive) for v in observed]
    fitted = request('clinical_logistic', data=tabular(data,features,categorical), target=y,
        features=features, reference=reference, config=dict(confidence=confidence,
        tolerance=tolerance,max_iterations=max_iterations,rank_tolerance=1e-12))
    table = pd.DataFrame(dict(term=fitted['terms'],coefficient=fitted['coefficient'],
        standard_error=fitted['standard_error'],z=fitted['z'],p_value=fitted['p_values'],
        odds_ratio=[float(v) for v in fitted['odds_ratio']],
        ci_low=[float(v) for v in fitted['odds_ratio_ci_lower']],
        ci_high=[float(v) for v in fitted['odds_ratio_ci_upper']]))
    fitted['analysis_unit']=analysis_unit
    table.attrs.update(analysis_unit=analysis_unit,n_used=fitted['n_used'],n_dropped=fitted['n_dropped'])
    return NativeLogisticAnalysis(table,fitted,fitted['n_used'],fitted['n_dropped'],
        deepcopy(fitted['reference_categories']),analysis_unit)


def univariate_logistic(data, target, features, categorical=None, reference=None,
                        confidence=.95, **kwargs):
    rows=[]
    for feature in features:
        result=multivariable_logistic(data,target,[feature],
            categorical=[feature] if feature in (categorical or []) else [],
            reference={feature:reference[feature]} if reference and feature in reference else None,
            confidence=confidence,**kwargs)
        table=result.table.loc[result.table.term!='const'].copy()
        table['feature'],table['n_used'],table['n_dropped']=feature,result.n_used,result.n_dropped
        rows.append(table)
    if not rows:
        raise ValueError('Select at least one clinical feature.')
    return pd.concat(rows,ignore_index=True)


def baseline_table(data,target,features,continuous,categorical=None,split=None,
                   continuous_test='auto',categorical_test='auto',*,analysis_unit='patient',
                   patient_id=None,id_col='ID',classes=None):
    _analysis_rows(data,analysis_unit,patient_id,id_col)
    categorical=list(categorical) if categorical is not None else [f for f in features if f not in continuous]
    features,categorical=_feature_roles(data,target,features,categorical,split)
    if set(continuous)&set(categorical) or set(continuous)|set(categorical)!=set(features):
        raise ValueError('Continuous and categorical names must partition the selected features.')
    observed=[string_value(v) for v in data[target]]
    classes=sorted({v for v in observed if v is not None}) if classes is None else [string_value(v) for v in classes]
    mapping={c:i for i,c in enumerate(classes)}
    if any(v is not None and v not in mapping for v in observed):
        raise ValueError('Observed baseline outcome is absent from declared classes.')
    continuous_test={'mannwhitney':'mann_whitney','t':'student_t','kruskal':'kruskal_wallis'}.get(continuous_test,continuous_test)
    categorical_test={'chi2':'chi_square'}.get(categorical_test,categorical_test)
    cohorts=None if split is None else [string_value(v) for v in data[split]]
    fitted=request('baseline',data=tabular(data,features,categorical),classes=classes,
        labels=[None if v is None else mapping[v] for v in observed],features=features,cohorts=cohorts,
        config=dict(continuous_test=continuous_test,categorical_test=categorical_test))
    rows=[]
    for native in fitted['rows']:
        row=dict(cohort=native['cohort'],feature=native['feature'],
            type='categorical' if native['feature'] in categorical else 'continuous',
            p_value=float(native['p_value']),test=native['method'])
        for group in native['groups']:
            prefix='class_'+group['class_label']+'_'
            for key,value in group.items():
                if key in ('type','class_index','class_label'):
                    continue
                key={'sd_ddof1':'sd','q25':'q1','q75':'q3'}.get(key,key)
                row[prefix+key]=float(value) if isinstance(value,str) and value in ('NaN','Inf','-Inf') else value
        rows.append(row)
    table=pd.DataFrame(rows)
    table.attrs.update(analysis_unit=analysis_unit,n_input=fitted['n_input'],native_report=fitted)
    return table


def icc(data,*,kind,confidence=.95):
    """Complete balanced subject×reader matrix, explicit ICC(2,1) or ICC(3,1)."""
    return request('icc',data=numeric_buffer(data),kind=kind,confidence=confidence)
