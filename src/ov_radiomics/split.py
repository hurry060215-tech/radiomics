"""Immutable Python view of the patient-level Rust SplitPlan contract."""
from dataclasses import dataclass
import json
import numbers
from ._backend import resolve_backend


def _native():
    native = resolve_backend('rust')
    if not hasattr(native, 'create_split_plan'):
        raise ImportError('This extension predates native SplitPlan; rebuild the current checkout.')
    return native


def _observations(data, target, id_col, patient_id, split, center):
    import pandas as pd
    from ._validation import unique_ids
    unique_ids(data, id_col)
    group = patient_id or ('patient_id' if 'patient_id' in data else id_col)
    required = [group, split] + ([center] if center else [])
    if any(column not in data for column in required):
        raise ValueError('Missing patient, cohort or center column.')
    if data[required].isna().any().any():
        raise ValueError('Patient, cohort and center identifiers must be complete.')
    rows = []
    columns=list(dict.fromkeys([id_col,group,split]+([center] if center else [])+([target] if target is not None and target in data else [])))
    for values in data[columns].itertuples(index=False,name=None):
        row=dict(zip(columns,values));value=row.get(target)
        rows.append(dict(sample_id=_label(row[id_col]),patient_id=_label(row[group]),
            cohort=_label(row[split]),class_label=None if pd.isna(value) else _label(value),
            center=_label(row[center]) if center else None))
    return rows


def _label(value):
    # pandas may promote the entire target column when only external labels become
    # missing. Preserve the training class identity of numerically equal 1 and 1.0.
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return str(int(value))
    return str(value)


@dataclass(frozen=True)
class SplitPlan:
    """Patient cohorts, actual outer/inner patient sets, class order and seed.

    All validation and split construction run in Rust. Repeated patients remain
    together, while sample IDs must be unique. External labels are prediction-only.
    ``to_dict()`` returns a copy; mutating it cannot change this plan.
    """
    _json: str

    def __post_init__(self):
        _native().validate_split_plan(self._json)

    @classmethod
    def from_data(cls, data, target=None, *, task='binary', classes=None, id_col='ID',
                  patient_id=None, split='group', center=None, outer_cv=5, inner_cv=3, seed=0):
        rows = _observations(data, target, id_col, patient_id, split, center)
        if classes is None:
            classes = [] if task == 'regression' else sorted({r['class_label'] for r in rows
                if r['cohort'] == 'train' and r['class_label'] is not None})
        classes = [_label(c) for c in classes]
        raw = _native().create_split_plan(json.dumps(rows, allow_nan=False), task,
            classes, outer_cv, inner_cv, seed)
        return cls(raw)

    @classmethod
    def from_dict(cls, value):
        return cls(json.dumps(value, allow_nan=False))

    def to_dict(self):
        return json.loads(self._json)

    @property
    def fingerprint(self):
        return _native().validate_split_plan(self._json)

    def assert_shared(self, other):
        if not isinstance(other, SplitPlan):
            raise TypeError('Expected a SplitPlan.')
        _native().assert_shared_split_plan(self._json, other._json)

    def validate_data(self, data, target=None, *, id_col='ID', patient_id=None,
                      split='group', center=None):
        rows = _observations(data, target, id_col, patient_id, split, center)
        _native().validate_split_observations(self._json, json.dumps(rows, allow_nan=False))

    def fold_indices(self, data, fold, *, patient_id=None, id_col='ID', inner=None):
        """Positions in ``data`` for a saved partition; no new random split is made.

        ``inner`` selects an inner fold within the selected outer fold. For final
        training use ``final_inner_indices``.
        """
        plan = self.to_dict()
        nested = plan['outer_folds'][fold]
        part = nested['outer'] if inner is None else nested['inner'][inner]
        return self._indices(data, part, patient_id or ('patient_id' if 'patient_id' in data else id_col))

    def final_inner_indices(self, data, fold, *, patient_id=None, id_col='ID'):
        return self._indices(data, self.to_dict()['final_inner_folds'][fold], patient_id or ('patient_id' if 'patient_id' in data else id_col))

    @staticmethod
    def _indices(data, part, group):
        if group not in data or data[group].isna().any():
            raise ValueError('Complete patient identifiers are required.')
        groups = data[group].map(_label)
        train, valid = set(part['training_patients']), set(part['validation_patients'])
        present = set(groups)
        if not (train | valid).issubset(present):
            raise ValueError('Data are missing patients required by the saved partition.')
        return ([i for i, g in enumerate(groups) if g in train],
                [i for i, g in enumerate(groups) if g in valid])
