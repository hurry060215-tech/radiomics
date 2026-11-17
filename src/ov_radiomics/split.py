"""Immutable Python view of the patient-level Rust SplitPlan contract."""

import json
import numbers
from dataclasses import dataclass

from ._backend import resolve_backend


def _native():
    native = resolve_backend("rust")
    if not hasattr(native, "create_split_plan"):
        raise ImportError("This extension predates native SplitPlan; rebuild the current checkout.")
    return native


def _observations(data, target, id_col, patient_id, split, center):
    import pandas as pd

    from ._validation import unique_ids

    unique_ids(data, id_col)
    if target is not None and target not in data:
        raise ValueError("The explicitly named outcome column is missing: " + str(target))
    group = patient_id or ("patient_id" if "patient_id" in data else id_col)
    center = center or ("center" if "center" in data else None)
    required = [group, split]
    if center is not None and center not in data:
        raise ValueError("Missing center column.")
    if any(column not in data for column in required):
        raise ValueError("Missing patient, cohort or center column.")
    if data[required].isna().any().any():
        raise ValueError("Patient and cohort identifiers must be complete.")
    rows = []
    columns = list(
        dict.fromkeys(
            [id_col, group, split]
            + ([center] if center else [])
            + ([target] if target is not None and target in data else [])
        )
    )
    for values in data[columns].itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        value = row.get(target)
        rows.append(
            dict(
                sample_id=_label(row[id_col]),
                patient_id=_label(row[group]),
                cohort=_label(row[split]),
                class_label=None if pd.isna(value) else _label(value),
                center=None if center is None or pd.isna(row[center]) else _label(row[center]),
            )
        )
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
    def from_data(
        cls,
        data,
        target=None,
        *,
        task="binary",
        classes=None,
        id_col="ID",
        patient_id=None,
        split="group",
        center=None,
        outer_cv=5,
        inner_cv=3,
        seed=0,
    ):
        rows = _observations(data, target, id_col, patient_id, split, center)
        if classes is None:
            classes = (
                []
                if task == "regression"
                else sorted(
                    {
                        r["class_label"]
                        for r in rows
                        if r["cohort"] == "train" and r["class_label"] is not None
                    }
                )
            )
        classes = [_label(c) for c in classes]
        raw = _native().create_split_plan(
            json.dumps(rows, allow_nan=False), task, classes, outer_cv, inner_cv, seed
        )
        return cls(raw)

    @classmethod
    def center_holdout(
        cls,
        data,
        target=None,
        *,
        held_out_centers,
        task="binary",
        classes=None,
        id_col="ID",
        patient_id=None,
        split="group",
        center="center",
        outer_cv=5,
        inner_cv=3,
        seed=0,
    ):
        """Hold out declared centers in Rust; preserve already external cohorts.

        Call ``plan.apply_cohorts(data)`` before training with this plan. Class
        inference reads only the resulting training centers, never held-out labels.
        """
        held_out_centers = (
            [held_out_centers] if isinstance(held_out_centers, str) else list(held_out_centers)
        )
        held_out_centers = [_label(value) for value in held_out_centers]
        prepared = data if split in data else data.assign(**{split: "unassigned"})
        rows = _observations(prepared, target, id_col, patient_id, split, center)
        held = set(held_out_centers)
        if classes is None:
            classes = (
                []
                if task == "regression"
                else sorted(
                    {
                        row["class_label"]
                        for row in rows
                        if row["cohort"] in ("train", "test", "unassigned")
                        and row["center"] not in held
                        and row["class_label"] is not None
                    }
                )
            )
        raw = _native().create_split_plan(
            json.dumps(rows, allow_nan=False),
            task,
            [_label(value) for value in classes],
            outer_cv,
            inner_cv,
            seed,
            held_out_centers,
        )
        return cls(raw)

    def apply_cohorts(self, data, *, id_col="ID", patient_id=None, split="group"):
        """Return identified rows with the saved patient memberships, without fitting."""
        from ._validation import unique_ids

        unique_ids(data, id_col)
        group = patient_id or ("patient_id" if "patient_id" in data else id_col)
        if group not in data or data[group].isna().any():
            raise ValueError("Complete patient identities are required.")
        plan = self.to_dict()
        memberships = {
            patient: name for name, patients in plan["cohorts"].items() for patient in patients
        }
        ids = data[id_col].map(_label)
        patients = data[group].map(_label)
        if any(
            plan["sample_patients"].get(sample) != patient for sample, patient in zip(ids, patients)
        ):
            raise ValueError("Sample/patient identities differ from the saved SplitPlan.")
        result = data.copy()
        result[split] = [memberships[patient] for patient in patients]
        return result

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
            raise TypeError("Expected a SplitPlan.")
        _native().assert_shared_split_plan(self._json, other._json)

    def validate_data(
        self, data, target=None, *, id_col="ID", patient_id=None, split="group", center=None
    ):
        rows = _observations(data, target, id_col, patient_id, split, center)
        _native().validate_split_observations(self._json, json.dumps(rows, allow_nan=False))

    def fold_indices(self, data, fold, *, patient_id=None, id_col="ID", inner=None):
        """Positions in ``data`` for a saved partition; no new random split is made.

        ``inner`` selects an inner fold within the selected outer fold. For final
        training use ``final_inner_indices``.
        """
        plan = self.to_dict()
        nested = plan["outer_folds"][fold]
        part = nested["outer"] if inner is None else nested["inner"][inner]
        return self._indices(
            data, part, patient_id or ("patient_id" if "patient_id" in data else id_col)
        )

    def final_inner_indices(self, data, fold, *, patient_id=None, id_col="ID"):
        return self._indices(
            data,
            self.to_dict()["final_inner_folds"][fold],
            patient_id or ("patient_id" if "patient_id" in data else id_col),
        )

    @staticmethod
    def _indices(data, part, group):
        if group not in data or data[group].isna().any():
            raise ValueError("Complete patient identifiers are required.")
        groups = data[group].map(_label)
        train, valid = set(part["training_patients"]), set(part["validation_patients"])
        present = set(groups)
        if not (train | valid).issubset(present):
            raise ValueError("Data are missing patients required by the saved partition.")
        return (
            [i for i, g in enumerate(groups) if g in train],
            [i for i, g in enumerate(groups) if g in valid],
        )
