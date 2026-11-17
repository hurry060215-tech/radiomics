"""Native CSV/Parquet readers; pandas is a result view, never the file parser."""

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ._native_common import buffer_array, request, string_value
from ._types import FeatureTable


class LoadedStudy:
    """Identified table and load audit, with outcome/IDs outside the feature list."""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return deepcopy(self._payload)

    @property
    def report(self):
        return deepcopy(self._payload["report"])

    @property
    def features(self):
        values = self._payload["data"]["features"]
        return list(values["numeric_names"]) + list(values["categorical_names"])

    @property
    def categorical(self):
        return list(self._payload["data"]["features"]["categorical_names"])

    @property
    def target(self):
        return self._payload["data"]["outcome"].get("name")

    @property
    def dataframe(self):
        data = self._payload["data"]
        features = data["features"]
        frame = pd.DataFrame(buffer_array(features["numeric"]), columns=features["numeric_names"])
        count = len(features["categorical_names"])
        for j, name in enumerate(features["categorical_names"]):
            frame[name] = [
                features["categorical_values"][i * count + j] for i in range(features["rows"])
            ]
        frame.insert(0, "ID", [sample["sample_id"] for sample in data["samples"]])
        frame["patient_id"] = [sample["patient_id"] for sample in data["samples"]]
        frame["group"] = data["cohorts"]
        for name in ("slide_id", "lesion_id", "modality", "region", "timepoint", "center"):
            values = [sample[name] for sample in data["samples"]]
            if any(value is not None for value in values):
                if name in frame:
                    if [string_value(value) for value in frame[name]] != values:
                        raise ValueError(
                            "Feature/metadata values collide in the canonical DataFrame view: "
                            + name
                        )
                else:
                    frame[name] = values
        outcome = data["outcome"]
        if outcome["task"] != "unlabeled":
            if outcome["name"] in frame:
                raise ValueError(
                    "The declared outcome name collides with canonical ID/cohort/feature columns."
                )
            frame[outcome["name"]] = outcome["values"]
        return frame


def read_study(spec, *, task="binary", base=None):
    """Read a schema-one StudyTableSpec, including explicit clinical/metadata joins.

    Paths resolve relative to a spec file, or to explicit base for a dictionary.
    No process working directory or optional pandas Parquet engine is changed.
    """
    if isinstance(spec, (str, Path)):
        path = Path(spec).resolve()
        if base is not None:
            raise ValueError("A specification file defines its own base directory.")
        spec = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
    elif base is None:
        base = Path.cwd()
    return LoadedStudy(request("read_study", spec=spec, task=task, base=str(Path(base).resolve())))


def read_feature_table(path, *, manifest=None):
    return FeatureTable.read(path, manifest=manifest)
