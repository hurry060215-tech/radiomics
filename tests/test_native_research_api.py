import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ov_radiomics as rad
from ov_radiomics._native_common import numeric_buffer

pytestmark = pytest.mark.usefixtures("native_backend")
DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def table(rows, names):
    return rad.FeatureTable(
        dict(
            samples=[{k: v for k, v in row.items() if k != "values"} for row in rows],
            features=[
                dict(name=name, definition="Frozen research specification", unit="synthetic_unit")
                for name in names
            ],
            data=numeric_buffer([[row["values"][name] for name in names] for row in rows]),
            effective_parameters={},
            quality=[],
        )
    )


def test_research_public_tables_preserve_keyed_order_and_delta_missing_policy():
    inputs = json.loads((DIRECTORY / "research_inputs.json").read_text())
    reference = json.loads((DIRECTORY / "research_reference.json").read_text())
    case = inputs["concat"]
    sources = {
        source["prefix"]: table(source["rows"], source["feature_names"])
        for source in case["sources"]
    }
    result = rad.research.concat(
        sources,
        key_fields=case["key_fields"],
        varying_fields=case["varying_fields"],
        already_registered=True,
    )
    assert result.features.columns[1:].tolist() == reference["concat"]["feature_names"]
    np.testing.assert_allclose(
        result.features.iloc[:, 1:].to_numpy(dtype=float),
        np.asarray([row["values"] for row in reference["concat"]["rows"]], dtype=float),
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="registration"):
        rad.research.concat(
            sources,
            key_fields=case["key_fields"],
            varying_fields=case["varying_fields"],
            already_registered=False,
        )
    case = inputs["delta"]
    result = rad.research.delta(
        table(case["rows"], case["feature_names"]),
        baseline=case["baseline"],
        followup=case["followup"],
        zero_denominator="missing",
    )
    np.testing.assert_allclose(
        result.features.iloc[:, 1:].to_numpy(dtype=float),
        np.asarray(
            [row["values"] for row in reference["delta_missing_policy"]["rows"]], dtype=float
        ),
        equal_nan=True,
    )


def test_habitat_public_scaling_centers_direct_distance_persistence_and_undefined_cluster_errors(
    tmp_path,
):
    inputs = json.loads((DIRECTORY / "research_inputs.json").read_text())["habitat"]
    reference = json.loads((DIRECTORY / "research_reference.json").read_text())["habitat"]
    feature_names = ["a", "b"]
    training = pd.DataFrame(
        inputs["training_values"], index=inputs["training_ids"], columns=feature_names
    )
    fitted = rad.habitat.fit(
        training,
        clusters=2,
        initial_centers=inputs["initial_centers_raw"],
        max_iterations=inputs["max_iter"],
    )
    assert fitted.report["converged"]
    np.testing.assert_allclose(fitted.centers_, reference["centers_raw"], atol=1e-10, rtol=1e-8)
    evaluation = pd.DataFrame(
        inputs["evaluation_values"], index=inputs["evaluation_ids"], columns=feature_names
    )
    result = fitted.predict(evaluation)
    assert result.cluster.tolist() == reference["evaluation_labels"]
    np.testing.assert_allclose(
        result.iloc[:, 2:],
        reference["evaluation_distances_euclidean_direct"],
        atol=1e-10,
        rtol=1e-8,
    )
    fitted.save(tmp_path / "habitat.json")
    pd.testing.assert_frame_equal(
        result, rad.habitat.HabitatModel.load(tmp_path / "habitat.json").predict(evaluation)
    )
    with pytest.raises(FileExistsError):
        fitted.save(tmp_path / "habitat.json")
    with pytest.raises(ValueError, match="complex"):
        rad.habitat.fit(training.astype(complex) + 1j)
    with pytest.raises(ValueError, match="empty"):
        rad.habitat.fit(training, initial_centers=inputs["empty_cluster_initial_centers_raw"])
