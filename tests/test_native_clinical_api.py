import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ov_radiomics as rad

pytestmark = pytest.mark.usefixtures("native_backend")


def test_identified_complete_case_clinical_fit_matches_frozen_reference():
    source = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "golden/python_golden_v1/statistics_reference.json"
        ).read_text()
    )["clinical_logistic"]
    frame = pd.DataFrame(source["rows"])
    # The independent numerical fixture deliberately repeats patients. Default
    # patient inference must reject; the fixture's row-level scope is explicit.
    with pytest.raises(ValueError, match="Repeated patients"):
        rad.multivariable_logistic(
            frame, "target", ["a", "b", "sex"], categorical=["sex"], backend="rust"
        )
    result = rad.multivariable_logistic(
        frame,
        "target",
        ["a", "b", "sex"],
        categorical=["sex"],
        reference={"sex": "F"},
        backend="rust",
        analysis_unit="sample",
    )
    expected = source["fits"]["multivariable"]
    assert result.n_used == expected["n_used"]
    assert result.n_dropped == expected["n_dropped"]
    assert result.table.term.tolist() == expected["terms"]
    np.testing.assert_allclose(
        result.table.coefficient, expected["coefficient"], atol=1e-6, rtol=1e-5
    )
    assert result.analysis_unit == "sample"
    assert result.model["report"]["converged"]
    with pytest.raises(ValueError, match="Repeated patients"):
        rad.baseline_table(frame, "target", ["a", "sex"], continuous=["a"], backend="rust")
    with pytest.raises(ValueError, match="nonmissing"):
        rad.baseline_table(
            frame, "target", ["a", "sex"], continuous=["a"], backend="rust", analysis_unit="sample"
        )
    table = rad.baseline_table(
        frame.dropna(subset=["target"]),
        "target",
        ["a", "sex"],
        continuous=["a"],
        backend="rust",
        analysis_unit="sample",
    )
    assert len(table) == 2
    assert table.attrs["analysis_unit"] == "sample"
    assert table.class_0_missing.sum() > 0


def test_typed_icc_and_patient_outcome_broadcast_guard():
    data = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    result = rad.icc(data, kind="ICC(3,1)")
    assert result["type"] == "ICC(3,1)"
    assert result["estimate"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rad.icc([[1.0, np.nan], [2.0, 3.0]], kind="ICC(2,1)")
    with pytest.raises(TypeError):
        rad.icc(data)
