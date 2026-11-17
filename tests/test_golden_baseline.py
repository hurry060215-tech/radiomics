"""Independent observed-count/denominator and descriptive-summary checks."""

import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read():
    return json.loads((DIRECTORY / "baseline_reference.json").read_text())


def test_baseline_retains_observed_counts_and_missing_denominators():
    for task, dataset in read()["datasets"].items():
        rows = dataset["rows"]
        assert len(rows) == dataset["n_input"] == (95 if task == "binary" else 96)
        assert all(row["target"] is not None for row in rows)
        for feature in dataset["table"]:
            for index, label in enumerate(dataset["classes"]):
                source = [r[feature["feature"]] for r in rows if r["target"] == index]
                values = [v for v in source if v is not None]
                summary = feature["groups"][label]
                assert summary["n"] == len(values)
                assert summary["missing"] == len(source) - len(values)
                if feature["type"] == "continuous":
                    if not values:
                        assert all(
                            summary[k] == "NaN"
                            for k in ["mean", "sd_ddof1", "median", "q25", "q75"]
                        )
                    else:
                        expected = [
                            np.mean(values),
                            np.std(values, ddof=1),
                            np.median(values),
                            np.quantile(values, 0.25),
                            np.quantile(values, 0.75),
                        ]
                        np.testing.assert_allclose(
                            [summary[k] for k in ["mean", "sd_ddof1", "median", "q25", "q75"]],
                            expected,
                            atol=1e-12,
                            rtol=1e-12,
                        )
                else:
                    assert sum(summary["counts"].values()) == len(values)
                    for category, count in summary["counts"].items():
                        assert values.count(category) == count
                        assert summary["proportions"][category] == count / len(values)


def test_baseline_defaults_sparse_multiclass_and_degenerate_rows_are_explicit():
    reference = read()
    for task, dataset in reference["datasets"].items():
        table = {row["feature"]: row for row in dataset["table"]}
        assert table["a"]["method"] == ("mann_whitney" if task == "binary" else "kruskal_wallis")
        assert table["all_missing"]["method"] == "insufficient_observations"
        assert table["all_missing"]["p_value"] == "NaN"
        assert table["constant_category"]["method"] == "constant_or_absent_class"
        assert table["constant_category"]["p_value"] == "NaN"
        assert table["missing_category"]["method"] == "insufficient_observations"
        assert table["sex"]["method"] == "chi_square_no_yates"
        rare = table["rare"]
        assert np.any(np.asarray(rare["expected_counts"]) < 5)
        assert rare["method"] == ("fisher_2x2" if task == "binary" else "chi_square_no_yates")
        assert rare["diagnostic"] is None
    assert (
        reference["datasets"]["binary"][
            "source_missing_target_rows_explicitly_omitted_when_constructing_fixture"
        ]
        == 1
    )
