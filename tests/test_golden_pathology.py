"""Hand checks on independently frozen pathology pixel reference artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parents[1] / "golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY / name).read_text())


def test_pathology_reference_provenance_and_hand_qc():
    ref = read("pathology_reference.json")
    hand = read("pathology_hand.json")
    assert ref["specification_sha256"] == hashlib.sha256((DIRECTORY / "pathology_spec.py").read_bytes()).hexdigest()
    assert ref["input_sha256"] == hashlib.sha256((DIRECTORY / "pathology_inputs.json").read_bytes()).hexdigest()
    for name in ("white_3x3", "black_3x3", "padding_excluded", "binary_grid_5x5"):
        qc = ref["patches"][name]["qc"]
        for key, expected in hand[name].items():
            if key.endswith("_mean"):
                actual = ref["patches"][name]["colors"]["channels"][key[0]]["statistics"]["mean"]
            else:
                actual = qc.get(key, qc["scores"].get(key))
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)
    for patch in ref["patches"].values():
        assert patch["default_no_exclusion"]["accepted"]
        assert not patch["default_no_exclusion"]["exclusion_reasons"]


def test_pathology_padding_histograms_and_circular_hue():
    ref = read("pathology_reference.json")
    pad = ref["patches"]["padding_excluded"]
    assert pad["colors"]["channels"]["R"]["histogram"]["counts"] == [0, 0, 6, 0]
    assert pad["colors"]["channels"]["G"]["histogram"]["denominator"] == 6
    assert pad["colors"]["channels"]["B"]["histogram"]["probabilities"] == [0, 0, 1, 0]
    neutral = ref["patches"]["neutral_histogram_boundary"]["colors"]
    for channel in ("a", "b"):
        assert neutral["channels"][channel]["statistics"]["minimum"] == 0
        assert neutral["channels"][channel]["statistics"]["maximum"] == 0
        assert neutral["channels"][channel]["histogram"]["counts"] == [0, 0, 4, 0]
    hue = ref["patches"]["hue_wrap"]["colors"]["hue"]
    assert min(abs(hue["mean_degrees"]), abs(360 - hue["mean_degrees"])) <= 1e-6
    assert hue["resultant_length"] > .999
    assert ref["patches"]["hue_undefined_gray"]["colors"]["hue"]["mean_degrees"] is None
    assert ref["patches"]["rgb_primary_and_complements"]["colors"]["hue"]["mean_degrees"] is None
    assert ref["patches"]["white_3x3"]["colors_tissue_only"]["status"] == "empty_statistical_mask"


def test_pathology_hand_objects_coordinates_and_units():
    ref = read("pathology_reference.json")
    hand = read("pathology_hand.json")
    for name in ("instance_fragmented", "semantic_diagonal_4", "semantic_diagonal_8", "mask_padding_excluded"):
        objects = ref["masks"][name]["objects"]
        assert objects["object_count"] == hand[name]["object_count"]
        assert objects["count_scope"] == "patch_local"
    instance = ref["masks"]["instance_fragmented"]["objects"]
    assert [row["area_px"] for row in instance["objects"]] == hand["instance_fragmented"]["areas_px"]
    assert [row["area_um2"] for row in instance["objects"]] == hand["instance_fragmented"]["areas_um2"]
    assert [row["centroid_pixel_centers"] for row in instance["objects"]] == hand["instance_fragmented"]["centroids"]
    assert [row["touches_valid_boundary"] for row in instance["objects"]] == hand["instance_fragmented"]["boundary"]
    assert instance["warnings"]
    coverage = ref["masks"]["mask_padding_excluded"]["coverage"]
    assert coverage["denominator"] == hand["mask_padding_excluded"]["denominator"]
    assert coverage["counts"] == hand["mask_padding_excluded"]["counts"]
    assert all(row["area_um2"] is None for row in ref["masks"]["semantic_diagonal_4"]["objects"]["objects"])
    scale = read("pathology_inputs.json")["pixel_scale_case"]
    mpp = [scale["level0_mpp_um"][0] * scale["tile_width0"] / scale["patch_width"], scale["level0_mpp_um"][1] * scale["tile_height0"] / scale["patch_height"]]
    assert mpp == hand["pixel_scale"]["patch_pixel_mpp_um"]
    assert 2 * mpp[0] * mpp[1] == hand["pixel_scale"]["area_of_two_patch_pixels_um2"]


def test_pathology_tiling_and_invalid_inputs_are_explicit():
    ref = read("pathology_reference.json")
    assert len(ref["tilings"][0]) == 2
    assert len(ref["tilings"][1]) == 6
    assert ref["tilings"][2] == []
    assert ref["tilings"][1][-1] == {"x0": 4, "y0": 3, "width0": 4, "height0": 6, "patch_width": 2, "patch_height": 2, "valid_pixels": [0, 0, 1, 1]}
    assert all(result["status"] == "error" for result in ref["failures"].values())
