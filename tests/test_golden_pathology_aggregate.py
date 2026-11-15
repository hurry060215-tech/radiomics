"""Hand checks for patient hierarchy, weighted CDF and spatial coverage references."""
import json
import math
from pathlib import Path

import numpy as np


DIRECTORY=Path(__file__).resolve().parents[1]/"golden/python_golden_v1"


def read(name):
    return json.loads((DIRECTORY/name).read_text())


def test_hierarchy_and_weighted_statistics_follow_independent_hand_values():
    data=read("pathology_aggregate_inputs.json");reference=read("pathology_aggregate_reference.json");hand=data["hand_expectations"]
    default=reference["aggregation"]["default_equal_slides"]
    slide=next(s for s in default["slides"] if s["slide_id"]=="A")
    for statistic,key in [("mean","slide_A_default_mean_embedding_0"),("std","slide_A_default_std_embedding_0"),("q25","slide_A_default_q25_embedding_0")]:
        assert slide["features"]["slide_tile_"+statistic+"(embedding_0)"]==hand[key]
    patient=next(p for p in default["patients"] if p["patient_id"]=="P1")
    assert patient["features"]["patient_mean(slide_tile_mean(embedding_0))"]==hand["patient_P1_equal_mean_of_slide_means_embedding_0"]
    weighted=reference["aggregation"]["valid_area_within_slide"]
    slide=next(s for s in weighted["slides"] if s["slide_id"]=="A")
    np.testing.assert_allclose(slide["features"]["slide_tile_mean(embedding_0)"],hand["slide_A_level0area_mean_embedding_0"],atol=1e-12,rtol=1e-12)
    assert slide["features"]["slide_tile_q50(embedding_0)"]==hand["slide_A_weighted_q50_embedding_0"]
    summaries=reference["summaries"]
    assert summaries["weighted_equal_uses_inverted_cdf"]["values"]["q50"]==1
    assert summaries["unweighted_uses_linear_quantile"]["values"]["q50"]==2
    assert summaries["decimal_cdf_half_boundary"]["values"]["q50"]==2
    assert summaries["decimal_cdf_reordered"]["values"]["q50"]==2
    for key in ["all_zero_weights_empty","all_missing_empty"]:
        assert summaries[key]["status"]=="empty"
        assert all(value is None for value in summaries[key]["values"].values())
    empty=reference["excluded_and_zero_tile_variants"]["zero_tile_slide"]
    slide=next(s for s in empty["slides"] if s["slide_id"]=="D")
    patient=next(p for p in empty["patients"] if p["patient_id"]=="P3")
    assert slide["patient_id"]=="P3" and slide["status"]=="no_accepted_tiles"
    assert patient["status"]=="no_accepted_slides"
    assert all(v is None for v in slide["features"].values())
    assert all(v is None for v in patient["features"].values())


def test_overlay_blocks_padding_global_coordinates_and_failures_are_explicit():
    data=read("pathology_aggregate_inputs.json");reference=read("pathology_aggregate_reference.json");whole=reference["overlay"]["whole"]
    assert whole["values"]==data["hand_expectations"]["overlay_values"]
    assert whole["coverage"]==data["hand_expectations"]["overlay_coverage"]
    reconstructed=[None]*len(whole["values"]);counts=[None]*len(reconstructed)
    for block in reference["overlay"]["blocks"]:
        for y in range(block["height"]):
            for x in range(block["width"]):
                source=y*block["width"]+x;target=(block["y"]+y)*whole["width"]+block["x"]+x
                reconstructed[target]=block["values"][source];counts[target]=block["coverage"][source]
    assert reconstructed==whole["values"] and counts==whole["coverage"]
    for value,count,alpha in zip(whole["values"],whole["coverage"],whole["alpha"]):
        assert (count==0)==(value=="NaN")== (alpha==0)
    offset=reference["overlay"]["offset_valid_rect"]
    assert [i for i,count in enumerate(offset["coverage"]) if count]==[5,6]
    context=reference["object_context"]
    assert context["local_coordinate_space"]=="patch_pixel_edges"
    assert context["coordinate_space"]=="level0_pixel_edges"
    assert context["objects"][0]["centroid0"]==data["hand_expectations"]["global_centroid0"]
    assert context["objects"][0]["bbox0"]==data["hand_expectations"]["global_bbox0"]
    assert all(result["status"]=="error" for result in reference["failures"].values())
    fractional=reference["overlay"]["fractional_boundary"]
    assert fractional["whole"]["coverage"]==data["fractional_boundary_overlay"]["expected_coverage"]
    rebuilt=[]
    for block in fractional["blocks"]:
        rebuilt.extend(block["coverage"])
    assert rebuilt==fractional["whole"]["coverage"]
    # This is deliberately an exact branch boundary, not a coordinate closeness test.
    assert 6.5*.1<.6500000000000001
    assert 6*.1+.5*.1==.6500000000000001
    assert math.nextafter(6.5*.1,math.inf)==.6500000000000001
    exact=reference["overlay"]["fractional_exact_edge"]
    assert exact["whole"]["coverage"]==data["fractional_boundary_exact_overlay"]["expected_coverage"]
    assert [count for block in exact["blocks"] for count in block["coverage"]]==exact["whole"]["coverage"]
