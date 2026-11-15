"""Run the independently reviewed hierarchy/overlay specification into a fresh candidate."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from golden_generate_reference import ROOT,observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing pathology aggregation candidate.")
    directory=ROOT/"golden/python_golden_v1"
    source=directory/"pathology_aggregate_spec.py"
    specification=importlib.util.spec_from_file_location("pathology_aggregate_spec",source)
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    data=json.loads((directory/"pathology_aggregate_inputs.json").read_text())
    tiles=data["tiles"]
    aligned=module.align_embeddings([t["tile_id"] for t in tiles],data["embedding_ids"],data["embedding_values"])
    outputs={}
    for variant in data["aggregation_variants"]:
        options={key:variant[key] for key in ["tile_weight","patient_weight","patient_stats"]}
        outputs[variant["id"]]=module.aggregate(tiles,aligned,data["feature_names"],data["slide_patients"],**options)
    exclusions={}
    for key in ["all_qc_excluded_slide_variant","all_qc_excluded_one_of_patient_slides"]:
        changed=copy.deepcopy(tiles)
        for tile in changed:
            if tile["slide_id"]==data[key]["slide_id"]:
                tile["exclusion_reasons"].append(data[key]["reason"])
        exclusions[key]=module.aggregate(changed,aligned,data["feature_names"],data["slide_patients"])
    parents=dict(data["slide_patients"])
    parents[data["zero_tile_slide_variant"]["slide_id"]]=data["zero_tile_slide_variant"]["patient_id"]
    exclusions["zero_tile_slide"]=module.aggregate(tiles,aligned,data["feature_names"],parents)
    exclusions["entire_input_has_no_tile_rows"]=module.aggregate([],[],data["feature_names"],parents)
    summaries={case["id"]:module.summary(case["values"],case["weights"],case.get("ids")) for case in data["summary_cases"]}
    overlay=data["overlay"]
    selected=[t for t in tiles if t["slide_id"]==overlay["slide_id"]]
    whole=module.rasterize(selected,overlay["score_ids"],overlay["scores"],overlay["canvas"])
    blocks=module.raster_blocks(selected,overlay["score_ids"],overlay["scores"],overlay["canvas"],overlay["block_size"])
    offset=data["offset_valid_rect_overlay"]
    offset_result=module.rasterize([offset["tile"]],offset["score_ids"],offset["scores"],offset["canvas"])
    fractional=data["fractional_boundary_overlay"]
    fractional_whole=module.rasterize([fractional["tile"]],fractional["score_ids"],fractional["scores"],fractional["canvas"])
    fractional_blocks=module.raster_blocks([fractional["tile"]],fractional["score_ids"],fractional["scores"],fractional["canvas"],fractional["block_size"])
    exact=data["fractional_boundary_exact_overlay"]
    exact_whole=module.rasterize([exact["tile"]],exact["score_ids"],exact["scores"],exact["canvas"])
    exact_blocks=module.raster_blocks([exact["tile"]],exact["score_ids"],exact["scores"],exact["canvas"],exact["block_size"])
    object_case=data["object_context"]
    tile=next(t for t in tiles if t["tile_id"]==object_case["tile_id"])
    global_objects=module.global_objects(tile,object_case["local_objects"])
    unknown_mpp=copy.deepcopy(tiles);unknown_mpp[0]["mpp_x_um"]=None
    duplicate=list(data["embedding_ids"]);duplicate[-1]=duplicate[0]
    wrong=list(data["embedding_ids"]);wrong[-1]="absent"
    failures={"duplicate_embedding_id":observed(lambda:module.align_embeddings([t["tile_id"] for t in tiles],duplicate,data["embedding_values"])),
              "same_count_wrong_embedding_id":observed(lambda:module.align_embeddings([t["tile_id"] for t in tiles],wrong,data["embedding_values"])),
              "unknown_mpp_physical":observed(lambda:module.aggregate(unknown_mpp,aligned,data["feature_names"],parents,tile_weight="physical_area")),
              "patient_level0_area_is_not_physical":observed(lambda:module.aggregate(tiles,aligned,data["feature_names"],parents,patient_weight="level0_area")),
              "broadcast_patient_probabilities":observed(lambda:module.rasterize(selected,overlay["score_ids"],overlay["scores"],overlay["canvas"],prediction_unit="patient")),
              "out_of_canvas_without_clip":observed(lambda:module.rasterize(selected,overlay["score_ids"],overlay["scores"],dict(overlay["canvas"],width=2,height=1))),
              "negative_summary_weight":observed(lambda:module.summary([1,3],[-1,2]))}
    clipped=module.rasterize(selected,overlay["score_ids"],overlay["scores"],dict(overlay["canvas"],width=2,height=1),clip=True)
    output={"schema":"python_golden_v1.pathology_aggregate.reference.v1",
            "specification_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "input_sha256":hashlib.sha256((directory/"pathology_aggregate_inputs.json").read_bytes()).hexdigest(),
            "alignment":{"tile_order":[t["tile_id"] for t in tiles],"aligned_values":aligned},
            "aggregation":outputs,"excluded_and_zero_tile_variants":exclusions,"summaries":summaries,
            "overlay":{"whole":whole,"blocks":blocks,"offset_valid_rect":offset_result,"explicit_clip":clipped,"fractional_boundary":{"whole":fractional_whole,"blocks":fractional_blocks},"fractional_exact_edge":{"whole":exact_whole,"blocks":exact_blocks}},
            "object_context":global_objects,"failures":failures,
            "numerical_contract":{"unweighted_quantiles":"linear","weighted_quantiles":"inverted CDF, stable(value,id)sort, total and prefix sequential_f64 in identical sorted order; first prefix>=q*total; no threshold epsilon",
                                  "valid_area":"sx=width0/patch_width; sy=height0/patch_height; area=(sx*valid_width)*(sy*valid_height); physical=(area*mpp_x_um)*mpp_y_um",
                                  "patient_default":"Equal weight of accepted slides, after tile-to-slide summary; never pool all patient tiles by default.",
                                  "empty":"Missing remains missing, with count0/weight0 and explicit no_accepted status. Zero-tile slides retain supplied slide_patients mapping.",
                                  "raster":"Actual accepted tile scores, center sampling of half-open valid rectangles, overlap arithmetic mean; missing value NaN and alpha0; coverage exact u64.",
                                  "raster_block_centers":"Always original_canvas_origin+(global_pixel_index+.5)*pixel_size, never subcanvas re-association. Reference may slice whole oracle; native must compute lazy blocks without allocating whole canvas.",
                                  "hue":"Generic numeric aggregation does not reinterpret circular angles; automatic H feature production excludes raw mean_degrees from ordinary modeling/aggregation."}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
