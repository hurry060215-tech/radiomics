"""Freeze a strict normalized WSIData provenance contract before its Rust builder."""
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
        raise SystemExit("Refusing existing normalized WSIData contract candidate.")
    directory=ROOT/"golden/python_golden_v1"
    source=directory/"wsidata_contract_spec.py"
    loader=importlib.util.spec_from_file_location("wsidata_contract_spec",source)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    data=json.loads((directory/"wsidata_contract_inputs.json").read_text())
    keys=["tile_key","shape_rows","tile_spec","shape_to_level0","slide_shape_wh","level0_mpp_xy","source"]
    payload={key:copy.deepcopy(data[key]) for key in keys}
    # Only real geometry enters the Rust builder; arbitrary upstream x/y columns do not.
    payload["shape_rows"]=[{key:row[key] for key in ["tile_id","geometry_corners","valid_pixels"] if key in row} for row in payload["shape_rows"]]
    current=module.snapshot(**payload)
    shape_ids=[row["tile_id"] for row in payload["shape_rows"]]
    def link(now=current,saved=current,region=data["table_region"],ids=shape_ids,tiles=data["obs_tile_ids"],rows=data["obs_row_ids"]):
        return module.validate_embedding_link(now,saved,region,data["tile_key"],ids,tiles,rows)
    ordering=link()
    assert ordering==data["expected_row_order"]
    variants={}
    explicit=copy.deepcopy(payload);explicit.update(copy.deepcopy(data["explicit_transform_variant"]))
    variants["positive_scale2"]=module.snapshot(**explicit)
    translation=copy.deepcopy(payload);translation.update(copy.deepcopy(data["translation_variant"]))
    variants["translation_point1"]=module.snapshot(**translation)
    reversed_rows=copy.deepcopy(payload);reversed_rows["shape_rows"].reverse()
    variants["reversed_shape_order"]={"snapshot":module.snapshot(**reversed_rows),"row_order":link(ids=list(reversed(shape_ids)))}
    assert variants["reversed_shape_order"]["snapshot"]["sha256"]==current["sha256"]
    edge=copy.deepcopy(payload);edge["shape_rows"]=copy.deepcopy(data["edge_validity_variant"]["shape_rows"])
    variants["explicit_valid_edge"]=module.snapshot(**edge)
    changed_geometry=copy.deepcopy(payload)
    for row in changed_geometry["shape_rows"]:
        row["geometry_corners"]=[[x+4,y] for x,y in row["geometry_corners"]]
    changed_spec=copy.deepcopy(payload);changed_spec["tile_spec"]["stride_width"]=1
    duplicate=copy.deepcopy(payload);duplicate["shape_rows"][1]["tile_id"]=duplicate["shape_rows"][0]["tile_id"]
    rotated=copy.deepcopy(payload);rotated["shape_rows"][0]["geometry_corners"]=[[0,2],[2,0],[4,2],[2,4],[0,2]]
    shear=copy.deepcopy(payload);shear["shape_to_level0"][0][1]=.1
    unknown_edge=copy.deepcopy(payload);unknown_edge["slide_shape_wh"]=None
    invalid_edge=copy.deepcopy(edge);invalid_edge["shape_rows"][0]["valid_pixels"]=[0,0,2,2]
    nonzero_level=copy.deepcopy(payload);nonzero_level["tile_spec"]["base_level"]=1
    boolean_level=copy.deepcopy(payload);boolean_level["tile_spec"]["base_level"]=False
    wrong_span=copy.deepcopy(payload);wrong_span["shape_rows"][0]["geometry_corners"]=[[0,0],[5,0],[5,4],[0,4],[0,0]]
    huge=copy.deepcopy(payload);huge["tile_spec"].update(width=3,height=4,base_downsample=1,mpp=.5);huge["slide_shape_wh"]=[10000000000000008,20]
    huge["shape_rows"]=[dict(tile_id="huge",geometry_corners=[[10000000000000000,0],[10000000000000004,0],[10000000000000004,4],[10000000000000000,4],[10000000000000000,0]])]
    corrupt_saved=copy.deepcopy(current);corrupt_saved["snapshot"]["source"]="different"
    corrupt_current=copy.deepcopy(current);corrupt_current["snapshot"]["source"]="different"
    unknown_schema=copy.deepcopy(current);unknown_schema["snapshot"]["schema"]="unknown"
    failures={"duplicate_shape_id":observed(lambda:module.snapshot(**duplicate)),"duplicate_obs_tile_id":observed(lambda:link(tiles=["T0","T0"])),
              "duplicate_obs_row_id":observed(lambda:link(rows=["same","same"])),"same_count_wrong_ids":observed(lambda:link(tiles=["T1","absent"])),
              "wrong_region":observed(lambda:link(region="other")),"missing_generation_manifest":observed(lambda:link(saved=None)),
              "same_ids_changed_geometry":observed(lambda:link(now=module.snapshot(**changed_geometry))),"same_ids_changed_tilespec":observed(lambda:link(now=module.snapshot(**changed_spec))),
              "corrupt_saved_digest":observed(lambda:link(saved=corrupt_saved)),"corrupt_current_digest":observed(lambda:link(now=corrupt_current)),
              "unknown_manifest_schema":observed(lambda:link(saved=unknown_schema)),"rotated_polygon":observed(lambda:module.snapshot(**rotated)),
              "sheared_transform":observed(lambda:module.snapshot(**shear)),"unknown_edge_validity":observed(lambda:module.snapshot(**unknown_edge)),
              "explicit_valid_rectangle_includes_padding":observed(lambda:module.snapshot(**invalid_edge)),"nonzero_base_level":observed(lambda:module.snapshot(**nonzero_level)),
              "boolean_base_level":observed(lambda:module.snapshot(**boolean_level)),"wrong_nominal_span":observed(lambda:module.snapshot(**wrong_span)),
              "roundoff_bound_too_large":observed(lambda:module.snapshot(**huge)),"empty_source_identity":observed(lambda:module.snapshot(**dict(payload,source="")))}
    if any(value["status"]!="error" for value in failures.values()):
        raise RuntimeError("A strict contract failure fixture was unexpectedly accepted.")
    canonical_float_cases=[{"value":value,"python_float_hex":value.hex(),"canonical":module._canonical(value)} for value in [0.,-0.,1.,.1,float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x1.fffffffffffffp+1023')]]
    output={"schema":"python_golden_v1.wsidata_contract.reference.v1","specification_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "input_sha256":hashlib.sha256((directory/"wsidata_contract_inputs.json").read_bytes()).hexdigest(),
            "snapshot_input_fields":keys,"snapshot_payload":payload,"current_snapshot":current,"expected_row_order":ordering,
            "aligned_feature_values":[data["embedding_values"][i] for i in ordering],
            "source_row_ids_in_shape_order":[data["obs_row_ids"][i] for i in ordering],"variants":variants,"failures":failures,"canonical_float_cases":canonical_float_cases,
            "contract":{"runtime":"Rust builds normalized payload/digest and validates alignment; Python reads actual GeoDataFrame geometry/SpatialData transforms/X/row IDs and performs conversion only.",
                        "geometry":"Actual hole-free axis-aligned closed rectangle polygon; raw x/y columns are ignored. Positive axis-aligned explicit transform only; actual transformed coordinates retained.",
                        "types":"Continuous bounds/transform/MPP and known TileSpec float roles normalized to f64 without losing integer precision; dims/levels/validpixels remain integers; unknown TileSpec JSON types retained.",
                        "span":"Six-operation conservative IEEE gamma bound, only metadata nominal extent comparison. Bound-relative-to-span cap1e-10 prevents unreliable large-coordinate acceptance. No coordinate/raster rounding or epsilon.",
                        "canonical":"Recursive float64_hex strings exactly matching Pythonfloat.hex including signed0/subnormals, sorted JSONkeys, UTF8compact JSON; no decimal formatting dependency.",
                        "cache":"Generation-time saved geometry/TileSpec/source manifest and its valid digest required. Current manifest is independently recomputed; sameN/sameIDs alone never bless stale cache.",
                        "row_alignment":"Authoritative shape tileID↔tableobs.tile_id bijection plus table region link; both obs row IDs and tile IDs unique; Xrows interpreted by that explicit correspondence.",
                        "edge_validity":"Unknown full/edge validity rejected unless slide shape proves full extent or an explicit valid pixel rectangle is recorded and inside known slide.",
                        "caller_arrays":"from_arrays is a separate caller-provided-input route, never automatic fallback when WSI provenance fails.",
                        "holes":"Real Python geometry extraction must reject non-Polygon, invalid polygon and any interior rings before building this normalized closed-rectangle input."}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
