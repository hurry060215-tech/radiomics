"""Verify independent IBSI source facts, licensed inputs and matrix invariants."""
import gzip
import hashlib
import json
from pathlib import Path
import struct


DIRECTORY=Path(__file__).resolve().parents[1]/"golden/ibsi_digital_v1"


def read(name):
    return json.loads((DIRECTORY/name).read_text())


def test_independent_ibsi_source_and_binary_input_integrity():
    manifest=read("manifest.json")
    assert manifest["status"]=="independent_source_frozen_no_native_results"
    for name,expected in manifest["files_sha256"].items():
        assert hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()==expected
    reference=read("published_reference.json")
    assert reference["license"]=="CC-BY-4.0"
    assert len(reference["features"])==55
    for feature in reference["features"]:
        family=feature["feature"].split("_")[1]
        required=["3D, averaged","3D, merged"] if family in ["glcm","glrlm"] else ["3D"]
        assert sorted(row["aggregation"] for row in feature["rows"])==sorted(required)
    for name in ("image.nii.gz","mask.nii.gz"):
        payload=gzip.decompress((DIRECTORY/name).read_bytes())
        assert struct.unpack_from("<i",payload,0)[0]==348
        assert struct.unpack_from("<4h",payload,40)==(3,5,4,4)
        assert payload[123]&7==2
    kurtosis=next(f for f in reference["features"] if f["feature"]=="original_firstorder_Kurtosis")
    assert kurtosis["definition_deviation"]
    assert all(row["published_tolerance"] is None or row["published_tolerance"]>=0 for feature in reference["features"] for row in feature["rows"])


def test_independent_appendix_integer_matrices_reconstruct_phantom_counts():
    matrices=read("published_reference.json")["matrices"]
    glcm={(int(r["values"][0]),int(r["values"][1])):int(r["values"][2]) for r in matrices["glcm_merged"]["rows"]}
    assert all(glcm[(j,i)]==count for (i,j),count in glcm.items())
    glrlm=[r["values"] for r in matrices["glrlm_merged"]["rows"]]
    assert sum(length*count for _,length,count in glrlm)==74*13
    assert sum(length*count for gray,length,count in glrlm if gray==1)==50*13
    glszm=[r["values"] for r in matrices["glszm"]["rows"]]
    assert sum(size*count for _,size,count in glszm)==74
    gldm=[r["values"] for r in matrices["gldm"]["rows"]]
    assert sum(count for _,_,count in gldm)==74
    ngtdm=[r["values"] for r in matrices["ngtdm"]["rows"]]
    assert sum(count for _,_,count in ngtdm)==74
    assert [gray for gray,_,_ in ngtdm]==[1,3,4,6]


def test_independent_fraction_firstorder_is_strict_and_not_only_printed_intervals():
    result=read("firstorder_fullprecision.json")
    assert result["roi_count"]==74 and result["intensity_sum"]==159 and result["sum_of_squares"]==567
    assert result["rational_reference"]["Mean"]=={"numerator":159,"denominator":74}
    assert result["rational_reference"]["Variance"]=={"numerator":16677,"denominator":5476}
    assert result["specification_sha256"]==hashlib.sha256((DIRECTORY/"firstorder_spec.py").read_bytes()).hexdigest()
    assert result["basic_voxel_shape"]=={"VoxelCount":74,"Volume_mm3":592.0}
    for name in result["exact_integer_features"]:
        rational=result["rational_reference"][name]
        assert rational["denominator"]==1
        assert result["ibsi_features"][name]==rational["numerator"]
    assert abs(result["native_raw_kurtosis_definition"]-result["ibsi_features"]["Kurtosis"]-3)<1e-15
    for feature in read("published_reference.json")["features"]:
        if not feature["feature"].startswith("original_firstorder_"):
            continue
        name=feature["feature"].removeprefix("original_firstorder_")
        row=feature["rows"][0]
        interval=row["published_tolerance"] if row["published_tolerance"] is not None else row["printed_half_unit"]
        assert abs(result["ibsi_features"][name]-row["value"])<=interval
