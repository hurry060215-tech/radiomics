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
