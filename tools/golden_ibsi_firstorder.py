"""Derive first-order expectations from the licensed phantom, without feature engines."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from golden_generate_reference import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing full-precision candidate.")
    if np.__version__!="1.23.0" or sitk.Version_VersionString()!="2.4.1":
        raise SystemExit("Input decoding requires the fixed NumPy1.23.0/SimpleITK2.4.1 environment.")
    directory=ROOT/"golden/ibsi_digital_v1"
    module_spec=importlib.util.spec_from_file_location("ibsi_firstorder_spec",directory/"firstorder_spec.py")
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    image=sitk.ReadImage(str(directory/"image.nii.gz"));mask=sitk.ReadImage(str(directory/"mask.nii.gz"))
    raw=sitk.GetArrayFromImage(image);labels=sitk.GetArrayFromImage(mask)
    values,counts=np.unique(raw[labels==1],return_counts=True)
    if any(int(v)!=v for v in values):
        raise RuntimeError("Original digital phantom intensities must be exact integers.")
    result=module.calculate({int(value):int(count) for value,count in zip(values,counts)})
    assert result["histogram"]==[{"gray":1,"count":50},{"gray":3,"count":1},{"gray":4,"count":16},{"gray":6,"count":7}]
    published=json.loads((directory/"published_reference.json").read_text())
    for feature in published["features"]:
        if not feature["feature"].startswith("original_firstorder_"):
            continue
        name=feature["feature"].split("original_firstorder_")[1]
        actual=result["ibsi_features"][name]
        reference=feature["rows"][0]
        error=reference["published_tolerance"] if reference["published_tolerance"] is not None else reference["printed_half_unit"]
        if abs(actual-reference["value"])>error:
            raise RuntimeError("Independent first-order math is outside published source interval: "+name)
    geometry={"shape_xyz":list(image.GetSize()),"spacing_mm":list(image.GetSpacing()),"origin_lps_mm":list(image.GetOrigin()),"direction_lps":list(image.GetDirection())}
    result.update(schema="ibsi_digital_v1.firstorder.fullprecision.v1",reference_type="Independent exact-histogram/Fraction specification; no PyRadiomics or native calculation.",
                  decoder_versions={"SimpleITK":"2.4.1","NumPy":"1.23.0"},geometry=geometry,
                  specification_sha256=hashlib.sha256((directory/"firstorder_spec.py").read_bytes()).hexdigest(),
                  input_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ["image.nii.gz","mask.nii.gz"]},
                  basic_voxel_shape={"VoxelCount":result["roi_count"],"Volume_mm3":result["roi_count"]*float(np.prod(image.GetSpacing()))},
                  tolerances={"exact":"ROI count, histogram, rational-derived integral features", "other_atol":1e-10,"other_rtol":1e-8},
                  kurtosis_policy="IBSI excess comparison is not passed by the native raw feature; difference3 is an explicit definition deviation. Independent raw definition is recorded only to explain and test this deviation, not to relabel the feature IBSI conformant.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
