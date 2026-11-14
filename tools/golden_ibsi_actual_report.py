"""Produce an actual-vs-independent-IBSI report; never modify reference files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from golden_generate_reference import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing actual report; choose a fresh output.")
    result=subprocess.run(["cargo","test","-p","ov-radiomics-core","--test","golden_ibsi_native","--features","serde_json/float_roundtrip","--","--nocapture"],cwd=ROOT,text=True,capture_output=True)
    if result.returncode:
        raise SystemExit(result.stdout+result.stderr)
    prefix="IBSI_NATIVE_ACTUAL_JSON:"
    actual=[json.loads(line.split(prefix,1)[1]) for line in result.stdout.splitlines() if prefix in line]
    if len(actual)!=2:
        raise RuntimeError("Native acceptance must emit both directional modes.")
    directory=ROOT/"golden/ibsi_digital_v1"
    published=json.loads((directory/"published_reference.json").read_text())
    precise=json.loads((directory/"firstorder_fullprecision.json").read_text())
    comparisons=[]
    for run in actual:
        for feature in published["features"]:
            key=feature["feature"]
            value=run["features"][key]
            if key.startswith("original_firstorder_"):
                name=key[len("original_firstorder_"):]
                expected=precise["ibsi_features"][name]
                comparison={"feature":key,"mode":run["mode"],"actual":value,"independent_reference":expected,"difference":value-expected,
                            "comparison":"exact" if name in precise["exact_integer_features"] else "atol1e-10_rtol1e-8","ibsi_conformant":name!="Kurtosis"}
                if name=="Kurtosis":
                    comparison.update(deviation_reason="python_golden_v1 raw kurtosis versus IBSI excess; expected difference3. Native output is retained and this feature is not IBSI conformant.",declared_offset_check=abs(value-expected-3)<=1e-10+1e-8*abs(expected))
                else:
                    comparison["passed"]=value==expected if name in precise["exact_integer_features"] else abs(value-expected)<=1e-10+1e-8*abs(expected)
                comparisons.append(comparison)
            else:
                reference=next(r for r in feature["rows"] if r["aggregation"] in ["3D",run["mode"]])
                tolerance=reference["published_tolerance"] if reference["published_tolerance"] is not None else reference["printed_half_unit"]
                comparisons.append({"feature":key,"mode":run["mode"],"actual":value,"printed_reference":reference["printed_value"],"reference_numeric":reference["value"],"difference":value-reference["value"],
                                    "comparison":reference["acceptance"],"published_interval_radius":tolerance,"passed":abs(value-reference["value"])<=tolerance,"ibsi_conformant":True})
    if any(not r.get("passed",r.get("declared_offset_check",False)) for r in comparisons):
        raise RuntimeError("An actual comparison failed; no accepted report is written.")
    files=["crates/radiomics-core/src/features.rs","crates/radiomics-core/src/imaging.rs","golden/ibsi_digital_v1/published_reference.json","golden/ibsi_digital_v1/firstorder_fullprecision.json"]
    output={"schema":"ibsi_digital_v1.actual_acceptance_report.v1","status":"passed_with_declared_raw_kurtosis_nonconformance",
            "scope":"IBSI1 digital phantom,3D,noresampling,FBS1,distance1,GLCM symmetric,GLDMalpha0; both per-direction mean and explicit matrix merging. No claim for2D/CTresampling/other preprocessing.",
            "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
            "native_test_result":"cargo test -p ov-radiomics-core --test golden_ibsi_native passed",
            "native_rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),
            "exact_matrix_checks":["GLCM3Dmerged","GLRLM3Dmerged","GLSZM3D","GLDM3D","NGTDM counts"],
            "ngtdm_difference_sum_check":"Declared published six-decimal precision interval, independent from PythonGolden tolerance.",
            "basic_shape_checks":{"voxel_count":74,"voxel_counting_volume_mm3":592,"comparison":"exact; no mesh morphology claim"},
            "feature_comparisons":comparisons}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(output,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
