"""Collect independent IBSI facts and licensed phantom bytes at pinned revisions.

No native radiomics output is read. Published precision is recorded explicitly;
it does not replace the much tighter independent Python numerical tolerances.
"""
import argparse
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


DOC_REV="b952e71e9d40569e8a9ff3c740bc389c04dd6c28"
DATA_REV="6da96021bc91faf4c0cb7fd7fa56a4225d2064a8"
MAPPING={
"firstorder":dict(Mean="stat_mean",Median="stat_median",Minimum="stat_minimum",Maximum="stat_maximum",**{"10Percentile":"stat_p10","90Percentile":"stat_p90"},Range="stat_range",InterquartileRange="stat_iqr",Variance="stat_variance",MeanAbsoluteDeviation="stat_mean_absolute_dev",Skewness="stat_skewness",Kurtosis="stat_kurtosis",Energy="stat_energy",RootMeanSquared="stat_root_mean_square",Entropy="int_hist_entropy",Uniformity="int_hist_uniformity"),
"glcm":dict(JointAverage="cm_joint_average",JointEnergy="cm_uniformity",JointEntropy="cm_joint_entropy",MaximumProbability="cm_joint_maximum",Contrast="cm_contrast",Correlation="cm_correlation",DifferenceAverage="cm_difference_average",Id="cm_inverse_difference",Idm="cm_inverse_difference_moment",SumSquares="cm_joint_variance"),
"glrlm":dict(ShortRunEmphasis="rlm_short_runs_emphasis",LongRunEmphasis="rlm_long_runs_emphasis",GrayLevelNonUniformity="rlm_grey_level_non_uniformity",GrayLevelNonUniformityNormalized="rlm_grey_level_non_uniformity_normalised",RunLengthNonUniformity="rlm_run_length_non_uniformity",RunLengthNonUniformityNormalized="rlm_run_length_non_uniformity_normalised",RunPercentage="rlm_run_percentage",RunEntropy="rlm_run_entropy"),
"glszm":dict(SmallAreaEmphasis="small_zone_emphasis",LargeAreaEmphasis="szm_large_zone_emphasis",GrayLevelNonUniformity="szm_grey_level_non_uniformity",GrayLevelNonUniformityNormalized="szm_grey_level_non_uniformity_normalised",SizeZoneNonUniformity="szm_zone_size_non_uniformity",SizeZoneNonUniformityNormalized="szm_zone_size_non_uniformity_normalised",ZonePercentage="szm_zone_percentage",ZoneEntropy="szm_zone_size_entropy"),
"gldm":dict(SmallDependenceEmphasis="ngldm_low_dependence_emphasis",LargeDependenceEmphasis="ngldm_high_dependence_emphasis",GrayLevelNonUniformity="ngldm_grey_level_non_uniformity",DependenceNonUniformity="ngldm_dependence_count_non_uniformity",DependenceNonUniformityNormalized="ngldm_normalised_dependence_count_non_uniformity",GrayLevelVariance="ngldm_grey_level_variance",DependenceVariance="ngldm_dependence_count_variance",DependenceEntropy="ngldm_dependence_count_entropy"),
"ngtdm":dict(Coarseness="ngtdm_coarseness",Contrast="ngtdm_contrast",Busyness="ngtdm_busyness",Complexity="ngtdm_complexity",Strength="ngtdm_strength")}


def blob(repo,revision,file):
    return subprocess.check_output(["git","-C",str(repo),"show",revision+":"+file])


def number(text):
    raw=text.replace(":math:`","").replace("`","").replace("\\","").replace(" ","")
    if "times10^" in raw:
        base,exponent=raw.split("times10^")
        value=Decimal(base)*(Decimal(10)**int(exponent.strip("{}")))
    else:
        value=Decimal(raw)
    half=(Decimal(10)**value.as_tuple().exponent)/2
    return float(value),float(half)


def sections(text):
    anchors=list(re.finditer(r"(?m)^\.\. _(feat_[A-Za-z0-9_]+):\s*$",text))
    result={}
    for i,anchor in enumerate(anchors):
        end=anchors[i+1].start() if i+1<len(anchors) else len(text)
        block=text[anchor.end():end]
        # The pinned manual omits an anchor for Small zone emphasis. A feature
        # block must end at the next actual feature title, even without an anchor.
        titles=list(re.finditer(r"(?m)^[^\n]+\n\^{4,}[ \t]*$",block))
        if len(titles)>1:
            block=block[:titles[1].start()]
        result[anchor.group(1)[5:]]=(block,text[:anchor.start()].count("\n")+1,text[:anchor.end()].count("\n")+1)
    start=text.index("Small zone emphasis\n")
    end=text.index(".. _feat_szm_large_zone_emphasis:",start)
    result["small_zone_emphasis"]=(text[start:end],text[:start].count("\n")+1,text[:start].count("\n")+1)
    return result


def published_features(text):
    blocks=sections(text)
    output=[]
    for family,mapping in MAPPING.items():
        for name,anchor in mapping.items():
            block,line,first_line=blocks[anchor]
            identifier=re.search(r'text-align:right">([A-Z0-9]{4})</p>',block)
            rows=[]
            for offset,raw in enumerate(block.splitlines()):
                if not raw.strip().startswith("| dig. phantom"):
                    continue
                cells=[x.strip() for x in raw.strip().strip("|").split("|")]
                if len(cells)==4:
                    aggregation="3D";value,tolerance,consensus=cells[1:]
                elif len(cells)==5:
                    aggregation,value,tolerance,consensus=cells[1:]
                else:
                    raise ValueError("Unexpected published table row: "+raw)
                if aggregation not in ["3D","3D, averaged","3D, merged"]:
                    continue
                numeric,half=number(value)
                tol=None if tolerance=="—" else number(tolerance)[0]
                rows.append({"aggregation":aggregation,"printed_value":value,"value":numeric,"printed_tolerance":tolerance,"published_tolerance":tol,
                             "printed_half_unit":half,"acceptance":"published_tolerance" if tol is not None else "published_printed_precision_interval",
                             "consensus":consensus,"source_line":first_line+offset})
                if text.splitlines()[first_line+offset-1].strip()!=raw.strip():
                    raise ValueError("Published row readback differs from its original absolute source line.")
            if not rows:
                raise ValueError("No independent 3D benchmark for "+family+"/"+name)
            required=["3D, averaged","3D, merged"] if family in ["glcm","glrlm"] else ["3D"]
            if sorted(row["aggregation"] for row in rows)!=sorted(required):
                raise ValueError("A source section has missing or cross-feature aggregation rows: "+family+"/"+name)
            for row in rows:
                if "dig. phantom" not in text.splitlines()[row["source_line"]-1]:
                    raise ValueError("Source line must identify the actual published data row.")
            output.append({"feature":"original_"+family+"_"+name,"source_anchor":"feat_"+anchor if anchor!="small_zone_emphasis" else "small-zone-emphasis",
                           "ibsi_identifier":identifier.group(1) if identifier else None,"source_section_line":line,"rows":rows,
                           "definition_deviation":"Native python_golden_v1 emits raw kurtosis; IBSI excess is raw minus3. Native profile is not conformant for this feature." if name=="Kurtosis" else None})
    return output


def matrix_rows(text,title):
    start=text.index(title+"\n")
    rest=text[start+len(title)+1:]
    end=re.search(r"\n[^\n]+\n-{8,}\n",rest)
    block=rest[:end.start()] if end else rest
    rows=[]
    for raw in block.splitlines():
        if not raw.strip().startswith("|"):
            continue
        cells=[x.strip() for x in raw.strip().strip("|").split("|")]
        if len(cells)!=3:
            continue
        try:
            values=[number(cell)[0] for cell in cells]
        except Exception:
            continue
        rows.append({"printed":cells,"values":values})
    if not rows:
        raise ValueError("No appendix matrix rows for "+title)
    return {"source_title":title,"source_line":text[:start].count("\n")+1,"rows":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-repo",required=True,type=Path)
    parser.add_argument("--doc-repo",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing IBSI candidate destination.")
    feature_bytes=blob(args.doc_repo,DOC_REV,"docs/03_Image_features.rst")
    matrix_bytes=blob(args.doc_repo,DOC_REV,"docs/06_Digital_phantom_texture_matrices.rst")
    feature_source=feature_bytes.decode("utf-8")
    matrix_source=matrix_bytes.decode("utf-8")
    data_path="ibsi_1_digital_phantom/"
    files={"image.nii.gz":blob(args.data_repo,DATA_REV,data_path+"nifti/image/phantom.nii.gz"),
           "mask.nii.gz":blob(args.data_repo,DATA_REV,data_path+"nifti/mask/mask.nii.gz"),
           "LICENSE.md":blob(args.data_repo,DATA_REV,data_path+"LICENSE.md"),
           "SOURCE_README.md":blob(args.data_repo,DATA_REV,data_path+"README.md")}
    matrices={name:matrix_rows(matrix_source,title) for name,title in [("glcm_merged","Grey level co-occurrence matrix (3D, merged)"),("glrlm_merged","Grey level run length matrix (3D, merged)"),("glszm","Grey level size zone matrix (3D)"),("gldm","Neighbouring grey level dependence matrix (3D)"),("ngtdm","Neighbourhood grey tone difference matrix (3D)")]}
    result={"schema":"ibsi_digital_v1.published_reference.v1","track":"Independent IBSI; no PyRadiomics or native outputs used to establish these expectations.",
            "document_repository":"https://github.com/theibsi/ibsi-doc","document_revision":DOC_REV,
            "source_sha256":{"03_Image_features.rst":hashlib.sha256(feature_bytes).hexdigest(),"06_Digital_phantom_texture_matrices.rst":hashlib.sha256(matrix_bytes).hexdigest()},
            "data_repository":"https://github.com/theibsi/data_sets","data_revision":DATA_REV,"license":"CC-BY-4.0","attribution":"IBSI-1 digital phantom, original design by Alex Zwanenburg; source NIfTI files unmodified.",
            "input_sha256":{name:hashlib.sha256(payload).hexdigest() for name,payload in files.items()},
            "spatial_unit_code":gzip.decompress(files["image.nii.gz"])[123]&7,
            "configuration":{"ndim":3,"bin_width":1,"resampling":None,"normalization":None,"distance":1,"distance_norm":"Chebyshev","gldm_alpha":0,"glcm_symmetric":True,"directional_modes":["3D, averaged","3D, merged"]},
            "precision_policy":"Published numeric tolerance is retained verbatim. When the digital table prints an em-dash tolerance, compare against its declared printed half-unit interval, not a fabricated full-precision scalar. This independent track does not change any PythonGolden atol/rtol. Integer count matrices are exact; NGTDM printed difference sums have explicitly recorded printed precision.",
            "features":published_features(feature_source),"matrices":matrices}
    args.output.mkdir(parents=True)
    for name,payload in files.items():
        (args.output/name).write_bytes(payload)
    (args.output/".gitattributes").write_bytes(b"*.nii.gz -text\n")
    (args.output/"published_reference.json").write_bytes((json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
