"""Observed-data binary/multiclass baseline specification, without imputation."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from scipy import stats

from golden_generate_reference import ROOT,clean


def summarize_numeric(values,group_size):
    values=np.asarray(values,dtype=float)
    return {"n":len(values),"missing":group_size-len(values),"mean":float(values.mean()) if len(values) else np.nan,
            "sd_ddof1":float(values.std(ddof=1)) if len(values)>1 else np.nan,
            "median":float(np.median(values)) if len(values) else np.nan,
            "q25":float(np.quantile(values,.25)) if len(values) else np.nan,
            "q75":float(np.quantile(values,.75)) if len(values) else np.nan}


def summarize_categorical(values,group_size):
    levels=sorted(set(values))
    counts={level:values.count(level) for level in levels}
    return {"n":len(values),"missing":group_size-len(values),"counts":counts,
            "proportions":{level:count/len(values) for level,count in counts.items()}}


def baseline(rows,classes):
    output=[]
    for feature in ["a","b","all_missing","sex","rare","constant_category","missing_category"]:
        continuous=feature in ["a","b","all_missing"]
        grouped=[[row[feature] for row in rows if row["target"]==index and row[feature] is not None] for index in range(len(classes))]
        groups={label:(summarize_numeric(values,sum(row["target"]==index for row in rows)) if continuous else summarize_categorical(values,sum(row["target"]==index for row in rows))) for index,(label,values) in enumerate(zip(classes,grouped))}
        statistic,pvalue=np.nan,np.nan
        contingency=None
        expected=None
        diagnostic=None
        if min(map(len,grouped))==0:
            method="insufficient_observations"
            diagnostic="At least one outcome group has no observed values for this variable. No imputation or cross-variable deletion."
        elif continuous:
            if len(classes)==2:
                result=stats.mannwhitneyu(*grouped,alternative="two-sided",method="auto",use_continuity=True)
                method="mann_whitney"
            else:
                result=stats.kruskal(*grouped)
                method="kruskal_wallis"
            statistic,pvalue=float(result.statistic),float(result.pvalue)
        else:
            levels=sorted({value for group in grouped for value in group})
            contingency=[[group.count(level) for group in grouped] for level in levels]
            if len(levels)<2:
                method="constant_or_absent_class"
                diagnostic="Categorical variable has fewer than two observed levels; no association test is defined."
            else:
                chi,p,df,expected_array=stats.chi2_contingency(contingency,correction=False)
                expected=expected_array.tolist()
                if expected_array.shape==(2,2) and np.any(expected_array<5):
                    fisher=stats.fisher_exact(contingency,alternative="two-sided")
                    method="fisher_2x2";statistic=float(fisher.statistic);pvalue=float(fisher.pvalue)
                else:
                    method="chi_square_no_yates";statistic=float(chi);pvalue=float(p)
        output.append({"feature":feature,"type":"continuous" if continuous else "categorical","groups":groups,
                       "method":method,"statistic":statistic,"p_value":pvalue,"contingency":contingency,"expected_counts":expected,"diagnostic":diagnostic})
    return output


def main():
    if metadata.version("scipy")!="1.13.1" or np.__version__!="1.23.0":
        raise SystemExit("Baseline reference requires SciPy1.13.1/NumPy1.23.0.")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing output; generate a fresh baseline candidate.")
    directory=ROOT/"golden/python_golden_v1"
    clinical=json.loads((directory/"statistics_reference.json").read_text())["clinical_logistic"]["rows"]
    data=json.loads((directory/"model_inputs.json").read_text())
    datasets={}
    for task in ["binary","multiclass"]:
        rows=[]
        source=clinical if task=="binary" else [r for r in data["datasets"][task]["records"] if r["cohort"]=="train"]
        omitted=0
        for record in source:
            if record["target"] is None:
                omitted+=1
                continue
            index=int(record["patient_id"].split("-P")[1])
            row={"sample_id":record["sample_id"],"patient_id":record["patient_id"],"cohort":"train","target":record["target"],
                 "a":record["a"] if task=="binary" else record["values"][0],"b":record["b"] if task=="binary" else record["values"][1],
                 "sex":record["sex"] if task=="binary" else "F" if index%3==0 else "M",
                 "rare":"R" if index%17==0 else "N","all_missing":None,"constant_category":"only","missing_category":None}
            rows.append(row)
        rows[4]["rare"]=None
        if task=="multiclass":
            rows[0]["a"]=None;rows[1]["b"]=None;rows[3]["sex"]=None
        classes=data["datasets"][task]["classes"]
        datasets[task]={"classes":classes,"rows":rows,"source_missing_target_rows_explicitly_omitted_when_constructing_fixture":omitted,
                        "analysis_unit":"sample","n_input":len(rows),"table":baseline(rows,classes),
                        "diagnostics":["Original data contains two lesions per patient: row-level numerical reference only; public patient inference must reject repeats or require explicit sample-unit analysis with dependence warning."]}
    output={"schema":"python_golden_v1.baseline.reference.v1","packages":{"numpy":"1.23.0","scipy":"1.13.1"},
            "reference_source_revision":"SciPy1.13.1 / 34a5ce2e0e51fc2005d3c2720bcd8296cd33af94",
            "source_sha256":{"model_inputs.json":hashlib.sha256((directory/"model_inputs.json").read_bytes()).hexdigest(),"statistics_reference.json":hashlib.sha256((directory/"statistics_reference.json").read_bytes()).hexdigest()},
            "policy":{"missing_outcome":"Error in baseline API; fixture assembly explicitly excludes its one source row with missing outcome, recorded separately. No implicit target deletion.",
                      "numeric":"Observed values separately for each variable/outcome group; sample SD ddof1 and linear q25/q50/q75. No imputation or complete-case deletion across variables.",
                      "categorical":"Counts and proportions divide by observed nonmissing variable values within outcome class; missing counts shown separately.",
                      "default_tests":"Binary continuous two-sided MW auto; multiclass continuous tie-corrected KW. Categorical auto: 2x2 with any expected<5 uses two-sided Fisher, otherwise Pearson chi-square without Yates, including multiclass tables.",
                      "degenerate":"Empty observed group or constant categorical variable retains counts and explicit undefined statistic/p plus diagnostic; never removes the variable or fills zeros as feature values.",
                      "unsupported":"Explicit Fisher on a non-2x2 table errors; no global multiclass Fisher approximation."},
            "datasets":datasets}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(clean(output),indent=2,allow_nan=False)+"\n").encode())
    print(args.output)


if __name__=="__main__":
    main()
