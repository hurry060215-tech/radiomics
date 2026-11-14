"""Generate review candidates; never overwrite an accepted python_golden_v1 fixture.

Run with the isolated PyRadiomics 3.0.1 reference interpreter. This script does not
import the native extension and refuses any other reference version. Candidate
updates require main-agent review; test execution never regenerates expected data.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import logging
import math
from pathlib import Path
import platform
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "2512e16979a544207f84c943e619ef33e02d3dc7"
PYRAD_REVISION = "c428a771c8cf682c462162b6babc836150d5d9d6"
FEATURES = {
    "firstorder": ["Mean", "Median", "Minimum", "Maximum", "10Percentile", "90Percentile", "Range", "InterquartileRange", "Variance", "MeanAbsoluteDeviation", "Skewness", "Kurtosis", "Energy", "RootMeanSquared", "Entropy", "Uniformity"],
    "glcm": ["JointAverage", "JointEnergy", "JointEntropy", "MaximumProbability", "Contrast", "Correlation", "DifferenceAverage", "Id", "Idm", "SumSquares"],
    "glrlm": ["ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "RunLengthNonUniformity", "RunLengthNonUniformityNormalized", "RunPercentage", "RunEntropy"],
    "glszm": ["SmallAreaEmphasis", "LargeAreaEmphasis", "GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "SizeZoneNonUniformity", "SizeZoneNonUniformityNormalized", "ZonePercentage", "ZoneEntropy"],
    "gldm": ["SmallDependenceEmphasis", "LargeDependenceEmphasis", "GrayLevelNonUniformity", "DependenceNonUniformity", "DependenceNonUniformityNormalized", "GrayLevelVariance", "DependenceVariance", "DependenceEntropy"],
    "ngtdm": ["Coarseness", "Contrast", "Busyness", "Complexity", "Strength"],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
    return value


def observed(call):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            result = {"status": "ok", "value": clean(call())}
        except Exception as exc:
            result = {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
        result["warnings"] = sorted(set(str(w.message) for w in recorded))
        return result


def reference_environment():
    import radiomics
    if radiomics.__version__ != "v3.0.1":
        raise RuntimeError("Reference generation requires public PyRadiomics v3.0.1, never 3.1.0.")
    names = ["numpy", "scipy", "pandas", "SimpleITK", "pyradiomics", "scikit-learn", "statsmodels", "PyWavelets", "pykwalify", "six", "setuptools", "wheel"]
    versions = {name: metadata.version(name) for name in names}
    package_root = Path(radiomics.__file__).parent
    source = {}
    for path in sorted(package_root.iterdir()):
        if path.suffix in (".py", ".pyd", ".so"):
            source[path.name] = sha(path.read_bytes())
    return {"python": platform.python_version(), "implementation": platform.python_implementation(), "packages": versions,
            "pyradiomics_revision": PYRAD_REVISION, "pyradiomics_tag": "v3.0.1", "pyradiomics_files_sha256": source,
            "scope": "Pinned imported numerical dependencies; pyd hashes are platform-build specific. No global environment modified."}


def feature_reference(cases):
    import numpy as np
    import radiomics
    import SimpleITK as sitk
    from radiomics import imageoperations, cMatrices
    radiomics.setVerbosity(logging.ERROR)
    output = {}
    for case in cases:
        array = np.asarray(case["values"], dtype=np.float64).reshape(tuple(reversed(case["shape"])))
        mask = np.asarray(case["mask"], dtype=np.uint8).reshape(array.shape)
        image, roi = sitk.GetImageFromArray(array), sitk.GetImageFromArray(mask)
        settings = {"label": 1, "distances": [1], "symmetricalGLCM": True, "weightingNorm": None,
                    "force2D": False, "force2Ddimension": 0, "gldm_a": 0, "voxelArrayShift": 0,
                    "normalize": False, "resampledPixelSpacing": None, "resegmentRange": None}
        settings.update({name: case[name] for name in ("binWidth", "binCount") if name in case})
        result = {"effective_settings": settings, "features": {}, "merged_features": {}}
        for family, names in FEATURES.items():
            module = importlib.import_module("radiomics." + family)
            klass = getattr(module, "Radiomics" + family.upper()) if family != "firstorder" else module.RadiomicsFirstOrder
            def compute(klass=klass, names=names, settings=settings):
                engine = klass(image, roi, **settings)
                for name in names:
                    engine.enableFeatureByName(name)
                return engine.execute()
            result["features"][family] = observed(compute)
            if family in ("glcm", "glrlm"):
                merged = dict(settings, weightingNorm="no_weighting")
                def compute_merged(klass=klass, names=names, merged=merged):
                    engine = klass(image, roi, **merged)
                    for name in names:
                        engine.enableFeatureByName(name)
                    return engine.execute()
                result["merged_features"][family] = observed(compute_merged)
        binned = observed(lambda: imageoperations.binImage(array, np.where(mask == 1), **settings)[0])
        result["discretized"] = binned
        if binned["status"] == "ok":
            labels = np.asarray(binned["value"], dtype=np.int32)
            if mask.any():
                ng = int(labels[mask == 1].max())
                common = (labels, mask.astype(bool))
                distances = np.asarray([1], dtype=int)
                def glcm():
                    matrix, angles = cMatrices.calculate_glcm(*common, distances, ng, False, 0)
                    matrix += matrix.transpose(0, 2, 1, 3).copy()
                    return {"angles_array_axes": angles, "angles_xyz": angles[:, ::-1], "counts": matrix[0]}
                def glrlm():
                    matrix, angles = cMatrices.calculate_glrlm(*common, ng, max(array.shape), False, 0)
                    return {"angles_array_axes": angles, "angles_xyz": angles[:, ::-1], "counts": matrix[0]}
                result["raw_matrices"] = {
                    "glcm": observed(glcm), "glrlm": observed(glrlm),
                    "glszm": observed(lambda: cMatrices.calculate_glszm(*common, ng, int(mask.sum()), False, 0)[0]),
                    "gldm": observed(lambda: cMatrices.calculate_gldm(*common, distances, ng, 0, False, 0)[0]),
                    "ngtdm": observed(lambda: cMatrices.calculate_ngtdm(*common, distances, ng, False, 0)[0]),
                }
        output[case["id"]] = result
    return output


def baseline_reference(baseline_source):
    import numpy as np
    sys.path.insert(0, str(baseline_source / "src"))
    from ov_radiomics import metrics
    source_hashes = {}
    for file in ("metrics.py", "_validation.py", "_backend.py"):
        path = "src/ov_radiomics/" + file
        blob = subprocess.check_output(["git", "show", BASELINE + ":" + path], cwd=baseline_source)
        if (baseline_source / path).read_bytes().replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
            raise RuntimeError("Generate baseline references from an unchanged 2512e16 worktree: " + path)
        source_hashes[path] = sha(blob)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    a = np.array([0., .4, .35, .8, .2, .6, .6, 1., .1, .9, .3, .7])
    b = np.array([.1, .3, .4, .7, .3, .5, .8, .9, .2, .85, .25, .75])
    parameters = {"threshold": .5, "confidence": .95, "n_bootstrap": 16, "random_state": 42}
    rng = np.random.RandomState(parameters["random_state"])
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    indices = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))].tolist() for _ in range(parameters["n_bootstrap"])]
    hl = metrics.hosmer_lemeshow(y, a, n_bins=4, backend="python")
    hl_table = hl.pop("table")
    hl["table"] = [{"bin": str(index), **row.to_dict()} for index, row in hl_table.iterrows()]
    return {"source_revision": BASELINE, "source_sha256": source_hashes,
            "inputs": {"y": y.tolist(), "a": a.tolist(), "b": b.tolist()}, "parameters": parameters,
            "binary_delong": metrics.binary_metrics(y, a, ci_method="delong", backend="python", **parameters).to_dict(),
            "binary_bootstrap": metrics.binary_metrics(y, a, ci_method="bootstrap", backend="python", **parameters).to_dict(),
            "bootstrap_indices": indices, "paired_delong": metrics.delong_test(y, a, b, backend="python").to_dict(),
            "dca": metrics.decision_curve(y, {"a": a, "b": b}, thresholds=[.5, .2, .5, .8], backend="python").to_dict("records"),
            "hosmer_lemeshow": hl,
            "failures": {"one_class": observed(lambda: metrics.binary_metrics([1, 1], [.1, .9], backend="python")),
                         "hl_constant": observed(lambda: metrics.hosmer_lemeshow(y, np.full(len(y), .5), backend="python")),
                         "invalid_probability": observed(lambda: metrics.binary_metrics(y, np.full(len(y), 1.01), backend="python"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh candidate directory; existing path is refused.")
    parser.add_argument("--baseline-source", type=Path, default=ROOT, help="Clean 2512e16 source worktree for existing metric API.")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing existing candidate output. Accepted Golden files are never overwritten.")
    environment = reference_environment()
    input_path = ROOT / "golden/python_golden_v1/cases.json"
    cases = json.loads(input_path.read_text())["cases"]
    args.output.mkdir(parents=True)
    result = {"schema": "python_golden_v1.reference.v1", "environment": environment,
              "input_file_sha256": sha(input_path.read_bytes()), "features": feature_reference(cases),
              "baseline_metrics": baseline_reference(args.baseline_source)}
    (args.output / "reference.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "requirements-reference.txt").write_text("# CPython " + environment["python"] + "\n" + "\n".join(name + "==" + version for name, version in environment["packages"].items()) + "\n", encoding="utf-8")
    print(str(args.output / "reference.json"))


if __name__ == "__main__":
    main()
