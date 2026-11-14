"""Execute the independently reviewed Python pixel specification into a fresh candidate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

from golden_generate_reference import ROOT, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing an existing output; accepted pathology Golden is never overwritten.")
    directory = ROOT / "golden/python_golden_v1"
    source = directory / "pathology_spec.py"
    inputs = directory / "pathology_inputs.json"
    specification = importlib.util.spec_from_file_location("pathology_spec_v1", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    data = json.loads(inputs.read_text())
    patches = {}
    for patch in data["patches"]:
        arguments = (patch["width"], patch["height"], patch["pixels"], patch["valid_pixels"])
        qc = module.qc(*arguments, data["qc_config"])
        no_filter_config = {k: (None if k.startswith(("max_", "min_")) else v) for k, v in data["qc_config"].items()}
        patches[patch["id"]] = {"qc": qc, "default_no_exclusion": module.qc(*arguments, no_filter_config),
                               "colors": module.colors(*arguments, data["fixed_edges"]),
                               "colors_tissue_only": module.colors(*arguments, data["fixed_edges"], tissue_mask=qc["tissue_mask"])}
    tilings = [module.tile_positions(**case) for case in data["tilings"]]
    masks = {}
    for mask in data["masks"]:
        arguments = (mask["width"], mask["height"], mask["labels"], mask["valid_pixels"])
        masks[mask["id"]] = {"coverage": module.annotation_coverage(*arguments, mask["classes"]),
                             "objects": module.objects(*arguments, mask["mask_kind"], mask["connectivity"], mask["mpp"], mask["overlapping_tiles"])}
    white = data["patches"][0]
    qc_arguments = (white["width"], white["height"], white["pixels"], white["valid_pixels"])
    failures = {"zero_border_band": observed(lambda: module.qc(*qc_arguments, dict(data["qc_config"], border_band_px=0))),
                "zero_downsample": observed(lambda: module.tile_positions(3, 2, [2, 2], [1, 1], "pad", [0, 1])),
                "nonfinite_histogram_edges": observed(lambda: module.histogram([0], [0, float("nan"), 1])),
                "negative_patch_mpp": observed(lambda: module.objects(1, 1, [1], [0, 0, 1, 1], "instance", 4, [-1, 1]))}
    result = {"schema": "python_golden_v1.pathology.reference.v1", "reference_python": platform.python_version(),
              "specification_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
              "source_references": {"sRGB_XYZ_matrix": "https://www.w3.org/TR/2026/CRD-css-color-4-20261112/#color-conversion-code",
                                    "HSV_BT601": "https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html",
                                    "Laplacian": "https://docs.opencv.org/4.13.0/d5/d0f/tutorial_py_gradients.html"},
              "coordinates": "Level0 pixel-edge x-right/y-down. Object centroid averages patch pixel centers x+.5,y+.5. MPP passed to objects is patch-pixel MPP, derived from level0 MPP times tile extent/patch dimensions.",
              "color_profile": "sRGB / D65 XYZ / Lab_D65 without D50 adaptation. For r=g=b the theoretical a=b=0 identity is exact; no epsilon rounding for other colors. Hue degrees circular statistics excludes S=0.",
              "comparison": "IDs, coordinate topology, masks, histogram counts, reasons and denominators exact; scalar QC/statistics atol1e-10 rtol1e-8; color/Lab/circular outputs atol1e-6 rtol1e-5. Missing states exact.",
              "patches": patches, "tilings": tilings, "masks": masks, "failures": failures,
              "restrictions": ["RGB8 only for QC/colors. Gray8/Gray16 require an explicit mapping; native errors Unsupported otherwise.",
                               "No inferred MPP from DPI or magnification.", "Patch-local object counts are not a global deduplicated cell count.",
                               "Patient model probabilities cannot be broadcast into local tile predictions.", "Default QC scores retain patches; only explicit limits exclude."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
