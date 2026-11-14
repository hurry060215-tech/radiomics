"""Freeze explicit generic-grid geometry references using SimpleITK 2.4.1.

This is not PyRadiomics's ROI-based resampling grid. Input/output affine values
are NIfTI RAS; SimpleITK image construction explicitly converts RAS to LPS.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def affine(origin, direction, spacing):
    n = len(spacing)
    matrix = np.eye(n + 1)
    matrix[:n, :n] = np.asarray(direction) * spacing
    matrix[:n, n] = origin
    return matrix.tolist()


def image_from_ras(values, shape, spacing, origin, direction, is_mask=False):
    flip = np.diag([-1, -1] + ([1] if len(shape) == 3 else []))
    image = sitk.GetImageFromArray(np.asarray(values, dtype=np.uint8 if is_mask else np.float64).reshape(tuple(reversed(shape))))
    image.SetSpacing(spacing)
    image.SetOrigin((flip @ np.asarray(origin)).tolist())
    image.SetDirection((flip @ np.asarray(direction)).ravel().tolist())
    return image


def source_case(identifier, shape, spacing, origin, direction, new_spacing, crop_index, crop_size):
    values = list(range(math.prod(shape)))
    mask = [int(i % 3 != 0) for i in values]
    image = image_from_ras(values, shape, spacing, origin, direction)
    roi = image_from_ras(mask, shape, spacing, origin, direction, is_mask=True)
    n = len(shape)
    flip = np.diag([-1, -1] + ([1] if n == 3 else []))
    out_shape = [math.ceil(size * old / new) for size, old, new in zip(shape, spacing, new_spacing)]
    out_origin = (np.asarray(origin) + np.asarray(direction) @ ((np.asarray(new_spacing) - spacing) / 2)).tolist()
    reference = image_from_ras([0] * math.prod(out_shape), out_shape, new_spacing, out_origin, direction)
    outputs = {}
    for name, source, method, fill in (("linear", image, sitk.sitkLinear, -999), ("nearest", image, sitk.sitkNearestNeighbor, -999), ("mask_nearest", roi, sitk.sitkNearestNeighbor, 0)):
        result = sitk.Resample(source, reference, sitk.Transform(n, sitk.sitkIdentity), method, fill, source.GetPixelID())
        outputs[name] = sitk.GetArrayFromImage(result).ravel().tolist()
    cropped = sitk.RegionOfInterest(image, crop_size, crop_index)
    cropped_origin = (flip @ np.asarray(cropped.GetOrigin())).tolist()
    cropped_expected_origin = (np.asarray(origin) + np.asarray(direction) @ (np.asarray(crop_index) * spacing)).tolist()
    np.testing.assert_allclose(cropped_origin, cropped_expected_origin, atol=1e-12, rtol=0)
    return {"id": identifier, "shape_xyz": shape, "values_x_fastest": values, "mask_x_fastest": mask,
            "spacing_mm": spacing, "origin_ras_mm": origin, "direction_ras": direction,
            "affine_ras": affine(origin, direction, spacing),
            "resample": {"spacing_mm": new_spacing, "shape_xyz": out_shape, "origin_ras_mm": out_origin,
                         "affine_ras": affine(out_origin, direction, new_spacing), "outputs_x_fastest": outputs,
                         "image_default_value": -999, "mask_default_value": 0},
            "crop": {"index_xyz": crop_index, "shape_xyz": crop_size, "origin_ras_mm": cropped_origin,
                     "affine_ras": affine(cropped_origin, direction, spacing),
                     "values_x_fastest": sitk.GetArrayFromImage(cropped).ravel().tolist()},
            "world_landmarks_ras_mm": [{"index_xyz": idx, "world": (np.asarray(origin) + np.asarray(direction) @ (np.asarray(idx) * spacing)).tolist()} for idx in ([0] * n, [x - 1 for x in shape])],
            "sitk_origin_lps_mm": image.GetOrigin(), "sitk_direction_lps": image.GetDirection()}


def boundary_cases():
    source = image_from_ras([0, 10, 20, 30, 40, 50], [3, 2], [1., 1.], [0., 0.], [[1., 0.], [0., 1.]])
    output = []
    for x in [-.50001, -.5, -.49999, 0., .49999, .5, .50001, 1.5, 2.49999, 2.5]:
        ref = image_from_ras([0], [1, 1], [1., 1.], [x, 0.], [[1., 0.], [0., 1.]])
        values = {}
        for name, method in (("linear", sitk.sitkLinear), ("nearest", sitk.sitkNearestNeighbor)):
            values[name] = float(sitk.GetArrayFromImage(sitk.Resample(source, ref, sitk.Transform(2, sitk.sitkIdentity), method, -999, sitk.sitkFloat64))[0, 0])
        output.append({"source_continuous_index_xy": [x, 0.], **values})
    return output


def main():
    if sitk.Version_VersionString() != "2.4.1" or np.__version__ != "1.23.0":
        raise SystemExit("Geometry reference requires SimpleITK2.4.1 and NumPy1.23.0.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a geometry reference.")
    result = {"schema": "python_golden_v1.geometry.v1", "units": "mm", "coordinate_system": "NIfTI RAS (explicit SimpleITK LPS conversion)",
              "source_versions": {"SimpleITK": "2.4.1", "NumPy": "1.23.0"},
              "grid_specification": {"name": "generic_outer_edge_preserved_v1", "shape": "ceil(n*old_spacing/new_spacing)",
                                     "origin": "origin + direction @ ((new_spacing-old_spacing)/2)", "direction": "unchanged",
                                     "distinction": "Not the PyRadiomics ROI resampling grid; extraction defaults to no resampling."},
              "cases": [source_case("flip_2d", [3, 2], [2., 3.], [11., -7.], [[-1., 0.], [0., 1.]], [1., 2.], [1, 0], [2, 2]),
                        source_case("rotate_3d", [3, 2, 2], [2., 1.5, 4.], [12., -9., 3.], [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], [1., 3., 2.], [1, 0, 1], [2, 2, 1])],
              "boundary_probe": {"source_shape_xyz": [3, 2], "source_values_x_fastest": [0, 10, 20, 30, 40, 50],
                                 "identity_ras_affine": True, "default_value": -999, "outputs": boundary_cases()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
