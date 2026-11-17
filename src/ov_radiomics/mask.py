"""Native ROI operations with explicit connectivity and distance units."""

from pathlib import Path

import numpy as np

from ._native_common import request
from ._types import MaskVolume
from .image import _volume_payload


def from_array(labels, *, geometry):
    array = np.asarray(labels)
    if np.iscomplexobj(array) or array.ndim not in (2, 3) or not np.isfinite(array).all():
        raise ValueError("Masks require finite scalar 2D/3D nonnegative integer labels.")
    if (
        np.any(array < 0)
        or np.any(array > np.iinfo(np.uint32).max)
        or np.any(array != np.floor(array))
    ):
        raise ValueError("Mask labels must be exact nonnegative uint32 values, without truncation.")
    expected = tuple(reversed(geometry["shape"][: geometry["ndim"]]))
    if tuple(array.shape) != expected:
        raise ValueError("Mask NumPy shape/axes differ from the declared native image geometry.")
    return MaskVolume(
        dict(geometry=geometry, labels=array.astype(np.uint32).ravel(order="C").tolist())
    )


def read_nifti(path):
    return MaskVolume._from_native(request("read_mask", path=str(Path(path).resolve())))


def select_labels(mask, labels):
    return MaskVolume._from_native(
        request("mask_select_labels", mask=_volume_payload(mask), labels=list(labels))
    )


def resample_like(mask, reference, *, outside=0):
    geometry = (
        reference.geometry
        if hasattr(reference, "geometry")
        else reference.get("geometry", reference)
    )
    return MaskVolume._from_native(
        request(
            "mask_resample_like", mask=_volume_payload(mask), reference=geometry, outside=outside
        )
    )


def peritumoral_ring(mask, *, inner_mm=0.0, outer_mm=3.0, organ_mask=None):
    return MaskVolume._from_native(
        request(
            "mask_peritumoral_ring",
            mask=_volume_payload(mask),
            inner_mm=inner_mm,
            outer_mm=outer_mm,
            organ_mask=None if organ_mask is None else _volume_payload(organ_mask),
        )
    )


def _connectivity(value):
    return {4: "four", 8: "eight", 6: "six", 18: "eighteen", 26: "twenty_six"}.get(value, value)


def largest_component(mask, *, connectivity):
    return MaskVolume._from_native(
        request(
            "mask_largest_component",
            mask=_volume_payload(mask),
            connectivity=_connectivity(connectivity),
        )
    )


def dilate_voxels(mask, *, iterations, connectivity):
    return MaskVolume._from_native(
        request(
            "mask_dilate_voxels",
            mask=_volume_payload(mask),
            iterations=iterations,
            connectivity=_connectivity(connectivity),
        )
    )


def resegment(image, mask, *, lower, upper):
    """Return a distinct intensity mask; the morphological mask is preserved."""
    return MaskVolume._from_native(
        request(
            "mask_resegment",
            image=_volume_payload(image),
            mask=_volume_payload(mask),
            lower=lower,
            upper=upper,
        )
    )
