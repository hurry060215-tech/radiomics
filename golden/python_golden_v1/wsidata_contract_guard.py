"""Required supplemental guard for the frozen WSI v1 canonical digest protocol.

The original specification and its expected snapshots remain immutable. A float
tag can also be supplied as an opaque extra JSON object, so digest equality alone
cannot establish equality of the original metadata types.
"""
import importlib.util
from pathlib import Path


_path = Path(__file__).with_name("wsidata_contract_spec.py")
_loader = importlib.util.spec_from_file_location("frozen_wsidata_contract", _path)
_base = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(_base)


def typed_equal(left, right):
    """Compare normalized JSON recursively without bool/int or int/float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_embedding_link(current, saved_at_generation, table_region, tile_key, shape_ids, obs_tile_ids, obs_row_ids):
    ordering = _base.validate_embedding_link(current, saved_at_generation, table_region, tile_key, shape_ids, obs_tile_ids, obs_row_ids)
    if not typed_equal(current["snapshot"], saved_at_generation["snapshot"]):
        raise ValueError("embedding snapshot metadata structure, type or value differs")
    return ordering
