"""API discovery must remain bounded by explicit public exports."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("api_sources", ROOT / "tools/build_api_sources.py")
api_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api_sources)


def test_api_discovery_matches_public_exports_without_reference_imports():
    before = set(sys.modules)
    import ov_radiomics

    normal, optional = api_sources.public_exports(ROOT)
    assert set(normal).isdisjoint(optional)
    assert set(normal) | set(optional) == set(ov_radiomics.__all__)
    assert set(optional) == {"TrainingResult", "FusionResult"}
    assert not any(name.startswith("ov_radiomics._reference") for name in set(sys.modules) - before)
