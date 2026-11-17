"""Drift checks compare rendered bytes as well as source hashes without writes."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "website_builder", Path(__file__).resolve().parents[1] / "tools/build_website_sources.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_check_rejects_source_and_output_drift_without_writing(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "website/tutorials").mkdir(parents=True)
    (tmp_path / "website/tutorials/07_public_ct.ipynb").write_text("{}")
    source = tmp_path / "docs/guide.md"
    source.write_text("# Original\n")
    monkeypatch.setattr(builder, "PAGES", ("guide.md",))
    builder.build(tmp_path, source_ref="revision")
    builder.build(tmp_path, source_ref="revision", check=True)
    output = tmp_path / "website/reference/guide.md"
    original = output.read_bytes()
    source.write_text("# Edited\n")
    with pytest.raises(ValueError, match="reference out of sync"):
        builder.build(tmp_path, source_ref="revision", check=True)
    assert output.read_bytes() == original
    source.write_text("# Original\n")
    output.write_text("Tampered output")
    with pytest.raises(ValueError, match="reference out of sync"):
        builder.build(tmp_path, source_ref="revision", check=True)
    assert output.read_text() == "Tampered output"
