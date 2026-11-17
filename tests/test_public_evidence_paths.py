"""Public evidence checks decode JSON, including compressed historical receipts."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evidence_redactor", ROOT / "tools/redact_public_evidence.py"
)
redactor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(redactor)


@pytest.mark.parametrize(
    "path",
    [
        "Q:/build/env/python.exe",
        r"Q:\build\env\python.exe",
        "/home/researcher/build/log.json",
        "/Users/researcher/data.json",
        "/root/build/data.json",
        "/mnt/work/data.json",
        "/opt/build/data.json",
        r"\\server\share\build\data.json",
    ],
)
def test_absolute_paths_are_redacted_in_keys_and_values(path):
    text = json.dumps({path: [path, 1.23456789, 3, None, True]})
    output = redactor.redact_text(text)
    assert path not in output
    assert "$WORKDIR" in output
    assert redactor.redact_text(output) == output
    assert redactor.observations(text) == redactor.observations(output)
    assert list(json.loads(output).values())[0][1:] == [1.23456789, 3, None, True]


def test_urls_relative_paths_and_numeric_spellings_stay_identical():
    text = '{"url":"https://example.org/path/data", "relative":"docs/data.json", "x":1.2300e-04}'
    assert redactor.redact_text(text) == text


def test_check_reports_a_residual_without_writing(tmp_path):
    import hashlib

    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs/report-redacted.json"
    path.write_text('{"path":"Q:/private/data.json", "count":12}', encoding="utf-8")
    raw = path.read_bytes()
    record = dict(
        public_path="docs/report-redacted.json",
        public_sha256=hashlib.sha256(raw).hexdigest(),
        public_bytes=len(raw),
    )
    (tmp_path / "docs/public-evidence-manifest.json").write_text(json.dumps({"files": [record]}))
    assert not redactor.process(tmp_path, check=True)
    assert path.read_bytes() == raw
    assert redactor.process(tmp_path)
    assert redactor.process(tmp_path, check=True)


def test_public_evidence_derivatives_have_no_absolute_paths():
    assert redactor.process(ROOT, check=True)
