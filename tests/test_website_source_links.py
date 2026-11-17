"""Published PR documentation must link to files in its own source revision."""

import hashlib
import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "tools/build_website_sources.py"
spec = importlib.util.spec_from_file_location("website_source_builder", PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
files_spec = importlib.util.spec_from_file_location(
    "public_source_files", PATH.parent / "public_source_files.py"
)
files_module = importlib.util.module_from_spec(files_spec)
files_spec.loader.exec_module(files_module)
public_files = files_module.public_files


def test_unmerged_documentation_source_links_are_pinned(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "website/tutorials").mkdir(parents=True)
    (tmp_path / "website/tutorials/07_public_ct.ipynb").write_text("{}")
    (tmp_path / "docs/guide.md").write_text("# Guide\n\n[Evidence](new-evidence.json)\n")
    (tmp_path / "docs/new-evidence.json").write_text("{}")
    monkeypatch.setattr(builder, "PAGES", ("guide.md",))
    revision = "a" * 40
    builder.build(tmp_path, source_ref=revision)
    rendered = (tmp_path / "website/reference/guide.md").read_text()
    assert "/blob/" + revision + "/docs/new-evidence.json" in rendered
    assert "/blob/main/" not in rendered
    manifest = json.loads((tmp_path / "website/reference/_sources.json").read_text())
    assert manifest["source_ref"] == revision


def test_unpacked_source_does_not_borrow_parent_git_identity(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    archive = tmp_path / "unpacked"
    archive.mkdir()

    def unexpected(*args, **kwargs):
        raise AssertionError("Do not consult a parent repository for archive identity.")

    monkeypatch.setattr(builder.subprocess, "run", unexpected)
    assert builder.source_revision(archive) == "main"


FORBIDDEN = (
    "Onekey",
    "onekey",
    "OnekeyPlatfrom50",
    "D:/Onekey",
    "D:\\Onekey",
    "audit_onekey",
)


def test_public_tree_has_no_alignment_material():
    root = Path(__file__).resolve().parents[1]
    test_file = Path(__file__).resolve()
    offenders = []
    for path in public_files(root):
        # This test necessarily names its forbidden tokens.
        if path.resolve() == test_file:
            continue
        if path.suffix not in {".md", ".py", ".toml", ".json", ".txt", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(root)}: {token}")
    assert not offenders, offenders


def test_public_evidence_manifest_matches_derived_bytes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/public-evidence-manifest.json").read_text())
    for record in manifest["files"]:
        data = (root / record["public_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["public_sha256"]
        assert len(data) == record["public_bytes"]


def test_archive_scan_ignores_build_products_and_parent_git(tmp_path):
    (tmp_path / ".git").mkdir()
    archive = tmp_path / "archive"
    (archive / "docs").mkdir(parents=True)
    source = archive / "docs" / "guide.md"
    source.write_text("Public source")
    (archive / ".build-tools").mkdir()
    (archive / ".build-tools" / "private.txt").write_text("Private artifact")
    assert public_files(archive) == [source]


def test_git_scan_only_reads_indexed_files(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "public.md"
    tracked.write_text("Public source")
    (tmp_path / "untracked.md").write_text("Private artifact")
    subprocess.run(["git", "add", "public.md"], cwd=tmp_path, check=True)
    assert public_files(tmp_path) == [tracked]
