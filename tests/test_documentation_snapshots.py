"""Release documentation must preserve the actual tagged source and published bytes."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "documentation_snapshots", ROOT / "tools/build_documentation_snapshots.py"
)
snapshots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshots)


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Documentation test",
            "-c",
            "user.email=documentation-test@example.invalid",
            *args,
        ],
        cwd=root,
        text=True,
    ).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "--quiet")
    calls = []

    def build_site(source, destination, version, source_revision, base_url):
        destination.mkdir(parents=True)
        content = (source / "marker.txt").read_bytes()
        (destination / "index.html").write_bytes(content)
        calls.append((version, source_revision))

    monkeypatch.setattr(snapshots, "build_site", build_site)

    def commit(content):
        (root / "marker.txt").write_text(content)
        git(root, "add", ".")
        git(root, "commit", "--quiet", "-m", content)
        return git(root, "rev-parse", "HEAD")

    return root, commit, calls


def publish(root, source, destination):
    git(root, "worktree", "add", "--detach", str(destination), "HEAD")
    git(destination, "switch", "--orphan", "gh-pages")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    git(destination, "add", ".")
    git(destination, "commit", "--quiet", "-m", "Publish snapshots")


def test_release_snapshot_uses_actual_tag_and_is_not_rebuilt(repository, tmp_path):
    root, commit, calls = repository
    release = commit("release source")
    git(root, "tag", "v0.1.0")
    current = commit("maintained source")
    first = tmp_path / "first"
    metadata = snapshots.build(root, output=first, published_ref="gh-pages")
    assert (first / "latest/index.html").read_text() == "maintained source"
    assert (first / "v0.1.0/index.html").read_text() == "release source"
    assert metadata["releases"] == {"v0.1.0": release}
    assert metadata["latest_revision"] == current
    publish(root, first, tmp_path / "published")
    commit("new maintained source")
    calls.clear()
    second = tmp_path / "second"
    snapshots.build(root, output=second, published_ref="gh-pages")
    assert (second / "v0.1.0/index.html").read_bytes() == (first / "v0.1.0/index.html").read_bytes()
    assert [version for version, _ in calls] == ["latest"]


def test_tag_event_preserves_the_newer_published_latest(repository, tmp_path):
    root, commit, calls = repository
    release = commit("release source")
    git(root, "tag", "v0.1.0")
    latest = commit("newer maintained source")
    first = tmp_path / "first"
    snapshots.build(root, output=first, published_ref="gh-pages")
    publish(root, first, tmp_path / "published")
    git(root, "checkout", "--detach", "--quiet", "v0.1.0")
    calls.clear()
    output = tmp_path / "tag-event"
    metadata = snapshots.build(
        root, output=output, event_ref="refs/tags/v0.1.0", published_ref="gh-pages"
    )
    assert metadata["build_revision"] == release
    assert metadata["latest_revision"] == latest
    assert (output / "latest/index.html").read_text() == "newer maintained source"
    assert calls == []


def test_a_missing_tag_remains_pending(repository, tmp_path):
    root, commit, calls = repository
    commit("candidate source")
    output = tmp_path / "candidate"
    metadata = snapshots.build(root, output=output, published_ref="gh-pages")
    assert not (output / "v0.1.0").exists()
    assert metadata["release_status"] == "pending actual release tag"
    assert "pending actual release tag" in (output / "index.html").read_text()


def test_a_moved_tag_cannot_replace_a_published_snapshot(repository, tmp_path):
    root, commit, calls = repository
    commit("release source")
    git(root, "tag", "v0.1.0")
    first = tmp_path / "first"
    snapshots.build(root, output=first, published_ref="gh-pages")
    publish(root, first, tmp_path / "published")
    commit("different source")
    git(root, "tag", "--force", "v0.1.0")
    with pytest.raises(ValueError, match="moved tag"):
        snapshots.build(root, output=tmp_path / "replacement", published_ref="gh-pages")


def test_archives_preserve_lf_blob_bytes_when_checkout_autocrlf_is_enabled(repository, tmp_path):
    root, commit, calls = repository
    commit("initial source")
    git(root, "config", "core.autocrlf", "true")
    original = {
        "v0.1.0/index.html": b"<html>\nrelease\n</html>\n",
        "v0.1.0/_static/site.css": b"body {\n  color: black;\n}\n",
        "v0.1.0/_sources/guide.md.txt": b"# Guide\n\nOriginal LF bytes.\n",
    }
    for relative, data in original.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "publication.json").write_bytes(b'{"releases": {}}\n')
    git(root, "-c", "core.autocrlf=false", "add", ".")
    git(root, "commit", "--quiet", "-m", "Publish exact LF artifacts")
    for relative, data in original.items():
        stored = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=root)
        assert stored == data
    with snapshots.source_at(root, "HEAD") as archived_source:
        for relative, data in original.items():
            assert (archived_source / relative).read_bytes() == data
    restored = tmp_path / "restored"
    snapshots.restore_published(root, restored, "HEAD", restore_latest=False)
    for relative, data in original.items():
        assert (restored / relative).read_bytes() == data


def test_flat_compatibility_copies_active_latest_files_without_restoring_old_root(
    repository, tmp_path, monkeypatch
):
    root, commit, calls = repository
    commit("maintained source")
    files = {
        "index.html": b"latest landing page",
        "installation.html": b"installation with latest canonical",
        "reference.html": b"public reference",
        "quickstart.html": b"quick start",
        "_static/site.css": b"body { color: black; }",
        "_sources/installation.md.txt": b"# Installation",
        "publication.json": b"not the root publication metadata",
        ".nojekyll": b"not the root marker",
    }

    def build_site(source, destination, version, source_revision, base_url):
        for relative, data in files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    monkeypatch.setattr(snapshots, "build_site", build_site)
    first = tmp_path / "first-flat"
    snapshots.build(root, output=first, published_ref="gh-pages")
    (first / "retired.html").write_bytes(b"obsolete published root")
    (first / "_sources/retired.md.txt").write_bytes(b"obsolete source")
    publish(root, first, tmp_path / "published-flat")
    output = tmp_path / "current-flat"
    metadata = snapshots.build(root, output=output, published_ref="gh-pages")
    assert b'<a href="latest/">latest</a>' in (output / "index.html").read_bytes()
    assert (output / "index.html").read_bytes() != files["index.html"]
    assert json.loads((output / "publication.json").read_text()) == metadata
    assert (output / ".nojekyll").read_bytes() == b""
    copied = set(files) - {"index.html", "publication.json", ".nojekyll"}
    assert set(metadata["flat_compatibility"]["files"]) == copied
    for relative in copied:
        data = (output / "latest" / relative).read_bytes()
        assert (output / relative).read_bytes() == data == files[relative]
        assert metadata["flat_compatibility"]["files"][relative] == {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    assert not (output / "retired.html").exists()
    assert not (output / "_sources/retired.md.txt").exists()
