"""Real-file tutorial contracts; public licensed voxel data stay outside test assets."""

import importlib.util
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "examples/real"
sys.path.insert(0, str(DIRECTORY))


def load(name):
    spec = importlib.util.spec_from_file_location(
        "real_example_" + name, DIRECTORY / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_entry_never_guesses_ids_layout_or_physical_scale(tmp_path):
    example = load("patch_features")
    manifest = tmp_path / "patches.csv"
    manifest.write_text(
        "path,patient_id,slide_id,tile_id\nimage.png,001,slide,0001\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="supplied x0"):
        example.load_manifest(manifest, "level0")
    rows, parents = example.load_manifest(manifest, "image-local")
    assert rows[0]["patient_id"] == "001" and rows[0]["tile_id"] == "0001"
    assert parents == {"slide": "001"}
    assert example.optional_number(rows[0], "mpp_x_um") is None
    manifest.write_text(
        "path,patient_id,slide_id,tile_id\na.png,P,S,T1\nb.png,P,S,T2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="one complete source-image"):
        example.load_manifest(manifest, "image-local")


def test_download_cache_refuses_git_worktree_and_unapproved_overwrite(tmp_path):
    example = load("download_public_ct")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").write_text("worktree marker", encoding="utf-8")
    with pytest.raises(ValueError, match="outside every Git"):
        example.reject_git_output(repository / "cache")
    cache = tmp_path / "external"
    cache.mkdir()
    target = cache / "NOTICE.md"
    example.exclusive_text(target, "fixed notice\n")
    example.exclusive_text(target, "fixed notice\n")
    with pytest.raises(FileExistsError):
        example.exclusive_text(target, "different\n")


def test_license_ack_precedes_directory_creation_network_and_fixed_size_and_sha_are_checked(
    tmp_path, monkeypatch
):
    example = load("download_public_ct")
    destination = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["download_public_ct.py", "--output", str(destination)])

    def network_must_not_run(*args, **kwargs):
        raise AssertionError("Network attempted before license ACK.")

    monkeypatch.setattr(example.urllib.request, "urlopen", network_must_not_run)
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2 and not destination.exists()
    path = tmp_path / "file"
    path.write_bytes(b"abc")
    digest = example.sha256(path)
    example.verify_source_file(path, dict(bytes=3, sha256=digest))
    with pytest.raises(ValueError, match="byte size"):
        example.verify_source_file(path, dict(bytes=4, sha256=digest))
    with pytest.raises(ValueError, match="SHA256"):
        example.verify_source_file(path, dict(bytes=3, sha256="0" * 64))
