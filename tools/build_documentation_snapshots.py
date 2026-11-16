"""Build latest and preserve documentation snapshots from actual release tags.

Existing published version directories are restored from the Pages branch. A
release snapshot is never relabeled from the current working tree or rebuilt
after publication. The output directory must be fresh to exclude stale pages.
"""

import argparse
import html
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://hurry060215-tech.github.io/radiomics/"
VERSION_DIRECTORY = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?")


def git(root, *arguments, required=True):
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if required and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout if result.returncode == 0 else None


def revision(root, ref):
    raw = git(root, "rev-parse", "--verify", ref + "^{commit}", required=False)
    return raw.decode().strip() if raw else None


def extract_files(data, destination, selected=lambda name: True):
    destination = Path(destination).resolve()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            if not selected(member.name):
                continue
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("An archived publication path escapes its destination.")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
            else:
                raise ValueError("Publication archives may contain only directories and files.")


@contextmanager
def source_at(root, ref):
    with tempfile.TemporaryDirectory(prefix="radiomics-doc-source-") as directory:
        source = Path(directory)
        extract_files(git(root, "archive", ref), source)
        yield source


def build_site(source, destination, version, source_revision, base_url):
    environment = dict(os.environ)
    environment["RADIOMICS_BUILD_REVISION"] = source_revision
    environment["RADIOMICS_DOC_VERSION"] = version
    subprocess.run(
        [
            sys.executable,
            str(source / "tools/build_website_sources.py"),
            "--source-ref",
            source_revision,
        ],
        cwd=source,
        env=environment,
        check=True,
    )
    api_builder = source / "tools/build_api_sources.py"
    if api_builder.exists():
        subprocess.run(
            [sys.executable, str(api_builder)],
            cwd=source,
            env=environment,
            check=True,
        )
    with tempfile.TemporaryDirectory(prefix="radiomics-doctrees-") as doctrees:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "--keep-going",
                "-b",
                "html",
                "-d",
                doctrees,
                "-D",
                "html_baseurl=" + base_url.rstrip("/") + "/" + version + "/",
                "-A",
                "build_revision=" + source_revision,
                str(source / "website"),
                str(destination),
            ],
            cwd=source,
            env=environment,
            check=True,
        )


def restore_published(root, output, published_ref, restore_latest):
    if revision(root, published_ref) is None:
        return {}
    metadata_raw = git(root, "show", published_ref + ":publication.json", required=False)
    metadata = json.loads(metadata_raw) if metadata_raw else {}

    def selected(name):
        top = name.split("/", 1)[0]
        return bool(VERSION_DIRECTORY.fullmatch(top)) or (
            restore_latest and metadata.get("latest_revision") and top == "latest"
        )

    extract_files(git(root, "archive", published_ref), output, selected)
    return metadata


def fetch_published(root, remote):
    branch = "refs/heads/gh-pages"
    listed = git(root, "ls-remote", "--heads", remote, branch)
    if listed.strip():
        git(root, "fetch", "--no-tags", remote, branch + ":refs/remotes/" + remote + "/gh-pages")


def build(
    root=ROOT,
    *,
    output,
    event_ref="",
    release_tag="v0.1.0",
    published_ref="origin/gh-pages",
    latest_ref="origin/main",
    base_url=BASE_URL,
):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh documentation publication output directory.")
    if not VERSION_DIRECTORY.fullmatch(release_tag):
        raise ValueError("Release snapshots require an explicit version tag.")
    output.mkdir(parents=True)
    tag_event = event_ref.startswith("refs/tags/")
    previous = restore_published(root, output, published_ref, restore_latest=tag_event)
    build_revision = revision(root, "HEAD")
    if tag_event and (output / "latest").is_dir():
        latest_revision = previous["latest_revision"]
    elif tag_event:
        latest_revision = revision(root, latest_ref)
        if latest_revision is None:
            raise ValueError("First tag publication requires the maintained branch source.")
        with source_at(root, latest_ref) as source:
            build_site(source, output / "latest", "latest", latest_revision, base_url)
    else:
        latest_revision = build_revision
        build_site(root, output / "latest", "latest", latest_revision, base_url)
    release_revision = revision(root, "refs/tags/" + release_tag)
    releases = dict(previous.get("releases", {}))
    release_output = output / release_tag
    if release_output.exists():
        if release_revision and releases.get(release_tag) != release_revision:
            raise ValueError("A published release snapshot cannot be replaced by a moved tag.")
    elif release_revision:
        with source_at(root, "refs/tags/" + release_tag) as source:
            build_site(source, release_output, release_tag, release_revision, base_url)
        releases[release_tag] = release_revision
    status = (
        "release snapshot available" if release_output.exists() else "pending actual release tag"
    )
    metadata = dict(
        schema="radiomics.documentation.publication.v1",
        build_revision=build_revision,
        latest_revision=latest_revision,
        releases=releases,
        release_status=status,
    )
    (output / "publication.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    release_link = (
        '<li><a href="' + html.escape(release_tag) + '/">' + html.escape(release_tag) + "</a></li>"
        if release_output.exists()
        else "<li>" + html.escape(release_tag) + ": " + status + "</li>"
    )
    index = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Radiomics documentation</title>"
        '<h1>Radiomics documentation</h1><ul><li><a href="latest/">latest</a></li>'
        + release_link
        + '</ul><p data-build-revision="'
        + html.escape(build_revision)
        + '">Build <code>'
        + html.escape(build_revision[:7])
        + "</code></p></html>\n"
    )
    (output / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--release-tag", default="v0.1.0")
    parser.add_argument("--published-ref", default="origin/gh-pages")
    parser.add_argument("--latest-ref", default="origin/main")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--fetch-published", action="store_true")
    parser.add_argument("--remote", default="origin")
    arguments = parser.parse_args()
    if arguments.fetch_published:
        fetch_published(arguments.root, arguments.remote)
    build(
        arguments.root,
        output=arguments.output,
        event_ref=arguments.event_ref,
        release_tag=arguments.release_tag,
        published_ref=arguments.published_ref,
        latest_ref=arguments.latest_ref,
        base_url=arguments.base_url,
    )
