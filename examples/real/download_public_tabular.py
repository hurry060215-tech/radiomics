"""Download/reuse hash-pinned UCI WDBC files in an external cache; standard library only."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import uuid
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external_path(path):
    path = Path(path).resolve()
    if any((parent / ".git").exists() for parent in [path, *path.parents]):
        raise ValueError("Public raw data belong in an external cache outside every Git worktree.")
    return path


def verify(path, spec):
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError("Fixed public source size/SHA256 differs: " + str(path))


def exclusive_text(path, text):
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing generated metadata differs: " + str(path))
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)


def fetch(output, *, from_cache=None):
    """Verify every existing file; publish verified downloads without overwrite."""
    output = external_path(output)
    facts_path = Path(__file__).with_name("public_tabular_source.json")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    print("Source:", facts["official_page"], "DOI:", facts["doi"])
    print("Dataset license:", facts["license"]["identifier"], facts["license"]["url"])
    print("Attribution:", facts["attribution"])
    output.mkdir(parents=True, exist_ok=True)
    for name, spec in facts["files"].items():
        target = output / name
        if target.exists():
            verify(target, spec)
            continue
        temporary = output / (name + "." + uuid.uuid4().hex + ".part")
        try:
            cached = None if from_cache is None else Path(from_cache) / name
            if cached is not None and cached.is_file():
                verify(cached, spec)
                with cached.open("rb") as source, temporary.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
            else:
                with (
                    urllib.request.urlopen(spec["url"], timeout=60) as source,
                    temporary.open("xb") as destination,
                ):
                    total = 0
                    while True:
                        block = source.read(65536)
                        if not block:
                            break
                        total += len(block)
                        if total > spec["bytes"]:
                            raise ValueError("Download exceeds frozen source byte count: " + name)
                        destination.write(block)
            verify(temporary, spec)
            # Same-directory hard link publishes atomically and refuses to
            # replace a target created by another process during the download.
            os.link(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    notice = (
        "# Public WDBC input\n\n"
        + facts["dataset"]
        + "\n\n"
        + facts["attribution"]
        + "\n\nLicense: "
        + facts["license"]["identifier"]
        + " "
        + facts["license"]["url"]
        + "\n\nData retained as downloaded, not covered by the software MIT license.\n"
        + "The tutorial renames columns and assigns a declared hash-based training/test split; it"
        " never invents diagnosis labels.\n"
    )
    exclusive_text(output / "NOTICE.md", notice)
    canonical = json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest = dict(
        schema="radiomics.real.public_tabular_cache.v1",
        source_facts_canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        dataset=facts["dataset"],
        doi=facts["doi"],
        license=facts["license"],
        files={
            name: {"sha256": spec["sha256"], "bytes": spec["bytes"], "url": spec["url"]}
            for name, spec in facts["files"].items()
        },
    )
    exclusive_text(
        output / "source-manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-cache", type=Path, help="Reuse matching original source files without network"
    )
    args = parser.parse_args()
    print("Verified cache:", fetch(args.output, from_cache=args.from_cache))


if __name__ == "__main__":
    main()
