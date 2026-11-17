"""Fetch/reuse fixed public PAT1 files in an external cache, preserving license and hashes."""

import argparse
import csv
import json
import shutil
import urllib.request
import uuid
from pathlib import Path

from _common import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def reject_git_output(output):
    output = Path(output).resolve()
    if any((parent / ".git").exists() for parent in [output, *output.parents]):
        raise ValueError(
            "CC-BY-NC raw/prepared data must use an external cache outside every Git worktree."
        )
    return output


def exclusive_text(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing generated metadata differs: " + str(path))
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def verify_source_file(path, spec):
    path = Path(path)
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("Fixed source byte size differs: " + path.name)
    if sha256(path) != spec["sha256"]:
        raise ValueError("Fixed source SHA256 differs: " + path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-cache",
        type=Path,
        help="Reuse already downloaded matching source files before network access.",
    )
    parser.add_argument(
        "--acknowledge-license",
        action="store_true",
        help=(
            "Required: acknowledge the bundled CC-BY-NC-3.0 notice before download/cache writes; no"
            " interactive prompt."
        ),
    )
    args = parser.parse_args()
    if not args.acknowledge_license:
        parser.error(
            "Read public_ct_source.json / CC-BY-NC-3.0 terms, then pass --acknowledge-license"
            " before downloading or writing the cache."
        )
    output = reject_git_output(args.output)
    protocol_path = Path(__file__).with_name("public_ct_source.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    print("Data license:", protocol["license"]["identifier"], protocol["license"]["url"])
    print("Noncommercial source data are separate from the software MIT license.")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, spec in protocol["source"]["files"].items():
        target = output / name
        method = "existing_verified"
        if not target.exists():
            temporary = output / (name + "." + uuid.uuid4().hex + ".part")
            try:
                cached = None if args.from_cache is None else args.from_cache / name
                if (
                    cached is not None
                    and cached.is_file()
                    and cached.stat().st_size == spec["bytes"]
                    and sha256(cached) == spec["sha256"]
                ):
                    shutil.copyfile(cached, temporary)
                    method = "copied_verified_external_cache"
                else:
                    with (
                        urllib.request.urlopen(spec["url"], timeout=60) as source,
                        temporary.open("xb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
                    method = "downloaded_fixed_revision"
                verify_source_file(temporary, spec)
                temporary.rename(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        verify_source_file(target, spec)
        records[name] = dict(
            path=str(target), sha256=spec["sha256"], source_url=spec["url"], method=method
        )
    notice = (
        "# Public PAT1 CT/ROI data\n\n"
        + protocol["dataset"]
        + "\n\nLicense: "
        + protocol["license"]["identifier"]
        + " "
        + protocol["license"]["url"]
        + "\n\nAttribution: Philippe Lambin / CancerData; NIfTI conversion Alex Zwanenburg. DOI "
        + protocol["source"]["original_doi"]
        + ".\n\nRaw/prepared images stay outside Git and installation packages. No clinical outcome"
        " or cohort label is provided or generated. Read SOURCE_LICENSE.md and SOURCE_README.md"
        " before use.\n"
    )
    exclusive_text(output / "NOTICE.md", notice)
    import io

    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["sample_id", "patient_id", "image", "mask", "roi_label", "modality", "region"])
    writer.writerow(["PAT1-GTV1", "PAT1", "image.nii.gz", "mask.nii.gz", "1", "CT", "GTV-1"])
    exclusive_text(output / "cohort.csv", handle.getvalue())
    manifest = dict(
        schema="radiomics.real.public_ct_cache.v1",
        source_facts_sha256=sha256(protocol_path),
        source=protocol["source"],
        license=protocol["license"],
        files=records,
        license_review_recorded=args.acknowledge_license,
        patient_id="Source dataset PAT1 identifier, not an invented outcome or clinical split.",
        cohort=str(output / "cohort.csv"),
    )
    manifest_path = output / "source-manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    else:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source") != manifest["source"] or any(
            existing.get("files", {}).get(name, {}).get("sha256") != record["sha256"]
            for name, record in records.items()
        ):
            raise FileExistsError(
                "Existing source-manifest differs; choose a fresh external cache."
            )
    print("cohort:", output / "cohort.csv")
    print("notice:", output / "NOTICE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
