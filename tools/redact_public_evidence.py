"""Redact absolute filesystem paths in public receipts without rerecording evidence.

JSON string tokens (including object keys) are decoded independently. All bytes
outside changed strings, including numeric spellings and observation order, stay
unchanged. Compressed derivatives use reproducible gzip headers.
"""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
# URL paths are excluded: a filesystem path must start a value or follow a
# delimiter, never a URL scheme/host. Cover Windows drives, UNC, and Unix roots.
ABSOLUTE_PATH = re.compile(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|/)(?:[^\s\"'<>|;,]*)")


def redact_string(value):
    def replace(match):
        path = match.group(0)
        parts = re.split(r"[\\/]+", path.rstrip("\\/"))
        # Retain a relative suffix for provenance; erase the machine's root.
        skip = 3 if path.startswith("/home/") or path.startswith("/Users/") else 2
        suffix = "/".join(parts[skip:])
        return "$WORKDIR" + ("/" + suffix if suffix else "")

    return ABSOLUTE_PATH.sub(replace, value)


def redact_text(text):
    def replace(match):
        original = json.loads(match.group(0))
        redacted = redact_string(original)
        return match.group(0) if redacted == original else json.dumps(redacted, ensure_ascii=False)

    return JSON_STRING.sub(replace, text)


def observations(text):
    """Identity of all non-string JSON bytes, including exact number spellings."""
    return hashlib.sha256(JSON_STRING.sub('""', text).encode("utf-8")).hexdigest()


def process(root, check=False):
    root = Path(root).resolve()
    manifest_path = root / "docs/public-evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending = []
    failures = []
    for record in manifest["files"]:
        path = (root / record["public_path"]).resolve()
        if not path.is_relative_to(root) or path == manifest_path:
            raise ValueError("Evidence derivative escapes its declared root.")
        raw = path.read_bytes()
        text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")
        redacted = redact_text(text)
        # Parse all lines to detect invalid JSON and redacted-key collisions.
        after_lines = redacted.splitlines()

        def unique(pairs):
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("Redaction would collide with an existing JSON key.")
            return result

        if path.suffix == ".gz":
            for line in after_lines:
                if line.strip():
                    json.loads(line, object_pairs_hook=unique)
        else:
            json.loads(redacted, object_pairs_hook=unique)
        digest = observations(text)
        if digest != observations(redacted):
            raise AssertionError("Non-string evidence bytes changed.")
        if text != redacted:
            failures.append(record["public_path"] + ": absolute filesystem path")
        if (
            hashlib.sha256(raw).hexdigest() != record["public_sha256"]
            or len(raw) != record["public_bytes"]
        ):
            failures.append(record["public_path"] + ": manifest identity mismatch")
        if check:
            if not record.get("values_unchanged") or record.get("observation_sha256") != digest:
                failures.append(record["public_path"] + ": missing observation identity")
            continue
        data = redacted.encode("utf-8")
        if path.suffix == ".gz":
            data = gzip.compress(data, compresslevel=9, mtime=0) if text != redacted else raw
        record.update(
            public_sha256=hashlib.sha256(data).hexdigest(),
            public_bytes=len(data),
            values_unchanged=True,
            observation_sha256=digest,
            redaction=(
                "Absolute filesystem path prefixes replaced with $WORKDIR; non-string bytes and"
                " observation order unchanged."
            ),
        )
        pending.append((path, data))
    if check:
        for failure in failures:
            print(failure)
        return not failures
    # Validate the whole set before writing any derivative.
    for path, data in pending:
        if path.read_bytes() != data:
            path.write_bytes(data)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Validated {len(pending)} evidence derivatives; updated path placeholders and identities."
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return 0 if process(args.root, args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
