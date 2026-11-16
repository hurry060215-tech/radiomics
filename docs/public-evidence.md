# Public historical evidence and redaction

Historical measurement evidence is immutable. Some original run receipts contain
machine-local working directories and interpreter locations. The public tree
provides separately named `*-redacted.json` and `*-redacted.jsonl.gz` derivatives
of those receipts, while the original files are preserved byte for byte in an
external evidence archive outside every Git repository.

The [public evidence manifest](public-evidence-manifest.json) records each
original repository-relative filename, original SHA256 and byte count, and its
public derivative's filename, SHA256 and byte count. The external archive's
`MANIFEST.json` also records the source paths and archival timestamps.
Archived originals preserve the checked-out working-tree bytes without newline
rewriting. A separate `source_git_blob_sha256` identifies Git-normalized bytes
when Windows checkout line endings differ from the stored Git blob.

## What changed in public derivatives

Only machine-local path prefixes and environment directory labels were replaced.
`$WORKDIR` denotes the original working directory; `reference-env` denotes an
external reference environment. These are provenance placeholders, not commands
that were executed during the historical experiment.

Feature values, timings, memory values, numerical types, observation counts and
observation order were retained. Object keys that were themselves absolute file
paths were redacted with the same rule. The derivative JSON serialization and
the compressed derivative's bytes therefore have their own SHA256 identities.
No experiment, Golden reference or benchmark observation was rerun or rerecorded
to create these public views.

## How to interpret historical artifact indexes

The historical `benchmarks/results/*/artifact-index.json` files remain unchanged.
Their original filenames, hashes and compressed/uncompressed byte identities
refer to the original archived artifacts. They do not claim that a public
derivative has the original artifact's hash.

Use the public evidence manifest to map an original hash to the separately
identified public derivative. In particular, the original compressed raw
observation SHA remains an identity of the archived original, while the public
compressed path-redacted view has a different identity. Numeric observations
retain their historical measurement scope and hardware limitations.

Public derivative receipts document past execution; they are not replacement
runtime configurations. Reproduction uses the maintained tools with explicit
input, interpreter, work-directory and output arguments.
