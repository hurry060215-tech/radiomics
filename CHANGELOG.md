# Changelog

## [Unreleased]

- Remove withdrawn audit material from public sources and publish separately
  identified path-redacted historical receipts; preserve original evidence in an
  external archive with original-to-public SHA256 mappings.
- Replace machine-specific paths in maintained documentation with workspace
  placeholders, add citation and package metadata, and explain license ownership.

## [0.1.0] - 2026-11-16

This section describes the candidate for the `v0.1.0` source tag. Registry
publication is separate; a source tag and CI artifacts do not claim a PyPI or
crates.io release.

- Share native computation between Rust core, Python bindings, and an independent
  CLI, with explicit optional Python reference backends.
- Read scalar NIfTI images and masks with full affine geometry; provide declared
  preparation operations and selected Original first-order, texture, and basic
  shape features.
- Keep patient-level SplitPlan provenance through native preprocessing,
  selection, supported classification/regression, evaluation, and probability
  fusion, with typed `.rad` persistence.
- Provide patch quality control, tile/slide/patient aggregation, explicit-ID
  external embedding adapters, and lightweight pathology overlays.
- Include frozen Python-compatibility and IBSI digital-reference validation,
  six native tutorials, and separate public CT and tabular examples with stated
  source and license boundaries.
- Record the independent Radiomics, learning/pathology, and deployment benchmark
  v2 results, including missed performance targets and the high-dynamic-range
  scalar compatibility limitation at their historical source revisions.

These entries are grounded in the existing completion audit and repository
history. Later numerical or performance changes must identify their own source
revision and acceptance evidence; historical results are not new measurements.

### Distribution acceptance and candidate artifacts

The first acceptance matrix is x86_64 Ubuntu (manylinux2014 wheels) and Windows,
each with Python 3.9, 3.10, 3.11, and 3.12. macOS and ARM are not accepted targets.
Each CI wheel artifact and the source artifact carry their own `SHA256SUMS`.

The following final stable-target candidates were built from clean source
`75c6738f82e550f397c90c278fb032ca0c28b719`, before this checksum-summary update:

| Candidate | Bytes | SHA256 |
|---|---:|---|
| `ov_radiomics-0.1.0-cp39-abi3-win_amd64.whl` | 3725488 | `0ff8767ed854447cfcd0e300ef70b7e9bdacec827c604e4a65f073d0a6086cb3` |
| `ov_radiomics-0.1.0.tar.gz` | 1899118 | `7a591131f8c95fbe765fe3647b7bdd0792718190945b562fd842f382086ef613` |

The release wheel was compiled with `--release --locked` in a previously absent
target directory and installed into a new virtual environment without system
site packages. The backend reported `metrics_backend: rust`; the version,
cancellation, extraction, and split-plan checks passed (13 tests). The complete
installed-wheel suite passed 257 tests with 5 optional-dependency skips and no
xfail. A final clean-source wheel rebuild passed the same 13 focused checks.

The original source archive was unpacked outside Git and installed with
`pip install .` in another new environment. The two acceptance modules were
copied to a separate directory containing no package source and passed all
10 tests against the installed native backend. Cargo naturally removed 16
unused CLI-only lock nodes in that unpacked artifact; every retained dependency
identity and checksum was verified unchanged, as was the repository lockfile.

These hashes identify the exact accepted candidates above. A source archive
containing this changelog has a different digest; its generated CI checksum
manifest is the identity of that later artifact. No archive hash is represented
as a checksum of an archive containing its own checksum.

The user-approved compensated Mean returns 0.5 for the cancellation regression,
matching the independent mathematical mean 0.5. The frozen Python observation
remains 0.0: this deliberately changes that high-dynamic-range compatibility
target without rewriting any frozen observation. Historical benchmark v2 data
remain unchanged and predate that repair. Buffer and parallel optimizations are
separate PRs and are not included in this release-candidate acceptance.
