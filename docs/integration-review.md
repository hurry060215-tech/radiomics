# Review-only integration of the radiomics PRs

All eight planned PRs in the independent MIT `radiomics` repository have now
merged into `main`, verified at
[`f3ec58cac4ed883331f56b30155ac80998056b80`](https://github.com/hurry060215-tech/radiomics/commit/f3ec58cac4ed883331f56b30155ac80998056b80).
This branch retains review and evidence documentation around that actual main
source. It is not a release branch, a deployment request, a PyPI publication, or
an additional feature PR. No algorithm or benchmark was rerun for this update.

The separate OmicVerse adapter is maintained in its GPL-3.0 repository:
[OmicVerse PR #1096](https://github.com/omicverse/omicverse/pull/1096). Its code
has not been copied into this MIT repository, and this package does not depend
on OmicVerse. The adapter at `c9f622b72928db0fe8a6e42d334d7dc34ca2d48d` is
**Ready for upstream review, not merged**. The contributor account lacks upstream
push/merge permission, so an upstream maintainer must perform that merge.

The adapter has its own independent **19-test** local Python 3.9 acceptance
(16 original cases plus three installation-gate cases), including a main-plus-PR9
candidate wheel. This is separate from the main-package test count below.
Its public-registry extra remains a commented placeholder until registry
publication and explicit activation. This branch does not promise a currently
installable registry extra or publish either package.

## Final merged component heads

These are the final component heads, each verified as an ancestor of main
`f3ec58c`. This review branch follows main through ordinary merges and adds only
review documentation, portable evidence and separately labeled snapshots.
It does not rewrite any component PR or modify runtime source, tests or tools.

| Planned PR | Scope | Merged source head |
|---|---|---|
| PR-1 | Public-content cleanup | `626baf204561abcf27b2c8e035af03e5b44796da` |
| PR-2 | Reproducibility and metadata | `ba873e6c87b22907dfafd9450c1d78286dcdea2e` |
| PR-3 | Consistent documentation publication | `ee7e80b968923b4795a3adaccac5a3390e5be2c8` |
| PR-4 | Stable Mean and numerical-boundary documentation | `95e3f3e4b7353b50fc34af31d323ab44cca87020` |
| PR-5 | Distribution readiness and release-scope notes | `df6cbbd7e2e57244b36bbc3d6f8bd6e2c08843cb` |
| PR-7 | Owned array buffers and bounded fallback | `4fa695e99ee8aeb69dffa2d45124dfd393b4ecfd` |
| PR-8 | Deterministic local feature threads | `154cf8bd558e486bc23682dc5f72cfbf79eafea8` |
| PR-9 | API and versioned documentation engineering | `0c57a1c7f3b77b23f7e14ac4c611ae5912e5bfed` |

The auxiliary language-statistics change
`26d87394ce04b55a89c78e077366c75247d6ec5b` is also included. It does not replace
the numerical or performance acceptance of any component.

## Source tag and numerical evidence

The `v0.1.0` source tag still resolves to
`bb2d3fe4f31e938c8aef8eb64c44987f3ed24c6e`. Its release scope is PR-1 through
PR-5; it does **not** include PR-7, PR-8 or PR-9. Current main and review builds
must not be substituted for that historical tagged candidate. The version
string alone does not identify source: record the commit, wheel SHA256 and
native implementation identity. This update does not retag or publish a package.

Main retains Neumaier compensation for Mean. For the documented cancellation
input, native Mean and independent `math.fsum` are both **0.5**. The frozen public
reference observation of **0.0** remains historical evidence, with the deliberate
compatibility boundary explained in the [scalar audit](scalar-accuracy-audit.md).
No broader claim of perfect mathematical summation is made.

Main also retains PR-7's immutable owned sample buffers, standalone typed input
snapshots and validation-preserving fallback, and PR-8's local thread pools,
explicit budgets and fixed-order directional reduction. Their earlier diagnostic
identities and observations remain those in their own reports; merging code is
not a new performance measurement or a change to a previously missed target.

Golden fixtures and original benchmark observations are immutable.
`golden/**` and `benchmarks/v2/**` remain the same 70 Git blobs as historical
source `20b7fc044b609822eed4c4c177aaae1b8803d9bd`. The 16 original/public
evidence relationships remain in the [redaction manifest](public-evidence-manifest.json).
Separately named public derivatives do not rewrite or rerecord archived originals.

## Historical checkpoint and current main verification

The [portable evidence](integration-review-evidence.json) has separate nodes:

- `historical_checkpoint` preserves the original source `77d5303`, old component
  heads, wheel/native identities and **280 passed / 5 skipped / zero xfail**
  local result. These numbers have not been reassigned to newer main source.
- `current_main_verified` records main `f3ec58c`, its actual
  [16 successful CI jobs](https://github.com/hurry060215-tech/radiomics/actions/runs/43884787523),
  and **290 passed / 5 skipped / zero xfail** after installing that CI wheel into
  a newly created environment without system site packages. The CI wheel SHA256
  is `624f5c0394388c2490f71e9ccba61c712c20fea4bbfdc631493587e36991cc1e`.

The two official documentation runs
[43884787496](https://github.com/hurry060215-tech/radiomics/actions/runs/43884787496)
and [43885696683](https://github.com/hurry060215-tech/radiomics/actions/runs/43885696683)
passed build, deployment and smoke checks. The separately recorded live HTTP
checks identify latest source `f3ec58c` and release source `bb2d3fe` on the
[production site](https://hurry060215-tech.github.io/radiomics/).
The **181 actual production release files** remained byte-identical across the
successful rebuild. Their production inventory SHA256 is
`a3695b76f2103d33745755bdde72ebdff8723bb8a59827662ec9b4ef292d4dbd`;
this is an actual GitHub Pages/Linux artifact comparison, not a local Windows
preview hash. These records describe completed main verification; this review
documentation update does not deploy anything or retag the source.

## Reproduce a local review build

Use a clean checkout and dedicated Python environment. Two distinct recorded
Windows x86_64 / Python 3.9 dependency snapshots are available:

- [Historical local-checkpoint snapshot](integration-review-requirements-win-py39.txt),
  preserved unchanged for source `77d5303` and its 280-case result.
- [Current-main fresh-environment snapshot](integration-review-requirements-main-win-py39.txt),
  for the actual main CI wheel and its 290-case result.

Neither is a cross-platform lock. Both exclude the package itself and local
wheel/direct references from dependency resolution. For a recorded environment,
choose the matching snapshot before building and install the local wheel with
`--no-deps`. Other platforms can use declared test/reference extras below and
must record their own results. Rust and Cargo must already be on the path.
Set Cargo cache/target directories to external working directories rather than
committing machine-specific paths.

All angle-bracket paths below are arguments to replace with your own locations:

```text
git fetch origin
git switch --detach origin/integration/review-nine-only
git rev-parse HEAD
git status --short
<review-python> -m pip install "maturin>=1.9,<2"
<review-python> tools/package_native.py --output "<external-validation>/package-assets"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
<review-python> -m maturin build --release --locked --out "<external-validation>/wheels"
<review-python> -m pip install --force-reinstall "<built-local-wheel.whl>[test,reference]"
<review-python> -m pytest --import-mode=importlib -o pythonpath= --basetemp "<external-validation>/pytest-temp" -q
```

No snapshot asks PyPI to resolve `ov-radiomics`; the general recipe installs
extras from the locally built wheel. There is no machine path or local file URL
in the public snapshots. Receipts keep source/wheel/native identities and exact
commands separate from immutable benchmark evidence. This branch adds only
public review documentation and does not trigger a new release or deployment.
