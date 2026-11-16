# Review-only integration of the radiomics PRs

This branch is a merged review tree for the eight PRs in the independent MIT
`radiomics` repository. It lets reviewers inspect their combined source and run
the combined tests. It is not a release branch, a deployment request, a PyPI
publication, or a new feature PR. No benchmark has been rerun for this branch.

The separate OmicVerse adapter is maintained in its GPL-3.0 repository:
[OmicVerse PR #1096](https://github.com/omicverse/omicverse/pull/1096). Its code
has not been copied into this MIT repository, and this package does not depend
on OmicVerse. That adapter has its own independent **19-test** acceptance
(16 original cases plus three installation-gate cases), against the accepted
local wheel. The public-registry extra remains a commented placeholder until
registry publication and explicit activation; this review tree does not promise
that a registry extra is currently installable.

## Exact component heads

These are the actual component heads included when this review tree was refreshed.
Each is an ancestor of this branch; the review branch adds merge commits and this
explanatory document rather than rewriting those PRs.

At this review checkpoint, **PR-1 has merged into `main`**, through merge commit
`8bf9d33400d9933662a08c152a678f15da3bb5e5`. The other seven radiomics PRs listed
below remain Draft and unmerged; the separate OmicVerse adapter is also Draft.
The corresponding main merge was included here as an ancestry-only wrapper:
its merge left the already-validated review tree exactly unchanged.

| Planned PR | Scope | Included source head |
|---|---|---|
| PR-1 | Public-content cleanup | `626baf204561abcf27b2c8e035af03e5b44796da` |
| PR-2 | Reproducibility and metadata | `ba873e6c87b22907dfafd9450c1d78286dcdea2e` |
| PR-3 | Consistent documentation publication | `ee7e80b968923b4795a3adaccac5a3390e5be2c8` |
| PR-4 | Stable Mean and numerical-boundary documentation | `95e3f3e4b7353b50fc34af31d323ab44cca87020` |
| PR-5 | Distribution readiness and release-scope notes | `89e723c6dfcb411a35837a34cf33327f8465b71d` |
| PR-7 | Owned array buffers | `69b252fd8c3575d2a24975bf424d211aa172be5b` |
| PR-8 | Deterministic local feature threads | `79bd90ff851db74f71baf41ae14ed8bb1d4fdf1f` |
| PR-9 | API and versioned documentation engineering | `1d5232992cea53511c679905071673a4a555d068` |

The auxiliary language-statistics change
`26d87394ce04b55a89c78e077366c75247d6ec5b` is also included. It does not replace
the numerical or performance acceptance of any component. The separate adapter
source head at this checkpoint is
`c9f622b72928db0fe8a6e42d334d7dc34ca2d48d` in the GPL repository.

## Source tag and numerical evidence

The `v0.1.0` source tag resolves to
`bb2d3fe4f31e938c8aef8eb64c44987f3ed24c6e`. Its release scope is PR-1 through
PR-5; it does **not** include PR-7, PR-8 or PR-9. A wheel built from this merged
review tree has a different source and native implementation identity and must
not be substituted for a tagged candidate. The version string alone does not
identify the source tree: record the commit, wheel SHA256 and native identity.

The merged numerical implementation retains Neumaier compensation for Mean.
For the documented cancellation input, native Mean and the independent
`math.fsum` result are both **0.5**. The frozen public reference observation of
**0.0** remains historical evidence, with the deliberate compatibility boundary
explained in the [scalar audit](scalar-accuracy-audit.md). No broader claim of
perfect mathematical summation is made.

The tree also retains PR-7's immutable owned sample buffers and standalone typed
input snapshots, and PR-8's local thread pools, explicit budgets and fixed-order
directional reduction. Their historical diagnostic identities and observations
remain those recorded in their own reports; merging them here is not a new
performance measurement or a change to a previously missed target.

Golden fixtures and all original benchmark observations are immutable.
`golden/**` and `benchmarks/v2/**` remain the same 70 Git blobs as historical
source `20b7fc044b609822eed4c4c177aaae1b8803d9bd`. The 16 original/public
evidence relationships remain recorded in the
[redaction manifest](public-evidence-manifest.json). Separately named public
derivatives do not replace, rewrite or rerecord the archived originals.

## Reproduce a local review build

Use a clean review checkout and a dedicated Python environment. The recorded
local acceptance uses Python 3.9 and keeps its pinned dependency snapshot in the
external validation receipt as `dependencies.txt`; use that snapshot when
reproducing the recorded environment. Other platforms and Python versions need
their own applicable checks. Rust and Cargo must already be on the path.
Set the Cargo cache and target directory to external working directories rather
than committing machine-specific paths.

All angle-bracket paths below are arguments to replace with your own locations:

```text
git fetch origin
git switch --detach origin/integration/review-nine-only
git rev-parse HEAD
git status --short
<review-python> -m pip install -r "<validation-receipt>/dependencies.txt"
<review-python> tools/package_native.py --output "<external-validation>/package-assets"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
<review-python> -m maturin build --release --locked --out "<external-validation>/wheels"
<review-python> -m pip install --no-deps --force-reinstall "<built-local-wheel.whl>"
<review-python> -m pytest --import-mode=importlib -o pythonpath= --basetemp "<external-validation>/pytest-temp" -q
```

The receipt records actual commands, output, dependency versions, component
commits, the built wheel and native hashes, the complete test result, frozen Git
object verification, and source/wheel publication-content scans. It is separate
from the historical benchmark evidence. Installing this local review wheel does
not publish it or establish a package-registry installation promise.
