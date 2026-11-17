# Contributing

Start with a reproducible issue or a narrowly scoped pull request. Keep the
package independently installable. Do not add deferred capabilities as placeholder
APIs or claim support from a third-party name appearing in a workflow.

## 与 omicverse 的关系

本包以 MIT 许可证独立发行；omicverse 使用 GPL-3.0。
由 omicverse 仓库维护的 `omicverse/radiomics/` 可以引用本包：MIT 代码可以
用于 GPL 项目，适配层按 omicverse 的许可证维护。
不得将 omicverse 的 GPL 实现复制回本包后按 MIT 发布。贡献本包时，应保持
独立实现和清晰的来源记录；适配层的修改提交到 omicverse 仓库。

## Development environment

Python3.9+, Rust1.85+ and a platform linker are required. Use a virtual environment;
install `.[test,reference]` for Python reference tests. The base/native-only
environment intentionally omits reference backends.

```bash
python -m pip install -e '.[test,reference]'
cargo fmt --all -- --check
cargo test -p ov-radiomics-core -p ov-radiomics-cli --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
python -m pytest -q
```

Optional medical-image reference paths additionally need `.[image]`. Installed-wheel
tests use `--import-mode=importlib -o pythonpath=` to avoid testing a different
extension in the source checkout.

## Tutorials and documentation

Author readable steps with input source/license, full parameters, intermediate
checks, expected output and replacement instructions. Use genuine outcome/identity
metadata for real-data claims. Keep synthetic data labeled synthetic. Never invent
clinical labels for the public CT or infer tile alignment from equal row counts.

The six introductory notebooks are generated from `examples/native/`; public CT
has a separately authored generator. Regeneration clears saved outputs, so execute
again before committing:

```bash
python tools/build_tutorial_notebooks.py --output artifacts/tutorial-inputs
python tools/build_public_ct_notebook.py
python tools/build_public_tabular_notebook.py --output artifacts/08_public_tabular.ipynb
python tools/run_tutorials.py --notebooks artifacts/tutorial-inputs --output artifacts/tutorial-execution
```

The public CT track is separate and explicitly accepts the dataset license;
choose an external cache and a fresh execution directory:

```bash
python tools/run_public_ct_tutorial.py --cache /outside/git/pat1 --output artifacts/public-ct-execution --acknowledge-license
```

The real UCI table lesson's input license, predeclared split and actual execution
record are described in [its guide](docs/user-guide/public-tabular.md). Source-record
groups do not establish independent patient identities or clinical validity.

Lessons 09–10 are authored, code-only local-data notebooks. Read their
[input contract](docs/user-guide/local-samples.md), then execute with externally
provided paths. Keep the input tree and all result files outside every Git worktree:

```bash
python tools/run_local_sample_tutorials.py --data-root "$DATA_ROOT" --output "$RESULTS_DIR"
```

Do not commit their executed notebooks, sample previews, feature/label tables,
models, individual predictions or machine-specific receipts. The shared executor
still requires genuine ordered execution and actual inline plots; code-only
publication does not relax local acceptance checks.

Use the executor's fresh output directory and a kernel containing the current
native wheel. Preserve failures and execution receipts. Public-data downloads
are explicit; do not make them a side effect of an ordinary documentation build.

```bash
python -m pip install -r website/requirements.txt
python tools/build_website_sources.py
python tools/build_api_sources.py
python -m sphinx -W --keep-going -b html website website/_build/html
```

`docs/` is maintained source. `website/reference/` is generated, ignored by Git
and rebuilt in CI. Edit the source rather than both copies. PRs upload HTML
previews only. Main pushes and explicit maintainer dispatches publish GitHub
Pages from allowed branches; manual publication does not merge a PR or release
a package version. Tests run once for each PR head and for main pushes, rather
than duplicating feature-branch push and PR runs.

Run `python tools/build_website_sources.py --check` after generation to detect
source or rendered-byte drift without writing anything. A fresh checkout first
generates the ignored reference tree; `docs-sync` then checks that exact tree.
There is no separately committed generated copy to keep synchronized.

The publication builder executes lessons 01–06 in fresh kernels through the
shared native executor, checks their actual artifacts and inline plots, and
renders a disposable website copy containing those outputs. Execution receipts
are separate CI artifacts, not public website files. The original notebooks and
historical execution receipts stay unchanged. Lessons 07–08 retain explicitly
licensed public-data execution, and 09–10 require external private inputs; normal
documentation builds never download those data or execute the private lessons.

## Numerical and provenance review

Do not update Golden outputs to make an implementation pass. Keep Python
compatibility separate from declared IBSI definitions. Geometry, discretization,
missing states, class order, splits, solvers and serialization require matching
independent checks. ID/mask/count/split checks remain exact; tolerance changes
need a documented definition-level reason.

Fit learned preprocessing and feature selection within patient-grouped training
splits. Keep actual patient sets, feature/class order, task, prediction unit and
model origin in exports. Native requests must never invoke a reference backend.

## Reporting results

Use approved benchmarks for performance claims; retain raw observations,
CPU/toolchain, threads, cache states and memory scope. Report suites separately.
Disclose missed targets and unmeasured stages. Synthetic accuracy/speed is not
clinical validation. See [SECURITY.md](SECURITY.md) before attaching data.
A PR does not authorize a version bump, registry publication or ov source changes.

## Scoped Python quality checks

Install the development tools in a separate environment:

```bash
python -m pip install pre-commit ruff==0.14.6 mypy==1.18.2
pre-commit install
python -m mypy
```

The two pre-commit hooks check staged Python files with Ruff E/F/I rules and
`ruff format --check`. They do not rewrite existing files or initiate a repository
wide formatting pass. Frozen reference implementations, Golden assets and
historical benchmark evidence are excluded.

`pre-commit install` is recommended locally. The `lint` CI job is mandatory:
it checks all maintained Python sources with the same pinned Ruff version,
including formatting, and runs the three strict Mypy entry points. The maintained
backlog was cleaned in separate tools/examples, tests and package commits, so CI
can check the full scope rather than allowing old errors through a diff-only gate.

Mypy strict checking is limited to the package entry point, `_backend.py` and
`interop.py`; other modules keep `ignore_errors = true` during gradual adoption.
Unannotated imported implementation modules are treated as external boundaries.
The three checked files retain explicit annotations for their dynamic external
objects. The future OmicVerse adapter is checked in its own repository.

Published documentation uses `latest/` for maintained sources and preserves a
`v0.1.0/` snapshot built from the actual release tag. Until that tag and publication
exist, the release snapshot is pending. Read the Docs, conda-forge and Docker
publication remain future work.

Current flat documentation URLs remain available as byte-identical copies of
the corresponding `latest/` files, including assets and source links. The root
index remains the version selector. Compatibility copies are generated only
from the current latest tree; removed pages and their sources are not restored.
`publication.json` records the copied file inventory and SHA256 values.

The GitHub Pages environment may restrict allowed deployment branches or tags.
An actual tag build can therefore produce a preview while deployment still needs
the maintainer's environment policy to allow that tag. The documentation workflow
does not change repository environment protection settings.

Frozen Golden generators (`tools/golden_*.py` and the frozen public CT validator)
retain their approved bytes alongside the Golden assets. Embedded Markdown and
notebook templates keep their literal line lengths; their other Ruff rules remain
enforced. Notebook JSON and historical execution outputs are not reformatted.
