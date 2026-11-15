# Contributing

Start with a reproducible issue or a narrowly scoped pull request. Keep the
package independently installable. Do not add deferred capabilities as placeholder
APIs or claim support from a third-party name appearing in a workflow.

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

Use the executor's fresh output directory and a kernel containing the current
native wheel. Preserve failures and execution receipts. Public-data downloads
are explicit; do not make them a side effect of an ordinary documentation build.

```bash
python -m pip install -r website/requirements.txt
python tools/build_website_sources.py
python -m sphinx -W --keep-going -b html website website/_build/html
```

`docs/` is maintained source. `website/reference/` is generated, ignored by Git
and rebuilt in CI. Edit the source rather than both copies. HTML preview artifacts
do not automatically publish a hosted website.

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
