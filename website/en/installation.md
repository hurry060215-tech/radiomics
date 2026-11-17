# Installation

[English home](index.md) · [Quickstart](quickstart.md) · [简体中文](../installation.md)

The distribution is **ov-radiomics**; import it as **ov_radiomics**. The current
package version is 0.1.0. A repository build or Actions artifact is a candidate
artifact, not evidence that the package has been published to a package index.

## Supported first-release platforms

The release matrix covers x86-64 Linux (manylinux wheels) and Windows with
Python 3.9, 3.10, 3.11 and 3.12. macOS and ARM installation are not claimed by
this matrix. Choose a wheel whose platform and interpreter tags match your
environment. The independent CLI is a separate artifact.

For a downloaded candidate wheel, use a fresh virtual environment:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install "dist/ov_radiomics-0.1.0-<matching-tags>.whl"
python -c "import ov_radiomics as r; print(r.__version__); print(r.backend_info())"
```

Check that `metrics_backend` reports `rust`. Compiled wheels do not require a
Rust toolchain. Base Python dependencies are NumPy, pandas and matplotlib.
The native NIfTI, feature, modeling and ordinary pathology workflows do not
require the optional reference or image backends.

## Source installation

Source builds require Python 3.9+, Rust 1.85+ and a system linker. Windows also
requires the Visual Studio C++ build tools and Windows SDK.

```bash
git clone https://github.com/hurry060215-tech/radiomics.git
cd radiomics
python -m pip install .
python -c "import ov_radiomics as r; print(r.backend_info())"
python examples/native/01_extract.py --output artifacts/first-extraction
```

Use a new output directory for each execution. If the native extension cannot
load, check the interpreter and wheel tags before rerunning the example.

## Frozen source-tag builds

The maintained branch includes the binding README required by PEP 517 metadata,
so its clean source checkout follows the ordinary installation steps above.
The immutable `v0.1.0` Git tag/GitHub source archive predates this repair. From
that tag's root, prepare its existing local packaging assets before installation:

```bash
python tools/package_native.py --output artifacts/tag-source-assets
python -m pip install .
```

Prepared installation sdists already carry these assets. A GitHub source zip or
tar archive is not the prepared sdist. The preparation step leaves the tag and
frozen references unchanged; the historical source tag is not being rewritten
or advertised as a repaired bare checkout.

## Optional backends and the CLI

Only install optional backends for workflows that explicitly request them:

```bash
python -m pip install '.[reference,image]'
```

The Python wheel does not install the independent `radiomics` executable.
Download the matching CLI artifact or build it separately:

```bash
cargo build --release -p ov-radiomics-cli --locked
```

The executable is `target/release/radiomics` on Linux and
`target/release/radiomics.exe` on Windows. It uses the same Rust core without a
Python runtime. See the [CLI reference](../reference/cli-workflows-native.md).

## Documentation versions

`latest/` follows the maintained branch. A `v0.1.0/` documentation snapshot is
generated from the actual release tag and then preserved; it remains pending
until that tag exists and the publication workflow runs. Read the displayed
build revision when comparing results across versions.

Read the [support matrix](../reference/support-matrix.md) before treating a
native subset, optional backend or historical validation result as broader
support. Read the [data provenance](../datasets.md) before using public examples.

Registry uploads require a separate approved workflow. See the
[release procedure](../reference/release.md).
