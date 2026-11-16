# ov-radiomics-python

PyO3 bindings for the independent `ov-radiomics` Rust core. The Python import is
`ov_radiomics`; numerical extraction and native modeling are implemented by the
shared core rather than duplicated in the bindings.

The software is MIT licensed; see the repository's root `LICENSE`. This binding
crate is not published independently. Build the Python distribution from the
repository root, using its `pyproject.toml` and a supported Rust toolchain and
platform linker. Python's base dependencies are NumPy, pandas and matplotlib;
reference and imaging backends remain optional.

This README is maintained in Git so PEP 517 metadata can be read from a clean
source checkout. Auxiliary `package-assets` are prepared separately for the
documented crate packaging and evidence workflows.
