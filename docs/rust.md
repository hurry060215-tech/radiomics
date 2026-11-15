# Rust core and native interfaces

The repository remains an independently importable Python package. OmicVerse only needs a normal `import ov_radiomics`; no code is copied into OmicVerse and no registration or monkey patch is involved.

## Architecture

```text
ov_radiomics Python API: array/table conversion, plotting, optional adapters
    |
    +-- _native (PyO3, built with maturin, ABI3 CPython 3.9+)
    |      |
    |      +-- ov-radiomics-core (Rust numerical/file-format libraries, no Python)
    |
    +-- Optional Python reference implementation selected explicitly

ov-radiomics-cli -> the same ov-radiomics-core (no Python process/runtime)
```

`crates/radiomics-core` accepts owned contiguous buffers, dimensions, missing
flags and typed contracts. It uses Rust format/numerical/serialization libraries;
third-party matrix types do not enter its public data contract. It does not depend
on Python, NumPy, PyRadiomics, sklearn or statsmodels. The CLI calls this same core.
An R binding has not been implemented. See the [acceptance ledger](native-development.md)
for the precise working scope and outstanding release gates.

`crates/radiomics-python` converts arrays/JSON into owned Rust buffers and releases
the GIL while computing. It does not claim zero-copy transfer. Patch and overlay
iterators retain owned native state and return one patch/block at a time. CDF,
quantiles, H–L bins and both legacy-compatible/new bootstrap schemes are native.

## What moved

| Public Python function | Rust calculation | Still in Python |
|---|---|---|
| `binary_metrics` | AUC/confusion/Brier/PR/calibration summaries, DeLong and bootstrap CIs including draws/quantiles | Array validation and legacy Series layout |
| `delong_test` | Tied placements, paired variance, normal tail | Input/output layout |
| `decision_curve` | Threshold counts/suffix sums, treat-all/treat-none curves | Named model/table layout |
| `hosmer_lemeshow` | Quantile bins/membership/contributions/df/chi-square tail | Legacy interval labels for display |
| Native `evaluate` | Three-task metrics, patient aggregation, native CIs | ID/outcome conversion, cohort display, flattened and complete report views |

DCA preserves threshold order, repeated thresholds and the `p >= threshold` rule. It uses O(n log n + t log n) work instead of scanning all observations for every threshold. The measured change combines algorithmic improvement and native execution.

Native geometry/preparation, the explicit Original feature subset, selection,
models/grouped workflows, clinical inference and lightweight pathology are now
integrated. See their dedicated [image](image-native.md), [features](features-native.md),
[learning](learning-native.md), [clinical](clinical-native.md) and [interop](interop.md)
specifications. Plotting stays in Python. Independent IBSI evidence is scoped to
the declared digital phantom/configuration; no blanket conformance or unmeasured
image-extraction/full-workflow speedup is claimed.

## Stable API and explicit reference execution

```python
import ov_radiomics as rad

print(rad.backend_info())
# {'metrics_backend': 'rust', 'rust_core_version': '0.1.0'}
result = rad.binary_metrics(y_true, y_score)                 # auto
native = rad.binary_metrics(y_true, y_score, backend='rust')
reference = rad.binary_metrics(y_true, y_score, backend='python')
```

The new argument is appended to existing signatures. Outputs retain column names, index order, threshold semantics, positive-class mapping, and seeded NumPy bootstrap draws. Floating-point reductions can differ at rounding precision; parity tests use 1e-12 tolerance for the checked inputs.

An unbuilt source checkout can execute the reference with `backend='python'`.
Normal wheel builds contain Rust. `auto` and `rust` both require the native
extension; neither an absent module nor a broken DLL triggers reference fallback.
`backend_info()` probes availability without running a backend and reports
`unavailable` if absent. The extension independently validates its inputs.

## Build and test

Install Rust >= 1.85 and the platform linker; use a Python virtual environment:

```bash
python -m pip install 'maturin>=1.9,<2'
maturin develop --release
cargo test -p ov-radiomics-core --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
python -m pytest -q
maturin build --release --locked --out artifacts/wheel-rust
```

The approved local Windows setup keeps Rust under ignored `.build-tools/` and Python build packages in `.venv/`. The virtual environment reuses read-only scientific packages from the existing Python environment; newly installed build/plotting packages remain local. No system PATH was modified. To use that setup in a new PowerShell process:

```powershell
. .\tools\windows-rust-env.ps1
python -c "import ov_radiomics as rad; print(rad.backend_info())"
```

The local inherited seaborn 0.11.2 is incompatible with matplotlib 3.9 (removed `register_cmap`). Local validation uses matplotlib 3.7.5 inside `.venv`, without altering the original environment. Fresh CI environments resolve their own modern plotting stack.

## Validation and benchmark

Local environment on 2026-11-14: Rust 1.100.0 MSVC, PyO3 0.30.1, maturin 1.16.0, Python 3.9.21. Rust unit tests cover exact AUC/placements/variance, confusion counts, DCA thresholds, H–L, invalid inputs, ties and signed zero. Python parity tests cover 10 random seeds, class imbalance, ties, boundary thresholds, positive-label strings, fixed-seed bootstrap, grouped H–L, backend dispatch and missing/broken extension handling.

```bash
python tools/benchmark_metrics.py
python examples/demo.py --output artifacts/rust-demo
```

The following table is **historical stage-one metric-binding evidence**, recorded
before the full native evaluator/JSON boundary. It is not the current performance
claim. The [Benchmark v2 independent report](acceptance-v2-report.md) retains
current raw pairs, spread and slower cases. Full workflow/peak-memory measurements
remain pending. The old table included validation/conversion/output and seven
alternating warm repetitions:

| Operation | n | Python ms | Rust ms | Python / Rust |
|---|---:|---:|---:|---:|
| Binary metrics with DeLong CI | 100 | 1.1405 | 0.1842 | 6.19 |
| Paired DeLong | 100 | 0.1868 | 0.1528 | 1.22 |
| DCA, 999 thresholds | 100 | 4.1650 | 0.8327 | 5.00 |
| Binary metrics with DeLong CI | 10,000 | 8.2618 | 1.2352 | 6.69 |
| Paired DeLong | 10,000 | 3.4428 | 1.2647 | 2.72 |
| DCA, 999 thresholds | 10,000 | 19.5143 | 1.2838 | 15.20 |

## Packaging status

The version remains 0.1.0; no PyPI or crates.io publication is performed by these changes. CI builds and tests Windows/Linux wheels, checks pure-Python fallback from source, and runs the existing image-enabled demo. Compiled wheels are development artifacts. A public package release should follow an explicit version/release decision.

Build-system references: [maturin mixed project layout](https://www.maturin.rs/project_layout.html), [PyO3 modules](https://pyo3.rs/v0.30.1/module.html).
