# Radiomics Benchmark v2

This specification replaces the original uniform extraction-speed acceptance
rule. The user approved this revision on 2026-11-14, before full-workflow
performance results were measured. Existing Golden inputs, expected values,
parameters, failure cases, seeds and numerical tolerances stay frozen.

## Release Gate and Performance Target

**Release Gate** requires all applicable predeclared correctness tests, API
compatibility, numerical stability and convergence, patient leakage controls,
deterministic reproducibility, independent operation, serialization, and
cross-platform installation. Fast execution cannot compensate for an incorrect
feature definition. The frozen RandomForest prediction-quality behavior
thresholds remain algorithm acceptance requirements; computational speed is a
separate concern.

**Performance Target** covers speed, memory efficiency and parallel scaling.
A missed target requires disclosure, analysis and an optimization task. It does
not automatically fail numerical correctness or block an accurately scoped
experimental release. The 1.10× runtime and 1.20× memory ratios are project
engineering regression thresholds, not industry standards. Exceeding either
must be reported. Exceeding a configured memory budget or crashing is still an
operational failure, distinct from an incorrect numerical definition.

Experimental builds may expose only capabilities passing their applicable
correctness and operation gates. Pending standards/performance work stays
explicit. Formal PyPI/crates.io publication, version changes and ov integration
still require separate user authorization.

## Three independent suites

| Suite | Reference | Main questions |
|---|---|---|
| Radiomics | Frozen public PyRadiomics 3.0.1; hand matrices/boundaries; independent IBSI data | Definitions, geometry, bins/counts/values, extraction speed and memory |
| Machine Learning & Statistics | Frozen sklearn/SciPy/statsmodels and mathematical objectives | Predictions, objectives/gradient/KKT, convergence/inference, leakage, persistence, runtime/memory |
| Pathology | Independently reviewed small Python specification and hand examples | QC/colors/textures, objects, aggregation, coordinates/coverage, throughput/memory |

No combined total score or aggregate speedup across these suites is allowed.
Report every dataset/task separately. Imaging I/O or favorable extraction speed
must not hide slower modeling, incorrect inference or pathology coordinates.
The other two complete workflows retain separate 1.10× runtime engineering
comparisons; misses are disclosed and investigated, not labeled algorithm errors.

## Correctness: independent compatibility and standards tracks

**PyRadiomics Compatibility** retains `python_golden_v1`, public 3.0.1 provenance,
effective parameters, direct feature-class behavior and explicit native failure
policies. **IBSI Conformance** separately declares the IBSI version/configuration,
feature identifiers/formulas, digital phantom and licensed reference data.
Reproducing PyRadiomics is not an IBSI test. Definition differences require an
explicit IBSI configuration/profile or validation mode; they do not silently
change `python_golden_v1` or mix the two definitions.

Each feature/configuration record contains at least these fields:

```json
{
  "feature": "original_firstorder_Kurtosis",
  "profile": "python_golden_v1",
  "pyradiomics_compatible": null,
  "ibsi_conformant": null,
  "deviation_reason": "Compatibility uses population kurtosis; IBSI excess kurtosis is a separately declared definition. This illustrative record asserts neither test has passed."
}
```

`true` means a frozen applicable test actually passed; `false` means it was
tested and failed; `null` means pending, untested or not applicable, with that
status and reason stated separately. Also preserve profile/definition version,
IBSI feature ID where applicable, effective parameters, input hashes, units,
reference provenance, observed values/errors/tolerances and evidence path.
Report pass/fail/pending/not-applicable counts and denominators for each track.
Declare applicability before execution; do not remove unfavorable results later.

Existing Python-compatible features do not yet establish IBSI conformance.
Until the independent mode/data checks pass, IBSI status remains pending with
no support claim. Use the [official IBSI reference-data specification](https://ibsi.readthedocs.io/en/master/05_Reference_data_sets.html)
and [documented PyRadiomics differences](https://pyradiomics.readthedocs.io/en/stable/faq.html)
as separate sources. Binning, resampling and kurtosis differences cannot be
reconciled by relaxing tolerances.

Retain exact IDs/orders/classes/masks/bins/counts/split sets and the existing
layered tolerances for geometry, deterministic statistics, interpolation/colors,
solver predictions/objectives/KKT and NaN/Inf positions. Keep all 2D/3D hand and
boundary tests, empty/singleton/constant/nonfinite inputs, anisotropy/affines,
class/patient failure cases, leakage perturbations and independent data checks.

## Radiomics performance levels

| Level | Measurement | Comparison / target |
|---|---|---|
| L1 Operator | First-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM separately; identify binning, matrix and scalar stages | Per-family latency, allocation/memory and exact feature scope; no uniform speedup gate |
| L2 Feature extraction | Identical fixed, already-prepared images/ROIs, complete explicit supported feature subset and matched parameters | Native/reference median runtime <=1.10× engineering threshold; >=1.5× speedup target |
| L3 End-to-end | NIfTI reading, explicit preparation, extraction, conversion and CSV/result export through public API/CLI | Full latency and stage times; >=1.5× speedup target |
| L4 Batch/scalability | Predeclared 100/500/1,000 ROI workloads; varied ROI sizes, gray levels, workers | Total latency, ROI/s, peak memory, thread scaling/efficiency and maximum stable throughput under a fixed memory budget |
| L5 Deployment | Windows/Linux Rust CLI and Python API | Python-free CLI; base install without reference/ov/WSIData; cold startup, import/call/conversion overhead, artifact/install cost |

L1/L2 isolate computation; L3 remains the primary public workflow measurement.
Internal-kernel timing cannot be advertised as complete API acceleration.
Compare identical enabled features: do not time a reference with extra filters,
mesh shape or unsupported families. Identify and consistently time any explicit
Python specification helper used to reference `shape_basic`.

Single-thread performance and multi-thread scaling are separate series. Only
the outer workload controls `n_jobs`; models/samples/textures must not each
create nested pools. Match effective reference thread limits, record requested
and observed concurrency, and retain failures within the predeclared grid.
Disclose repeated synthetic ROIs used to reach batch sizes and disable content
cache shortcuts in uncached throughput runs.

## Measurement protocol and uncertainty

Freeze dataset hashes, ROI/voxel/gray-level counts, enabled features, preparation,
parameters/seeds, output format, thread grid, memory budget and repetition policy
before measuring. Record CPU model, physical/logical cores, RAM, OS/build, power
mode and load; filesystem/storage and input/output bytes; Rust compiler/target,
optimization/LTO/codegen flags, source/implementation and Cargo.lock hashes;
Python/NumPy/SimpleITK/PyRadiomics versions and compiled-extension hashes;
sklearn/SciPy/statsmodels for their suite; thread-library settings and allocator
if changed. State whether OS file-cache state was controlled or uncontrolled.

After declared warmup, measure at least seven pairs in alternating backend
order, on fixed hardware and equal thread/memory budgets. Keep cold process
startup, hot uncached execution and verified content-cache hits in separate
series, including cache preparation cost and hit/miss counts. Never pool them.
Seven observations give a coarse p95: label it empirical and show n. Larger
repetition counts must be chosen before measurement, not to chase a favorable
result or stop at a favorable speedup.

For every backend/dataset/configuration/state/thread count, publish **median,
q25/q75, IQR, empirical p95, minimum/maximum and n**, plus all raw elapsed times.
Use a declared linear quantile interpolation method. Primary speedup is
`median(reference_seconds) / median(native_seconds)`; report both medians,
paired ratios and execution order. An optional fixed-seed paired-bootstrap
interval describes uncertainty; with seven pairs it is exploratory, not a
precise tail guarantee. Gains smaller than observed noise are not promoted.

Measure full-process peak RSS/private memory consistently, declare the metric
and sampling granularity, and report the <=1.20× engineering protection separately
from correctness. Allocation count/bytes, when instrumented, are additional L1
diagnostics; otherwise label them unmeasured. Python traced allocations cannot
substitute for total RSS when comparing against native process memory.

Retain raw per-run records: suite/level/dataset/config/state/backend/pair/order,
thread count, elapsed/stage seconds, peak memory, output digest, status/error;
also retain machine/toolchain manifests and derived summaries. Correct matching
outputs are required for a valid speed result. Incorrect output is a correctness
failure with diagnostic timing; correct slow output is a disclosed performance
miss. Neither permits changing datasets, seeds, numerical tolerances or Golden.

## Report and release evidence

Deliver separate compatibility and IBSI matrices, per-suite results, L1–L5
tables, raw data/provenance, uncertainty/limitations, deployment/reproducibility
evidence, engineering regressions and optimization follow-ups. Mark capabilities
native, optional external, experimental or deferred according to actual evidence.
Local Windows results never imply Linux installation, Python-free execution,
IBSI conformance or unmeasured throughput has passed.

## Executable evidence utilities

`tools/benchmark_protocol.py` records machine/dependency/binary hashes and raw
paired observations, and computes the declared linear-quantile median/IQR/p95.
It rejects duplicate/incomplete pairs or fewer than seven declared pairs. A
failed or numerically inconsistent observation suppresses a speed claim; it
does not change correctness tolerances or produce a combined suite score.

The existing `tools/benchmark_metrics.py` is retained as one **ML & Statistics
public-metric microbenchmark**, now emitting the v2 raw-record format. It measures
one warmup pair followed by seven alternating pairs, n=100/10,000 and the same
three operations/seed. It includes conversion/output, retains per-observation
input/output hashes and errors, and refuses output replacement by default.
It does not represent radiomics L1–L4, training or pathology performance. Memory,
cold/cache states, observed concurrency and currently unrecorded machine fields
remain explicit missing evidence. These utilities are not completion of the
full performance acceptance suite.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python tools/benchmark_metrics.py --output artifacts/benchmark-v2/metrics-warm.json
```
