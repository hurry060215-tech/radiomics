# Executable Radiomics Benchmark v2

The immutable case definition is
[`benchmarks/v2/radiomics-plan.json`](../benchmarks/v2/radiomics-plan.json).
It was reviewed by the main integrator before algorithm timing. The worker
requires its exact approved file SHA256 and refuses a changed plan. The plan's
historical `predeclared_pending_main_review` text stays immutable; approval is
recorded externally by the accepted hash and in execution provenance.

`tools/benchmark_radiomics.py --prepare` creates and hashes deterministic original
CC0 synthetic images/ROIs and copies the independent IBSI digital phantom without
altering its source bytes. The artifact retains Alex Zwanenburg's CC-BY-4.0
license and source README. It does not use CT data with unclear or incompatible
redistribution rights. Preparing the corpus does **not** run or time feature
extraction. Synthetic formulas, seed, image/ROI shapes and every case are fixed
in the plan; every file gets a SHA256 manifest.

```powershell
python tools/benchmark_radiomics.py --prepare --fixtures E:/radiomics-build/benchmark-v2/radiomics-inputs
```

Measurement requires two separate interpreters. The reference interpreter must
actually import public PyRadiomics 3.0.1, NumPy 1.23.0 and SimpleITK 2.4.1, with
their source/extension hashes retained. The existing isolated Golden reference
environment is read-only; it is not replaced by local 3.1.0 or used for native
calls. The native interpreter must load a dedicated normally installed wheel,
with every package `.py/.pyd/.so/.dll` byte matching the fixed wheel. An editable
checkout is refused. The supplied native build manifest and wheel are copied to
the run's snapshot directory. Backend versions, effective settings, internal
thread-library settings where observable, source/extension hashes and executable
paths are retained in non-timed preflight provenance. Every recorded observation
has coordinator byte audits immediately before launch and after exit; these
whole-file audits are outside the cold wall clock. The reference preflight checks
source **and compiled-extension hashes** against the accepted public 3.0.1
Golden provenance, not just `__version__`. An unreviewed different build is an
error/pending state, not an assumed equivalent reference.

```powershell
python tools/benchmark_radiomics.py --run `
  --approved-plan-sha256 ACTUAL_REVIEWED_FILE_SHA256 `
  --fixtures E:/radiomics-build/benchmark-v2/radiomics-inputs `
  --output E:/radiomics-build/benchmark-v2/fresh-measured-run `
  --native-python E:/radiomics-build/benchmark-v2/native-snapshot/Scripts/python.exe `
  --reference-python E:/radiomics-build/golden/env-301/Scripts/python.exe `
  --native-wheel E:/radiomics-build/fixed-snapshot/ov_radiomics-0.1.0-cp39-abi3-win_amd64.whl `
  --native-build-manifest E:/radiomics-build/fixed-snapshot/build-info.json
```

The exact output path must be new. No favorable-run overwrite or repetition-count
adjustment is supported. Every paired observation, failure, stderr stream,
request, numerical comparison artifact and public output remains available. Each
case completes all seven alternating pairs; numerical mismatch or execution
failure suppresses its speed claim. The native and reference result values are
compared under existing scalar/nonfinite/exact-ID rules. This runner never writes
Golden expected values or changes tolerance/seed/data after observing speed.

## What each level actually measures

- **L1**: five fixed 2D/3D/IBSI datasets, each selected family separately plus a
  distinct project `shape_basic` specification row. Each warmup/timed call really
  extracts 16 ROIs. Native selection skips other families' matrix/scalar work.
  Timing includes the public array/image conversion and actual selected
  discretization/matrix/scalar work; it is not mislabeled as an internal-kernel
  or separately instrumented substage measurement. There is no uniform speed
  threshold for these rows.
- **L2**: the same five datasets, fixed prepared buffers, full explicit 55
  PyRadiomics-compatible features plus the separately documented 11 project
  voxel-support descriptors on both sides. Each call really processes 16 ROIs,
  including public conversion/table construction. The reference uses actual
  public PyRadiomics feature classes; the independent project shape specification
  uses mask indices and complete RAS affine, not mesh morphology.
- **L3**: three fixed 2D/3D/IBSI cases, eight image/ROI pairs each. It includes
  cohort CSV, NIfTI reads, explicit range clipping and ROI bbox crop/affine update,
  complete feature extraction, table/diagnostic conversion, and CSV/config
  export. Native uses `read_cohort → extract → ExtractionResult.save`; reference
  uses public SimpleITK read/ROI, NumPy clipping, public
  `RadiomicsFeatureExtractor.execute`, the independent project shape specification,
  and DataFrame/CSV export. Only Original and the 55 declared PyRadiomics features
  are enabled. The entire effective 3.0.1 extractor settings are retained,
  including its provenance diagnostics setting. No extra mesh/filter feature
  can be silently calculated and filtered away afterward.
- **L4**: two predeclared synthetic 3D cases at 100, 500 and 1,000 ROIs, outer
  workers 1 and 4. It times complete batch I/O/extraction/export. Repeated source
  images/ROIs are disclosed, sample/patient IDs remain distinct, and feature
  content caching is disabled. Scaling/ROI/s is reported independently for each
  dataset and backend; no combined ML/pathology score exists.

The primary grid therefore has 58 cases and 812 recorded backend observations.
Each warm observation is a **fresh interpreter**, importing/loading its fixed
data and performing one complete warmup call before one timed call. For L1/L2
that means 16 real warmup ROIs and 16 real measured ROIs, not a `timeit` minimum
or a selected best iteration. Warmup output artifacts are retained and must match
the measured call exactly in that same worker. Seven pairs remain only seven
independent recorded timings; the 16 ROIs per call are not falsely counted as 112
independent latency observations.

L3 cold rows have no warmup and use coordinator process-launch-to-exit time,
including imports, required local resource configuration, full API work and
numerical report transport. Package/input audits occur before launch/after exit;
comparison-sidecar persistence, file re-reading and output digest calculation
occur after this clock stops. Application-internal provenance/hash work performed
by the public extraction API remains part of that API. These rows are
fresh-process measurements, not flushed disk caches or Python-free CLI startup.
The OS page cache remains explicitly uncontrolled. L5 CLI/install evidence is
separate.

The native content-cache series is also separate: seven independent cache
directories, each with one population observation and one hit observation. Actual
core hit/miss/rejected/write-failure counts must match the declared state, input
and implementation identities. Population cost remains in raw data. A cached
native call is never advertised as a speedup over uncached Python.

## Memory, concurrency and uncertainty

Both backends use outer workers in the **same worker process**. The reference
uses an explicitly bounded ThreadPoolExecutor rather than a process pool whose
children would be missed by process RSS. Inner OpenMP/BLAS/ITK thread settings are
fixed to 1 before import; SimpleITK's effective thread count is also checked.
Only outer worker counts differ in the declared scaling series. The isolated
measurement interpreter uses threadpoolctl as a probe/local limiter, without
adding it to the project runtime dependencies or installing SciPy. Missing
runtime BLAS/OpenMP observability or an effective count above 1 fails the
matched-resource check; effective reference ITK threads must also equal 1.

Full-process lifetime peak resident memory uses Windows `PeakWorkingSetSize` or
Linux `ru_maxrss` converted from KiB to bytes. It covers imports, fixture buffers,
warmup, timed workload and common result/protocol bookkeeping on both backends.
It is not `tracemalloc`, private memory, or per-call allocation count. The
coordinator samples the owned worker every 20 ms, enforces a fixed 2 GiB budget
and fixed observation timeout, and also checks the kernel lifetime peak after
completion. A worker footer records the kernel peak after result serialization
and transport. Final peak is the maximum of that worker kernel peak and the
coordinator's observed resident/high-water peak; both original measurements are
retained. Only its own worker process can be stopped by this guard.

Raw seconds, median, q25/q75, IQR, empirical linear-interpolation p95, min/max,
paired ratios, ROI/s, process RSS and each status/error are retained. Seven
observations do not establish a precise tail estimate. Source/build metadata,
CPU/core/RAM data, active power scheme when obtainable, input/output byte counts
and hashes are retained. Uncontrolled host background load, disk cache and
unrecorded device details remain limitations. Noise-scale differences must not be
advertised as acceleration.

Correctness and performance remain separate: L2 runtime ratio >1.10 or memory
ratio >1.20 are engineering misses; >=1.5× L2/L3 speedup is a target. Correct slow
results are disclosed and analyzed, not redefined as numerical algorithm failure.
Actual memory-budget failure is an operational failure. All 14 cache population/
hit observations participate in the global correctness/operation verdict, so a
failed cache series cannot coexist with an overall passed declaration. Comparison
artifact errors retain both completed backend observations and continue the
fixed grid rather than dropping the pair. This report cannot
establish blanket IBSI conformance, medical effectiveness, other benchmark
suites, allocator counts, or unmeasured deployment gates.
