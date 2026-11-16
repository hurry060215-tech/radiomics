# Radiomics Benchmark v2: native snapshot 48ecb310

**All 854 recorded observations completed within the fixed budget and passed
their applicable numerical comparisons. Performance targets were mixed.**
The real CT prepared-array call was 2.50× slower than the reference, and its
complete I/O workflow was 1.225× faster, below the 1.5× target. This is a correct
but disclosed performance limitation of this snapshot; it is not evidence of an
incorrect feature definition, nor permission to lower the target.

This is the Radiomics suite only. It must not be pooled with Machine Learning &
Statistics or Pathology into a total acceleration score. It does not establish
blanket project Release Gate completion, new IBSI conformance, clinical validity,
or a formal publication/release.

## Reproducible evidence

- Native source: `48ecb310a2d9ccb5da821733ee1190ce1764f27a`, optimized release,
  Rust1.100.0/LLVM23.1.5, `lto=thin`, `codegen-units=1`, target
  `x86_64-pc-windows-msvc`; explicit Rust flag states/build commands are retained.
- Fixed wheel SHA256:
  `f2c8e0cbf55bfdf56ee4efb7b4d664112028618cb6bf03e0761170730214c160`;
  installed in a dedicated isolated interpreter, with package bytes and runtime
  implementation constant checked against the immutable build receipt.
- Actual public PyRadiomics3.0.1 source revision `c428a771...`, source/compiled
  hashes matched accepted Golden provenance; NumPy1.23.0/Pandas1.3.5 on both sides,
  SimpleITK2.4.1 reference. The reference environment was not modified or replaced
  by the locally available3.1.0.
- Intel Core Ultra9 285K, 24 physical/logical cores, 33,663,320,064 physical RAM
  bytes, Windows build22631, balanced power-scheme GUID
  `381b4222-f694-41f0-9685-ff5bb260df2e`. Actual BLAS/OpenMP/ITK inner limits1 were
  probed; only the declared outer workers1/4 changed.
- Original base-plan SHA `a37b0339...`, pre-measurement CT append `14192fe4...`,
  explicit true2D inactive-axis binding `36730dda...`. Source/prepared arrays,
  IDs, feature list, seed, cases, repeats, tolerances and thresholds stayed fixed
  throughout measurement. The earlier failed untimed2D configuration and the
  zero-observation localized metadata-probe failure remain retained evidence.

The [versioned artifact directory](../benchmarks/results/radiomics-native-48ecb310/)
contains the full [machine/build/run provenance](../benchmarks/results/radiomics-native-48ecb310/provenance-redacted.json),
[60-case report](../benchmarks/results/radiomics-native-48ecb310/report-redacted.json),
[human-readable all-case table](../benchmarks/results/radiomics-native-48ecb310/report.md),
and compressed [all observation records, with machine paths redacted](../benchmarks/results/radiomics-native-48ecb310/raw-observations-redacted.jsonl.gz).
The [artifact index](../benchmarks/results/radiomics-native-48ecb310/artifact-index.json)
records hashes, observed request/report timestamps and the external per-worker
CSV/config/comparison/stdout/stderr location. No raw CT voxels or prepared arrays
were added to the repository/package. Public CT terms remain CC-BY-NC-3.0;
IBSI digital phantom terms remain CC-BY-4.0 with their original attribution.

## Correctness and observation integrity

The fixed grid produced 60 primary cases ×7 alternating pairs ×2 backends =
**840 primary observations**, plus **14 native-only cache observations**. No
observation was dropped, replaced, re-seeded or selected as a favorable sample.
Every warmup was a complete real call, not an implicit `timeit` best iteration.
Warmup/timed values matched exactly within each worker. All primary pairs passed
the existing exact-ID/order/count/bbox/nonfinite and scalar/geometry rules;
both CT paths additionally matched the independently frozen66-column tight-crop
reference. Cache population/hit state matched actual native counters. Global
correctness and operation verdicts include that cache series.

## Prepared-array and complete workflow results

The following are **per complete call** medians, not per-ROI kernel times. L2
processes16 ROIs/call for synthetic/phantom data and1 ROI/call for CT. L3 processes
8 ROIs/call for synthetic/phantom data and1 ROI/call for CT. Both sides compute
the explicit55 features plus the separately identified11 project `shape_basic`
descriptors. Mesh/filter features are disabled. Seven observations per backend
give coarse empirical p95 estimates; full q25/q75/IQR/min/max and paired data are
retained in JSON/raw evidence.

| Case | ROIs/call | Python median ms | Rust median ms | Rust IQR ms | Rust empirical p95 ms | Python/Rust speedup | Engineering/target result |
|---|---:|---:|---:|---:|---:|---:|---|
| L2 2D32×24, gray8 | 16 | 47.754 | 24.877 | 0.236 | 25.643 | 1.920× | Runtime≤1.10×; target≥1.5× met |
| L2 2D48×40, gray32 | 16 | 67.714 | 47.022 | 0.451 | 47.998 | 1.440× | Runtime passes; speed target missed |
| L2 3D16×14×12, gray8 | 16 | 58.811 | 59.718 | 0.154 | 61.860 | 0.985× | Runtime passes; near parity, speed target missed |
| L2 3D24×20×16, gray32 | 16 | 107.711 | 166.168 | 2.836 | 170.146 | 0.648× | Runtime≈1.543×, regression and speed target missed |
| L2 IBSI digital phantom | 16 | 42.366 | 13.444 | 0.191 | 13.675 | 3.151× | Runtime and speed target met in this scope |
| L2 public CT tight bbox100×99×26, ROI125256 | 1 | 157.666 | 394.163 | 3.494 | 397.686 | 0.400× | Runtime≈2.500×, regression and speed target missed |
| L3 2D32×24 clip/crop/read/export | 8 | 64.200 | 8.315 | 0.113 | 8.652 | 7.721× | Target met for this small synthetic workflow |
| L3 3D24×20×16 clip/crop/read/export | 8 | 94.868 | 29.617 | 0.530 | 30.605 | 3.203× | Target met for this synthetic workflow |
| L3 IBSI digital read/clip/crop/export | 8 | 59.485 | 7.893 | 0.234 | 8.059 | 7.536× | Target met in this phantom scope |
| L3 public CT original NIfTI/crop0/export | 1 | 225.082 | 183.807 | 0.700 | 184.903 | 1.225× | Correct result;1.5× target missed |

The small synthetic/phantom L3 gains cannot be advertised as a universal
radiomics or real-CT speedup. L2 and L3 also have different declared preparation
for synthetic data, so their difference is not a measured decomposition into
conversion, I/O and computation. CT L2/L3 preserve the same ROI/scalars/world
geometry and illustrate the cost difference between the public array and NIfTI
paths, but a stage profiler is still needed to assign that cost precisely.

## Operators, scaling, memory and cache

All35 L1 rows are actual selected-family public-array calls: other matrices and
scalars are skipped in the shared Rust implementation. Many are slower than the
C-backed Python reference, particularly at larger buffer sizes. Their timing
includes each public buffer/image conversion and validation. They are not pure
kernel measurements, and allocator count/bytes remain unmeasured. `shape_basic`
is a separate native-versus-independent-spec row, never a relabeled IBSI mesh
family. Small near-equality differences are not promoted as acceleration.

The L4 synthetic workloads deliberately repeat a disclosed small image/ROI with
distinct IDs, cache disabled. They are useful for scheduling/API throughput and
cannot stand in for a large clinical cohort. For the1,000-ROI series:

| Synthetic case | Rust1 worker ROI/s | Rust4 workers ROI/s | Native4/1 scaling | Python4 workers ROI/s | Native/reference4-worker ratio |
|---|---:|---:|---:|---:|---:|
| 8³, gray8 | 1933.6 | 2654.3 | 1.373× | 213.5 | 12.431× |
| 12³, gray32 | 875.1 | 1934.3 | 2.210× | 134.6 | 14.375× |

Neither native nor reference scaling is linear. These are maximum **observed**
stable rates within the frozen count/worker grid and2GiB budget, not theoretical
maximum engine throughput. The12 separate L4 cases retain all medians/spread,
ROI/s and worker efficiencies; no overall averaging hides an unfavorable row.

All60 median native/reference peak-RSS ratios were below the1.20 engineering
threshold. The largest observed whole-worker peak was152.22MiB, below2GiB.
For CT, median native/reference RSS was109.44/143.68MiB on L2 and
107.87/147.14MiB on L3. Memory means complete process lifetime resident high-water
mark, including imports, fixture buffers, warmup, call and numerical transport;
worker kernel/footer peak and coordinator sampled high-water were combined by
maximum. These values do not measure allocation count, Python-only memory,
process-private bytes, or arbitrary large-image memory scaling.

Seven independent native cache populations took median36.810ms (IQR8.107ms,
empirical p9545.295ms) for the declared eight-ROI synthetic3D workflow. The seven
verified hit observations took median6.789ms (IQR0.132ms, p956.896ms), with actual
hits8/misses0/rejected0/write-failures0. Population cost remains recorded. No
cached-Rust-versus-uncached-Python speed claim is made.

## Optimization tasks and remaining evidence

The two L2 engineering regressions and real-CT L3 target miss require explicit
optimization work. Code inspection shows the public array path repeatedly
serializes owned image/mask buffers and performs validation across JSON/PyO3
requests. The size-dependent L1/L2 pattern and the CT array/NIfTI path contrast
make boundary overhead a reasonable **hypothesis**, not a measured attribution.
Do not claim the numerical kernel itself is slower/faster by that amount.

1. Profile conversion, buffer ownership/validation, binning, matrices/scalars and
   output on the same frozen cases before selecting an optimization.
2. Investigate one shared validated contiguous-buffer/PyO3 path or a native owned
   image object to avoid repeated full-volume JSON round trips. Preserve the
   current contract, exact bin/matrix behavior, units, NaN positions and errors;
   do not create a second Python calculation engine.
3. If software changes, produce a new immutable source/wheel receipt and rerun
   the **same** predeclared grid, retaining this report as the previous snapshot.
   Do not alter datasets, seeds, ROI sizes, active features, repetitions,
   tolerances or targets to improve the measured result.

L5 CLI startup/install-cost measurements, allocator diagnostics, separately
instrumented stages, exclusive control of unrelated host processes and disk
cache, and detailed storage hardware remain separate missing evidence. The task
team held builds/installs/other suite timing during this run, but unrelated OS
load and file-cache state were not controlled. This snapshot's numerical and
operation checks pass within the measured scope; the disclosed performance
misses remain performance work, and formal publication/version changes still
require separate authorization.

### Subsequent Windows process-supervision finding

Later inspection identified that this Windows venv's `Scripts/python.exe` is
a launcher with a separate base-Python worker. Coordinator polling of the
initial Popen PID therefore does not by itself prove real-time supervision of
the actual worker's RSS or complete descendant cleanup on timeout. The recorded
peak still incorporates the real worker's own post-transport lifetime high-water
footer, so the observed peak values and post-completion 2 GiB comparisons remain
valid. They must not be presented as a tested guarantee of immediate termination
at the budget boundary. A subsequent owned-process-tree watchdog repair is being
validated separately; no original observation or frozen measurement is replaced.
