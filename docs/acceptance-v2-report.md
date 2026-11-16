# Independent Benchmark v2 audit, 2026-11-14

以下路径为记录时的本机位置，复现时请替换为自己的工作目录。`$WORKDIR` 表示仓库外的构建、缓存和证据目录；源码相对路径以仓库根目录为基准。这里只替换维护文档的路径，不改写历史测量文件。

This audit starts from integration checkpoint
`597c78feed96d9af27799d89b85cc34357f89f24`. It changes benchmark evidence
handling and its tests, not numerical algorithms, Golden expected values,
datasets, seeds, tolerances, performance targets or release policy. It is a
scoped independent audit, not acceptance of all M0–M5 deliverables.

## Policy and feature evidence

The [Benchmark v2 specification](benchmark-v2.md) correctly separates numerical,
API, leakage and deployment release gates from runtime/memory targets. It keeps
Radiomics, ML & Statistics and Pathology independent and does not use fast
execution to excuse incorrect features. The old universal 1.5× extraction
speed requirement is replaced by the declared L1–L5 policy.

The [feature validation matrix](../golden/python_golden_v1/feature_validation_matrix.json)
has 55 explicit first-order/texture features: 55 Python-compatible statuses,
54 independently conformant IBSI statuses in the **declared digital phantom
scope**, and one known nonconformance, raw population Kurtosis. The [actual
IBSI report](golden-ibsi-digital-report.json) has 110 comparisons across two
aggregation modes. Raw Kurtosis remains nonconformant in both; checking the
declared subtraction of three does not relabel its native output as IBSI
excess kurtosis. No whole-family, all-preprocessing, 2D, resampling or CT
conformance follows from these results.

Independent checks performed in this audit:

| Check | Result |
|---|---|
| Four feature/imaging/reference hashes stored in the actual IBSI report | All match checkpoint files |
| Nine files hashed by the independent IBSI source manifest | All match, including original NIfTI bytes, attribution/license and source correction record |
| Existing compiled `golden_ibsi_native` harness | 1 test passed; both actual-output maps exactly reproduce all 110 stored report values |
| Existing compiled `golden_native` harness | 3 tests passed: discrete bins, raw matrices and all frozen public 3.0.1 feature cases |
| Benchmark evidence regression tests | 12 tests passed, including mixed-case rejection, typed flags/order, and Python/Rust warmup failures |

The Rust harnesses were rerun directly from the integration build without
recompiling or modifying the main worktree. Their SHA256 values are:

```text
golden_ibsi_native-6a4a9ee6d34997d0.exe
5353e2a5feb4c2ab2db80fcee8e319bdcb3fd192a41907e77663c158e64c3deb

golden_native-c45166a61905c3e1.exe
81d9b957fc01b0bbd4fa14a07586129ac32525f46702cc0ea8d5bcd65d855936
```

The integration owner separately reported 121 Python tests, 143 Rust core tests,
6 CLI tests and 25 installed-wheel tests in a Windows environment without
reference backends. This audit does not present those entire suites as freshly
rerun by this worker. Linux installation remains unverified here.

## Benchmark evidence repairs

Three issues in the initial v2 utility implementation could weaken evidence:

1. Matching pair numbers could join rows from different suites, datasets,
   operations, states, thread counts or inputs. The summarizer now requires
   one complete case identity, including input and effective-configuration
   SHA256, and the declared alternating backend order. Nonboolean correctness
   flags and malformed digests are rejected.
2. Warmup calls were outside exception capture. A workload failure during
   warmup could abort without an artifact. Both warmups and all measured calls
   are now captured. Failed warmup evidence prevents a speed claim even when
   later calls happen to succeed. Correctness comparison and output hashing
   are outside the timed call.
3. `threads=1` previously relied only on caller setup. The reference numerical
   libraries are now loaded before a locally restored `threadpoolctl` limit.
   Available BLAS/OpenMP pools are checked as one thread and recorded before
   and after measurement. Actual OS concurrency is still explicitly unmeasured.

Output hashes now preserve full float64 JSON representations with explicit
nonfinite states rather than pandas' decimal-rounded JSON formatting. They are
trace identifiers; numerical equivalence continues to use the unchanged
`atol=1e-12, rtol=1e-10` comparator. Expected hashes are not substituted for
correctness checks.

## One scoped metrics microbenchmark

The repaired runner was executed once with the existing seed 2026, n=100/10,000,
three operations, one warmup pair and seven alternating measured pairs. It ran
on Windows 10.0.22631, Intel Core Ultra 9 285K, Python 3.9.21, NumPy 1.23.0,
pandas 1.3.5, SciPy 1.13.1 and sklearn 1.0.2. The toolchain query recorded Rust
1.100.0, x86_64-pc-windows-msvc. Reference thread limits were one; both recorded
pool snapshots confirm this. The integration owner held heavy root build/test
work during measurement; overall background load was not instrumented.

The artifact contains all 84 measured observations and 12 warmup executions,
input/configuration/output hashes, compiler/dependency/binary metadata and
per-backend n, median, q25/q75, IQR, empirical p95, min and max. All comparisons
passed. Summary values were independently recomputed from its raw observations.

| Public call | n | Python median ms | Rust median ms | Rust IQR ms | Rust empirical p95 ms | Python/Rust median ratio |
|---|---:|---:|---:|---:|---:|---:|
| binary_metrics | 100 | 1.1816 | 0.3915 | 0.02105 | 0.40678 | 3.0181 |
| delong_test | 100 | 0.2240 | 0.3086 | 0.01310 | 0.33237 | 0.7259 |
| decision_curve, 999 thresholds | 100 | 4.0543 | 1.7255 | 0.04680 | 1.78367 | 2.3496 |
| binary_metrics | 10,000 | 8.3018 | 18.7690 | 0.12775 | 18.99741 | 0.4423 |
| delong_test | 10,000 | 3.4395 | 16.6767 | 0.09635 | 16.79166 | 0.2062 |
| decision_curve, 999 thresholds | 10,000 | 19.3744 | 10.5257 | 0.13470 | 10.80750 | 1.8407 |

Ratios below one are disclosed slowdowns: large-n native binary metrics take
about 2.26× reference time; large-n DeLong about 4.85×. Profiling boundary
serialization/conversion and repeated metric/rank work is an optimization task,
not a demonstrated diagnosis. These misses do not fail feature correctness.
No aggregate speedup across cases or suites is reported.

Raw artifact location on the acceptance machine:

```text
$WORKDIR/benchmark-v2/metrics-warm-acceptance-v2-20261114.json
SHA256 e738b2673617b7fda7258b138ff85dfd7d9f393d43a195fe280fb2409e268242
```

The earlier `metrics-warm-20261114.json` is retained unchanged as an initial
diagnostic run with incomplete compiler metadata. The new run is a separately
identified evidence repair, not replacement of an unfavorable result.

The new artifact also records the exact measured runner source hashes:

```text
tools/benchmark_protocol.py
5d8bec3f06e72a1d8160dfa46e128d3105f08449efd20ab14e5ddb45a74a52c1

tools/benchmark_metrics.py
6fadd78890ee7349dd7b7efcd403835d76c227f0a9c58f067f6cc02419f5ce4b
```

This is **ML & Statistics warm public-metric latency only**. It does not establish
radiomics extraction L1–L4 targets, model-training performance, pathology
throughput, peak-memory ratios, cold startup, cache performance or multicore
scaling. Seven pairs give a coarse empirical p95. Physical-core count, power
mode, storage details and background load remain unrecorded. The queried
compiler and source Cargo release settings are useful provenance but do not
replace a captured binary build command/effective compiler flags. PyRadiomics
3.1.0 is visible in environment metadata but is never invoked by these metric
cases and is not substituted for the independent 3.0.1 feature Golden.

## Remaining integration and release evidence

The following public documentation was stale at checkpoint 597c78f. The
integration owner was notified; these files are outside this worker's edit scope:

- `docs/api.md` still describes one patient row, binary train/test-only data,
  fold-number-only fusion, joblib-only persistence and Python/SciPy quantile/
  bootstrap computation. It needs explicit legacy/backend scope and native API
  links or regenerated signatures.
- README Method boundaries still says `patient_id` rejects repeat patients,
  describes all LASSO as squared loss and all classifiers through sklearn
  defaults. Those statements need legacy Python qualification; native grouped
  pipelines and multiclass OvR sparse selection differ.
- `docs/provenance.md` still describes the first metric-only Rust migration and
  all computation through installed Python engines. Native implementation and
  optional reference dependencies need distinct descriptions.
- `docs/validation.md` presents the 6-Rust/37-Python metric migration as its
  November 14 section. Keep that evidence historical and link the new ledger.
  The integration owner is also updating the old stage-one `docs/rust.md`.

Full M0–M5 acceptance still needs fixed public CT coverage, the outstanding
research combinations, pathology CLI/tutorial/live-container checks, all six
runnable tutorials, complete three-suite L1–L5/peak-memory evidence, and actual
Windows/Linux installation/independent-operation records for final artifacts.
No PyPI/crates.io publication or version change is authorized by this audit.

## Native API follow-up audit: c245959 plus integration fixes

The preceding integration gaps describe the initial 597c78f snapshot. The owner
subsequently integrated M3/CLI work and corrected shared API boundaries. This
follow-up independently exercises the current **uncommitted integration
snapshot above `c245959e67e9b18c53fcdb24bf195476cdfaeb93`**, using its matching
rebuilt extension. It does not count the newly integrated M3/pathology CLI as
independently audited here merely because the owner reports their tests passed.

Read-only adversarial calls first reproduced these actual failures, which the
integration owner then fixed without changing Golden expected values:

- pandas `iterrows()` promoted integer sample/patient IDs to floats in a purely
  numeric external frame. Training patient `1000` became `1000.0`, bypassing
  the saved training-patient overlap check. Column-preserving conversion now
  retains the same canonical identities in splitting and prediction.
- New patients could be labeled `train`/`test`, and existing patients could be
  moved between cohorts. Saved plan membership and actual fitted sample/patient
  provenance are now checked at standalone and fusion prediction boundaries.
- A selector accepted reversed target Series indices, and declared multiclass
  or unknown tasks could silently fit as binary. Index order and task class
  cardinality are now checked.
- A 3×2 tissue mask was flattened onto a 2×3 patch, and an all-NaN float mask
  became all tissue. Shape/axis and finite binary semantics are now required.
- Explicit binary class order and `positive_label` could disagree while the
  saved API configuration described the opposite of the returned probability.
  The order is canonicalized and conflicts are rejected. Numeric class lists
  now use the same string mapping as their saved SplitPlan.
- An explicitly misspelled outcome column produced `prediction_only` rather
  than an error. Explicit missing names now fail; intentional `target=None`
  remains a valid unlabeled-cohort path.
- Aggregating OOF rows to patients changed their displayed type to `apparent`.
  Aggregation now preserves OOF origin and avoids rewriting already patient-unit
  provenance.
- OOF fusion views validated all study outcomes, including held-out patients.
  Changing only one held-out lesion label could prevent fusion even though
  models and OOF predictions were unchanged. Truth checks are now scoped to the
  selected OOF population; the held-out inconsistency is left to its evaluation.
- Native `ExtractionResult.save()` silently replaced existing result files.
  Default saving now refuses replacement and explicit overwrite is required.
- Clinical patient-level summaries accepted an empty patient identifier and
  counted it as a patient. Canonical nonempty identity checks now reject it.
- Missing-native `auto` execution could enter optional Python algorithms. A
  separate read-only availability probe now serves inspection/skip logic;
  explicit `auto` requires native execution and never runs an implicit reference.

The independent regression file is
[`tests/test_native_api_audit.py`](../tests/test_native_api_audit.py). **All 36
tests passed in both environments**: Python 3.9.21 / NumPy 1.23.0 / pandas 1.3.5
with reference packages available, and Python 3.9.21 / NumPy 2.0.2 / pandas 2.3.3
with sklearn, statsmodels, SciPy, PyRadiomics, SimpleITK, joblib, ov and WSIData
absent. Both runs explicitly imported the integration worktree's source and
matching `.pyd` through `PYTHONPATH`; this is dependency-isolation evidence,
not a fresh installation test of a final wheel.

Sixteen tests cover the boundary findings and default saved threshold, original
patient grouping, model/output replacement protection, and no implicit reference
dispatch. Twenty additional cases independently compare every selected feature
family to the frozen public 3.0.1 reference: ten 2D/3D/boundary cases, each under
direction averaging and matrix merging. The 18 nonempty cases compare all **990
scalar feature values/states**, including NaN positions, with the unchanged
`atol=1e-10, rtol=1e-8`; the empty ROI case rejects each family in both modes.
Every six-family union is exactly equal to its separate-family projections.
This uses the frozen Python values, not newly recorded native expectations.

The current compiled IBSI harness was also independently rerun. All **110
observed values are exactly unchanged** from the prior actual report. Only its
feature-source hash and observed-executable/recheck provenance were refreshed
in [the actual report](golden-ibsi-digital-report.json). Published IBSI values,
the source freeze, Python Golden, tolerances and the 54-conformant/one-deviation
feature matrix remain unchanged.

Exact audited snapshot hashes:

```json
{
  "src/ov_radiomics/_native_learning.py": "f7db1c806dc0f66f729548421e9c294012ded6a8b756f23a96982fb6f2536c5c",
  "src/ov_radiomics/_native_selection.py": "cbfe43579a5b40751b1f123565b015984697733206426786fce97d5c6efd4f8f",
  "src/ov_radiomics/_native_common.py": "1f383ded882ae60e98c7f3bc49b1736991ebff416b7ce637e3157eea715240be",
  "src/ov_radiomics/_types.py": "5991f50f956694e4d8486a26e03259e136377aaf6684c054f19c2780b4ef76d3",
  "src/ov_radiomics/_native_clinical.py": "9d02ea5c3ff68bb960c9cba6938a556b0d69b66c5f8a519ec545033c7955ecd9",
  "src/ov_radiomics/pathology.py": "1a2cf8a38c6cc3d531c0ce83e0cd2bd8d53398f222662a7cc1aae860cb4feff7",
  "src/ov_radiomics/_backend.py": "1de353d7e9205257fc66fb1130b60e3f0d4bf27a49472d7a8ec7e73f3a8db058",
  "src/ov_radiomics/image.py": "ae37cd7b901691a56b55043a8f482a281a7f140555f7da88f76888fef35a2b99",
  "src/ov_radiomics/modeling.py": "f19e9ca3fe0753087fb70f880b2c8a60ad48a77b2809cc19af1bd82fa8180473",
  "src/ov_radiomics/split.py": "f0090ab15a820ccc6e0009b0c9b39e013fc44203b07d2f9cac3684860475c249",
  "crates/radiomics-core/src/learning/population.rs": "f1faacbf349863b86f842317e2ec49e1ec0cfa3a9754cae8c2f31063e10b3fb1",
  "crates/radiomics-core/src/learning/fusion.rs": "44c497655cfbcef66e06225ed79a9f734b0e0ae672480beb751a76d7a37673d2",
  "crates/radiomics-core/src/learning/training.rs": "8facc0f7a7c452523248392877eb9d0b29b3faf705a7ab26672961912de81831",
  "crates/radiomics-core/src/features.rs": "bf83322ba590b19c0b10ea5801edb287b3bdf986ad7290cdf3eedee91afdc57a",
  "native_extension": "a4a2a6231cc5fd8db83e1c062af578272c20bbaf6ede18cdea2ad60aa1cc796d",
  "golden_ibsi_native_executable": "1e0d28a6a9d1c18426ea5f269e430d88abbfa7ef101579280e30a51a505c0584"
}
```

These targeted results do not close the remaining public CT, full three-suite
performance/memory, final artifact installation, Linux, or complete M0–M5 release
gates. The original metrics timing artifact above remains a measurement of its
own earlier binary/source hashes, not a timing result for this later snapshot.
