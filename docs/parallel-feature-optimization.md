# Deterministic local feature parallelism

Feature extraction accepts an explicit native `threads` argument. Its default
is one. The change introduces Rayon scheduling in `features.rs`; it does not
change the formulas, discretization, BTreeMap representation or modeling layer.
This is a separate diagnostic from the original 854 benchmark v2 observations.

```python
table = sample.extract(config, threads=8)
result = ov_radiomics.extract(samples, params=batch_config,
    backend='rust', n_jobs=1, threads=8)
```

## Task granularity and reproducibility

Only GLCM and GLRLM have independent half-direction outputs in the current
implementation. For a non-singleton 3D input, each has thirteen ordered direction
tasks. GLSZM, GLDM and NGTDM each use one whole-family task. First-order and basic
shape keep their existing scalar path. There are no per-voxel scheduling jobs.
Discretization, gray-level validation and rank lookup are shared before dispatch.

The indexed parallel iterator collects each result into its fixed task slot.
Directional features are then reduced in the original order. Within every
direction or family, the original voxel order and floating accumulation order
remain unchanged. Three repeated runs and serial/parallel comparisons use
`np.array_equal`; a subprocess test also compares `RAYON_NUM_THREADS=1` with `8`.
Both aggregation modes and the typed/JSON fallback gateways are covered.

Each call with `threads > 1` creates a local `ThreadPoolBuilder` pool. No global
Rayon pool is resized. For this opt-in inner parallelism, `n_jobs * threads` must
not exceed the logical CPUs available to the process. Zero counts and multiplied
budget overflow are errors. The default `threads=1` preserves the existing outer
`n_jobs` behavior, including its historical allowance of more outer workers than
available CPUs. A Rust test explicitly simulates two CPUs and checks that legacy
`n_jobs=3, threads=1` still passes while `n_jobs=3, threads=2` is rejected.

## Measured comparison with the PR-7 baseline

The baseline was the clean-base PR-7 wheel at `a8f7966`; the candidate contained
the implementation and tests through `12d74cc`. Both were separately installed
with identical numerical dependencies: Python 3.9.21, NumPy 1.23.0, pandas 1.3.5
and threadpoolctl 3.6.0. Measurements ran on the same Windows system, Intel Core
Ultra 9 285K with 24 logical CPUs. Other development workers were paused.
The baseline used one feature thread; the candidate explicitly used eight.
Both used one outer worker and limited numerical library pools to one.

Inputs, prepared geometry, CT66 definitions and extraction parameters came from
the unchanged fixed PAT1 plan. Each runtime worker ran one warmup and seven
recorded calls. L2 used preloaded tight-crop arrays and the complete public array
API; L3 read original NIfTI inputs, prepared the ROI and exported the result.
Three separate fresh-process workers per implementation and level measured RSS.
All outputs passed the frozen 66-value reference at the original tolerances.
Their ordered float64 feature buffers were also byte-identical across every
recorded baseline/candidate worker and repetition.

| Metric | PR-7 baseline | Parallel candidate | Candidate / baseline |
|---|---:|---:|---:|
| L2 runtime median ms | 194.5244 | 136.0858 | 0.69958 |
| L2 peak RSS median bytes | 84652032 | 84180992 | 0.99444 |
| L3 runtime median ms | 175.9637 | 115.1640 | 0.65448 |
| L3 peak RSS median bytes | 106958848 | 106995712 | 1.00034 |

Both levels passed the unchanged **1.10x runtime / 1.20x memory** regression
limits. L2 speedup was **1.4294x**, so its **1.5x target was not met**. L3 speedup
was **1.5279x**, meeting that target on this input and machine. The target and
thresholds were not relaxed. These results do not replace formal L2/L3 values
and do not establish a new comparison with a reference backend.

RSS is the actual worker's lifetime `PeakWorkingSetSize`, including imports and
setup. It is not a resettable feature-only allocation peak. Runtime and RSS were
measured in separate workers; RSS-worker call timings are retained but excluded
from the runtime summaries.

## Reproduction and retained evidence

```text
python tools/profile_parallel_extraction.py --fixtures <fixed-fixture-directory> --output <new-external-directory> --baseline-python <pr7-python> --candidate-python <candidate-python> --threads 8 --repeats 7 --warmups 1 --rss-repeats 3 --acknowledge-license
```

All machine paths are arguments. The coordinator validates frozen input hashes
before measurement and retains raw commands, worker PIDs, runtime rounds, RSS
snapshots, source-file identities and numerical dependency versions. Every
output directory must be new and outside Git. No historical measurement or
reference file is overwritten, and image pixels are not copied into Git.

- Raw comparison report SHA256: `803dcc6ef9f66176d035b687c7d59cdefb87371c7c2180eb3a1c977d450fed7f`.
- Baseline wheel SHA256: `92505ef4b20f9babd77b1497255f1b3ce289b654b341c0cc9044bfb4a8e4a6cd`.
- Candidate wheel SHA256: `d85f56bf0083a8ec79b52f22729479c6a059b2e080b81cac774f9323236e353d`.

The accepted candidate passed the complete Python suite: **263 passed, 5 skipped,
1 xfailed**. Core feature tests passed **8 tests**, including the new budget
compatibility test. Existing assertions, Golden and historical benchmark data
were preserved. Scanning every candidate wheel member found no forbidden public
publication tokens, including in its native binary and package metadata.

The subsequent PR-7 strict-Clippy repair was merged after measurement. It only
uses Rust's automatic Arc dereference for the same borrowed array slice. No
parallel algorithm, arithmetic or measured evidence was changed. The wheel and
report identities above remain the actual pre-lint-repair measured artifacts;
strict workspace Clippy also passed on the final merged source.

A later standalone typed-entry ownership repair snapshots caller-owned NumPy
input before releasing the GIL. The measured public `ImageSample` and cohort
extraction paths already own their buffers and do not call that separate private
entry. Their reported measurements and artifact identities remain unchanged.
