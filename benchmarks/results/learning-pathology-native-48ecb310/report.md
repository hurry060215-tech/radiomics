# Independent learning and pathology benchmark results

Fixed native source `48ecb310a2d9ccb5da821733ee1190ce1764f27a`. All observations, including misses/failures, are retained. These two suites have separate result rows and no pooled acceleration score.

Grid: 16 case/state combinations, 224 backend observations; successful execution 224, matching output 224.

## machine_learning_statistics

| Case | State | Python median s | Native median s | Python/native ratio | Native/reference RSS | Runtime >1.10× | RSS >1.20× |
|---|---|---:|---:|---:|---:|---|---|
| learning_binary | warm_uncached | 0.0705954 | 0.0432611 | 1.632 | 0.5776 | False | False |
| learning_binary | cold_process_total | 0.867617 | 0.439036 | 1.976 | 0.5767 | False | False |
| learning_multiclass | warm_uncached | 0.0977131 | 0.0467946 | 2.088 | 0.5805 | False | False |
| learning_multiclass | cold_process_total | 0.904804 | 0.436395 | 2.073 | 0.5772 | False | False |
| learning_regression | warm_uncached | 0.0382445 | 0.038985 | 0.981 | 0.5801 | False | False |
| learning_regression | cold_process_total | 0.856173 | 0.438519 | 1.952 | 0.5734 | False | False |
| clinical_observed_complete_case | warm_uncached | 0.0217229 | 0.0101452 | 2.141 | 0.4818 | False | False |
| clinical_observed_complete_case | cold_process_total | 1.03511 | 0.414289 | 2.499 | 0.4814 | False | False |
| pathology_patient_binary_model | warm_uncached | 0.0422024 | 0.02119 | 1.992 | 0.5628 | False | False |
| pathology_patient_binary_model | cold_process_total | 0.872388 | 0.418875 | 2.083 | 0.5593 | False | False |

## pathology

| Case | State | Python median s | Native median s | Python/native ratio | Native/reference RSS | Runtime >1.10× | RSS >1.20× |
|---|---|---:|---:|---:|---:|---|---|
| pathology_decoded_frontend | warm_uncached | 10.5146 | 2.32751 | 4.518 | 1.05 | False | False |
| pathology_decoded_frontend | cold_process_total | 9.15867 | 3.17559 | 2.884 | 1.06 | False | False |
| pathology_png_frontend | warm_uncached | 8.39243 | 2.23391 | 3.757 | 1.035 | False | False |
| pathology_png_frontend | cold_process_total | 9.30028 | 3.0739 | 3.026 | 1.039 | False | False |
| pathology_png_full_scene_diagnostic | warm_uncached | 8.44623 | 2.28658 | 3.694 | 0.782 | False | False |
| pathology_png_full_scene_diagnostic | cold_process_total | 9.6509 | 3.08106 | 3.132 | 0.7174 | False | False |

## Interpretation and limits

- Values below one in the Python/native ratio are native slowdowns, retained rather than optimized away by changing the experiment.
- Runtime1.10× and memory1.20× are engineering targets, not algorithm-correctness release gates. Per-backend median/IQR/p95/min/max/n and paired ratios are in report.json; p95 is empirical with n=7.
- Each primary pathology front-end row contains QC/colors/ID alignment/hierarchy/actual tile-feature overlays/exports and excludes model fitting. The independent patient-model row belongs to ML & Statistics. The extra full-scene diagnostic retains separate stage columns; it does not replace the front-end row.
- Predictive evaluation CI=None: no bootstrap/DeLong-interval performance claim. Ordinary OR/Wald intervals are included only in the separately named clinical case.
- The pathology comparator is a frozen simple Python specification, not an optimized competing WSI engine. Data are synthetic; no clinical validation or WSI/GPU throughput claim.
- Cold means fresh interpreter with common numeric result transport; OS disk cache and background load remain uncontrolled. Source/package/input/hash audits are outside the measured envelope.
- Full-process resident memory includes imports, fixed buffers, warmup and result transport, using worker/footer/coordinator high-water maximum. This is not tracemalloc or private memory.
- On this Windows host the venv executable launches a base-Python child: worker self/footer high-water measurements cover the actual interpreter, while the coordinator polls the launcher PID. Final memory checks remain valid; immediate parent memory/timeout termination is not independently validated for that child. Operational failures, if present, remain in raw records. Future process-tree supervision is separate engineering work; this grid is not replaced.
- Single-thread workload only; no multicore scaling inference. Native computation/CLI deployment, Windows/Linux installation and other modules retain separately scoped acceptance evidence.
