# ML & Statistics / Pathology predeclaration

This is a proposed experiment plan, not a measured result. The exact
[`learning-pathology-plan.json`](../benchmarks/v2/learning-pathology-plan.json)
must be reviewed by the main integration agent before timing. A fixed installed
wheel, build manifest and approved environment/input hashes must then be bound
to it. No performance observation has been collected for this plan.

The eight workload rows remain separate:

| Suite | Workload | Fixed scope |
|---|---|---|
| ML & Statistics | Binary learning | 66 patients / 132 lesions / four features, frozen 3×2 nested patient CV, LR lambda [.05,.1] |
| ML & Statistics | Multiclass learning | Same counts and actual frozen patient sets, softmax LR |
| ML & Statistics | Regression learning | Same counts and actual frozen patient sets, Ridge |
| ML & Statistics | Clinical secondary | Frozen 96-row mathematical fixture: observed baseline and univariate/multivariable unpenalized Logit, explicit sample-unit scope |
| Pathology | Decoded front end | 12 study patients × two slides plus an all-QC-excluded sentinel: 25 slides / 600 tiles |
| Pathology | PNG front end | Identical input pixels/parameters, with actual PNG reading inside the workflow |
| ML & Statistics | Pathology patient model | Separately frozen identical 12-patient aggregate input; eight train, two test, two unlabeled external |
| Pathology | Full-scene diagnostic | PNG front end followed by its own aggregate→model→predict chain, with each stage shown independently |

Every row has distinct cold and warm series, seven alternating backend pairs,
one outer worker, verified single-thread BLAS/OpenMP and a 2 GiB resident-memory
budget. This yields 16 case/state combinations and 224 measured worker attempts,
plus the separately retained warmups. No number combines these suites with
Radiomics. Missed 1.10× runtime or 1.20× memory engineering targets are disclosed
per row, not treated as an algorithm-correctness failure.

The three learning rows cover conversion, complete train-only grouped
preprocessing/screening/tuning, final model fit, actual model persistence and
loading, prediction without outcomes, patient-level evaluation and table exports.
Model evaluation uses `ci_method=None` in all tasks: bootstrap and DeLong interval
runtime is not being measured. The separate clinical row naturally includes its
ordinary OR/Wald intervals and complete-case counts.

The existing Golden workflow generator is a correctness specification rather
than a production timing adapter. The benchmark reference will preserve its
imputation, scaling, correlation screen, objective/penalty mappings, actual folds,
candidate order, tie rule and public sklearn solver settings. It will not time
fixture generation, perturbation refits, redundant final-score calculations,
whole-Golden comparisons or independent gradient/KKT oracles. Those checks occur
in untimed preflight. Ordinary solver convergence checks and the audit/model/table
work actually persisted by the reference remain part of its runtime. No
statsmodels/HiGHS separation oracle is added merely to penalize reference speed;
the native clinical API's stronger diagnostics are real API costs and disclosed.

For pathology, the RGB images are deterministically derived from the frozen
binary input vectors using bounded chromatic midtones plus a fixed modular spatial
gradient. No outcome enters the generator. Images are 128×96 pixels; patches are
32×32, stride 24, padded at the edge. Each slide has 24 tiles, including nine with
padding. The source/coordinate-based tile-ID byte specification is explicit, and
feature rows are deliberately reversed with their IDs before strict alignment.
QC reasons, histogram counts and missing states remain exact. All frozen color
statistics are computed on accepted valid pixels, but only the declared RGB means,
Hue resultant length and tissue fraction enter generic aggregation. Raw Hue
angles are never averaged linearly. Excluded tiles retain metadata and five
missing cells; the sentinel's empty slide/patient remains in exports.

The two front-end rows include QC/colors, ID alignment, six tile→slide summaries,
equal-slide patient means, and real tile-R-mean overlays for a fixed ordinary slide
and the empty sentinel. Each overlay is twelve 32×32 blocks on the original global
128×96 canvas. Padding is excluded, overlaps are averaged, coverage is exact, and
uncovered pixels remain NaN/transparent. Whole/block consistency is checked
outside timing; the reference does not calculate a second whole-image oracle in
the timed call. These rows exclude training and must not be presented as latency
for the full pathology-modeling scene.

The separately reported patient-model row consumes one frozen common patient
matrix prepared from the independent Python specification before measurements.
The additional full-scene diagnostic instead trains from each run's own aggregate
features and records pixel, aggregation, overlay/export, fitting, persistence,
prediction and extra table-export stages separately. Its total never replaces
the front-end or model rows. Public `TrainingResult.save()` combines persistence
and its builtin CSV exports, so that stage is named accordingly rather than
inventing an internal split.

Both full-process RSS and failures include warmup and transport bookkeeping;
byte hashing/audit work and oracle comparisons stay outside the timed envelope.
Cold means a fresh interpreter with uncontrolled disk cache. Raw records retain
every attempt, both backend versions/source/binary hashes, stage and process
times, memory sampling policy, output bytes and exact input/configuration identity.
The simple Python pathology specification is not an optimized competing engine;
speed results may only describe this declared reference and synthetic workload.
