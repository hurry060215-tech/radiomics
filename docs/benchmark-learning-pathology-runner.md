# Learning / Pathology measurement runner

以下路径为记录时的本机位置，复现时请替换为自己的工作目录。`$WORKDIR` 表示仓库外的构建、缓存和证据目录；源码相对路径以仓库根目录为基准。这里只替换维护文档的路径，不改写历史测量文件。

The main-reviewed plan is
`33911b9585e934a43ec08e620bae37fcfd1061293799023b14243e49b80b874b`.
The runner keeps eight declared workload rows, two separate cold/warm states,
seven alternating backend pairs and **224 observations**. ML & Statistics and
Pathology have separate results; there is no pooled acceleration number.

Implementation is in `tools/benchmark_learning_pathology.py` with small learning,
independent public-reference, pathology and process-protocol helpers. The reference
matches the frozen public sklearn workflow's medians, scaling, correlation screen,
patient partitions, two penalties, objectives and solver settings. It omits
fixture generation, redundant final-score/oracle computation and perturbation
refits from timed work. Runtime convergence checks and exported audit/persistence
state remain included. Independent pathology math imports only the frozen Python
specifications. No native package is installed in that reference environment.

The fixed source is native snapshot `48ecb310a2d9ccb5da821733ee1190ce1764f27a`,
core implementation `a39c66ee16dcf04aefd79f2b3996e1be5666b2903b0f40e991aaf5097d54c2a8`,
wheel SHA `f2c8e0cbf55bfdf56ee4efb7b4d664112028618cb6bf03e0761170730214c160`.
Dedicated measurement environments are separate from the Radiomics/reference301
environments. Both facade environments use NumPy1.23.0/pandas1.3.5; the native
environment has no sklearn, SciPy, statsmodels, PyRadiomics, SimpleITK, joblib,
ov or WSIData. Metadata/source/shared-library/wheel hashes are bound before timing.

Stages cover actual public calls and declared exports. The whole-pathology
diagnostic shows pixel, aggregation, overlay, export, fitting, persistence and
prediction work separately. Its public `TrainingResult.save()` combines model
persistence and builtin tables, so that stage is labeled as combined. Canonical
comparison construction/file hashing is outside warm API timing; cold process
time includes required numeric result transport. Neither state claims cold disk
cache. Predictive evaluation uses CI=None and does not measure bootstrap/DeLong
interval runtime. Clinical OR/Wald inference is a separately named row.

Comparisons keep exact IDs/classes/folds/features/counts/missing states. Frozen
color descriptors use their declared color tolerance. Model predictions and
model-dependent point metrics use the declared solver/prediction tolerance;
independent QC/aggregation/statistical scalars retain their tighter tolerance.
Whole/block raster validation uses actual tile features, never patient probability
broadcasting. Golden data, expected values and tolerances are not regenerated.

Operations are intentionally staged:

1. `--prepare-inputs`: deterministic bytes only, fresh output. Completed inputs
   are under `$WORKDIR/benchmark-v2/learning-pathology-inputs-v1`.
2. `--freeze-environments`: validate immutable build receipt/wheel, actual package
   bytes/versions and numerical resources; produces a reviewable binding.
3. `--freeze-derived`: one untimed independent pathology-spec pass generates
   the common 12-patient feature input; no native expected values are recorded.
4. `--validate`: eight untimed public-output comparisons. No algorithm elapsed
   fields are emitted; keep original failures and repair only runner adapters.
5. `--run`: exact plan/binding/input hashes, fresh output, all 224 attempts.

The initial untimed validation retained three successful clinical/pathology rows
and five reference CSV-export adapter failures (`n` supplied twice to `dict`).
That adapter and CLI failure exit propagation were corrected without algorithm,
input, parameter or threshold changes. Only the five failed rows were rechecked;
all eight combined comparisons then passed. Existing frozen workflow final
predictions, outer selected supports and candidate choices were independently
checked for all three task references. The initial failure directory remains
`ml-path-validation-v1`; repaired evidence is
`ml-path-validation-v2-failed-cases` and `ml-path-validation-combined-v2.json`.

Numerical validation is distinct from speed. Missed runtime1.10× or memory1.20×
engineering targets stay visible per row. Process startup, owned-worker errors,
timeouts, memory budget, malformed protocol/comparison files and nonfinite
mismatches cannot silently disappear. Raw data retain warmups, actual execution
order, process/API/stage durations, full-process RSS and artifact hashes.
