# Native learning CLI workflows

The `train` and `predict` command adapters call the same Rust table loader,
patient-grouped learning workflow and typed `.rad` loader as the Python boundary.
They do not import or invoke Python, sklearn or a subprocess analysis backend.
Command settings, input load reports and numerical provenance are exported as
separate artifacts.

## Train

```sh
radiomics train --data study.json --config training.json --output radiomics-models
radiomics train --data study.json --config training.json \
  --split-plan split_plan.json --output radiomics-models
```

`--data` is a schema-one `StudyTableSpec` JSON file. Its CSV/Parquet and auxiliary
paths are resolved relative to that specification, including a bare filename.
`--config` is a full native `TrainingConfig`: task, explicit class order,
fit/scoring units, threshold and finite candidate grids remain explicit.

Without a supplied plan, defaults are five outer folds, three inner folds, seed
zero and one outer worker. `--outer-cv`, `--inner-cv`, `--seed` and `--n-jobs`
make those choices explicit. A supplied plan is validated against the actual
study and used without regeneration; its seed/folds are saved as the effective
settings. Insufficient patient/class counts or an incompatible plan are errors,
not reasons to search for a favorable split.

The output contains:

- `split_plan.json`, full `training.json`, independent `load_report.json` and
  `command_config.json`.
- One `model-NNN.rad` per requested model, with actual model names mapped in
  `summary.json` rather than used as filesystem paths.
- Model-specific OOF and final prediction JSON/CSV, outer CV audit JSON and a
  compact CV CSV linking scores to their fitted training provenance.
- Best model filename, comparison metric and direction, class order and the
  declared fit/scoring units in `summary.json`.

Regression RMSE is minimized, while binary AUC and multiclass macro OvR AUC are
maximized. The CLI does not negate metrics or add a second selection routine.
Full fitted audit states and patient sets remain in the native training result.
Input file SHA values can include held-out data, so they remain in the separate
load report rather than being inserted into the fitted model. Changing held-out
features/labels therefore does not alter fitted `.rad` state.

## Predict

```sh
radiomics predict --model radiomics-models/model-000.rad \
  --data external-study.json --output radiomics-predictions
```

For a pipeline model, `--data` is a `StudyTableSpec`. An omitted outcome column
loads `Outcome::Unlabeled`; prediction does not fit or impute new training
statistics. Required features are aligned through the saved native pipeline.
The model's fitted prediction unit is retained; the command does not broadcast
a patient probability onto samples or tiles.

The manifest's explicit `model_kind` selects the pipeline or fusion typed
loader. Unknown model/schema kinds fail. For a fusion model, the input file is
schema one with a `components` map from exactly the fitted component names to
their study-spec paths. Each nested study resolves its own source paths relative
to that study spec. Duplicate, missing or extra component names are errors.
The core checks component identities, patient overlap and fitted provenance.

Prediction outputs are `predictions.json`, `predictions.csv`, separate load
reports and `prediction_metadata.json`, including task/class order, actual unit,
model threshold and a `fit_performed=false` marker. CSV probability columns are
named with their class labels and escaped by the CSV library. Regression values
and full sample/patient/slide/tile/cohort/fold/training-source identities are
retained. Unlabeled prediction is a normal successful state.

## Output and executable checks

Existing output directories require explicit `--overwrite`. Only fixed known
files for the current command are replaced; unrelated files remain untouched,
and directories are never recursively cleared. Inputs/configurations and output
destinations are checked before results are written. JSON/CSV publish through
same-directory temporary files. Actual model serialization uses the core typed
bundle implementation.

Integration tests execute the compiled CLI with PATH restricted to an empty
directory. All three tasks train, save and predict from CSV and Parquet, with
identical `.rad` state across formats and identical labeled/unlabeled predictions.
They also verify preserved supplied plans, scoped overwrite, user-note
preservation, unknown schema/kind errors, bare relative paths and fitted-state
invariance after held-out file changes. These are native subprocess tests, not
Python wrappers or mocked command calls.
