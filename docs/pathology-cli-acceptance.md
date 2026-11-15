# Native pathology CLI integration evidence

Scope: `patches`, `aggregate`, `overlay`; shared core algorithms and unchanged
`python_golden_v1` fixtures. This report concerns adapter/orchestration behavior,
not a second numerical implementation or complete project release acceptance.

## Frozen reference comparison

| Check | Reference | Rule | Result |
|---|---|---|---|
| External embedding row alignment and two-stage slide summaries | `pathology_aggregate_inputs.json`, `pathology_aggregate_reference.json` | IDs/order/missing states exact; existing scalar atol 1e-10, rtol 1e-8 | Passed |
| Overlap values and coverage, actual tile PredictionTable vs explicit tile scores | Same independent frozen overlay | Values/coverage/counts exact; patient broadcast error | Passed |
| Color configuration | Frozen `pathology_inputs.json` fixed edges | No re-recording or implicit bin choices; field mapping only | Passed adapter checks; underlying color Golden remains the common core's acceptance |
| Gray16 traditional texture | Hand input [1,2,3;4,5,6], explicit grayscale companion | First-order means 3.0 and 4.5, padding excluded; RGB without companion errors | Passed |
| Semantic mask | Hand [1,0,2;0,1,2], four/eight connectivity, explicit MPP | Patch-local counts [2,2,1] in four-connectivity; final padded label-2 denominator=2/fraction=1; 1-pixel area=1 µm² | Passed |

No Golden data/values/tolerances or public shared contracts were edited. Numeric
JSON integer-versus-f64 encoding is normalized only by typed canvas/numeric
comparison; IDs and exact counts retain exact comparison. Native feature
undefined states remain missing flags; no zeros are substituted as valid data.

## Engineering checks

Seven subprocess E2E tests pass with `PATH` pointing to an empty directory and
`PYTHONPATH` empty. They cover 32-patient patient-level model training/persistence/
prediction/evaluation, scoped overwrite, no patch-pixel accumulation or patch
file deletion, missing images, zero tiles, all-QC-excluded slides, retained parent
IDs, explicit grayscale inputs, mask semantics/denominators/units, strict external
embedding ID sets and true tile-level overlay provenance.

`cargo clippy -p ov-radiomics-cli --all-targets --locked -- -D warnings` passes.
Commands were run on local Windows using the current Rust stable toolchain.
The parent integrator registers the command in `main.rs` and
`workflow_commands/mod.rs`; this work package only commits its new module,
tests and documentation. Worker test runs used the corresponding temporary
registration, provided explicitly with the handoff.

## Limits and outstanding gates

- Ordinary decoded images occupy memory as one primary image plus optional
  explicitly co-gridded gray and label companions. Patch buffers and overlay
  blocks are streamed; selected scalar matrices and tile metadata remain in
  memory for aggregation.
- Masks read through the ordinary image reader are Gray8/Gray16 TIFF, not a
  vendor/WSI decoder or arbitrary UInt32 raster format.
- Caller-supplied primary/gray/mask grids must already align. Equal decoded
  dimensions are a necessary check for this explicit contract, not proof of
  upstream registration. Shared TileRecord geometry/identity is checked at each
  patch; no row-count-only embedding alignment is used.
- Each failed image retains a slide/patient result. A late failure invalidates
  its earlier scalar rows; raw QC JSONL remains a stage measurement and the final
  TileTable/slide status controls eligibility. A partially written run lacking a
  final overlay manifest is incomplete.
- Overlapping instances remain patch-local, with no whole-slide cell total.
  Hierarchical summaries of patch counts describe patch count distributions.
- No benchmark speedup, memory target, Linux install, WSIData runtime or clinical
  effectiveness claim is made. Those remain separate project gates/targets.
