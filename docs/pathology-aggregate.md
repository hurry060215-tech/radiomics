# Native tile identity, hierarchy summaries and local overlays

The `pathology_aggregate` and `overlay` Rust modules consume the shared `TileTable`
and `PredictionTable` contracts. They do not read outcomes, fit a transformation,
or execute Python. They use the independently frozen
`python_golden_v1/pathology_aggregate_spec.py` definition. Optional WSIData
adaptation, artifact export and public Python/CLI availability require their own
integration checks; they are not established by these core unit tests.

## Identity and missing values

`align_embeddings(table, ids, buffer, feature_names)` validates nonempty unique
string IDs, exact tile-ID sets, feature names and buffer dimensions. It reorders
the new numeric matrix into the existing tile order. Equal row counts are never
evidence of alignment. It returns a new `TileTable`, preserving all original tile
metadata, QC reasons, slide status and patient parents. A source table that already
has features is an explicit conflict; `replace_embeddings` is the separately named
operation for replacing them in a returned copy.

Both inputs and outputs use the shared `NumericBuffer`: finite stored values,
one missing flag per element, and a canonical stored value of zero when missing.
Missing is not a measured zero. Empty tile sets can retain a known feature width
and `slide_patients`, so zero-tile slides never lose their patient identity.

## Two explicit aggregation stages

`aggregate(&TileTable, &AggregateConfig)` first summarizes accepted tiles within
each slide. A tile is accepted when its explicit `exclusion_reasons` is empty.
All original tile rows remain in the input and are not deleted. The six summaries
are mean, population standard deviation, maximum and q25/q50/q75.

The second stage summarizes these slide-level features within each patient. Every
accepted slide has equal weight by default; tile counts do not change its patient
weight. With the default six statistics at both stages, two source dimensions yield
12 slide columns and 72 patient columns. For example,
`patient_mean(slide_tile_std(embedding_0))` means the mean of slide-specific tile
standard deviations; it is not the standard deviation of all pooled patient tiles.
`patient_stats=[mean]` is an explicit option for a smaller feature set.

Feature order is declared in `slide_feature_names` and `patient_feature_names`:
source feature order, then the frozen six slide summaries, then configured patient
summary order. Group rows are sorted by slide/patient ID. Reductions use tile-ID
order and slide-ID order, so incoming row reordering does not change grouping.

Each output feature separately records the number of observed contributions and
its weight sum. Missing values do not contribute a value or weight. A slide with
no accepted tiles has `no_accepted_tiles`; a patient with no accepted slides has
`no_accepted_slides`. Their features remain missing, not all-zero measurements.
The explicit `TileTable.slide_patients` mapping preserves entirely zero-tile slides.
An accepted slide can still have an all-missing individual feature; that feature
retains count zero and missing summaries.

`HierarchyAggregate::to_feature_tables()` produces shared slide and patient
`FeatureTable` objects for the native modeling workflow. It preserves hierarchy,
ordered feature definitions, missing flags and empty-result quality records.
It does not silently remove empty-result samples or assign them invented targets.

## Weights, units and quantiles

Tile weights are explicit `equal`, `level0_area` or `physical_area`. The valid area
is computed in this fixed order:

```text
sx = width0 / patch_width
sy = height0 / patch_height
area_level0 = (sx * valid_width) * (sy * valid_height)
area_um2 = (area_level0 * mpp_x_um) * mpp_y_um
```

Padding never contributes area. Level-0 area is measured in squared level-0 pixels.
Physical area requires both recorded positive finite MPP axes. No scale is guessed
from magnification or DPI. Patient-level area weighting only accepts physical area
with complete MPP; differently resolved slides' pixel areas are not compared as
if they were physical units.

Weighted mean and population variance use normalized positive weights. Zero
weights are ignored; a summary with no positive observed weight is explicitly
empty. Overlapping patches contribute their valid patch areas separately. This is
`sum_of_valid_patch_contributions_not_deduplicated_slide_area`, not unique tissue
area and not a whole-slide cell count.

Unweighted quantiles use the shared linear interpolation primitive. Weighted
quantiles use an explicitly different inverted-CDF definition: sort by `(value,
row ID)`, accumulate the total and each prefix sequentially in `f64` in the same
sorted order, and select the first value satisfying `prefix >= q * total`.
There is no threshold epsilon and no assumption that equally weighted inverted
CDF equals unweighted linear interpolation. The frozen decimal-weight tests
check boundary behavior before native implementation.

Generic aggregation treats explicitly supplied numeric columns as linear values.
It does not reinterpret them based on column names. The default H feature producer
must therefore omit raw circular `Hue.mean_degrees` from ordinary model/aggregate
columns; Hue resultant length and circular variance can be numeric features.
The angle can remain in per-tile display/QC data. Adding directional sin/cos
features requires its own explicit specification and reference fixtures.

## Real tile overlays and lazy blocks

`raster_blocks(table, score_ids, scores, canvas, block_size, config)` requires
actual finite tile-level scores and an exact bijection with accepted tile IDs for
one slide. `rasterize` is a convenience operation that requests the whole canvas
as one block. `rasterize_features` selects a named aligned feature; a missing value
on an accepted tile is an error rather than a silent omission. `rasterize_predictions`
requires `PredictionUnit::Tile`, validates the tile/slide/patient hierarchy,
and requires an explicit class for classification or the actual regression value.
Unknown, excluded or mismatched prediction tiles are rejected. Selecting one
slide does not silently validate unrelated unknown prediction IDs.

**Patient or slide model probabilities cannot be broadcast into a local tile
probability map.** A genuine tile-level prediction or tile feature is required.

The canvas has explicit dimensions, level-0 origin and positive per-axis level-0
pixel size. The valid tile rectangle is half open on its right and bottom edges.
Every output pixel samples its original global center:

```text
x0 = original_canvas_origin_x + (global_pixel_index_x + 0.5) * pixel_size0_x
y0 = original_canvas_origin_y + (global_pixel_index_y + 0.5) * pixel_size0_y
```

All matching tile scores contribute to the arithmetic mean; coverage is their
exact `u64` count. Uncovered values are NaN with `missing=true` and alpha zero.
Covered values have `missing=false` and alpha 255. JSON uses the shared explicit
float-state serialization, preserving NaN as a state rather than converting it
to a measured zero. Python conversion can expose the same values as actual NaNs.

The iterator validates metadata once, stores tile bounds/scores, and allocates only
the requested block. It does not first allocate the entire canvas. Block centers
are never recomputed by adding a local offset to the rounded block origin. The
frozen `0.1`-pixel and `nextafter(0.65,+Inf)` cases check that even these boundary
coverage decisions are exactly identical between whole and blocked output.
Negative tile coordinates fail the shared contract. Tiles outside the requested
canvas require an explicit `clip=true`; clipping does not invent padded coverage.

The present evaluator scans contributing tile bounds per output pixel. Its memory
use is bounded by tile metadata plus one block, but the implementation makes no
claim of spatial-index acceleration. Performance belongs to the independent
Pathology suite under Benchmark v2, with measured raw repetitions and dispersion.

## Local object records and level-0 context

`global_object_context(tile, local_objects)` retains each H `ObjectRecord`
unchanged under `local`, explicitly labels it `patch_pixel_edges`, and adds
`coordinate_space="level0_pixel_edges"` with:

```text
centroid0 = (x0 + local_x * width0/patch_width,
             y0 + local_y * height0/patch_height)
bbox0 = (x0 + left * width0/patch_width,
         y0 + top * height0/patch_height,
         width * width0/patch_width,
         height * height0/patch_height)
```

Local objects must have unique nonzero IDs, nonempty area and geometry within the
tile's valid rectangle. This context mapping does not deduplicate overlapping
instances and does not reinterpret patch-local counts as global cell totals.

## Validation evidence

Seven native tests directly consume A's frozen input/reference fixtures. They
cover alignment, all default and weighted hierarchy outputs and column orders,
per-feature missing denominators, all-QC-excluded and entirely zero-tile inputs,
decimal/tied weighted quantiles, physical-area errors, global/local object fields,
patient probability rejection and exact overlay coverage for overlaps, padding,
clipping and floating-point boundaries. A billion-pixel-wide canvas test reads
only a two-pixel block, checking the lazy API without whole-image allocation.
Feature-driven and genuine regression-tile prediction overlays are compared.

No WSI decoder, model training, automatic segmentation, inferred MPP, hidden
category-name behavior, upstream embedding-cache trust bypass, or performance
claim is introduced by these modules.
