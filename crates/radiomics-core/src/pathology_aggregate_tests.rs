use super::*;
use crate::contracts::{PredictionRow, PredictionTable, PredictionUnit, Task};
use crate::overlay::{self, Canvas, OverlayConfig};
use serde_json::{json, Value};
fn fixtures() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/pathology_aggregate_inputs.json"
        )))
        .unwrap(),
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/pathology_aggregate_reference.json"
        )))
        .unwrap(),
    )
}
fn table_from(tiles: &Value, parents: &Value) -> TileTable {
    let mut tiles = tiles.clone();
    for tile in tiles.as_array_mut().unwrap() {
        let row = tile.as_object_mut().unwrap();
        row.entry("source")
            .or_insert_with(|| json!("synthetic_fixture"));
        row.entry("qc").or_insert_with(|| json!({}));
    }
    let slide_patients: BTreeMap<String, String> = serde_json::from_value(parents.clone()).unwrap();
    TileTable {
        tiles: serde_json::from_value(tiles).unwrap(),
        features: None,
        feature_names: vec![],
        slide_status: slide_patients
            .keys()
            .map(|s| (s.clone(), "provided".into()))
            .collect(),
        slide_patients,
    }
}
fn numeric(rows: &Value, cols: usize) -> NumericBuffer {
    let mut values = vec![];
    let mut missing = vec![];
    for row in rows.as_array().unwrap() {
        for value in row.as_array().unwrap() {
            values.push(value.as_f64().unwrap_or(0.));
            missing.push(value.is_null());
        }
    }
    NumericBuffer {
        rows: rows.as_array().unwrap().len(),
        cols,
        values,
        missing,
    }
}
fn aligned(input: &Value) -> TileTable {
    let table = table_from(&input["tiles"], &input["slide_patients"]);
    let names: Vec<String> = serde_json::from_value(input["feature_names"].clone()).unwrap();
    align_embeddings(
        &table,
        &serde_json::from_value::<Vec<String>>(input["embedding_ids"].clone()).unwrap(),
        &numeric(&input["embedding_values"], names.len()),
        &names,
    )
    .unwrap()
}
fn compare(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "keys {path}"
            );
            for (k, v) in b {
                compare(&a[k], v, &format!("{path}.{k}"));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "len {path}");
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let exact = path.contains("count")
                || path.contains("coverage")
                || path.contains("alpha")
                || path.contains("bbox")
                || path.contains("centroid")
                || path.contains("origin0")
                || path.contains("pixel_size0")
                || path.contains("width")
                || path.contains("height");
            let (atol, rtol) = if exact { (0., 0.) } else { (1e-10, 1e-8) };
            assert!((a - b).abs() <= atol + rtol * b.abs(), "{path}: {a} != {b}");
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}
fn reference_overlay(block: &overlay::OverlayBlock, include_offset: bool) -> Value {
    let mut result = serde_json::to_value(block).unwrap();
    let m = result.as_object_mut().unwrap();
    m.remove("missing");
    if !include_offset {
        m.remove("x");
        m.remove("y");
    }
    result
}
fn subset(table: &TileTable, slide: &str) -> TileTable {
    let mut subset = table.clone();
    subset.tiles.retain(|t| t.slide_id == slide);
    subset.features = None;
    subset.feature_names.clear();
    subset.slide_status.retain(|s, _| s == slide);
    subset.slide_patients.retain(|s, _| s == slide);
    subset
}

#[test]
fn frozen_alignment_two_stage_summaries_and_feature_orders() {
    let (input, reference) = fixtures();
    let table = aligned(&input);
    let data = table.features.as_ref().unwrap();
    for (i, row) in reference["alignment"]["aligned_values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        for (j, value) in row.as_array().unwrap().iter().enumerate() {
            assert_eq!(data.missing[i * data.cols + j], value.is_null());
            assert_eq!(data.values[i * data.cols + j], value.as_f64().unwrap_or(0.));
        }
    }
    for variant in input["aggregation_variants"].as_array().unwrap() {
        let config:AggregateConfig=serde_json::from_value(json!({"tile_weight":variant["tile_weight"],"patient_weight":variant["patient_weight"],"patient_stats":variant["patient_stats"]})).unwrap();
        let result = aggregate(&table, &config).unwrap();
        compare(
            &serde_json::to_value(&result).unwrap(),
            &reference["aggregation"][variant["id"].as_str().unwrap()],
            variant["id"].as_str().unwrap(),
        );
        let (slides, patients) = result.to_feature_tables().unwrap();
        assert_eq!(slides.data.cols, 12);
        assert_eq!(
            patients.data.cols,
            if config.patient_stats.len() == 1 {
                12
            } else {
                72
            }
        );
        assert_eq!(patients.samples[0].sample_id, "P1");
    }
    assert!(align_embeddings(
        &table,
        &[],
        &NumericBuffer {
            rows: 0,
            cols: 1,
            values: vec![],
            missing: vec![]
        },
        &["x".into()]
    )
    .is_err());
}
#[test]
fn frozen_excluded_slides_zero_tiles_and_missing_rows_keep_parents() {
    let (input, reference) = fixtures();
    for key in [
        "all_qc_excluded_slide_variant",
        "all_qc_excluded_one_of_patient_slides",
    ] {
        let mut table = aligned(&input);
        for tile in &mut table.tiles {
            if tile.slide_id == input[key]["slide_id"].as_str().unwrap() {
                tile.exclusion_reasons
                    .push(input[key]["reason"].as_str().unwrap().into());
            }
        }
        let result = aggregate(&table, &AggregateConfig::default()).unwrap();
        compare(
            &serde_json::to_value(result).unwrap(),
            &reference["excluded_and_zero_tile_variants"][key],
            key,
        );
    }
    let mut table = aligned(&input);
    table.slide_patients.insert("D".into(), "P3".into());
    table.slide_status.insert("D".into(), "no_tiles".into());
    let result = aggregate(&table, &AggregateConfig::default()).unwrap();
    compare(
        &serde_json::to_value(&result).unwrap(),
        &reference["excluded_and_zero_tile_variants"]["zero_tile_slide"],
        "zero_tile_slide",
    );
    let (_, patients) = result.to_feature_tables().unwrap();
    assert!(patients.data.missing[2 * patients.data.cols..]
        .iter()
        .all(|v| *v));
    assert!(patients.data.values[2 * patients.data.cols..]
        .iter()
        .all(|v| *v == 0.));
    table.tiles.clear();
    table.features = Some(NumericBuffer {
        rows: 0,
        cols: 2,
        values: vec![],
        missing: vec![],
    });
    let result = aggregate(&table, &AggregateConfig::default()).unwrap();
    compare(
        &serde_json::to_value(result).unwrap(),
        &reference["excluded_and_zero_tile_variants"]["entire_input_has_no_tile_rows"],
        "no_rows",
    );
}
#[test]
fn frozen_weighted_inverse_cdf_decimal_ties_and_missing_zero_weights() {
    let (input, reference) = fixtures();
    for case in input["summary_cases"].as_array().unwrap() {
        let values: Vec<Option<f64>> = serde_json::from_value(case["values"].clone()).unwrap();
        let weights: Option<Vec<f64>> = serde_json::from_value(case["weights"].clone()).unwrap();
        let ids: Vec<String> = if case.get("ids").is_some() {
            serde_json::from_value(case["ids"].clone()).unwrap()
        } else {
            (0..values.len()).map(|i| format!("row{i:012}")).collect()
        };
        let result = summarize(&values, weights.as_deref(), &ids).unwrap();
        compare(
            &serde_json::to_value(result).unwrap(),
            &reference["summaries"][case["id"].as_str().unwrap()],
            case["id"].as_str().unwrap(),
        );
    }
    assert!(summarize(
        &[Some(1.), Some(3.)],
        Some(&[-1., 2.]),
        &["a".into(), "b".into()]
    )
    .is_err());
}
#[test]
fn frozen_overlaps_valid_padding_clipping_and_exact_fractional_blocks() {
    let (input, reference) = fixtures();
    let table = aligned(&input);
    let case = &input["overlay"];
    let slide = subset(&table, "A");
    let ids: Vec<String> = serde_json::from_value(case["score_ids"].clone()).unwrap();
    let scores: Vec<f64> = serde_json::from_value(case["scores"].clone()).unwrap();
    let canvas: Canvas = serde_json::from_value(case["canvas"].clone()).unwrap();
    let config = OverlayConfig::default();
    let whole = overlay::rasterize(&slide, &ids, &scores, &canvas, &config).unwrap();
    compare(
        &reference_overlay(&whole, false),
        &reference["overlay"]["whole"],
        "whole",
    );
    let blocks = overlay::raster_blocks(&slide, &ids, &scores, &canvas, [2, 2], &config)
        .unwrap()
        .collect::<RadResult<Vec<_>>>()
        .unwrap();
    compare(
        &Value::Array(blocks.iter().map(|b| reference_overlay(b, true)).collect()),
        &reference["overlay"]["blocks"],
        "blocks",
    );
    let clipped = Canvas {
        width: 2,
        height: 1,
        ..canvas.clone()
    };
    assert!(overlay::rasterize(&slide, &ids, &scores, &clipped, &config).is_err());
    let clip = OverlayConfig {
        clip: true,
        ..config.clone()
    };
    compare(
        &reference_overlay(
            &overlay::rasterize(&slide, &ids, &scores, &clipped, &clip).unwrap(),
            false,
        ),
        &reference["overlay"]["explicit_clip"],
        "clipped",
    );
    for (key, reference_key) in [
        ("offset_valid_rect_overlay", "offset_valid_rect"),
        ("fractional_boundary_overlay", "fractional_boundary"),
        ("fractional_boundary_exact_overlay", "fractional_exact_edge"),
    ] {
        let case = &input[key];
        let tile = table_from(&json!([case["tile"]]), &json!({"A":"P1"}));
        let ids: Vec<String> = serde_json::from_value(case["score_ids"].clone()).unwrap();
        let scores: Vec<f64> = serde_json::from_value(case["scores"].clone()).unwrap();
        let canvas: Canvas = serde_json::from_value(case["canvas"].clone()).unwrap();
        let whole = overlay::rasterize(&tile, &ids, &scores, &canvas, &config).unwrap();
        if key == "offset_valid_rect_overlay" {
            compare(
                &reference_overlay(&whole, false),
                &reference["overlay"][reference_key],
                key,
            );
        } else {
            compare(
                &reference_overlay(&whole, false),
                &reference["overlay"][reference_key]["whole"],
                key,
            );
            let bs: [usize; 2] = serde_json::from_value(case["block_size"].clone()).unwrap();
            let blocks = overlay::raster_blocks(&tile, &ids, &scores, &canvas, bs, &config)
                .unwrap()
                .collect::<RadResult<Vec<_>>>()
                .unwrap();
            compare(
                &Value::Array(blocks.iter().map(|b| reference_overlay(b, true)).collect()),
                &reference["overlay"][reference_key]["blocks"],
                key,
            );
            let rebuilt: Vec<_> = blocks
                .iter()
                .flat_map(|b| b.coverage.iter().copied())
                .collect();
            assert_eq!(rebuilt, whole.coverage);
        }
    }
}
#[test]
fn frozen_global_context_keeps_local_fields_and_explicit_spaces() {
    let (input, reference) = fixtures();
    let table = aligned(&input);
    let case = &input["object_context"];
    let tile = table
        .tiles
        .iter()
        .find(|t| t.tile_id == case["tile_id"].as_str().unwrap())
        .unwrap();
    let local: Vec<ObjectRecord> = serde_json::from_value(case["local_objects"].clone()).unwrap();
    let result = global_object_context(tile, &local).unwrap();
    compare(
        &serde_json::to_value(&result).unwrap(),
        &reference["object_context"],
        "object_context",
    );
    assert_eq!(result.objects[0].local, local[0]);
}
#[test]
fn identity_scale_errors_and_patient_prediction_broadcast_are_rejected() {
    let (input, _) = fixtures();
    let original = table_from(&input["tiles"], &input["slide_patients"]);
    let buffer = numeric(&input["embedding_values"], 2);
    let mut ids: Vec<String> = serde_json::from_value(input["embedding_ids"].clone()).unwrap();
    let names: Vec<String> = serde_json::from_value(input["feature_names"].clone()).unwrap();
    ids[0] = "absent".into();
    assert!(align_embeddings(&original, &ids, &buffer, &names).is_err());
    ids[0] = ids[1].clone();
    assert!(align_embeddings(&original, &ids, &buffer, &names).is_err());
    let mut table = aligned(&input);
    table.tiles[0].mpp_x_um = None;
    assert!(aggregate(
        &table,
        &AggregateConfig {
            tile_weight: WeightMode::PhysicalArea,
            ..AggregateConfig::default()
        }
    )
    .is_err());
    assert!(aggregate(
        &table,
        &AggregateConfig {
            patient_weight: WeightMode::Level0Area,
            ..AggregateConfig::default()
        }
    )
    .is_err());
    let predictions = PredictionTable {
        task: Task::Binary,
        unit: PredictionUnit::Patient,
        classes: vec!["negative".into(), "positive".into()],
        rows: vec![PredictionRow {
            sample_id: "P1".into(),
            patient_id: "P1".into(),
            slide_id: None,
            tile_id: None,
            cohort: "external".into(),
            fold: None,
            training_source: "fitted_model".into(),
            probabilities: vec![0.2, 0.8],
            value: None,
        }],
    };
    let canvas = Canvas {
        width: 5,
        height: 3,
        origin0: [0., 0.],
        pixel_size0: [1., 1.],
    };
    assert!(overlay::rasterize_predictions(
        &table,
        &predictions,
        "A",
        Some("positive"),
        &canvas,
        [2, 2],
        false
    )
    .is_err());
}
#[test]
fn lazy_overlay_allocates_only_block_and_feature_path_equals_actual_tile_predictions() {
    let (input, _) = fixtures();
    let table = aligned(&input);
    let canvas = Canvas {
        width: 1_000_000_000,
        height: 1,
        origin0: [0., 0.],
        pixel_size0: [1., 3.],
    };
    let mut blocks =
        overlay::rasterize_features(&table, "A", "embedding_0", &canvas, [2, 1], true).unwrap();
    let first = blocks.next().unwrap().unwrap();
    assert_eq!(first.values.len(), 2);
    assert_eq!(first.coverage, vec![1, 1]);
    let canvas = Canvas {
        width: 5,
        height: 3,
        origin0: [0., 0.],
        pixel_size0: [1., 1.],
    };
    let rows = table
        .tiles
        .iter()
        .filter(|t| t.slide_id == "A" && t.exclusion_reasons.is_empty())
        .map(|t| PredictionRow {
            sample_id: t.tile_id.clone(),
            patient_id: t.patient_id.clone(),
            slide_id: Some(t.slide_id.clone()),
            tile_id: Some(t.tile_id.clone()),
            cohort: "external".into(),
            fold: None,
            training_source: "external_tile_model".into(),
            probabilities: vec![],
            value: Some(if t.tile_id == "A1" { 1. } else { 3. }),
        })
        .collect();
    let predictions = PredictionTable {
        task: Task::Regression,
        unit: PredictionUnit::Tile,
        classes: vec![],
        rows,
    };
    let actual =
        overlay::rasterize_predictions(&table, &predictions, "A", None, &canvas, [2, 2], false)
            .unwrap()
            .collect::<RadResult<Vec<_>>>()
            .unwrap();
    let features = overlay::rasterize_features(&table, "A", "embedding_0", &canvas, [2, 2], false)
        .unwrap()
        .collect::<RadResult<Vec<_>>>()
        .unwrap();
    for (a, b) in actual.iter().zip(&features) {
        assert_eq!(a.coverage, b.coverage);
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!(x == y || (x.is_nan() && y.is_nan()));
        }
    }
}
