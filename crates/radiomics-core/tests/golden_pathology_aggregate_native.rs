//! Acceptance authored independently of the aggregation/overlay implementation.
use ov_radiomics_core::contracts::{NumericBuffer, PredictionUnit, TileTable};
use ov_radiomics_core::overlay::{self, Canvas, OverlayConfig};
use ov_radiomics_core::pathology_aggregate::{
    self as agg, AggregateConfig, SummaryStatistic, WeightMode,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn fixtures() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(
            "../../../golden/python_golden_v1/pathology_aggregate_inputs.json"
        ))
        .unwrap(),
        serde_json::from_str(include_str!(
            "../../../golden/python_golden_v1/pathology_aggregate_reference.json"
        ))
        .unwrap(),
    )
}
fn table(mut tiles: Value, parents: Value) -> TileTable {
    for row in tiles.as_array_mut().unwrap() {
        let row = row.as_object_mut().unwrap();
        row.entry("source")
            .or_insert_with(|| json!("synthetic acceptance source"));
        row.entry("qc").or_insert_with(|| json!({}));
    }
    let parents: BTreeMap<String, String> = serde_json::from_value(parents).unwrap();
    TileTable {
        tiles: serde_json::from_value(tiles).unwrap(),
        features: None,
        feature_names: vec![],
        slide_status: parents
            .keys()
            .map(|s| (s.clone(), "supplied".into()))
            .collect(),
        slide_patients: parents,
    }
}
fn compare(actual: &Value, expected: &Value, path: &str) {
    match expected {
        Value::Object(expected) => {
            let actual = actual
                .as_object()
                .unwrap_or_else(|| panic!("{path}: expected object"));
            assert_eq!(actual.len(), expected.len(), "{path} key count");
            for (key, value) in expected {
                compare(&actual[key], value, &format!("{path}.{key}"));
            }
        }
        Value::Array(expected) => {
            let actual = actual
                .as_array()
                .unwrap_or_else(|| panic!("{path}: expected array"));
            assert_eq!(actual.len(), expected.len(), "{path} length");
            for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                compare(a, e, &format!("{path}[{i}]"));
            }
        }
        Value::Number(expected) => {
            let a = actual.as_f64().unwrap();
            let e = expected.as_f64().unwrap();
            let exact = path.contains("count")
                || path.contains("coverage")
                || path.contains("alpha")
                || path.contains("width")
                || path.contains("height")
                || path.contains("centroid")
                || path.contains("bbox");
            let tolerance = if exact { 0. } else { 1e-10 + 1e-8 * e.abs() };
            assert!((a - e).abs() <= tolerance, "{path}: {a} vs {e}");
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
fn aligned(input: &Value) -> TileTable {
    let bare = table(input["tiles"].clone(), input["slide_patients"].clone());
    let rows = input["embedding_values"].as_array().unwrap();
    let values: Vec<_> = rows
        .iter()
        .flat_map(|row| row.as_array().unwrap())
        .collect();
    let buffer = NumericBuffer {
        rows: rows.len(),
        cols: input["feature_names"].as_array().unwrap().len(),
        values: values.iter().map(|v| v.as_f64().unwrap_or(0.)).collect(),
        missing: values.iter().map(|v| v.is_null()).collect(),
    };
    agg::align_embeddings(
        &bare,
        &serde_json::from_value::<Vec<String>>(input["embedding_ids"].clone()).unwrap(),
        &buffer,
        &serde_json::from_value::<Vec<String>>(input["feature_names"].clone()).unwrap(),
    )
    .unwrap()
}

#[test]
fn independent_alignment_hierarchical_weights_missing_and_empty_parents() {
    let (input, reference) = fixtures();
    let source = aligned(&input);
    for (i, row) in reference["alignment"]["aligned_values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        for (j, value) in row.as_array().unwrap().iter().enumerate() {
            let buffer = source.features.as_ref().unwrap();
            assert_eq!(buffer.missing[i * 2 + j], value.is_null());
            assert_eq!(buffer.values[i * 2 + j], value.as_f64().unwrap_or(0.));
        }
    }
    for variant in input["aggregation_variants"].as_array().unwrap() {
        let config:AggregateConfig=serde_json::from_value(json!({"tile_weight":variant["tile_weight"],"patient_weight":variant["patient_weight"],"patient_stats":variant["patient_stats"]})).unwrap();
        let actual = agg::aggregate(&source, &config).unwrap();
        let id = variant["id"].as_str().unwrap();
        compare(
            &serde_json::to_value(&actual).unwrap(),
            &reference["aggregation"][id],
            id,
        );
        let (slide, patient) = actual.to_feature_tables().unwrap();
        slide.validate().unwrap();
        patient.validate().unwrap();
        assert_eq!(slide.data.cols, 12);
        assert_eq!(
            patient.data.cols,
            if id == "mean_only_patient" { 12 } else { 72 }
        );
        assert_eq!(
            actual,
            serde_json::from_slice(&serde_json::to_vec(&actual).unwrap()).unwrap()
        );
    }
    for case in input["summary_cases"].as_array().unwrap() {
        let values: Vec<Option<f64>> = serde_json::from_value(case["values"].clone()).unwrap();
        let weights: Option<Vec<f64>> = serde_json::from_value(case["weights"].clone()).unwrap();
        let ids = case
            .get("ids")
            .map(|x| serde_json::from_value::<Vec<String>>(x.clone()).unwrap())
            .unwrap_or_else(|| (0..values.len()).map(|n| format!("{n}")).collect());
        let actual = agg::summarize(&values, weights.as_deref(), &ids).unwrap();
        let id = case["id"].as_str().unwrap();
        compare(
            &serde_json::to_value(&actual).unwrap(),
            &reference["summaries"][id],
            id,
        );
        if id.starts_with("decimal_cdf_") {
            assert_eq!(actual.values[&SummaryStatistic::Q50], Some(2.));
        }
    }
    for name in [
        "all_qc_excluded_slide_variant",
        "all_qc_excluded_one_of_patient_slides",
    ] {
        let mut modified = source.clone();
        let slide = input[name]["slide_id"].as_str().unwrap();
        for tile in &mut modified.tiles {
            if tile.slide_id == slide {
                tile.exclusion_reasons.push("acceptance-excluded".into());
            }
        }
        compare(
            &serde_json::to_value(agg::aggregate(&modified, &AggregateConfig::default()).unwrap())
                .unwrap(),
            &reference["excluded_and_zero_tile_variants"][name],
            name,
        );
    }
    let mut zero = source.clone();
    let slide = input["zero_tile_slide_variant"]["slide_id"]
        .as_str()
        .unwrap();
    let patient = input["zero_tile_slide_variant"]["patient_id"]
        .as_str()
        .unwrap();
    zero.slide_patients.insert(slide.into(), patient.into());
    zero.slide_status.insert(slide.into(), "supplied".into());
    compare(
        &serde_json::to_value(agg::aggregate(&zero, &AggregateConfig::default()).unwrap()).unwrap(),
        &reference["excluded_and_zero_tile_variants"]["zero_tile_slide"],
        "zero tile",
    );
    zero.tiles.clear();
    zero.features = Some(NumericBuffer {
        rows: 0,
        cols: 2,
        values: vec![],
        missing: vec![],
    });
    compare(
        &serde_json::to_value(agg::aggregate(&zero, &AggregateConfig::default()).unwrap()).unwrap(),
        &reference["excluded_and_zero_tile_variants"]["entire_input_has_no_tile_rows"],
        "entire empty",
    );
    let mut no_mpp = source.clone();
    no_mpp.tiles[0].mpp_x_um = None;
    assert!(agg::aggregate(
        &no_mpp,
        &AggregateConfig {
            tile_weight: WeightMode::PhysicalArea,
            ..Default::default()
        }
    )
    .is_err());
    assert!(agg::aggregate(
        &source,
        &AggregateConfig {
            patient_weight: WeightMode::Level0Area,
            ..Default::default()
        }
    )
    .is_err());
    assert!(agg::summarize(&[Some(1.)], Some(&[-1.]), &["x".into()]).is_err());
    let mut reordered = input.clone();
    reordered["tiles"].as_array_mut().unwrap().reverse();
    assert_eq!(
        agg::aggregate(&source, &AggregateConfig::default()).unwrap(),
        agg::aggregate(&aligned(&reordered), &AggregateConfig::default()).unwrap()
    );
    let mut bare = source.clone();
    bare.features = None;
    bare.feature_names.clear();
    let buffer = source.features.as_ref().unwrap();
    let mut ids = source
        .tiles
        .iter()
        .map(|t| t.tile_id.clone())
        .collect::<Vec<_>>();
    ids[0] = "absent".into();
    assert!(agg::align_embeddings(&bare, &ids, buffer, &source.feature_names).is_err());
    ids[0] = ids[1].clone();
    assert!(agg::align_embeddings(&bare, &ids, buffer, &source.feature_names).is_err());
}

fn normalized_overlay(block: &overlay::OverlayBlock, with_offset: bool) -> Value {
    let mut value = serde_json::to_value(block).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("missing");
    if !with_offset {
        object.remove("x");
        object.remove("y");
    }
    value
}
#[test]
fn independent_overlay_coverage_half_open_global_blocks_and_global_objects() {
    let (input, reference) = fixtures();
    let mut source = aligned(&input);
    source.features = None;
    source.feature_names.clear();
    source.tiles.retain(|tile| tile.slide_id == "A");
    source.slide_status.retain(|slide, _| slide == "A");
    source.slide_patients.retain(|slide, _| slide == "A");
    let spec = &input["overlay"];
    let ids: Vec<String> = serde_json::from_value(spec["score_ids"].clone()).unwrap();
    let scores: Vec<f64> = serde_json::from_value(spec["scores"].clone()).unwrap();
    let canvas: Canvas = serde_json::from_value(spec["canvas"].clone()).unwrap();
    let actual =
        overlay::rasterize(&source, &ids, &scores, &canvas, &OverlayConfig::default()).unwrap();
    compare(
        &normalized_overlay(&actual, false),
        &reference["overlay"]["whole"],
        "whole",
    );
    let blocks = overlay::raster_blocks(
        &source,
        &ids,
        &scores,
        &canvas,
        serde_json::from_value(spec["block_size"].clone()).unwrap(),
        &OverlayConfig::default(),
    )
    .unwrap()
    .map(|block| normalized_overlay(&block.unwrap(), true))
    .collect::<Vec<_>>();
    compare(&json!(blocks), &reference["overlay"]["blocks"], "blocks");
    assert!(overlay::rasterize(
        &source,
        &ids,
        &scores,
        &canvas,
        &OverlayConfig {
            prediction_unit: PredictionUnit::Patient,
            clip: false
        }
    )
    .is_err());
    for (input_key, reference_key) in [
        ("offset_valid_rect_overlay", "offset_valid_rect"),
        ("fractional_boundary_overlay", "fractional_boundary"),
        ("fractional_boundary_exact_overlay", "fractional_exact_edge"),
    ] {
        let spec = &input[input_key];
        let table = table(json!([spec["tile"]]), json!({"A":"P1"}));
        let canvas: Canvas = serde_json::from_value(spec["canvas"].clone()).unwrap();
        let ids: Vec<String> = serde_json::from_value(spec["score_ids"].clone()).unwrap();
        let scores: Vec<f64> = serde_json::from_value(spec["scores"].clone()).unwrap();
        let actual =
            overlay::rasterize(&table, &ids, &scores, &canvas, &OverlayConfig::default()).unwrap();
        let expected = &reference["overlay"][reference_key];
        compare(
            &normalized_overlay(&actual, false),
            if expected.get("whole").is_some() {
                &expected["whole"]
            } else {
                expected
            },
            input_key,
        );
        if let Some(block_size) = spec.get("block_size") {
            let blocks = overlay::raster_blocks(
                &table,
                &ids,
                &scores,
                &canvas,
                serde_json::from_value(block_size.clone()).unwrap(),
                &OverlayConfig::default(),
            )
            .unwrap();
            let mut coverage = vec![0; canvas.width * canvas.height];
            for block in blocks {
                let block = block.unwrap();
                for y in 0..block.height {
                    for x in 0..block.width {
                        coverage[(block.y + y) * canvas.width + block.x + x] =
                            block.coverage[y * block.width + x];
                    }
                }
            }
            assert_eq!(
                coverage, actual.coverage,
                "global-index block sampling {input_key}"
            );
            assert_eq!(json!(coverage), spec["expected_coverage"]);
        }
    }
    let full = aligned(&input);
    let tile = full
        .tiles
        .iter()
        .find(|tile| tile.tile_id == input["object_context"]["tile_id"].as_str().unwrap())
        .unwrap();
    let objects = serde_json::from_value::<Vec<ov_radiomics_core::pathology_pixels::ObjectRecord>>(
        input["object_context"]["local_objects"].clone(),
    )
    .unwrap();
    compare(
        &serde_json::to_value(agg::global_object_context(tile, &objects).unwrap()).unwrap(),
        &reference["object_context"],
        "global0",
    );
    // A trillion-pixel canvas must remain lazy: obtaining its first 2x2 block
    // allocates no full-canvas value/coverage arrays.
    let canvas = Canvas {
        width: 1_000_000,
        height: 1_000_000,
        origin0: [0., 0.],
        pixel_size0: [1., 1.],
    };
    let first = overlay::raster_blocks(
        &source,
        &ids,
        &scores,
        &canvas,
        [2, 2],
        &OverlayConfig::default(),
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap();
    assert_eq!(first.values.len(), 4);
    assert_eq!(first.coverage.len(), 4);
}
