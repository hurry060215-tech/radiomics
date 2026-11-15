use super::*;
fn fixtures() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/wsidata_contract_inputs.json"
        )))
        .unwrap(),
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/wsidata_contract_reference.json"
        )))
        .unwrap(),
    )
}
fn merge(target: &mut Value, source: &Value) {
    for (k, v) in source.as_object().unwrap() {
        target[k] = v.clone();
    }
}
fn strings(v: &Value) -> Vec<String> {
    serde_json::from_value(v.clone()).unwrap()
}
#[test]
fn frozen_snapshot_bytes_float_hex_and_normalized_geometry() {
    let (inputs, reference) = fixtures();
    let payload = reference["snapshot_payload"].clone();
    let current = snapshot(&payload).unwrap();
    assert_eq!(
        serde_json::to_value(&current).unwrap(),
        reference["current_snapshot"]
    );
    for case in reference["canonical_float_cases"].as_array().unwrap() {
        let value = case["value"].as_f64().unwrap();
        assert_eq!(
            float_hex(value).unwrap(),
            case["python_float_hex"].as_str().unwrap()
        );
        assert_eq!(canonical(&case["value"]).unwrap(), case["canonical"]);
    }
    for (key, input_key) in [
        ("positive_scale2", "explicit_transform_variant"),
        ("translation_point1", "translation_variant"),
    ] {
        let mut value = payload.clone();
        merge(&mut value, &inputs[input_key]);
        assert_eq!(
            serde_json::to_value(snapshot(&value).unwrap()).unwrap(),
            reference["variants"][key]
        );
    }
    let mut edge = payload.clone();
    edge["shape_rows"] = inputs["edge_validity_variant"]["shape_rows"].clone();
    assert_eq!(
        serde_json::to_value(snapshot(&edge).unwrap()).unwrap(),
        reference["variants"]["explicit_valid_edge"]
    );
    let mut reversed = payload;
    reversed["shape_rows"].as_array_mut().unwrap().reverse();
    assert_eq!(
        serde_json::to_value(snapshot(&reversed).unwrap()).unwrap(),
        reference["variants"]["reversed_shape_order"]["snapshot"]
    );
}
#[test]
fn native_authoritative_rows_namespace_and_embedding_values() {
    let (inputs, reference) = fixtures();
    let payload = reference["snapshot_payload"].clone();
    let current = snapshot(&payload).unwrap();
    let shape_ids: Vec<_> = payload["shape_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["tile_id"].as_str().unwrap().to_string())
        .collect();
    let ids = strings(&inputs["obs_tile_ids"]);
    let row_ids = strings(&inputs["obs_row_ids"]);
    let order = validate_embedding_link(
        &current,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &row_ids,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(order).unwrap(),
        reference["expected_row_order"]
    );
    let feature_values = inputs["embedding_values"].as_array().unwrap();
    let cols = feature_values[0].as_array().unwrap().len();
    let buffer = NumericBuffer {
        rows: feature_values.len(),
        cols,
        values: feature_values
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
        missing: vec![false; feature_values.len() * cols],
    };
    let input = WsiInput {
        geometry: payload,
        saved_at_generation: Some(current),
        table_region: json!("tiles"),
        table_region_key: "library_id".into(),
        table_instance_key: "tile_id".into(),
        obs_regions: vec!["tiles".into(); ids.len()],
        obs_tile_ids: ids,
        obs_row_ids: row_ids,
        patient_id: "P".into(),
        slide_id: "Slideα".into(),
        feature_key: "embedding_tiles".into(),
        feature_names: (0..cols).map(|i| format!("embedding_{i}")).collect(),
    };
    let result = adapt(&input, &buffer).unwrap();
    for (tile, raw) in result.table.tiles.iter().zip(&shape_ids) {
        assert_eq!(
            tile.tile_id,
            serde_json::to_string(&("Slideα", raw)).unwrap()
        );
    }
    assert_eq!(
        result.table.features.unwrap().values,
        reference["aligned_feature_values"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        result.provenance["source_row_ids_in_shape_order"],
        reference["source_row_ids_in_shape_order"]
    );
}
#[test]
fn malformed_geometry_levels_padding_and_history_are_rejected() {
    let (inputs, reference) = fixtures();
    let base = reference["snapshot_payload"].clone();
    let current = snapshot(&base).unwrap();
    let shape_ids = vec!["T0".into(), "T1".into()];
    let ids = strings(&inputs["obs_tile_ids"]);
    let rows = strings(&inputs["obs_row_ids"]);
    assert!(
        validate_embedding_link(&current, None, "tiles", "tiles", &shape_ids, &ids, &rows).is_err()
    );
    assert!(validate_embedding_link(
        &current,
        Some(&current),
        "wrong",
        "tiles",
        &shape_ids,
        &ids,
        &rows
    )
    .is_err());
    assert!(validate_embedding_link(
        &current,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &["T0".into(), "T0".into()],
        &rows
    )
    .is_err());
    assert!(validate_embedding_link(
        &current,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &["same".into(), "same".into()]
    )
    .is_err());
    assert!(validate_embedding_link(
        &current,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &["T1".into(), "absent".into()],
        &rows
    )
    .is_err());
    let mut saved = current.clone();
    saved.snapshot["source"] = json!("changed");
    assert!(validate_embedding_link(
        &current,
        Some(&saved),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &rows
    )
    .is_err());
    assert!(validate_embedding_link(
        &saved,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &rows
    )
    .is_err());
    saved = current.clone();
    saved.snapshot["schema"] = json!("unknown");
    assert!(validate_embedding_link(
        &current,
        Some(&saved),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &rows
    )
    .is_err());
    let mut geometry = base.clone();
    for row in geometry["shape_rows"].as_array_mut().unwrap() {
        for p in row["geometry_corners"].as_array_mut().unwrap() {
            p[0] = json!(p[0].as_f64().unwrap() + 4.);
        }
    }
    let changed = snapshot(&geometry).unwrap();
    assert!(validate_embedding_link(
        &changed,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &rows
    )
    .is_err());
    let mut spec = base.clone();
    spec["tile_spec"]["stride_width"] = json!(1);
    let changed = snapshot(&spec).unwrap();
    assert!(validate_embedding_link(
        &changed,
        Some(&current),
        "tiles",
        "tiles",
        &shape_ids,
        &ids,
        &rows
    )
    .is_err());
    let mut cases = vec![];
    let mut value = base.clone();
    value["shape_rows"][1]["tile_id"] = json!("T0");
    cases.push(value);
    let mut value = base.clone();
    value["shape_rows"][0]["geometry_corners"] = json!([[0, 2], [2, 0], [4, 2], [2, 4], [0, 2]]);
    cases.push(value);
    let mut value = base.clone();
    value["shape_to_level0"][0][1] = json!(0.1);
    cases.push(value);
    let mut value = base.clone();
    value["slide_shape_wh"] = Value::Null;
    cases.push(value);
    let mut value = base.clone();
    value["shape_rows"] = inputs["edge_validity_variant"]["shape_rows"].clone();
    value["shape_rows"][0]["valid_pixels"] = json!([0, 0, 2, 2]);
    cases.push(value);
    for level in [json!(1), json!(false)] {
        let mut value = base.clone();
        value["tile_spec"]["base_level"] = level;
        cases.push(value);
    }
    let mut value = base.clone();
    value["shape_rows"][0]["geometry_corners"] = json!([[0, 0], [5, 0], [5, 4], [0, 4], [0, 0]]);
    cases.push(value);
    let mut value = base.clone();
    value["tile_spec"]["width"] = json!(3);
    value["tile_spec"]["height"] = json!(4);
    value["tile_spec"]["base_downsample"] = json!(1);
    value["slide_shape_wh"] = json!([10000000000000008u64, 20]);
    value["shape_rows"] = json!([{"tile_id":"huge","geometry_corners":[[10000000000000000u64,0],[10000000000000004u64,0],[10000000000000004u64,4],[10000000000000000u64,4],[10000000000000000u64,0]]}]);
    cases.push(value);
    let mut value = base.clone();
    value["source"] = json!("");
    cases.push(value);
    for (i, value) in cases.iter().enumerate() {
        assert!(snapshot(value).is_err(), "accepted malformed case{i}");
    }
    let mut value = base;
    value["shape_to_level0"][0][2] = json!(9007199254740993u64);
    assert!(snapshot(&value).is_err());
    assert!(real(&json!(u64::MAX)).is_err());
}

#[test]
fn independent_frozen_float_tag_collision_requires_typed_snapshot_guard() {
    let reference: Value = serde_json::from_str(include_str!(concat!(
        env!("RAD_GOLDEN_DIR"),
        "/python_golden_v1/wsidata_collision_reference.json"
    )))
    .unwrap();
    let saved = snapshot(&reference["saved_generation_input"]).unwrap();
    let current = snapshot(&reference["current_input"]).unwrap();
    assert_eq!(
        serde_json::to_value(&saved).unwrap(),
        reference["saved_snapshot"]
    );
    assert_eq!(
        serde_json::to_value(&current).unwrap(),
        reference["current_snapshot"]
    );
    assert_eq!(saved.sha256, current.sha256);
    assert_ne!(saved.snapshot, current.snapshot);
    assert!(validate_embedding_link(
        &current,
        Some(&saved),
        "tiles",
        "tiles",
        &["T0".into(), "T1".into()],
        &["T1".into(), "T0".into()],
        &["rowB".into(), "rowA".into()]
    )
    .is_err());
}
