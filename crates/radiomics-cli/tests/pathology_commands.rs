use ov_radiomics_core::{contracts::*, pathology_pixels, table_io};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "radiomics-pathology-cli-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        fs::create_dir(path.join("empty-bin")).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn json(&self, name: &str, value: &Value) {
        fs::write(self.path(name), serde_json::to_vec(value).unwrap()).unwrap();
    }
    fn read(&self, name: &str) -> Value {
        serde_json::from_slice(&fs::read(self.path(name)).unwrap()).unwrap()
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_radiomics"))
            .args(args)
            .current_dir(&self.0)
            .env("PATH", self.path("empty-bin"))
            .env("PYTHONPATH", "")
            .output()
            .unwrap()
    }
    fn checked(&self, args: &[&str], status: i32) -> Value {
        let output = self.run(args);
        assert_eq!(
            output.status.code(),
            Some(status),
            "args {args:?}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let root = fs::canonicalize(std::env::temp_dir()).unwrap();
        if let Ok(path) = fs::canonicalize(&self.0) {
            if path.starts_with(&root)
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("radiomics-pathology-cli-")
            {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
}
/// Hand-built little-endian, uncompressed single-page TIFF fixture. No image/Python writer dependency.
fn tiff(path: &Path, width: u32, height: u32, rgb: bool, values: &[u16]) {
    let count = if rgb { 3 } else { 1 };
    let bits = if rgb { 8 } else { 16 };
    let entries = 10u16;
    let extra_offset = 8 + 2 + u32::from(entries) * 12 + 4;
    let pixels_offset = extra_offset + if rgb { 6 } else { 0 };
    let mut data = vec![b'I', b'I', 42, 0];
    data.extend(8u32.to_le_bytes());
    data.extend(entries.to_le_bytes());
    let tags = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, count, if rgb { extra_offset } else { bits }),
        (259, 3, 1, 1),
        (262, 3, 1, if rgb { 2 } else { 1 }),
        (273, 4, 1, pixels_offset),
        (277, 3, 1, count),
        (278, 4, 1, height),
        (279, 4, 1, width * height * count * (bits / 8)),
        (284, 3, 1, 1),
    ];
    for (tag, kind, count, value) in tags {
        data.extend((tag as u16).to_le_bytes());
        data.extend((kind as u16).to_le_bytes());
        data.extend(count.to_le_bytes());
        data.extend(value.to_le_bytes());
    }
    data.extend(0u32.to_le_bytes());
    if rgb {
        for _ in 0..3 {
            data.extend(8u16.to_le_bytes());
        }
    }
    if rgb {
        data.extend(values.iter().map(|v| *v as u8));
    } else {
        for value in values {
            data.extend(value.to_le_bytes());
        }
    }
    fs::write(path, data).unwrap();
}
fn golden(name: &str) -> Value {
    serde_json::from_slice(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../golden/python_golden_v1")
                .join(name),
        )
        .unwrap(),
    )
    .unwrap()
}
fn json_close(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for key in b.keys() {
                json_close(&a[key], &b[key]);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                json_close(a, b);
            }
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!((a - b).abs() <= 1e-10 + 1e-8 * b.abs(), "{a} != {b}");
        }
        _ => assert_eq!(actual, expected),
    }
}
fn config() -> Value {
    json!({"schema_version":1,"tiling":{"patch_size":[2,2],"stride":[2,2],"edge":"pad","pad_value":0},
        "qc":pathology_pixels::QcConfig::default(),"color":{"fixed_edges":golden("pathology_inputs.json")["fixed_edges"]},
        "texture":null,"tissue_only":false,"features":[{"kind":"color_statistic","channel":"R","statistic":"mean"},{"kind":"qc","score":"white_fraction"}],
        "aggregation":{"tile_weight":"equal","patient_weight":"equal","patient_stats":["mean"]}})
}
fn slide(id: &str, patient: &str, path: &str) -> Value {
    json!({"image":path,"texture_image":null,"metadata":{"patient_id":patient,"slide_id":id,"source":format!("synthetic:{path}"),"origin0":[0,0],"pixel_size0":[1,1],"mpp_x_um":null,"mpp_y_um":null}})
}

#[test]
fn ordinary_image_to_native_patient_feature_tables_has_no_python_and_no_padding_bias() {
    let work = Work::new();
    tiff(&work.path("white.tif"), 3, 2, true, &[255; 18]);
    work.json("cohort.json", &json!([slide("S", "P", "white.tif")]));
    work.json("config.json", &config());
    let summary = work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "config.json",
            "--output",
            "out",
        ],
        0,
    );
    assert_eq!(summary["tiles"], 2);
    assert_eq!(summary["accepted_tiles"], 2);
    let table: TileTable = serde_json::from_value(work.read("out/tiles.json")).unwrap();
    table.validate().unwrap();
    assert_eq!(table.tiles[1].valid_pixels, [0, 0, 1, 2]);
    assert_eq!(table.tiles[1].mpp_x_um, None);
    assert_eq!(table.features.unwrap().values, vec![255., 1., 255., 1.]);
    let csv = table_io::read_feature_table(work.path("out/patient-features.csv"), None).unwrap();
    let parquet =
        table_io::read_feature_table(work.path("out/patient-features.parquet"), None).unwrap();
    assert_eq!(csv, parquet);
    assert_eq!(csv.samples[0].patient_id, "P");
    assert!(csv.data.values.contains(&255.));
    let lines = fs::read_to_string(work.path("out/qc.jsonl")).unwrap();
    assert_eq!(lines.lines().count(), 2);
    let last: Value = serde_json::from_str(lines.lines().last().unwrap()).unwrap();
    assert_eq!(last["scores"]["white_fraction"], 1.);
    assert!(!last.as_object().unwrap().contains_key("tissue_mask"));
    assert!(!work.path("out/patches").exists());
    fs::write(work.path("out/unrelated.txt"), "retain").unwrap();
    assert_eq!(
        work.run(&[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "config.json",
            "--output",
            "out"
        ])
        .status
        .code(),
        Some(1)
    );
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "config.json",
            "--output",
            "out",
            "--overwrite",
        ],
        0,
    );
    assert_eq!(
        fs::read_to_string(work.path("out/unrelated.txt")).unwrap(),
        "retain"
    );
}

#[test]
fn ordinary_patch_patient_tables_feed_native_training_persistence_prediction_and_evaluation() {
    let work = Work::new();
    let mut cohort = vec![];
    let mut clinical = csv::Writer::from_path(work.path("clinical.csv")).unwrap();
    clinical
        .write_record(["patient_id", "cohort", "label"])
        .unwrap();
    for i in 0..32 {
        let id = format!("P{i:03}");
        let filename = format!("image-{i:03}.tif");
        let value = 40 + (i % 2) * 120 + (i / 2) % 30;
        tiff(&work.path(&filename), 3, 2, true, &[value as u16; 18]);
        cohort.push(slide(&format!("S{i:03}"), &id, &filename));
        clinical
            .write_record([
                id,
                if i < 24 {
                    "train".into()
                } else {
                    "test".into()
                },
                (i % 2).to_string(),
            ])
            .unwrap();
    }
    clinical.flush().unwrap();
    work.json("cohort.json", &json!(cohort));
    work.json("patch-config.json", &config());
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "patch-config.json",
            "--output",
            "patches",
        ],
        0,
    );
    let feature = "patient_mean(slide_tile_mean(color_R_mean))";
    let mut spec = json!({"schema_version":1,"table":"patches/patient-features.parquet","clinical":"clinical.csv","clinical_key":"patient","numeric_features":[feature],"outcome_column":"label"});
    work.json("study.json", &spec);
    work.json("train-config.json",&json!({"task":"binary","classes":["0","1"],"fit_unit":"patient","scoring_unit":"patient","threshold":0.5,"selection_tie_atol":1e-12,"models":[{"name":"LR","candidates":[{"preprocessing":{"scaling":"standard","all_missing":"error"},"selector":null,"model":{"model":"LR","lambda":0.1,"tolerance":1e-10,"max_iterations":5000}}]}]}));
    work.checked(
        &[
            "train",
            "--data",
            "study.json",
            "--config",
            "train-config.json",
            "--outer-cv",
            "3",
            "--inner-cv",
            "2",
            "--output",
            "models",
        ],
        0,
    );
    assert!(work.path("models/model-000.rad").exists());
    spec["outcome_column"] = Value::Null;
    work.json("unlabeled.json", &spec);
    work.checked(
        &[
            "predict",
            "--model",
            "models/model-000.rad",
            "--data",
            "unlabeled.json",
            "--output",
            "predicted",
        ],
        0,
    );
    let actual: PredictionTable =
        serde_json::from_value(work.read("predicted/predictions.json")).unwrap();
    assert_eq!(actual.unit, PredictionUnit::Patient);
    assert_eq!(actual.rows.len(), 32);
    let expected = work.read("models/model-000-predictions.json");
    assert_eq!(serde_json::to_value(&actual).unwrap(), expected);
    work.checked(
        &[
            "evaluate",
            "--predictions",
            "predicted/predictions.json",
            "--data",
            "study.json",
            "--n-bootstrap",
            "16",
            "--output",
            "evaluated",
        ],
        0,
    );
}

#[test]
fn excluded_missing_and_zero_tile_slides_keep_parents_and_missing_patient_features() {
    let work = Work::new();
    tiff(&work.path("white.tif"), 3, 2, true, &[255; 18]);
    tiff(&work.path("tiny.tif"), 1, 1, true, &[255; 3]);
    let mut config = config();
    config["tiling"]["edge"] = json!("drop");
    config["qc"]["max_white_fraction"] = json!(0.8);
    work.json("config.json", &config);
    work.json(
        "cohort.json",
        &json!([
            slide("excluded", "P1", "white.tif"),
            slide("missing", "P2", "missing.tif"),
            slide("tiny", "P3", "tiny.tif")
        ]),
    );
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "config.json",
            "--output",
            "out",
        ],
        2,
    );
    let table: TileTable = serde_json::from_value(work.read("out/tiles.json")).unwrap();
    assert_eq!(table.slide_status["excluded"], "all_excluded");
    assert_eq!(table.slide_status["missing"], "failed");
    assert_eq!(table.slide_status["tiny"], "no_tiles");
    assert_eq!(table.slide_patients.len(), 3);
    let patients: FeatureTable =
        serde_json::from_value(work.read("out/patient-features.json")).unwrap();
    assert_eq!(patients.samples.len(), 3);
    assert!(patients.data.missing.iter().all(|v| *v));
    let mut bad = config.clone();
    bad["features"] = json!([{"kind":"color_statistic","channel":"H","statistic":"mean"}]);
    work.json("bad.json", &bad);
    let failed = work.run(&[
        "pathology",
        "patches",
        "--cohort",
        "cohort.json",
        "--config",
        "bad.json",
        "--output",
        "invalid",
    ]);
    assert_eq!(failed.status.code(), Some(1));
    assert!(!work.path("invalid").exists());
}

#[test]
fn explicit_gray16_texture_preserves_intensity_and_rgb_conversion_is_never_inferred() {
    let work = Work::new();
    tiff(&work.path("rgb.tif"), 3, 2, true, &[255; 18]);
    tiff(&work.path("gray.tif"), 3, 2, false, &[1, 2, 3, 4, 5, 6]);
    let mut c = config();
    c["color"] = Value::Null;
    c["texture"] = json!({"profile":"python_golden_v1","discretization":{"method":"fixed_width","width":1},"aggregation":"direction_mean","gldm_alpha":0,"voxel_array_shift":0});
    c["features"] = json!([{"kind":"texture","name":"original_firstorder_Mean"}]);
    work.json("config.json", &c);
    let mut s = slide("S", "P", "rgb.tif");
    s["texture_image"] = json!("gray.tif");
    work.json("cohort.json", &json!([s]));
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "config.json",
            "--output",
            "out",
        ],
        0,
    );
    let table: TileTable = serde_json::from_value(work.read("out/tiles.json")).unwrap();
    assert_eq!(table.features.unwrap().values, vec![3., 4.5]);
    work.json("no-gray.json", &json!([slide("S", "P", "rgb.tif")]));
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "no-gray.json",
            "--config",
            "config.json",
            "--output",
            "rejected",
        ],
        2,
    );
    assert!(work.read("rejected/summary.json")["slides"][0]["error"]
        .as_str()
        .unwrap()
        .contains("explicit co-gridded grayscale"));
}

#[test]
fn existing_semantic_mask_uses_valid_pixels_explicit_connectivity_mpp_and_patch_local_counts() {
    let work = Work::new();
    tiff(&work.path("rgb.tif"), 3, 2, true, &[100; 18]);
    tiff(&work.path("mask.tif"), 3, 2, false, &[1, 0, 2, 0, 1, 2]);
    let mut c = config();
    c["color"] = Value::Null;
    c["tiling"]["stride"] = json!([1, 2]);
    c["mask_analysis"] = json!({"kind":"semantic","connectivity":"four","classes":[0,1,2]});
    c["features"] = json!([{"kind":"patch_object_count"},{"kind":"annotation_fraction","label":2}]);
    work.json("config.json", &c);
    let mut s = slide("S", "P", "rgb.tif");
    s["mask"] = json!("mask.tif");
    s["metadata"]["mpp_x_um"] = json!(0.5);
    s["metadata"]["mpp_y_um"] = json!(2.);
    work.json("cohort.json", &json!([s]));
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "config.json",
            "--output",
            "out",
        ],
        0,
    );
    let table: TileTable = serde_json::from_value(work.read("out/tiles.json")).unwrap();
    assert_eq!(table.tiles.len(), 3);
    let data = table.features.unwrap();
    assert_eq!(data.values, vec![2., 0., 2., 0.5, 1., 1.]);
    let lines = fs::read_to_string(work.path("out/patch-details.jsonl")).unwrap();
    let details: Vec<Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(details[2]["annotation"]["denominator"], 2);
    assert_eq!(details[2]["annotation"]["fractions"]["2"], 1.);
    assert_eq!(details[0]["objects"]["count_scope"], "patch_local");
    assert_eq!(details[0]["objects"]["objects"][0]["area_um2"], 1.);
    assert!(!details[0]["objects"]["warnings"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(work.read("out/manifest.json")["objects"]
        .as_str()
        .unwrap()
        .contains("No whole-slide"));
    let mut eight = c.clone();
    eight["mask_analysis"]["connectivity"] = json!("eight");
    work.json("eight.json", &eight);
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "cohort.json",
            "--config",
            "eight.json",
            "--output",
            "eight",
        ],
        0,
    );
    let data: TileTable = serde_json::from_value(work.read("eight/tiles.json")).unwrap();
    assert_eq!(data.features.unwrap().values[0], 1.);
    tiff(&work.path("wrong.tif"), 1, 1, false, &[1]);
    let mut bad = slide("bad", "P", "rgb.tif");
    bad["mask"] = json!("wrong.tif");
    work.json("bad-cohort.json", &json!([bad]));
    work.checked(
        &[
            "pathology",
            "patches",
            "--cohort",
            "bad-cohort.json",
            "--config",
            "config.json",
            "--output",
            "bad",
        ],
        2,
    );
    assert!(work.read("bad/summary.json")["slides"][0]["error"]
        .as_str()
        .unwrap()
        .contains("matching primary image dimensions"));
}

#[test]
fn external_embedding_feature_tables_align_by_tile_id_and_reject_mismatched_sets() {
    let work = Work::new();
    let input = golden("pathology_aggregate_inputs.json");
    let expected = golden("pathology_aggregate_reference.json");
    let parents: BTreeMap<String, String> =
        serde_json::from_value(input["slide_patients"].clone()).unwrap();
    let table:TileTable=serde_json::from_value(json!({"tiles":input["tiles"],"features":null,"feature_names":[],"slide_patients":parents,"slide_status":parents.keys().map(|k|(k.clone(),"provided")).collect::<BTreeMap<_,_>>()})).unwrap();
    work.json("tiles.json", &serde_json::to_value(&table).unwrap());
    work.json("config.json",&json!({"tile_weight":"equal","patient_weight":"equal","patient_stats":["mean","std","max","q25","q50","q75"]}));
    let ids: Vec<String> = serde_json::from_value(input["embedding_ids"].clone()).unwrap();
    let names: Vec<String> = serde_json::from_value(input["feature_names"].clone()).unwrap();
    let mut values = vec![];
    let mut missing = vec![];
    for row in input["embedding_values"].as_array().unwrap() {
        for value in row.as_array().unwrap() {
            values.push(value.as_f64().unwrap_or(0.));
            missing.push(value.is_null());
        }
    }
    let samples = ids
        .iter()
        .map(|id| {
            let tile = table.tiles.iter().find(|t| &t.tile_id == id).unwrap();
            SampleMetadata {
                sample_id: id.clone(),
                patient_id: tile.patient_id.clone(),
                slide_id: Some(tile.slide_id.clone()),
                lesion_id: None,
                modality: None,
                region: None,
                timepoint: None,
                center: None,
            }
        })
        .collect();
    let features = FeatureTable {
        data: NumericBuffer {
            rows: ids.len(),
            cols: names.len(),
            values,
            missing,
        },
        samples,
        features: names
            .iter()
            .map(|name| FeatureDefinition {
                name: name.clone(),
                definition: "frozen external synthetic embedding".into(),
                unit: None,
            })
            .collect(),
        effective_parameters: BTreeMap::new(),
        quality: vec![],
    };
    table_io::write_feature_table(&features, work.path("embedding.parquet"), false, "\\N").unwrap();
    work.checked(
        &[
            "pathology",
            "aggregate",
            "--table",
            "tiles.json",
            "--config",
            "config.json",
            "--embeddings",
            "embedding.parquet",
            "--output",
            "out",
        ],
        0,
    );
    let report = work.read("out/aggregation.json");
    json_close(
        &report["slides"],
        &expected["aggregation"]["default_equal_slides"]["slides"],
    );
    let mut wrong = features.clone();
    wrong.samples[0].sample_id = "unknown".into();
    table_io::write_feature_table(&wrong, work.path("wrong.csv"), false, "\\N").unwrap();
    assert_eq!(
        work.run(&[
            "pathology",
            "aggregate",
            "--table",
            "tiles.json",
            "--config",
            "config.json",
            "--embeddings",
            "wrong.csv",
            "--output",
            "bad"
        ])
        .status
        .code(),
        Some(1)
    );
    assert!(!work.path("bad").exists());
}

#[test]
fn chunked_true_tile_overlay_matches_frozen_coverage_and_rejects_patient_broadcast() {
    let work = Work::new();
    let input = golden("pathology_aggregate_inputs.json");
    let expected = golden("pathology_aggregate_reference.json");
    let case = &input["overlay"];
    let tiles: Vec<Value> = input["tiles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["slide_id"] == case["slide_id"])
        .cloned()
        .collect();
    work.json("tiles.json",&json!({"tiles":tiles,"features":null,"feature_names":[],"slide_status":{"A":"provided"},"slide_patients":{"A":"P1"}}));
    let mut config = json!({"schema_version":1,"canvas":case["canvas"],"block_size":case["block_size"],"clip":false,"source":{"kind":"scores","tile_ids":case["score_ids"],"scores":case["scores"],"prediction_unit":"tile"}});
    work.json("config.json", &config);
    work.checked(
        &[
            "pathology",
            "overlay",
            "--table",
            "tiles.json",
            "--config",
            "config.json",
            "--output",
            "out",
        ],
        0,
    );
    let manifest = work.read("out/manifest.json");
    let canvas: ov_radiomics_core::overlay::Canvas =
        serde_json::from_value(manifest["canvas"].clone()).unwrap();
    assert_eq!(
        canvas,
        serde_json::from_value(case["canvas"].clone()).unwrap()
    );
    assert_eq!(manifest["block_index"], "blocks.jsonl");
    let (width, height) = (
        case["canvas"]["width"].as_u64().unwrap() as usize,
        case["canvas"]["height"].as_u64().unwrap() as usize,
    );
    let mut values = vec![Value::Null; width * height];
    let mut coverage = vec![0u64; width * height];
    for line in fs::read_to_string(work.path("out/blocks.jsonl"))
        .unwrap()
        .lines()
    {
        let entry: Value = serde_json::from_str(line).unwrap();
        let block = work.read(&format!("out/{}", entry["path"].as_str().unwrap()));
        let (x, y, w, h) = (
            block["x"].as_u64().unwrap() as usize,
            block["y"].as_u64().unwrap() as usize,
            block["width"].as_u64().unwrap() as usize,
            block["height"].as_u64().unwrap() as usize,
        );
        for j in 0..h {
            for i in 0..w {
                let target = (y + j) * width + x + i;
                values[target] = block["values"][j * w + i].clone();
                coverage[target] = block["coverage"][j * w + i].as_u64().unwrap();
            }
        }
    }
    assert_eq!(json!(values), expected["overlay"]["whole"]["values"]);
    assert_eq!(json!(coverage), expected["overlay"]["whole"]["coverage"]);
    let rows:Vec<_>=case["score_ids"].as_array().unwrap().iter().zip(case["scores"].as_array().unwrap()).map(|(id,score)| {
        let p=score.as_f64().unwrap();json!({"sample_id":id,"patient_id":"P1","slide_id":"A","tile_id":id,"cohort":"external","fold":null,"training_source":"external:actual_tile_fixture","probabilities":[1.-p,p],"value":null})
    }).collect();
    work.json(
        "predictions.json",
        &json!({"task":"binary","unit":"tile","classes":["negative","positive"],"rows":rows}),
    );
    let mut prediction_config = config.clone();
    prediction_config["source"] =
        json!({"kind":"predictions","path":"predictions.json","slide_id":"A","class":"positive"});
    work.json("prediction-config.json", &prediction_config);
    work.checked(
        &[
            "pathology",
            "overlay",
            "--table",
            "tiles.json",
            "--config",
            "prediction-config.json",
            "--output",
            "predicted",
        ],
        0,
    );
    assert_eq!(
        fs::read(work.path("predicted/blocks.jsonl")).unwrap(),
        fs::read(work.path("out/blocks.jsonl")).unwrap()
    );
    for line in fs::read_to_string(work.path("out/blocks.jsonl"))
        .unwrap()
        .lines()
    {
        let entry: Value = serde_json::from_str(line).unwrap();
        let name = entry["path"].as_str().unwrap();
        assert_eq!(
            work.read(&format!("out/{name}")),
            work.read(&format!("predicted/{name}"))
        );
    }
    config["source"]["prediction_unit"] = json!("patient");
    work.json("patient.json", &config);
    let failed = work.run(&[
        "pathology",
        "overlay",
        "--table",
        "tiles.json",
        "--config",
        "patient.json",
        "--output",
        "bad",
    ]);
    assert_eq!(failed.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failed.stderr).contains("tile-level"));
    assert!(!work.path("bad").exists());
}
