use super::*;
use serde_json::{json, Value};
#[test]
fn parquet_row_groups_and_nullable_column_batches_preserve_all_rows_and_signed_zero() {
    let work = Workspace::new();
    let mut table = feature_table();
    let n = 4101;
    table.samples = (0..n)
        .map(|i| metadata(&format!("{i:05}"), &format!("P{i:05}")))
        .collect();
    table.data.rows = n;
    table.data.values.clear();
    table.data.missing.clear();
    for i in 0..n {
        for j in 0..table.data.cols {
            let missing = (i + j) % 17 == 0;
            table.data.missing.push(missing);
            table.data.values.push(if missing {
                0.0
            } else if j == 1 {
                -0.0
            } else {
                i as f64 + j as f64 / 10.0
            });
        }
    }
    let path = work.path("batches.parquet");
    write_feature_table(&table, &path, false, "").unwrap();
    let restored = read_feature_table(&path, None).unwrap();
    assert_eq!(restored.data.missing, table.data.missing);
    assert_eq!(restored.samples, table.samples);
    assert_eq!(
        restored
            .data
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        table
            .data
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("radiomics-table-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn csv(&self, name: &str, value: &str) {
        fs::write(self.path(name), value).unwrap();
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let root = fs::canonicalize(std::env::temp_dir());
        let target = fs::canonicalize(&self.0);
        if let (Ok(root), Ok(target)) = (root, target) {
            if target.starts_with(root)
                && target
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("radiomics-table-"))
            {
                let _ = fs::remove_dir_all(target);
            }
        }
    }
}
fn spec(value: Value) -> StudyTableSpec {
    serde_json::from_value(value).unwrap()
}
fn frozen(name: &str) -> Value {
    serde_json::from_str(
        &fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../golden/python_golden_v1")
                .join(name),
        )
        .unwrap(),
    )
    .unwrap()
}
fn metadata(id: &str, patient: &str) -> SampleMetadata {
    SampleMetadata {
        sample_id: id.into(),
        patient_id: patient.into(),
        slide_id: None,
        lesion_id: None,
        modality: None,
        region: None,
        timepoint: None,
        center: None,
    }
}
fn feature_table() -> FeatureTable {
    let input = frozen("table_inputs.json");
    let names: Vec<String> = serde_json::from_value(input["numeric_names"].clone()).unwrap();
    let rows = input["numeric_train"].as_array().unwrap();
    let mut values = Vec::new();
    let mut missing = Vec::new();
    for row in rows {
        for value in row.as_array().unwrap() {
            values.push(value.as_f64().unwrap_or(0.0));
            missing.push(value.is_null());
        }
    }
    FeatureTable {
        data: NumericBuffer {
            rows: rows.len(),
            cols: names.len(),
            values,
            missing,
        },
        samples: (0..rows.len())
            .map(|i| metadata(&format!("{i:03}"), &format!("P{i:03}")))
            .collect(),
        features: names
            .into_iter()
            .map(|name| FeatureDefinition {
                name,
                definition: "Original synthetic frozen numeric input".into(),
                unit: None,
            })
            .collect(),
        effective_parameters: BTreeMap::from([("profile".into(), json!("python_golden_v1"))]),
        quality: vec![QualityRecord {
            code: "synthetic".into(),
            message: "Original licensed synthetic fixture".into(),
            severity: "info".into(),
            sample_id: None,
        }],
    }
}
#[test]
fn feature_csv_parquet_round_trip_preserves_contract_nulls_hash_and_metadata() {
    let work = Workspace::new();
    let mut table = feature_table();
    table.samples[0].center = Some(String::new());
    table.samples[1].center = Some("NA".into());
    for name in ["features.csv", "features.parquet"] {
        let path = work.path(name);
        let manifest = write_feature_table(&table, &path, false, "\\N").unwrap();
        assert_eq!(
            manifest.table_sha256.as_deref(),
            Some(sha256_file(&path).unwrap().as_str())
        );
        assert_eq!(table, read_feature_table(&path, None).unwrap());
        assert!(write_feature_table(&table, &path, false, "\\N").is_err());
        let input = spec(
            json!({"schema_version":1,"table":name,"feature_manifest":format!("{name}.manifest.json"),"default_cohort":"external_A"}),
        );
        let loaded = read_study(&input, Task::Binary, &work.0).unwrap();
        assert_eq!(loaded.data.features.numeric, table.data);
        assert_eq!(loaded.data.samples, table.samples);
        assert!(matches!(loaded.data.outcome, Outcome::Unlabeled));
        assert_eq!(loaded.data.cohorts, vec!["external_A"; table.samples.len()]);
        assert_eq!(loaded.feature_manifest, Some(manifest));
    }
    let path = work.path("features.csv");
    let mut bytes = fs::read(&path).unwrap();
    bytes.push(b'\n');
    fs::write(&path, bytes).unwrap();
    assert!(read_feature_table(&path, None)
        .unwrap_err()
        .message
        .contains("SHA256"));
    assert!(write_feature_table(&table, work.path("collision.csv"), false, "0.01").is_err());
    assert!(!work.path("collision.csv").exists());
}
#[test]
fn feature_parquet_lowlevel_reader_handles_all_enabled_codecs_without_arrow() {
    use parquet::basic::{Compression, GzipLevel, ZstdLevel};
    let work = Workspace::new();
    let table = feature_table();
    for (name, codec) in [
        ("raw", Compression::UNCOMPRESSED),
        ("snappy", Compression::SNAPPY),
        ("gzip", Compression::GZIP(GzipLevel::default())),
        ("zstd", Compression::ZSTD(ZstdLevel::default())),
    ] {
        let path = work.path(&format!("{name}.parquet"));
        let manifest = FeatureTableManifest {
            format: "radiomics.feature-table".into(),
            schema_version: 1,
            samples: table.samples.clone(),
            features: table.features.clone(),
            effective_parameters: table.effective_parameters.clone(),
            quality: table.quality.clone(),
            csv_null_value: None,
            table_sha256: None,
        };
        feature_io::parquet_write(&table, &path, &manifest, codec).unwrap();
        assert_eq!(table, read_feature_table(&path, None).unwrap());
    }
}
#[test]
fn sample_and_explicit_patient_joins_use_keys_preserve_order_and_report_left_unmatched_ids() {
    let work = Workspace::new();
    work.csv("base.csv", "sample_id,x\n001,1\n002,2\n003,\n");
    work.csv(
        "metadata.csv",
        "sample_id,patient_id,cohort,center\n003,P2,test,B\n001,P1,train,A\n002,P1,train,A\n",
    );
    work.csv(
        "clinical.csv",
        "patient_id,age,label,x,unused_predictor\nP2,60,positive,3,999\nP1,50,negative,,999\n",
    );
    let input = spec(
        json!({"schema_version":1,"table":"base.csv","metadata":"metadata.csv","clinical":"clinical.csv","clinical_key":"patient","outcome_column":"label","numeric_features":["x","age"]}),
    );
    let loaded = read_study(&input, Task::Binary, &work.0).unwrap();
    assert_eq!(
        loaded
            .data
            .samples
            .iter()
            .map(|s| s.sample_id.as_str())
            .collect::<Vec<_>>(),
        vec!["001", "002", "003"]
    );
    assert_eq!(
        loaded
            .data
            .samples
            .iter()
            .map(|s| s.patient_id.as_str())
            .collect::<Vec<_>>(),
        vec!["P1", "P1", "P2"]
    );
    assert_eq!(
        loaded.data.features.numeric.values,
        vec![1., 50., 2., 50., 3., 60.]
    );
    assert_eq!(loaded.report.joins[1].matched_base_rows, 3);
    assert!(loaded.report.inputs[2]
        .ignored_columns
        .contains(&"unused_predictor".into()));
    let mut automatic = input.clone();
    automatic.numeric_features = None;
    assert_eq!(
        read_study(&automatic, Task::Binary, &work.0)
            .unwrap()
            .data
            .features
            .numeric_names,
        vec!["x"]
    );
    work.csv(
        "clinical.csv",
        "patient_id,age,label,x\nP2,60,positive,3\nP1,50,negative,\nP1,50,negative,\n",
    );
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("unique"));
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,x\n001,P1,train,1\n002,P1,train,2\n003,P2,test,3\n",
    );
    work.csv("metadata.csv", "sample_id,center\n001,A\n004,Z\n");
    let mut left = spec(
        json!({"schema_version":1,"table":"base.csv","metadata":"metadata.csv","join_policy":"left"}),
    );
    let loaded = read_study(&left, Task::Binary, &work.0).unwrap();
    assert_eq!(loaded.data.samples.len(), 3);
    assert_eq!(
        loaded.report.joins[0].missing_base_sample_ids,
        vec!["002", "003"]
    );
    assert_eq!(loaded.report.joins[0].unused_auxiliary_keys, vec!["004"]);
    left.join_policy = JoinPolicy::Exact;
    assert!(read_study(&left, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("sets differ"));
}
#[test]
fn base_source_manifest_null_encoding_preserves_empty_and_na_values_over_spec_defaults() {
    let work = Workspace::new();
    work.csv("encoded.csv","sample_id,patient_id,cohort,x,cat,center\n001,P1,external,1,,\n002,P2,external,2,NA,NA\n003,P3,external,\\N,\\N,\\N\n");
    let mut samples = vec![
        metadata("001", "P1"),
        metadata("002", "P2"),
        metadata("003", "P3"),
    ];
    samples[0].center = Some(String::new());
    samples[1].center = Some("NA".into());
    let manifest = FeatureTableManifest {
        format: "radiomics.feature-table".into(),
        schema_version: 1,
        samples,
        features: vec![FeatureDefinition {
            name: "x".into(),
            definition: "Synthetic value".into(),
            unit: None,
        }],
        effective_parameters: BTreeMap::new(),
        quality: vec![],
        csv_null_value: Some("\\N".into()),
        table_sha256: Some(sha256_file(&work.path("encoded.csv")).unwrap()),
    };
    fs::write(
        work.path("source.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let input = spec(
        json!({"schema_version":1,"table":"encoded.csv","feature_manifest":"source.json","numeric_features":["x"],"categorical_features":["cat"],"missing_values":["","NA"]}),
    );
    let loaded = read_study(&input, Task::Binary, &work.0).unwrap();
    assert_eq!(
        loaded.data.features.categorical_values,
        vec![Some(String::new()), Some("NA".into()), None]
    );
    assert_eq!(loaded.data.samples[0].center, Some(String::new()));
    assert_eq!(loaded.data.samples[1].center, Some("NA".into()));
    assert_eq!(
        loaded.data.features.numeric.missing,
        vec![false, false, true]
    );
    assert_eq!(loaded.report.base_missing_values, vec!["\\N"]);
    assert_eq!(
        loaded.report.base_missing_source,
        "feature_manifest.csv_null_value"
    );
    let mut no_manifest = input;
    no_manifest.feature_manifest = None;
    assert!(read_study(&no_manifest, Task::Binary, &work.0).is_err());
}

enum Column {
    Text(Vec<Option<String>>),
    Int(Vec<Option<i64>>, Option<parquet::basic::LogicalType>),
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
}
fn strings(values: &[&str]) -> Column {
    Column::Text(values.iter().map(|v| Some((*v).into())).collect())
}
fn parquet_fixture(path: &Path, columns: Vec<(&str, Column)>) {
    use parquet::{
        basic::{LogicalType, Repetition, Type as Physical},
        data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type},
        file::{properties::WriterProperties, writer::SerializedFileWriter},
        schema::types::Type,
    };
    use std::sync::Arc;
    let fields = columns
        .iter()
        .map(|(name, column)| {
            let (physical, logical) = match column {
                Column::Text(_) => (Physical::BYTE_ARRAY, Some(LogicalType::String)),
                Column::Int(_, logical) => (Physical::INT64, logical.clone()),
                Column::Float(_) => (Physical::DOUBLE, None),
                Column::Bool(_) => (Physical::BOOLEAN, None),
            };
            Arc::new(
                Type::primitive_type_builder(name, physical)
                    .with_repetition(Repetition::OPTIONAL)
                    .with_logical_type(logical)
                    .build()
                    .unwrap(),
            )
        })
        .collect();
    let schema = Arc::new(
        Type::group_type_builder("synthetic_external")
            .with_fields(fields)
            .build()
            .unwrap(),
    );
    let mut writer = SerializedFileWriter::new(
        File::create(path).unwrap(),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    let mut group = writer.next_row_group().unwrap();
    macro_rules! primitive {
        ($column:expr,$type:ty,$values:expr) => {{
            let values = $values;
            let defs: Vec<_> = values.iter().map(|v| i16::from(v.is_some())).collect();
            let values: Vec<_> = values.iter().flatten().copied().collect();
            $column
                .typed::<$type>()
                .write_batch(&values, Some(&defs), None)
                .unwrap();
        }};
    }
    for (_, data) in columns {
        let mut column = group.next_column().unwrap().unwrap();
        match data {
            Column::Text(values) => {
                let defs: Vec<_> = values.iter().map(|v| i16::from(v.is_some())).collect();
                let values: Vec<_> = values
                    .iter()
                    .flatten()
                    .map(|v| ByteArray::from(v.as_str()))
                    .collect();
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&values, Some(&defs), None)
                    .unwrap();
            }
            Column::Int(values, _) => primitive!(column, Int64Type, values),
            Column::Float(values) => primitive!(column, DoubleType, values),
            Column::Bool(values) => primitive!(column, BoolType, values),
        }
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
}
#[test]
fn parquet_explicit_integer_bool_categories_and_null_empty_strings_are_type_safe() {
    let work = Workspace::new();
    let path = work.path("typed.parquet");
    parquet_fixture(
        &path,
        vec![
            ("sample_id", strings(&["001", "002", "003"])),
            ("patient_id", strings(&["P1", "P2", "P3"])),
            ("cohort", strings(&["test", "test", "test"])),
            (
                "x",
                Column::Int(
                    vec![Some(9_007_199_254_740_994), Some(i64::MIN), None],
                    None,
                ),
            ),
            (
                "cat",
                Column::Text(vec![Some(String::new()), Some("NA".into()), None]),
            ),
            ("center", Column::Int(vec![Some(10), Some(20), None], None)),
            ("label", Column::Bool(vec![Some(false), Some(true), None])),
        ],
    );
    let input = spec(
        json!({"schema_version":1,"table":"typed.parquet","numeric_features":["x"],"categorical_features":["cat","center"],"outcome_column":"label","missing_values":["","NA"]}),
    );
    let loaded = read_study(&input, Task::Binary, &work.0).unwrap();
    assert_eq!(loaded.data.samples[0].sample_id, "001");
    assert_eq!(
        loaded.data.features.numeric.values,
        vec![9_007_199_254_740_994f64, i64::MIN as f64, 0.]
    );
    assert_eq!(
        loaded.data.features.numeric.missing,
        vec![false, false, true]
    );
    assert_eq!(
        loaded.data.features.categorical_values,
        vec![
            Some(String::new()),
            Some("10".into()),
            Some("NA".into()),
            Some("20".into()),
            None,
            None
        ]
    );
    assert_eq!(loaded.data.samples[0].center, Some("10".into()));
    let Outcome::Classification { values, .. } = loaded.data.outcome else {
        unreachable!()
    };
    assert_eq!(values, vec![Some("0".into()), Some("1".into()), None]);
    assert!(loaded.report.inputs[0]
        .conversions
        .iter()
        .any(|c| c.column == "label" && c.conversion.contains("canonical")));
    assert_eq!(
        loaded.report.base_missing_source,
        "parquet_definition_levels"
    );
}
#[test]
fn parquet_precision_loss_non_utf8_ids_float_categories_and_nonfinite_features_error() {
    let work = Workspace::new();
    let common = || {
        vec![
            ("sample_id", strings(&["001", "002"])),
            ("patient_id", strings(&["P1", "P2"])),
            ("cohort", strings(&["test", "test"])),
        ]
    };
    for (name, values) in [
        ("signed", Column::Int(vec![Some(i64::MAX), Some(2)], None)),
        (
            "unsigned",
            Column::Int(
                vec![Some(-1), Some(2)],
                Some(parquet::basic::LogicalType::Integer {
                    bit_width: 64,
                    is_signed: false,
                }),
            ),
        ),
    ] {
        let mut columns = common();
        columns.push(("x", values));
        parquet_fixture(&work.path(&format!("{name}.parquet")), columns);
        let input = spec(
            json!({"schema_version":1,"table":format!("{name}.parquet"),"numeric_features":["x"]}),
        );
        assert!(read_study(&input, Task::Binary, &work.0)
            .unwrap_err()
            .message
            .contains("precision loss"));
    }
    let mut columns = common();
    columns[0] = ("sample_id", Column::Int(vec![Some(1), Some(2)], None));
    columns.push(("x", Column::Float(vec![Some(1.), Some(2.)])));
    parquet_fixture(&work.path("ids.parquet"), columns);
    let input = spec(json!({"schema_version":1,"table":"ids.parquet","numeric_features":["x"]}));
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("UTF8"));
    let mut columns = common();
    columns.push(("x", Column::Float(vec![Some(1.), Some(2.)])));
    columns.push(("cat", Column::Float(vec![Some(0.), Some(1.)])));
    parquet_fixture(&work.path("cat.parquet"), columns);
    let input = spec(
        json!({"schema_version":1,"table":"cat.parquet","numeric_features":["x"],"categorical_features":["cat"]}),
    );
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("string encoding"));
    for (name, value) in [("nan", f64::NAN), ("inf", f64::INFINITY)] {
        let mut columns = common();
        columns.push(("x", Column::Float(vec![Some(value), Some(2.)])));
        parquet_fixture(&work.path(&format!("{name}.parquet")), columns);
        let mut input = spec(
            json!({"schema_version":1,"table":format!("{name}.parquet"),"numeric_features":["x"]}),
        );
        assert!(read_study(&input, Task::Binary, &work.0).is_err());
        input.nan_as_missing = true;
        if value.is_nan() {
            let data = read_study(&input, Task::Binary, &work.0).unwrap();
            assert_eq!(data.data.features.numeric.values, vec![0., 2.]);
            assert_eq!(data.data.features.numeric.missing, vec![true, false]);
        } else {
            assert!(read_study(&input, Task::Binary, &work.0).is_err());
        }
    }
}

#[test]
fn frozen_csv_inputs_train_save_reload_and_predict_all_three_tasks_without_python() {
    use crate::{
        bundle::ModelBundle,
        learning::*,
        models::ModelSpec,
        preprocessing::{
            AllMissingPolicy, PreprocessConfig, Scaling, SelectorConfig, UnivariateMethod,
        },
        split::SplitPlan,
    };
    let work = Workspace::new();
    let input = frozen("model_inputs.json");
    let plans = frozen("workflow_plans.json");
    for (name, task) in [
        ("binary", Task::Binary),
        ("multiclass", Task::Multiclass),
        ("regression", Task::Regression),
    ] {
        let dataset = &input["datasets"][name];
        let classes: Vec<String> = serde_json::from_value(dataset["classes"].clone()).unwrap();
        let records = dataset["records"].as_array().unwrap();
        let features: Vec<String> =
            serde_json::from_value(dataset["feature_names"].clone()).unwrap();
        let path = work.path(&format!("{name}.csv"));
        let mut writer = csv::Writer::from_path(&path).unwrap();
        writer
            .write_record([
                "sample_id",
                "patient_id",
                "cohort",
                "center",
                "lesion_id",
                "signal_a",
                "signal_b",
                "noise_a",
                "noise_b",
                "outcome",
            ])
            .unwrap();
        for row in records {
            let mut values: Vec<String> =
                ["sample_id", "patient_id", "cohort", "center", "lesion_id"]
                    .into_iter()
                    .map(|key| row[key].as_str().unwrap().to_string())
                    .collect();
            values.extend(
                row["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap().to_string()),
            );
            values.push(if task == Task::Regression {
                row["target"]
                    .as_f64()
                    .map(|v| v.to_string())
                    .unwrap_or_default()
            } else {
                row["target"]
                    .as_u64()
                    .map(|c| classes[c as usize].clone())
                    .unwrap_or_default()
            });
            writer.write_record(values).unwrap();
        }
        writer.flush().unwrap();
        let config = spec(
            json!({"schema_version":1,"table":format!("{name}.csv"),"outcome_column":"outcome"}),
        );
        let loaded = read_study(&config, task, &work.0).unwrap();
        assert_eq!(loaded.data.features.numeric_names, features);
        let expected: Vec<_> = records
            .iter()
            .flat_map(|r| {
                r["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap().to_bits())
            })
            .collect();
        assert_eq!(
            loaded
                .data
                .features
                .numeric
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(loaded.report.inputs[0].sha256, sha256_file(&path).unwrap());
        let mut columns = vec![
            (
                "sample_id",
                Column::Text(
                    records
                        .iter()
                        .map(|r| r["sample_id"].as_str().map(str::to_string))
                        .collect(),
                ),
            ),
            (
                "patient_id",
                Column::Text(
                    records
                        .iter()
                        .map(|r| r["patient_id"].as_str().map(str::to_string))
                        .collect(),
                ),
            ),
            (
                "cohort",
                Column::Text(
                    records
                        .iter()
                        .map(|r| r["cohort"].as_str().map(str::to_string))
                        .collect(),
                ),
            ),
            (
                "center",
                Column::Text(
                    records
                        .iter()
                        .map(|r| r["center"].as_str().map(str::to_string))
                        .collect(),
                ),
            ),
            (
                "lesion_id",
                Column::Text(
                    records
                        .iter()
                        .map(|r| r["lesion_id"].as_str().map(str::to_string))
                        .collect(),
                ),
            ),
        ];
        for (j, feature) in features.iter().enumerate() {
            columns.push((
                feature.as_str(),
                Column::Float(records.iter().map(|r| r["values"][j].as_f64()).collect()),
            ));
        }
        columns.push((
            "outcome",
            if task == Task::Regression {
                Column::Float(records.iter().map(|r| r["target"].as_f64()).collect())
            } else {
                Column::Text(
                    records
                        .iter()
                        .map(|r| r["target"].as_u64().map(|c| classes[c as usize].clone()))
                        .collect(),
                )
            },
        ));
        parquet_fixture(&work.path(&format!("{name}.parquet")), columns);
        let mut parquet_spec = config.clone();
        parquet_spec.table = PathBuf::from(format!("{name}.parquet"));
        let parquet_data = read_study(&parquet_spec, task, &work.0).unwrap();
        assert_eq!(loaded.data, parquet_data.data);
        let plan: SplitPlan = serde_json::from_value(plans[name].clone()).unwrap();
        let training = TrainingConfig {
            task,
            classes,
            fit_unit: PredictionUnit::Sample,
            scoring_unit: ScoreUnit::Patient,
            threshold: (task == Task::Binary).then_some(0.5),
            selection_tie_atol: 1e-12,
            models: vec![ModelGrid {
                name: if task == Task::Regression {
                    "Ridge".into()
                } else {
                    "LR".into()
                },
                candidates: [0.05, 0.1]
                    .into_iter()
                    .map(|lambda| PipelineSpec {
                        preprocessing: PreprocessConfig {
                            scaling: Scaling::Standard,
                            all_missing: AllMissingPolicy::Error,
                        },
                        selector: Some(SelectorConfig {
                            univariate: UnivariateMethod::None,
                            ..SelectorConfig::default()
                        }),
                        model: if task == Task::Regression {
                            ModelSpec::Ridge {
                                lambda,
                                rank_tolerance: 1e-12,
                            }
                        } else {
                            ModelSpec::LR {
                                lambda,
                                tolerance: 1e-10,
                                max_iterations: 5000,
                            }
                        },
                    })
                    .collect(),
            }],
        };
        let result = train(&loaded.data, &plan, &training, 1).unwrap();
        let model = &result.models[0].model;
        let predictions = predict(model, &loaded.data).unwrap();
        let file = work.path(&format!("{name}.rad"));
        model.save(&file, false).unwrap();
        let restored = ModelBundle::<FittedPipeline>::load(&file).unwrap();
        assert_eq!(predictions, predict(&restored, &loaded.data).unwrap());
        assert_eq!(predictions, predict(&restored, &parquet_data.data).unwrap());
        let mut unlabeled_spec = config;
        unlabeled_spec.outcome_column = None;
        unlabeled_spec.numeric_features = Some(features);
        let unlabeled = read_study(&unlabeled_spec, task, &work.0).unwrap();
        assert!(matches!(unlabeled.data.outcome, Outcome::Unlabeled));
        assert_eq!(predictions, predict(&restored, &unlabeled.data).unwrap());
        assert!(train(&unlabeled.data, &plan, &training, 1).is_err());
    }
}
#[test]
fn bare_feature_paths_do_not_change_cwd_or_treat_empty_parent_as_a_directory() {
    struct Bare {
        root: PathBuf,
        paths: Vec<PathBuf>,
    }
    impl Drop for Bare {
        fn drop(&mut self) {
            for path in &self.paths {
                if let Ok(path) = fs::canonicalize(path) {
                    if path.parent() == Some(self.root.as_path())
                        && path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("radiomics-table-bare-"))
                    {
                        let _ = fs::remove_file(path);
                    }
                }
            }
        }
    }
    let cwd = fs::canonicalize(std::env::current_dir().unwrap()).unwrap();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = PathBuf::from(format!(
        "radiomics-table-bare-{}-{stamp}.csv",
        std::process::id()
    ));
    let sidecar = feature_manifest_path(&path);
    let _cleanup = Bare {
        root: cwd.clone(),
        paths: vec![path.clone(), sidecar.clone()],
    };
    let table = feature_table();
    write_feature_table(&table, &path, false, "").unwrap();
    assert_eq!(table, read_feature_table(&path, None).unwrap());
    let input = spec(
        json!({"schema_version":1,"table":path,"feature_manifest":sidecar,"default_cohort":"external"}),
    );
    assert_eq!(
        read_study(&input, Task::Binary, Path::new(""))
            .unwrap()
            .data
            .features
            .numeric,
        table.data
    );
    assert_eq!(
        fs::canonicalize(std::env::current_dir().unwrap()).unwrap(),
        cwd
    );
}
#[test]
fn missing_schema_conflicting_fields_present_missing_cohort_and_csv_nonfinite_are_errors() {
    let work = Workspace::new();
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,x\n001,P1,train,1\n002,P2,test,2\n",
    );
    work.csv("aux.csv", "sample_id,patient_id,x\n002,P2,2\n001,P9,1\n");
    let input = spec(
        json!({"schema_version":1,"table":"base.csv","clinical":"aux.csv","numeric_features":["x"]}),
    );
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("Conflicting"));
    work.csv("aux.csv", "sample_id,patient_id,x\n002,P2,3\n001,P1,1\n");
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("Conflicting"));
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,x\n001,P1,train,1\n002,P2,,2\n",
    );
    let input = spec(json!({"schema_version":1,"table":"base.csv","default_cohort":"external"}));
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("only when no cohort column exists"));
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,x,x\n001,P1,train,1,2\n",
    );
    assert!(read_study(&input, Task::Binary, &work.0).is_err());
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,x\n001,P1,train,NaN\n",
    );
    let mut input = spec(json!({"schema_version":1,"table":"base.csv","numeric_features":["x"]}));
    assert!(read_study(&input, Task::Binary, &work.0).is_err());
    input.nan_as_missing = true;
    assert!(
        read_study(&input, Task::Binary, &work.0)
            .unwrap()
            .data
            .features
            .numeric
            .missing[0]
    );
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,x\n001,P1,train,inf\n",
    );
    assert!(read_study(&input, Task::Binary, &work.0).is_err());
    work.csv(
        "base.csv",
        "sample_id,patient_id,cohort,center\n001,P1,train,1\n",
    );
    let input = spec(json!({"schema_version":1,"table":"base.csv","numeric_features":["center"]}));
    assert!(read_study(&input, Task::Binary, &work.0)
        .unwrap_err()
        .message
        .contains("simultaneously"));
}
