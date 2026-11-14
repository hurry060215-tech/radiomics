use ov_radiomics_core::{contracts::*, learning::*, table_io};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "radiomics-cli-workflows-{}-{stamp}",
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
        fs::write(self.path(name), serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }
    fn cli(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_radiomics"))
            .current_dir(&self.0)
            .env("PATH", self.path("empty-bin"))
            .env("PYTHONPATH", "")
            .args(args)
            .output()
            .unwrap()
    }
    fn success(&self, args: &[&str]) -> Value {
        let output = self.cli(args);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{:?}\nstdout {}\nstderr {}",
            args,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        if let (Ok(root), Ok(path)) = (
            fs::canonicalize(std::env::temp_dir()),
            fs::canonicalize(&self.0),
        ) {
            if path.starts_with(root)
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("radiomics-cli-workflows-"))
            {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
}
fn frozen(name: &str) -> Value {
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
fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn fixture(work: &Workspace, name: &str, changed_test: bool) {
    let input = frozen("model_inputs.json");
    let dataset = &input["datasets"][name];
    let rows = dataset["records"].as_array().unwrap();
    let classes: Vec<String> = serde_json::from_value(dataset["classes"].clone()).unwrap();
    let features: Vec<String> = serde_json::from_value(dataset["feature_names"].clone()).unwrap();
    let mut writer = csv::Writer::from_path(work.path(&format!("{name}.csv"))).unwrap();
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
    let mut values = Vec::new();
    for row in rows {
        let mut record: Vec<_> = ["sample_id", "patient_id", "cohort", "center", "lesion_id"]
            .into_iter()
            .map(|key| row[key].as_str().unwrap().to_string())
            .collect();
        let numeric: Vec<f64> = row["values"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(j, v)| {
                v.as_f64().unwrap()
                    + if changed_test && row["cohort"] == "test" && j == 0 {
                        1000.0
                    } else {
                        0.0
                    }
            })
            .collect();
        record.extend(numeric.iter().map(ToString::to_string));
        values.extend(numeric);
        record.push(if name == "regression" {
            row["target"]
                .as_f64()
                .map(|v| v.to_string())
                .unwrap_or_default()
        } else {
            row["target"]
                .as_u64()
                .map(|v| {
                    classes[if changed_test && name == "binary" && row["cohort"] == "test" {
                        1 - v as usize
                    } else {
                        v as usize
                    }]
                    .clone()
                })
                .unwrap_or_default()
        });
        writer.write_record(record).unwrap();
    }
    writer.flush().unwrap();
    let mut spec =
        json!({"schema_version":1,"table":format!("{name}.csv"),"outcome_column":"outcome"});
    work.json(&format!("{name}-data.json"), &spec);
    spec["outcome_column"] = Value::Null;
    spec["numeric_features"] = json!(features);
    work.json(&format!("{name}-unlabeled.json"), &spec);
    let samples: Vec<_> = rows
        .iter()
        .map(|r| SampleMetadata {
            sample_id: r["sample_id"].as_str().unwrap().into(),
            patient_id: r["patient_id"].as_str().unwrap().into(),
            slide_id: None,
            lesion_id: r["lesion_id"].as_str().map(str::to_string),
            modality: None,
            region: None,
            timepoint: None,
            center: r["center"].as_str().map(str::to_string),
        })
        .collect();
    let table = FeatureTable {
        data: NumericBuffer {
            rows: rows.len(),
            cols: 4,
            missing: vec![false; values.len()],
            values,
        },
        samples,
        features: features
            .iter()
            .map(|name| FeatureDefinition {
                name: name.clone(),
                definition: "Original frozen synthetic model input".into(),
                unit: None,
            })
            .collect(),
        effective_parameters: BTreeMap::new(),
        quality: vec![],
    };
    let parquet = work.path(&format!("{name}.parquet"));
    table_io::write_feature_table(&table, &parquet, true, "").unwrap();
    let mut metadata = csv::Writer::from_path(work.path(&format!("{name}-metadata.csv"))).unwrap();
    metadata
        .write_record(["sample_id", "cohort", "outcome"])
        .unwrap();
    for row in rows.iter().rev() {
        metadata
            .write_record([
                row["sample_id"].as_str().unwrap().to_string(),
                row["cohort"].as_str().unwrap().to_string(),
                if name == "regression" {
                    row["target"]
                        .as_f64()
                        .map(|v| v.to_string())
                        .unwrap_or_default()
                } else {
                    row["target"]
                        .as_u64()
                        .map(|c| classes[c as usize].clone())
                        .unwrap_or_default()
                },
            ])
            .unwrap();
    }
    metadata.flush().unwrap();
    work.json(&format!("{name}-parquet-data.json"),&json!({"schema_version":1,"table":format!("{name}.parquet"),"metadata":format!("{name}-metadata.csv"),"outcome_column":"outcome"}));
    let task: Task = serde_json::from_value(json!(name)).unwrap();
    let config = TrainingConfig {
        task,
        classes,
        fit_unit: PredictionUnit::Sample,
        scoring_unit: ScoreUnit::Patient,
        threshold: (task == Task::Binary).then_some(0.5),
        selection_tie_atol: 1e-12,
        models: vec![ModelGrid {
            name: if name == "regression" {
                "Ridge".into()
            } else {
                "LR".into()
            },
            candidates: [0.05, 0.1]
                .into_iter()
                .map(|lambda| PipelineSpec {
                    preprocessing: ov_radiomics_core::preprocessing::PreprocessConfig {
                        scaling: ov_radiomics_core::preprocessing::Scaling::Standard,
                        all_missing: ov_radiomics_core::preprocessing::AllMissingPolicy::Error,
                    },
                    selector: Some(ov_radiomics_core::preprocessing::SelectorConfig {
                        univariate: ov_radiomics_core::preprocessing::UnivariateMethod::None,
                        ..Default::default()
                    }),
                    model: if task == Task::Regression {
                        ov_radiomics_core::models::ModelSpec::Ridge {
                            lambda,
                            rank_tolerance: 1e-12,
                        }
                    } else {
                        ov_radiomics_core::models::ModelSpec::LR {
                            lambda,
                            tolerance: 1e-10,
                            max_iterations: 5000,
                        }
                    },
                })
                .collect(),
        }],
    };
    work.json(
        &format!("{name}-config.json"),
        &serde_json::to_value(config).unwrap(),
    );
    work.json(
        &format!("{name}-plan.json"),
        &frozen("workflow_plans.json")[name],
    );
}

#[test]
fn native_cli_three_tasks_train_predict_save_load_csv_parquet_and_no_python_path() {
    let work = Workspace::new();
    for name in ["binary", "multiclass", "regression"] {
        fixture(&work, name, false);
        let data = format!("{name}-data.json");
        let config = format!("{name}-config.json");
        let plan = format!("{name}-plan.json");
        let models = format!("{name}-models");
        work.success(&[
            "train",
            "--data",
            &data,
            "--config",
            &config,
            "--split-plan",
            &plan,
            "--output",
            &models,
        ]);
        assert_eq!(
            json_file(&work.path(&format!("{models}/split_plan.json"))),
            frozen("workflow_plans.json")[name]
        );
        let summary = json_file(&work.path(&format!("{models}/summary.json")));
        let model_file = format!("{models}/{}", summary["best_model_file"].as_str().unwrap());
        assert!(work.path(&model_file).exists());
        assert!(work.path(&format!("{models}/model-000-oof.csv")).exists());
        let prediction = format!("{name}-prediction");
        work.success(&[
            "predict",
            "--model",
            &model_file,
            "--data",
            &data,
            "--output",
            &prediction,
        ]);
        let expected = json_file(&work.path(&format!("{prediction}/predictions.json")));
        let unlabeled = format!("{name}-unlabeled.json");
        let no_labels = format!("{name}-prediction-unlabeled");
        work.success(&[
            "predict",
            "--model",
            &model_file,
            "--data",
            &unlabeled,
            "--output",
            &no_labels,
        ]);
        assert_eq!(
            expected,
            json_file(&work.path(&format!("{no_labels}/predictions.json")))
        );
        assert_eq!(
            json_file(&work.path(&format!("{no_labels}/prediction_metadata.json")))
                ["fit_performed"],
            false
        );
        let parquet = format!("{name}-parquet-data.json");
        let parquet_predictions = format!("{name}-prediction-parquet");
        work.success(&[
            "predict",
            "--model",
            &model_file,
            "--data",
            &parquet,
            "--output",
            &parquet_predictions,
        ]);
        assert_eq!(
            expected,
            json_file(&work.path(&format!("{parquet_predictions}/predictions.json")))
        );
        let parity_models = format!("{name}-parquet-models");
        work.success(&[
            "train",
            "--data",
            &parquet,
            "--config",
            &config,
            "--split-plan",
            &plan,
            "--output",
            &parity_models,
        ]);
        assert_eq!(
            fs::read(work.path(&model_file)).unwrap(),
            fs::read(work.path(&format!("{parity_models}/model-000.rad"))).unwrap()
        );
    }
}
#[test]
fn cli_overwrite_is_scoped_unknown_models_fail_and_holdout_sha_never_changes_fitted_state() {
    let work = Workspace::new();
    fixture(&work, "binary", false);
    let args = [
        "train",
        "--data",
        "binary-data.json",
        "--config",
        "binary-config.json",
        "--split-plan",
        "binary-plan.json",
        "--output",
        "models",
    ];
    work.success(&args);
    let before = fs::read(work.path("models/model-000.rad")).unwrap();
    let before_report = fs::read(work.path("models/load_report.json")).unwrap();
    fs::write(work.path("models/user-notes.txt"), b"keep").unwrap();
    assert_eq!(work.cli(&args).status.code(), Some(1));
    fixture(&work, "binary", true);
    let mut overwrite = args.to_vec();
    overwrite.push("--overwrite");
    work.success(&overwrite);
    assert_eq!(before, fs::read(work.path("models/model-000.rad")).unwrap());
    assert_ne!(
        before_report,
        fs::read(work.path("models/load_report.json")).unwrap()
    );
    assert_eq!(
        fs::read(work.path("models/user-notes.txt")).unwrap(),
        b"keep"
    );
    let mut bad = json_file(&work.path("models/model-000.rad"));
    bad["manifest"]["training_configuration"]["model_kind"] = json!("unknown");
    work.json("unknown.rad", &bad);
    let output = work.cli(&[
        "predict",
        "--model",
        "unknown.rad",
        "--data",
        "binary-data.json",
        "--output",
        "bad-model",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Unknown model_kind"));
    assert!(!work.path("bad-model").exists());
    bad["manifest"]["schema_version"] = json!(999);
    work.json("unknown.rad", &bad);
    assert_eq!(
        work.cli(&[
            "predict",
            "--model",
            "unknown.rad",
            "--data",
            "binary-data.json",
            "--output",
            "bad-schema"
        ])
        .status
        .code(),
        Some(1)
    );
}
