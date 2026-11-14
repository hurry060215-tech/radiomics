use ov_radiomics_core::{contracts::*, learning};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
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
            "radiomics-cli-centers-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        fs::create_dir(path.join("empty-bin")).unwrap();
        Self(path)
    }
    fn json(&self, name: &str, value: &impl serde::Serialize) {
        fs::write(self.0.join(name), serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }
    fn read(&self, name: &str) -> Value {
        serde_json::from_slice(&fs::read(self.0.join(name)).unwrap()).unwrap()
    }
    fn cli(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_radiomics"))
            .current_dir(&self.0)
            .env("PATH", self.0.join("empty-bin"))
            .env("PYTHONPATH", "")
            .args(args)
            .output()
            .unwrap()
    }
    fn success(&self, args: &[&str]) {
        let result = self.cli(args);
        assert_eq!(
            result.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
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
                    .is_some_and(|n| n.starts_with("radiomics-cli-centers-"))
            {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
}

fn fixture(work: &Workspace, task: Task, missing_center: bool) -> PredictionTable {
    let classes = match task {
        Task::Binary => vec!["negative".into(), "positive".into()],
        Task::Multiclass => vec!["A".into(), "B".into(), "C".into()],
        Task::Regression => vec![],
    };
    let mut writer = csv::Writer::from_path(work.0.join("study.csv")).unwrap();
    writer
        .write_record([
            "sample_id",
            "patient_id",
            "slide_id",
            "cohort",
            "center",
            "x",
            "outcome",
        ])
        .unwrap();
    let mut predictions = PredictionTable {
        task,
        unit: PredictionUnit::Sample,
        classes: classes.clone(),
        rows: vec![],
    };
    for (center_index, center) in ["Center/A", "Center B"].iter().enumerate() {
        for cohort in ["train", "test", "external_unlabeled"] {
            for patient in 0..6 {
                // A true patient may have finer-unit observations at both centers.
                let patient_id = format!("{cohort}-P{patient}");
                let slide = format!("{patient_id}-slide");
                for lesion in 0..2 {
                    let id = format!("{patient_id}-C{center_index}-L{lesion}");
                    let class = patient % classes.len().max(1);
                    let target = if cohort == "external_unlabeled" {
                        String::new()
                    } else if task == Task::Regression {
                        (patient as f64 * 2.).to_string()
                    } else {
                        classes[class].clone()
                    };
                    writer
                        .write_record([
                            id.clone(),
                            patient_id.clone(),
                            slide.clone(),
                            cohort.into(),
                            if missing_center && center_index == 0 && patient == 0 && lesion == 0 {
                                String::new()
                            } else {
                                center.to_string()
                            },
                            patient.to_string(),
                            target,
                        ])
                        .unwrap();
                    let probabilities = match task {
                        Task::Binary => {
                            let p = (if class == 0 { 0.2 } else { 0.8 })
                                + if center_index == 1 {
                                    if class == 0 {
                                        0.2
                                    } else {
                                        -0.2
                                    }
                                } else {
                                    0.
                                }
                                + 0.05 * lesion as f64;
                            vec![1. - p, p]
                        }
                        Task::Multiclass => (0..3)
                            .map(|c| {
                                if c == class {
                                    if center_index == 0 {
                                        0.8
                                    } else {
                                        0.6
                                    }
                                } else if center_index == 0 {
                                    0.1
                                } else {
                                    0.2
                                }
                            })
                            .collect(),
                        Task::Regression => vec![],
                    };
                    predictions.rows.push(PredictionRow {
                        sample_id: id,
                        patient_id: patient_id.clone(),
                        slide_id: Some(slide.clone()),
                        tile_id: None,
                        cohort: cohort.into(),
                        fold: None,
                        training_source: "provided:fixed-center-fixture".into(),
                        probabilities,
                        value: (task == Task::Regression).then_some(
                            patient as f64 * 2. + 0.2 + center_index as f64 + 0.2 * lesion as f64,
                        ),
                    });
                }
            }
        }
    }
    writer.flush().unwrap();
    work.json("study.json",&json!({"schema_version":1,"table":"study.csv","numeric_features":["x"],"outcome_column":"outcome"}));
    predictions.validate().unwrap();
    work.json("predictions.json", &predictions);
    predictions
}
fn evaluate_args<'a>(source: &'a str, output: &'a str, by_center: bool) -> Vec<&'a str> {
    let mut args = vec![
        "evaluate",
        "--predictions",
        source,
        "--data",
        "study.json",
        "--n-bootstrap",
        "8",
        "--output",
        output,
    ];
    if by_center {
        args.push("--by-center");
    }
    args
}

#[test]
fn all_tasks_preserve_overall_results_and_export_true_center_patient_scores() {
    for task in [Task::Binary, Task::Multiclass, Task::Regression] {
        let work = Workspace::new();
        let source = fixture(&work, task, false);
        work.success(&evaluate_args("predictions.json", "overall", false));
        work.success(&evaluate_args("predictions.json", "by-center", true));
        assert_eq!(
            work.read("overall/predictions.json"),
            work.read("by-center/predictions.json")
        );
        let overall = work.read("overall/summary.json");
        let centered = work.read("by-center/summary.json");
        assert!(overall.get("centers").is_none());
        assert!(overall.get("by_center").is_none());
        assert_eq!(overall["cohorts"], centered["cohorts"]);
        assert_eq!(centered["centers"].as_array().unwrap().len(), 2);
        for cohort in overall["cohorts"].as_array().unwrap() {
            let file = cohort["file"].as_str().unwrap();
            assert_eq!(
                work.read(&format!("overall/{file}")),
                work.read(&format!("by-center/{file}"))
            );
        }
        let mut center_tables = BTreeMap::new();
        for center in centered["centers"].as_array().unwrap() {
            let name = center["center"].as_str().unwrap();
            let table: PredictionTable = serde_json::from_value(work.read(&format!(
                "by-center/{}",
                center["predictions_file"].as_str().unwrap()
            )))
            .unwrap();
            table.validate().unwrap();
            assert_eq!(table.unit, PredictionUnit::Patient);
            assert_eq!(table.rows.len(), 18);
            let expected_source = PredictionTable {
                rows: source
                    .rows
                    .iter()
                    .filter(|row| {
                        row.sample_id
                            .contains(if name == "Center/A" { "-C0-" } else { "-C1-" })
                    })
                    .cloned()
                    .collect(),
                ..source.clone()
            };
            assert_eq!(
                table,
                learning::aggregate_patient_predictions(&expected_source).unwrap()
            );
            for report in center["cohorts"].as_array().unwrap() {
                let value = work.read(&format!("by-center/{}", report["file"].as_str().unwrap()));
                assert_eq!(value["center"], name);
                assert_eq!(value["cohort"], report["cohort"]);
                assert_eq!(value["n_predictions"], 6);
                if report["cohort"] == "external_unlabeled" {
                    assert_eq!(value["status"], "prediction_only");
                    assert!(value["metrics"].is_null());
                } else {
                    assert_eq!(value["status"], "evaluated");
                    if task == Task::Binary {
                        let expected = if name == "Center/A" {
                            0.040625
                        } else {
                            0.160625
                        };
                        assert!(
                            (value["metrics"]["brier"].as_f64().unwrap() - expected).abs() < 1e-12
                        );
                    } else if task == Task::Regression {
                        let expected = if name == "Center/A" { 0.3 } else { 1.3 };
                        assert!(
                            (value["metrics"]["rmse"].as_f64().unwrap() - expected).abs() < 1e-12
                        );
                    }
                }
            }
            center_tables.insert(name.to_owned(), table);
        }
        assert_ne!(
            center_tables["Center/A"].rows[0],
            center_tables["Center B"].rows[0]
        );
        // The center name containing '/' is data, never a path component.
        assert!(!work.0.join("by-center/Center").exists());
    }
}

#[test]
fn center_metadata_is_required_only_when_requested_and_cannot_split_patient_or_slide_scores() {
    let work = Workspace::new();
    fixture(&work, Task::Binary, true);
    work.success(&evaluate_args(
        "predictions.json",
        "default-without-center",
        false,
    ));
    let failed = work.cli(&evaluate_args("predictions.json", "missing-center", true));
    assert_eq!(failed.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failed.stderr).contains("one known center"));
    assert!(!work.0.join("missing-center").exists());
    let work = Workspace::new();
    let source = fixture(&work, Task::Binary, false);
    let patient = learning::aggregate_patient_predictions(&source).unwrap();
    work.json("patient.json", &patient);
    let failed = work.cli(&evaluate_args("patient.json", "mixed-patient", true));
    assert_eq!(failed.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failed.stderr).contains("already aggregated"));
    assert!(!work.0.join("mixed-patient").exists());
    let slide = PredictionTable {
        unit: PredictionUnit::Slide,
        rows: patient
            .rows
            .iter()
            .map(|row| PredictionRow {
                sample_id: format!("{}-slide", row.patient_id),
                slide_id: Some(format!("{}-slide", row.patient_id)),
                ..row.clone()
            })
            .collect(),
        ..patient
    };
    work.json("slide.json", &slide);
    let failed = work.cli(&evaluate_args("slide.json", "mixed-slide", true));
    assert_eq!(failed.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&failed.stderr).contains("already aggregated"));
    assert!(!work.0.join("mixed-slide").exists());
}

#[test]
fn already_patient_scores_are_valid_when_each_patient_has_one_recorded_center() {
    let work = Workspace::new();
    let source = fixture(&work, Task::Regression, false);
    let patient = learning::aggregate_patient_predictions(&source).unwrap();
    work.json("patient.json", &patient);
    let mut reader = csv::Reader::from_path(work.0.join("study.csv")).unwrap();
    let headers = reader.headers().unwrap().clone();
    let rows: Vec<_> = reader.records().map(Result::unwrap).collect();
    let mut writer = csv::Writer::from_path(work.0.join("one-center-per-patient.csv")).unwrap();
    writer.write_record(&headers).unwrap();
    for row in rows {
        let mut values: Vec<_> = row.iter().map(str::to_owned).collect();
        let id = values[1]
            .rsplit_once('P')
            .unwrap()
            .1
            .parse::<usize>()
            .unwrap();
        values[4] = if id < 3 { "OnlyA" } else { "OnlyB" }.into();
        writer.write_record(values).unwrap();
    }
    writer.flush().unwrap();
    work.json("study.json",&json!({"schema_version":1,"table":"one-center-per-patient.csv","numeric_features":["x"],"outcome_column":"outcome"}));
    work.success(&evaluate_args("patient.json", "patient-centers", true));
    let summary = work.read("patient-centers/summary.json");
    for center in summary["centers"].as_array().unwrap() {
        let table: PredictionTable = serde_json::from_value(work.read(&format!(
            "patient-centers/{}",
            center["predictions_file"].as_str().unwrap()
        )))
        .unwrap();
        assert_eq!(table.unit, PredictionUnit::Patient);
        assert_eq!(table.rows.len(), 9);
        for row in &table.rows {
            assert_eq!(
                patient
                    .rows
                    .iter()
                    .find(|original| original.patient_id == row.patient_id)
                    .unwrap(),
                row
            );
        }
    }
}
