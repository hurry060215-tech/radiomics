//! Independent workflow acceptance of every nested patient fit and held-out invariance.
use ov_radiomics_core::{
    contracts::*,
    learning::*,
    models::ModelSpec,
    preprocessing::{
        AllMissingPolicy, PreprocessConfig, Scaling, SelectorConfig, UnivariateMethod,
    },
    split::SplitPlan,
};
use serde_json::Value;
use std::collections::BTreeMap;

fn read(name: &str) -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().into())
        .collect()
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * b.abs(),
        "{a:.17e} != {b:.17e}"
    )
}
fn study(task: &str) -> StudyData {
    let input = read("model_inputs.json");
    let d = &input["datasets"][task];
    let records = d["records"].as_array().unwrap();
    let classes = strings(&d["classes"]);
    let values: Vec<_> = records
        .iter()
        .flat_map(|r| {
            r["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
        })
        .collect();
    StudyData {
        samples: records
            .iter()
            .map(|r| SampleMetadata {
                sample_id: r["sample_id"].as_str().unwrap().into(),
                patient_id: r["patient_id"].as_str().unwrap().into(),
                slide_id: None,
                lesion_id: Some(r["lesion_id"].as_str().unwrap().into()),
                modality: None,
                region: None,
                timepoint: None,
                center: Some(r["center"].as_str().unwrap().into()),
            })
            .collect(),
        cohorts: records
            .iter()
            .map(|r| r["cohort"].as_str().unwrap().into())
            .collect(),
        features: TabularData {
            rows: records.len(),
            numeric_names: strings(&d["feature_names"]),
            numeric: NumericBuffer {
                rows: records.len(),
                cols: 4,
                missing: vec![false; values.len()],
                values,
            },
            categorical_names: vec![],
            categorical_values: vec![],
        },
        outcome: if task == "regression" {
            Outcome::Regression {
                name: "outcome".into(),
                values: records.iter().map(|r| r["target"].as_f64()).collect(),
            }
        } else {
            Outcome::Classification {
                name: "outcome".into(),
                values: records
                    .iter()
                    .map(|r| r["target"].as_u64().map(|i| classes[i as usize].clone()))
                    .collect(),
            }
        },
    }
}
fn configuration(task: &str) -> TrainingConfig {
    let kind: Task = serde_json::from_value(task.into()).unwrap();
    TrainingConfig {
        task: kind,
        classes: strings(&read("model_inputs.json")["datasets"][task]["classes"]),
        fit_unit: PredictionUnit::Sample,
        scoring_unit: ScoreUnit::Patient,
        threshold: (task == "binary").then_some(0.5),
        selection_tie_atol: 1e-12,
        models: vec![ModelGrid {
            name: if task == "regression" { "Ridge" } else { "LR" }.into(),
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
                    model: if task == "regression" {
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
    }
}
fn check_fit(actual: &FitAudit, expected: &Value) {
    assert_eq!(
        actual.training_sample_ids,
        strings(&expected["fitted_sample_ids"])
    );
    assert_eq!(
        actual.training_patients.iter().cloned().collect::<Vec<_>>(),
        strings(&expected["fitted_patient_ids"])
    );
    assert_eq!(
        actual.selected_features,
        strings(&expected["selected_names"])
    );
    for (j, state) in actual.preprocessing.numeric.iter().enumerate() {
        close(state.median, expected["medians"][j].as_f64().unwrap());
        close(state.mean, expected["means"][j].as_f64().unwrap());
        close(state.scale, expected["scales"][j].as_f64().unwrap());
    }
    close(
        actual.model_report.objective.unwrap(),
        expected["numerical_state"]["objective"].as_f64().unwrap(),
    );
    assert!(actual.model_report.gradient_or_kkt_inf.unwrap() <= 1e-7);
}
fn check_tuning(actual: &TuningReport, expected: &Value) {
    assert_eq!(
        actual.selected_candidate,
        expected["selected_candidate_index"].as_u64().unwrap() as usize
    );
    assert_eq!(actual.selection_tie_atol, 1e-12);
    for (candidate, reference) in actual
        .candidates
        .iter()
        .zip(expected["candidates"].as_array().unwrap())
    {
        close(
            candidate.mean_score.unwrap(),
            reference["mean_score"].as_f64().unwrap(),
        );
        for (fold, expected) in candidate
            .inner_folds
            .iter()
            .zip(reference["inner_folds"].as_array().unwrap())
        {
            assert_eq!(
                fold.partition
                    .training_patients
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                strings(&expected["training_patients"])
            );
            assert_eq!(
                fold.partition
                    .validation_patients
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                strings(&expected["validation_patients"])
            );
            check_fit(&fold.fit, &expected["fit_state"]);
            close(fold.score, expected["score"].as_f64().unwrap());
        }
    }
}
fn check_predictions(actual: &PredictionTable, expected: &Value, regression: bool) {
    let map: BTreeMap<_, _> = actual
        .rows
        .iter()
        .map(|r| (r.sample_id.as_str(), r))
        .collect();
    assert_eq!(map.len(), expected.as_array().unwrap().len());
    for row in expected.as_array().unwrap() {
        let id = row["sample_id"].as_str().unwrap();
        let found = map[id];
        assert_eq!(found.patient_id, row["patient_id"].as_str().unwrap());
        assert_eq!(found.cohort, row["cohort"].as_str().unwrap());
        if let Some(fold) = row["fold"].as_u64() {
            assert_eq!(found.fold, Some(fold as usize));
        }
        if regression {
            let b = row["value"].as_f64().unwrap();
            assert!((found.value.unwrap() - b).abs() <= 1e-6 + 1e-5 * b.abs());
        } else {
            assert_eq!(
                found.probabilities.len(),
                row["probabilities"].as_array().unwrap().len()
            );
            for (i, &a) in found.probabilities.iter().enumerate() {
                let b = row["probabilities"][i].as_f64().unwrap();
                assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs());
            }
        }
    }
}

#[test]
fn every_nested_fit_tuning_and_prediction_matches_frozen_independent_workflow() {
    let references = read("workflow_reference.json");
    let plans = read("workflow_plans.json");
    for task in ["binary", "multiclass", "regression"] {
        let data = study(task);
        let config = configuration(task);
        let plan: SplitPlan = serde_json::from_value(plans[task].clone()).unwrap();
        let result = train(&data, &plan, &config, 1).unwrap();
        result.validate().unwrap();
        let run = &result.models[0];
        let expected = &references["datasets"][task];
        run.validate_provenance(&plan).unwrap();
        for (fold, reference) in run
            .outer_folds
            .iter()
            .zip(expected["outer_folds"].as_array().unwrap())
        {
            check_tuning(&fold.tuning, &reference["tuning"]);
            check_fit(&fold.fit, &reference["fit_state"]);
            close(fold.score, reference["score"].as_f64().unwrap());
        }
        check_tuning(&run.final_tuning, &expected["final_tuning"]);
        close(
            run.mean_cv_score,
            expected["mean_outer_patient_score"].as_f64().unwrap(),
        );
        check_predictions(
            &run.oof,
            &expected["oof_sample_predictions"],
            task == "regression",
        );
        check_predictions(
            &run.predictions,
            &expected["final_sample_predictions"],
            task == "regression",
        );
        assert_eq!(
            run.model.fitted.selected_features,
            strings(&expected["final_fit_state"]["selected_names"])
        );
        for (j, state) in run.model.fitted.preprocessing.numeric.iter().enumerate() {
            close(
                state.mean,
                expected["final_fit_state"]["means"][j].as_f64().unwrap(),
            );
            close(
                state.scale,
                expected["final_fit_state"]["scales"][j].as_f64().unwrap(),
            );
        }
    }
}

#[test]
fn heldout_unknown_labels_and_large_features_never_change_fits_oof_or_threshold() {
    let data = study("binary");
    let config = configuration("binary");
    let plan: SplitPlan =
        serde_json::from_value(read("workflow_plans.json")["binary"].clone()).unwrap();
    let baseline = train(&data, &plan, &config, 1).unwrap();
    let mut changed = data.clone();
    for i in 0..changed.samples.len() {
        if changed.cohorts[i] != "train" {
            for j in 0..4 {
                changed.features.numeric.values[i * 4 + j] = 1e9 + j as f64;
            }
            if let Outcome::Classification { values, .. } = &mut changed.outcome {
                values[i] = Some("new_unseen_heldout_label".into());
            }
        }
    }
    let modified = train(&changed, &plan, &config, 3).unwrap();
    assert_eq!(baseline.models[0].model, modified.models[0].model);
    assert_eq!(baseline.models[0].oof, modified.models[0].oof);
    assert_eq!(
        baseline.models[0].outer_folds,
        modified.models[0].outer_folds
    );
    assert_eq!(baseline.models[0].model.manifest.threshold, Some(0.5));
    let indices: Vec<_> = (0..data.samples.len()).rev().collect();
    let reversed = train(&data.take_rows(&indices).unwrap(), &plan, &config, 3).unwrap();
    assert_eq!(baseline, reversed);
    let external_indices: Vec<_> = data
        .cohorts
        .iter()
        .enumerate()
        .filter_map(|(i, c)| (c == "external_A").then_some(i))
        .collect();
    let mut external = data.take_rows(&external_indices).unwrap();
    external.outcome = Outcome::Classification {
        name: "outcome".into(),
        values: vec![None; external.samples.len()],
    };
    let path = std::env::temp_dir().join(format!(
        "golden-pipeline-independent-{}.rad",
        std::process::id()
    ));
    baseline.models[0].model.save(&path, true).unwrap();
    let loaded = ov_radiomics_core::bundle::ModelBundle::<FittedPipeline>::load(&path).unwrap();
    assert_eq!(
        predict(&baseline.models[0].model, &external).unwrap(),
        predict(&loaded, &external).unwrap()
    );
    std::fs::remove_file(path).unwrap();
}
