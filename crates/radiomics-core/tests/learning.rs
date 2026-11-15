use ov_radiomics_core::{
    bundle::ModelBundle,
    contracts::*,
    learning::*,
    models::ModelSpec,
    preprocessing::{
        AllMissingPolicy, PreprocessConfig, Scaling, SelectorConfig, UnivariateMethod,
    },
    split::SplitPlan,
};
use serde_json::Value;
fn load(name: &str) -> Value {
    let root = std::path::Path::new(env!("RAD_GOLDEN_DIR")).join("python_golden_v1");
    serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap()
}
fn data(task: &str) -> StudyData {
    let input = load("model_inputs.json");
    let d = &input["datasets"][task];
    let records = d["records"].as_array().unwrap();
    let classes: Vec<String> = serde_json::from_value(d["classes"].clone()).unwrap();
    let values: Vec<f64> = records
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
            numeric_names: serde_json::from_value(d["feature_names"].clone()).unwrap(),
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
fn config(task: &str) -> TrainingConfig {
    let input = load("model_inputs.json");
    let task_enum: Task = serde_json::from_value(task.into()).unwrap();
    TrainingConfig {
        task: task_enum,
        classes: serde_json::from_value(input["datasets"][task]["classes"].clone()).unwrap(),
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
fn plan(task: &str) -> SplitPlan {
    serde_json::from_value(load("workflow_plans.json")[task].clone()).unwrap()
}
fn close(a: f64, b: f64, atol: f64, rtol: f64) {
    assert!(
        (a - b).abs() <= atol + rtol * b.abs(),
        "{a:.17e} != {b:.17e}"
    );
}
fn predictions(table: &PredictionTable, expected: &Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(table.rows.len(), expected.len());
    for row in &table.rows {
        let r = expected
            .iter()
            .find(|r| r["sample_id"] == row.sample_id)
            .unwrap();
        assert_eq!(row.patient_id, r["patient_id"].as_str().unwrap());
        assert_eq!(row.cohort, r["cohort"].as_str().unwrap());
        if let Some(value) = row.value {
            close(value, r["value"].as_f64().unwrap(), 1e-6, 1e-5);
        } else {
            for (p, q) in row
                .probabilities
                .iter()
                .zip(r["probabilities"].as_array().unwrap())
            {
                close(*p, q.as_f64().unwrap(), 1e-6, 1e-5);
            }
        }
        if let Some(fold) = row.fold {
            assert_eq!(fold, r["fold"].as_u64().unwrap() as usize);
        }
    }
}
fn fit_statistics(fit: &FitAudit, reference: &Value) {
    assert_eq!(
        fit.training_sample_ids,
        serde_json::from_value::<Vec<String>>(reference["fitted_sample_ids"].clone()).unwrap()
    );
    assert_eq!(
        fit.selected_features,
        serde_json::from_value::<Vec<String>>(reference["selected_names"].clone()).unwrap()
    );
    for (i, c) in fit.preprocessing.numeric.iter().enumerate() {
        for (actual, key) in [
            (c.median, "medians"),
            (c.mean, "means"),
            (c.scale, "scales"),
        ] {
            close(actual, reference[key][i].as_f64().unwrap(), 1e-10, 1e-8);
        }
    }
}
#[test]
fn all_three_nested_workflows_match_implementation_independent_reference() {
    let reference = load("workflow_reference.json");
    for task in ["binary", "multiclass", "regression"] {
        let result = train(&data(task), &plan(task), &config(task), 2).unwrap();
        result.validate().unwrap();
        let model = &result.models[0];
        let expected = &reference["datasets"][task];
        predictions(&model.oof, &expected["oof_sample_predictions"]);
        predictions(&model.predictions, &expected["final_sample_predictions"]);
        assert_eq!(
            model.final_tuning.selected_candidate,
            expected["final_tuning"]["selected_candidate_index"]
                .as_u64()
                .unwrap() as usize
        );
        close(
            model.mean_cv_score,
            expected["mean_outer_patient_score"].as_f64().unwrap(),
            1e-10,
            1e-8,
        );
        for (i, fold) in model.outer_folds.iter().enumerate() {
            let r = &expected["outer_folds"][i];
            assert_eq!(
                fold.tuning.selected_candidate,
                r["tuning"]["selected_candidate_index"].as_u64().unwrap() as usize
            );
            fit_statistics(&fold.fit, &r["fit_state"]);
            close(fold.score, r["score"].as_f64().unwrap(), 1e-10, 1e-8);
            for (j, candidate) in fold.tuning.candidates.iter().enumerate() {
                let r = &r["tuning"]["candidates"][j];
                close(
                    candidate.mean_score.unwrap(),
                    r["mean_score"].as_f64().unwrap(),
                    1e-10,
                    1e-8,
                );
                for (k, inner) in candidate.inner_folds.iter().enumerate() {
                    fit_statistics(&inner.fit, &r["inner_folds"][k]["fit_state"]);
                    close(
                        inner.score,
                        r["inner_folds"][k]["score"].as_f64().unwrap(),
                        1e-10,
                        1e-8,
                    );
                }
            }
        }
    }
}
#[test]
fn holdout_and_external_perturbations_cannot_change_state_threshold_or_oof() {
    for task in ["binary", "multiclass", "regression"] {
        let original = data(task);
        let first = train(&original, &plan(task), &config(task), 1).unwrap();
        let mut changed = original.clone();
        for i in 0..changed.samples.len() {
            if changed.cohorts[i] != "train" {
                for j in 0..changed.features.numeric.cols {
                    changed.features.numeric.values[i * 4 + j] = 123456. * (j as f64 + 1.);
                }
                match &mut changed.outcome {
                    Outcome::Unlabeled => unreachable!(),
                    Outcome::Classification { values, .. } => {
                        values[i] = Some("unseen-external-label".into())
                    }
                    Outcome::Regression { values, .. } => values[i] = Some(987654.),
                }
            }
        }
        let second = train(&changed, &plan(task), &config(task), 3).unwrap();
        assert_eq!(first.models[0].model, second.models[0].model);
        assert_eq!(first.models[0].oof, second.models[0].oof);
        assert_eq!(first.models[0].outer_folds, second.models[0].outer_folds);
        assert_eq!(first.models[0].final_tuning, second.models[0].final_tuning);
        assert_eq!(first.best_model, second.best_model);
        let reverse = original
            .take_rows(&(0..original.samples.len()).rev().collect::<Vec<_>>())
            .unwrap();
        let reordered = train(&reverse, &plan(task), &config(task), 2).unwrap();
        assert_eq!(first, reordered);
    }
}
#[test]
fn portable_pipeline_bundle_reloads_and_predicts_without_labels() {
    let data = data("binary");
    let result = train(&data, &plan("binary"), &config("binary"), 1).unwrap();
    let model = &result.models[0];
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let file = std::env::temp_dir().join(format!(
        "native-pipeline-{}-{nonce}.rad",
        std::process::id()
    ));
    model.model.save(&file, false).unwrap();
    let restored = ModelBundle::<FittedPipeline>::load(&file).unwrap();
    assert_eq!(model.model, restored);
    let mut unlabeled = data.clone();
    unlabeled.outcome = Outcome::Unlabeled;
    assert_eq!(predict(&restored, &unlabeled).unwrap(), model.predictions);
    let mut overlapping = unlabeled.clone();
    let last = overlapping.cohorts.len() - 1;
    overlapping.cohorts[last] = "another_external_cohort".into();
    assert!(predict(&restored, &overlapping).is_err());
    std::fs::remove_file(file).unwrap();
    let mut bad = result.clone();
    bad.models[0].outer_folds[0]
        .partition
        .training_patients
        .clear();
    assert!(bad.validate().is_err());
    let mut bad = result.clone();
    bad.models[0].oof.rows[0].training_source = "oof:wrong-patients".into();
    assert!(bad.validate().is_err());
    let mut bad = result.clone();
    bad.models[0].outer_folds[0].tuning.candidates[0].inner_folds[0]
        .partition
        .training_patients
        .clear();
    assert!(bad.validate().is_err());
    let mut bad = result.clone();
    bad.models[0].oof.classes.swap(0, 1);
    assert!(bad.validate().is_err());
    let mut bad = result.clone();
    bad.models[0].final_tuning.selected_candidate =
        1 - bad.models[0].final_tuning.selected_candidate;
    assert!(bad.validate().is_err());
}

fn subset_features(data: &StudyData, indices: &[usize]) -> StudyData {
    let mut result = data.clone();
    result.features.numeric_names = indices
        .iter()
        .map(|&i| data.features.numeric_names[i].clone())
        .collect();
    let values: Vec<_> = (0..data.features.rows)
        .flat_map(|r| {
            indices
                .iter()
                .map(move |&c| data.features.numeric.values[r * data.features.numeric.cols + c])
        })
        .collect();
    result.features.numeric = NumericBuffer {
        rows: data.features.rows,
        cols: indices.len(),
        missing: vec![false; values.len()],
        values,
    };
    result
}
fn fixed_config(task: &str) -> TrainingConfig {
    let mut c = config(task);
    c.models[0].candidates = vec![c.models[0].candidates[1].clone()];
    c
}

#[test]
fn patient_fusion_matches_frozen_reference_and_never_saves_holdout_numerics() {
    let original = data("binary");
    let plan = plan("binary");
    let cfg = fixed_config("binary");
    let signal_data = subset_features(&original, &[0, 1]);
    let noise_data = subset_features(&original, &[2, 3]);
    let signal = train(&signal_data, &plan, &cfg, 1).unwrap();
    let noise = train(&noise_data, &plan, &cfg, 1).unwrap();
    let inputs = [
        FusionInput {
            name: "signal",
            training: &signal,
            model_name: "LR",
        },
        FusionInput {
            name: "noise",
            training: &noise,
            model_name: "LR",
        },
    ];
    let reference = load("fusion_reference.json");
    for method in [
        FusionMethod::Weighted {
            weights: vec![0.5, 0.5],
        },
        FusionMethod::Logistic {
            lambda: 0.1,
            tolerance: 1e-10,
            max_iterations: 5000,
        },
    ] {
        let logistic = matches!(method, FusionMethod::Logistic { .. });
        let config = FusionConfig {
            method,
            threshold: Some(0.5),
        };
        let result = fit_fusion(&inputs, &config).unwrap();
        result.model.validate().unwrap();
        for row in &result.predictions.rows {
            let expected = reference["final_patient_predictions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["patient_id"] == row.patient_id)
                .unwrap();
            assert_eq!(row.cohort, expected["cohort"].as_str().unwrap());
            let name = if logistic {
                "logistic_probabilities"
            } else {
                "weighted_probabilities"
            };
            for (p, q) in row
                .probabilities
                .iter()
                .zip(expected[name].as_array().unwrap())
            {
                close(*p, q.as_f64().unwrap(), 1e-6, 1e-5);
            }
        }
        assert_eq!(result.predictions.unit, PredictionUnit::Patient);
        assert_eq!(
            result.training_prediction_type,
            if logistic {
                "apparent_meta_training"
            } else {
                "oof_fixed_weight"
            }
        );
        if logistic {
            let model = result.model.fitted.meta_model.as_ref().unwrap();
            close(
                model.report.objective.unwrap(),
                reference["meta_fit"]["numerical_state"]["objective"]
                    .as_f64()
                    .unwrap(),
                1e-10,
                1e-8,
            );
            assert!(model.report.gradient_or_kkt_inf.unwrap() <= 1e-7);
        }
        let mut changed_signal = signal_data.clone();
        let mut changed_noise = noise_data.clone();
        for dataset in [&mut changed_signal, &mut changed_noise] {
            for i in 0..dataset.samples.len() {
                if dataset.cohorts[i] != "train" {
                    for j in 0..dataset.features.numeric.cols {
                        dataset.features.numeric.values[i * dataset.features.numeric.cols + j] =
                            98765.;
                    }
                    if let Outcome::Classification { values, .. } = &mut dataset.outcome {
                        values[i] = Some("unknown-test-outcome".into());
                    }
                }
            }
        }
        let changed_signal = train(&changed_signal, &plan, &cfg, 2).unwrap();
        let changed_noise = train(&changed_noise, &plan, &cfg, 2).unwrap();
        let changed = fit_fusion(
            &[
                FusionInput {
                    name: "signal",
                    training: &changed_signal,
                    model_name: "LR",
                },
                FusionInput {
                    name: "noise",
                    training: &changed_noise,
                    model_name: "LR",
                },
            ],
            &config,
        )
        .unwrap();
        assert_eq!(result.model, changed.model);
        assert_eq!(result.training, changed.training);
        let mut prediction_inputs = std::collections::BTreeMap::from([
            ("signal".into(), signal_data.clone()),
            ("noise".into(), noise_data.clone()),
        ]);
        for d in prediction_inputs.values_mut() {
            d.outcome = Outcome::Unlabeled;
        }
        assert_eq!(
            predict_fusion(&result.model, &prediction_inputs).unwrap(),
            result.predictions
        );
        let file = std::env::temp_dir().join(format!(
            "fusion-{}-{}.rad",
            std::process::id(),
            if logistic { "logistic" } else { "weighted" }
        ));
        result.model.save(&file, true).unwrap();
        let loaded = ModelBundle::<FittedFusion>::load(&file).unwrap();
        assert_eq!(loaded, result.model);
        assert_eq!(
            predict_fusion(&loaded, &prediction_inputs).unwrap(),
            result.predictions
        );
        std::fs::remove_file(file).unwrap();
    }
}

#[test]
fn fusion_requires_real_shared_nested_partitions_even_when_outer_fold_numbers_match() {
    let dataset = data("binary");
    let original = plan("binary");
    let cfg = fixed_config("binary");
    let first = train(&dataset, &original, &cfg, 1).unwrap();
    let mut other = original.clone();
    other.outer_folds[0].inner.swap(0, 1);
    other.validate().unwrap();
    let second = train(&dataset, &other, &cfg, 1).unwrap();
    assert_eq!(
        first.models[0]
            .oof
            .rows
            .iter()
            .map(|r| (&r.sample_id, r.fold))
            .collect::<Vec<_>>(),
        second.models[0]
            .oof
            .rows
            .iter()
            .map(|r| (&r.sample_id, r.fold))
            .collect::<Vec<_>>()
    );
    assert!(fit_fusion(
        &[
            FusionInput {
                name: "first",
                training: &first,
                model_name: "LR"
            },
            FusionInput {
                name: "second",
                training: &second,
                model_name: "LR"
            }
        ],
        &FusionConfig {
            method: FusionMethod::Weighted {
                weights: vec![1., 1.]
            },
            threshold: Some(0.5)
        }
    )
    .is_err());
}
