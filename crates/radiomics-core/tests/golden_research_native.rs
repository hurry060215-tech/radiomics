//! Native results compared to independently frozen keyed/Euclidean references.
use ov_radiomics_core::{
    contracts::{FeatureDefinition, FeatureTable, NumericBuffer, SampleMetadata},
    errors::ErrorCode,
    habitat::{self, HabitatConfig, Initialization},
    preprocessing::Scaling,
    research::{
        self, ConcatConfig, DeltaConfig, MetadataField, NamedFeatureTable, ZeroDenominator,
    },
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn fixtures() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/research_inputs.json"
        )))
        .unwrap(),
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/research_reference.json"
        )))
        .unwrap(),
    )
}
fn names(value: &Value) -> Vec<String> {
    serde_json::from_value(value.clone()).unwrap()
}
fn numeric(rows: &Value) -> NumericBuffer {
    let data = rows.as_array().unwrap();
    let cols = data[0].as_array().unwrap().len();
    NumericBuffer {
        rows: data.len(),
        cols,
        values: data
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
        missing: vec![false; data.len() * cols],
    }
}
fn table(rows: &Value, columns: &Value) -> FeatureTable {
    let columns = names(columns);
    let mut values = vec![];
    let mut missing = vec![];
    let mut samples = vec![];
    for row in rows.as_array().unwrap() {
        let mut sample = row.clone();
        sample.as_object_mut().unwrap().remove("values");
        samples.push(serde_json::from_value::<SampleMetadata>(sample).unwrap());
        for name in &columns {
            let value = &row["values"][name];
            values.push(value.as_f64().unwrap_or(0.));
            missing.push(value.is_null());
        }
    }
    FeatureTable {
        data: NumericBuffer {
            rows: samples.len(),
            cols: columns.len(),
            values,
            missing,
        },
        samples,
        features: columns
            .into_iter()
            .map(|name| FeatureDefinition {
                definition: format!("frozen {name}"),
                name,
                unit: Some("synthetic_unit".into()),
            })
            .collect(),
        effective_parameters: BTreeMap::from([("synthetic".into(), json!(true))]),
        quality: vec![],
    }
}
fn optional_row(table: &FeatureTable, index: usize) -> Value {
    Value::Array(
        (0..table.data.cols)
            .map(|column| {
                let k = index * table.data.cols + column;
                if table.data.missing[k] {
                    Value::Null
                } else {
                    json!(table.data.values[k])
                }
            })
            .collect(),
    )
}
fn assert_optional_row(table: &FeatureTable, index: usize, expected: &Value) {
    let actual = optional_row(table, index);
    let actual = actual.as_array().unwrap();
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.is_null(), b.is_null());
        if !b.is_null() {
            assert_eq!(a.as_f64().unwrap(), b.as_f64().unwrap());
        }
    }
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-10 + 1e-8 * expected.abs(),
        "{actual:.17e} != {expected:.17e}"
    );
}
fn vec_close(actual: &[f64], reference: &Value) {
    let expected: Vec<_> = reference
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        close(*a, b);
    }
}
fn concat_case(input: &Value) -> (Vec<NamedFeatureTable>, ConcatConfig) {
    let case = &input["concat"];
    let sources = case["sources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| NamedFeatureTable {
            prefix: s["prefix"].as_str().unwrap().into(),
            table: table(&s["rows"], &s["feature_names"]),
        })
        .collect();
    let config = ConcatConfig {
        key_fields: serde_json::from_value(case["key_fields"].clone()).unwrap(),
        varying_fields: serde_json::from_value(case["varying_fields"].clone()).unwrap(),
        already_registered: true,
    };
    (sources, config)
}

#[test]
fn strict_key_concat_matches_frozen_order_values_metadata_and_source_provenance() {
    let (input, reference) = fixtures();
    let (sources, config) = concat_case(&input);
    let output = research::concat(&sources, &config).unwrap();
    let expected = &reference["concat"];
    assert_eq!(
        output
            .table
            .features
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>(),
        names(&expected["feature_names"])
    );
    for (i, row) in expected["rows"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            serde_json::to_value(&output.table.samples[i]).unwrap(),
            row["metadata"]
        );
        assert_optional_row(&output.table, i, &row["values"]);
        assert_eq!(
            serde_json::to_value(&output.rows[i].source_sample_ids).unwrap(),
            row["source_sample_ids"]
        );
        assert_eq!(
            serde_json::to_value(&output.rows[i].source_metadata).unwrap(),
            row["source_metadata"]
        );
    }
    assert!(output
        .table
        .features
        .iter()
        .all(|f| f.unit.as_deref() == Some("synthetic_unit")));
    assert_eq!(
        output.table.effective_parameters["sources"][0]["effective_parameters"]["synthetic"],
        true
    );
    let mut shuffled = sources.clone();
    shuffled[1].table.samples.swap(0, 1);
    for column in 0..shuffled[1].table.data.cols {
        let next = shuffled[1].table.data.cols + column;
        shuffled[1].table.data.values.swap(column, next);
        shuffled[1].table.data.missing.swap(column, next);
    }
    assert_eq!(research::concat(&shuffled, &config).unwrap(), output);
}

#[test]
fn keyed_concat_rejects_population_missing_known_metadata_and_prefix_conflicts() {
    let (input, _) = fixtures();
    let (sources, config) = concat_case(&input);
    let mut changed = sources.clone();
    let n = changed[1].table.data.cols;
    changed[1].table.samples.pop();
    changed[1].table.data.rows -= 1;
    let retained = changed[1].table.data.rows * n;
    changed[1].table.data.values.truncate(retained);
    changed[1].table.data.missing.truncate(retained);
    assert_eq!(
        research::concat(&changed, &config).unwrap_err().code,
        ErrorCode::Alignment
    );
    let mut changed = sources.clone();
    changed[1].table.samples[0].lesion_id = None;
    assert_eq!(
        research::concat(&changed, &config).unwrap_err().code,
        ErrorCode::Alignment
    );
    let mut changed = sources.clone();
    changed[1].table.samples[0].center = None;
    assert_eq!(
        research::concat(&changed, &config).unwrap_err().code,
        ErrorCode::Alignment
    );
    let mut changed = sources.clone();
    changed[1].prefix = changed[0].prefix.clone();
    assert!(research::concat(&changed, &config).is_err());
    let mut no_registration = config.clone();
    no_registration.already_registered = false;
    assert!(research::concat(&sources, &no_registration).is_err());
    let mut invalid_vary = config.clone();
    invalid_vary.varying_fields.push(MetadataField::Center);
    assert!(research::concat(&sources, &invalid_vary).is_err());
    let mut duplicate = sources.clone();
    duplicate[1].table.samples[1].patient_id = duplicate[1].table.samples[0].patient_id.clone();
    duplicate[1].table.samples[1].lesion_id = duplicate[1].table.samples[0].lesion_id.clone();
    duplicate[1].table.samples[1].timepoint = duplicate[1].table.samples[0].timepoint.clone();
    assert_eq!(
        research::concat(&duplicate, &config).unwrap_err().code,
        ErrorCode::Alignment
    );
}

#[test]
fn delta_matches_fixed_pairs_missing_before_zero_and_individual_units() {
    let (input, reference) = fixtures();
    let case = &input["delta"];
    let source = table(&case["rows"], &case["feature_names"]);
    let config = DeltaConfig {
        baseline: "T0".into(),
        followup: "T1".into(),
        zero_denominator: ZeroDenominator::Missing,
    };
    let output = research::delta(&source, &config).unwrap();
    let expected = &reference["delta_missing_policy"];
    assert_eq!(
        output
            .table
            .features
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>(),
        names(&expected["feature_names"])
    );
    for (i, pair) in output.pairs.iter().enumerate() {
        assert_eq!(
            serde_json::to_value(pair).unwrap(),
            json!({"key":expected["rows"][i]["key"],"baseline_sample_id":expected["rows"][i]["baseline_sample_id"],"followup_sample_id":expected["rows"][i]["followup_sample_id"]})
        );
        assert_optional_row(&output.table, i, &expected["rows"][i]["values"]);
    }
    assert_eq!(
        output.table.features[0].unit.as_deref(),
        Some("synthetic_unit")
    );
    assert_eq!(
        output.table.features[1].unit.as_deref(),
        Some("dimensionless")
    );
    let mut fail = config.clone();
    fail.zero_denominator = ZeroDenominator::Error;
    assert!(research::delta(&source, &fail)
        .unwrap_err()
        .message
        .contains("Zero baseline"));
    let before = source
        .samples
        .iter()
        .position(|s| s.timepoint.as_deref() == Some("T0"))
        .unwrap();
    let mut missing = source.clone();
    missing.samples = vec![source.samples[before].clone(); 2];
    missing.samples[1].sample_id = "missing-followup".into();
    missing.samples[1].timepoint = Some("T1".into());
    missing.features = vec![source.features[0].clone()];
    missing.data = NumericBuffer {
        rows: 2,
        cols: 1,
        values: vec![0., 0.],
        missing: vec![false, true],
    };
    let result = research::delta(&missing, &fail).unwrap();
    assert_eq!(optional_row(&result.table, 0), json!([null, null, null]));
    let mut extra = missing.clone();
    extra.samples[1].timepoint = Some("T2".into());
    assert_eq!(
        research::delta(&extra, &config).unwrap_err().code,
        ErrorCode::Alignment
    );
    let mut overflow = missing.clone();
    overflow.data.values = vec![-1e308, 1e308];
    overflow.data.missing = vec![false, false];
    assert!(research::delta(&overflow, &config)
        .unwrap_err()
        .message
        .contains("overflow"));
    let mut conflict = missing.clone();
    conflict.samples[1].center = None;
    assert!(research::delta(&conflict, &config).is_err());
}

fn habitat_case(input: &Value) -> (NumericBuffer, Vec<String>, Vec<String>, HabitatConfig) {
    let case = &input["habitat"];
    let config = HabitatConfig {
        clusters: case["initial_centers_raw"].as_array().unwrap().len(),
        scaling: Scaling::Standard,
        initialization: Initialization::Explicit {
            centers_raw: numeric(&case["initial_centers_raw"]),
        },
        seed: 0,
        tolerance: case["tol"].as_f64().unwrap(),
        max_iterations: case["max_iter"].as_u64().unwrap() as usize,
    };
    (
        numeric(&case["training_values"]),
        names(&case["training_ids"]),
        vec!["x".into(), "y".into()],
        config,
    )
}
#[test]
fn habitat_matches_frozen_scaler_centers_inertia_labels_and_direct_euclidean_distances() {
    let (input, reference) = fixtures();
    let (data, ids, features, config) = habitat_case(&input);
    let fitted = habitat::fit(&data, &ids, &features, &config).unwrap();
    let expected = &reference["habitat"];
    vec_close(
        &fitted
            .preprocessing
            .numeric
            .iter()
            .map(|s| s.mean)
            .collect::<Vec<_>>(),
        &expected["scaler"]["mean"],
    );
    vec_close(
        &fitted
            .preprocessing
            .numeric
            .iter()
            .map(|s| s.variance)
            .collect::<Vec<_>>(),
        &expected["scaler"]["variance"],
    );
    vec_close(
        &fitted
            .preprocessing
            .numeric
            .iter()
            .map(|s| s.scale)
            .collect::<Vec<_>>(),
        &expected["scaler"]["scale"],
    );
    for (index, center) in fitted.centers_scaled.values.chunks(data.cols).enumerate() {
        vec_close(center, &expected["centers_scaled"][index]);
    }
    for (index, center) in fitted
        .centers_raw()
        .unwrap()
        .values
        .chunks(data.cols)
        .enumerate()
    {
        vec_close(center, &expected["centers_raw"][index]);
    }
    close(fitted.report.inertia, expected["inertia"].as_f64().unwrap());
    assert!(fitted.report.converged && fitted.report.assignment_fixed_point);
    assert_eq!(
        serde_json::to_value(fitted.predict(&data, &ids, &features).unwrap().labels).unwrap(),
        expected["labels"]
    );
    let prediction = fitted
        .predict(
            &numeric(&input["habitat"]["evaluation_values"]),
            &names(&input["habitat"]["evaluation_ids"]),
            &features,
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(prediction.labels).unwrap(),
        expected["evaluation_labels"]
    );
    for (index, row) in prediction
        .distances
        .values
        .chunks(config.clusters)
        .enumerate()
    {
        vec_close(
            row,
            &expected["evaluation_distances_euclidean_direct"][index],
        );
    }
}

#[test]
fn habitat_exact_ties_convergence_empty_clusters_and_fit_scope_are_explicit() {
    let (input, reference) = fixtures();
    let (data, ids, features, config) = habitat_case(&input);
    let fitted = habitat::fit(&data, &ids, &features, &config).unwrap();
    let saved = serde_json::to_string(&fitted).unwrap();
    let loaded: habitat::FittedHabitat = serde_json::from_str(&saved).unwrap();
    loaded.validate().unwrap();
    assert_eq!(fitted, loaded);
    let mut reordered = data.clone();
    reordered.values = vec![];
    let mut reverse_ids = ids.clone();
    reverse_ids.reverse();
    for row in data.values.chunks(data.cols).rev() {
        reordered.values.extend_from_slice(row);
    }
    assert_eq!(
        habitat::fit(&reordered, &reverse_ids, &features, &config).unwrap(),
        fitted
    );
    let external = numeric(&json!([[1e6, -1e6], [-1000., 2000.]]));
    loaded
        .predict(&external, &["V0".into(), "V1".into()], &features)
        .unwrap();
    assert_eq!(serde_json::to_string(&loaded).unwrap(), saved);
    let mut reversed = data.clone();
    for row in reversed.values.chunks_mut(data.cols) {
        row.reverse();
    }
    let mut reversed_features = features.clone();
    reversed_features.reverse();
    assert_eq!(
        fitted.predict(&reversed, &ids, &reversed_features).unwrap(),
        fitted.predict(&data, &ids, &features).unwrap()
    );
    let early = &input["habitat"]["nonconvergence"];
    let early_config = HabitatConfig {
        initialization: Initialization::Explicit {
            centers_raw: numeric(&early["initial_centers_raw"]),
        },
        max_iterations: 1,
        ..config.clone()
    };
    let early_data = numeric(&early["training_values"]);
    let early_ids = (0..early_data.rows)
        .map(|i| format!("T{i}"))
        .collect::<Vec<_>>();
    assert_eq!(
        habitat::fit(&early_data, &early_ids, &features, &early_config)
            .unwrap_err()
            .code,
        ErrorCode::Convergence
    );
    let mut duplicate = config.clone();
    if let Initialization::Explicit { centers_raw } = &mut duplicate.initialization {
        centers_raw.values = [0.; 4].to_vec();
    }
    assert!(habitat::fit(&data, &ids, &features, &duplicate).is_err());
    let exact = &reference["habitat_exact_tie"];
    let centers = numeric(&exact["centers_scaled"]);
    let tie_config = HabitatConfig {
        clusters: 2,
        scaling: Scaling::None,
        initialization: Initialization::Explicit {
            centers_raw: centers.clone(),
        },
        ..config.clone()
    };
    let tie = habitat::fit(&centers, &["a".into(), "b".into()], &features, &tie_config).unwrap();
    let prediction = tie
        .predict(
            &numeric(&exact["evaluation_scaled"]),
            &["query".into()],
            &features,
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(prediction.labels).unwrap(),
        exact["labels"]
    );
    assert_eq!(prediction.distances.values, vec![1., 1.]);
    let mut invalid = fitted.clone();
    invalid.report.converged = false;
    assert_eq!(invalid.validate().unwrap_err().code, ErrorCode::Convergence);
    let mut invalid = fitted.clone();
    invalid.schema_version = 2;
    assert_eq!(invalid.validate().unwrap_err().code, ErrorCode::Schema);
}

#[test]
fn recorded_kmeans_plus_plus_is_reproducible_across_worker_counts() {
    let (input, _) = fixtures();
    let (data, ids, features, config) = habitat_case(&input);
    for seed in 0..5 {
        let config = HabitatConfig {
            seed,
            initialization: Initialization::KmeansPlusPlus,
            ..config.clone()
        };
        let first = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| habitat::fit(&data, &ids, &features, &config))
            .unwrap();
        let second = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| habitat::fit(&data, &ids, &features, &config))
            .unwrap();
        assert_eq!(first, second);
        close(first.report.inertia, 4. / 16.25);
        let saved: habitat::FittedHabitat =
            serde_json::from_str(&serde_json::to_string(&first).unwrap()).unwrap();
        assert_eq!(
            saved.predict(&data, &ids, &features).unwrap(),
            first.predict(&data, &ids, &features).unwrap()
        );
    }
    let mut missing = data.clone();
    missing.missing[0] = true;
    missing.values[0] = 0.;
    assert!(habitat::fit(&missing, &ids, &features, &config).is_err());
}
