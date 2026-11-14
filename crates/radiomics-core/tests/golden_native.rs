//! Independent acceptance against frozen public Python output. This module never
//! generates fixtures and does not share formulas with the feature implementation.
use ov_radiomics_core::contracts::{
    Discretization, ExtractionConfig, Geometry, ImageVolume, MaskVolume, SpatialUnit,
    TextureAggregation,
};
use ov_radiomics_core::features::{self, CountMatrix};
use serde_json::Value;
use std::collections::BTreeMap;

fn fixture(name: &str) -> Value {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../golden/python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap()
}

fn scalar_matches(actual: f64, expected: &Value, atol: f64, rtol: f64) -> bool {
    match expected.as_str() {
        Some("NaN") => actual.is_nan(),
        Some("Inf") => actual == f64::INFINITY,
        Some("-Inf") => actual == f64::NEG_INFINITY,
        _ => {
            let expected = expected.as_f64().unwrap();
            actual.is_finite() && (actual - expected).abs() <= atol + rtol * expected.abs()
        }
    }
}

fn input(case: &Value) -> (ImageVolume, MaskVolume, ExtractionConfig) {
    let dimensions = case["shape"].as_array().unwrap();
    let mut shape = [1; 3];
    for (i, size) in dimensions.iter().enumerate() {
        shape[i] = size.as_u64().unwrap() as usize;
    }
    let geometry = Geometry {
        shape,
        ndim: dimensions.len() as u8,
        affine: [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        unit: SpatialUnit::Millimeter,
        diagnostics: vec![],
    };
    let image = ImageVolume {
        geometry: geometry.clone(),
        values: case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
    };
    let mask = MaskVolume {
        geometry,
        labels: case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect(),
    };
    let discretization = if let Some(count) = case["binCount"].as_u64() {
        Discretization::FixedCount {
            count: count as usize,
        }
    } else {
        Discretization::FixedWidth {
            width: case["binWidth"].as_f64().unwrap(),
        }
    };
    let config = ExtractionConfig {
        profile: "python_golden_v1".into(),
        discretization,
        aggregation: TextureAggregation::DirectionMean,
        gldm_alpha: 0,
        voxel_array_shift: 0.,
    };
    (image, mask, config)
}

fn count_entries(matrix: &CountMatrix) -> BTreeMap<(usize, usize), u64> {
    assert_eq!(
        matrix.counts.len(),
        matrix.row_values.len() * matrix.column_values.len()
    );
    let mut entries = BTreeMap::new();
    for (row, &r) in matrix.row_values.iter().enumerate() {
        for (col, &c) in matrix.column_values.iter().enumerate() {
            let count = matrix.counts[row * matrix.column_values.len() + col];
            if count > 0 {
                assert!(entries.insert((r as usize, c as usize), count).is_none());
            }
        }
    }
    entries
}

fn reference_counts(raw: &Value, angle: Option<usize>) -> BTreeMap<(usize, usize), u64> {
    let mut entries = BTreeMap::new();
    for (r, row) in raw.as_array().unwrap().iter().enumerate() {
        for (c, cell) in row.as_array().unwrap().iter().enumerate() {
            let count = if let Some(a) = angle {
                cell[a].as_f64().unwrap()
            } else {
                cell.as_f64().unwrap()
            };
            assert!(count >= 0. && count.fract() == 0.);
            if count > 0. {
                entries.insert((r + 1, c + 1), count as u64);
            }
        }
    }
    entries
}

#[test]
fn original_features_match_public_301_all_frozen_cases() {
    let inputs = fixture("cases.json");
    let reference = fixture("reference.json");
    let mut failures = vec![];
    for case in inputs["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let (image, mask, mut config) = input(case);
        if id == "empty_roi" {
            assert!(features::extract(&image, &mask, None, 1, &config).is_err());
            continue;
        }
        for (mode, aggregation) in [
            ("features", TextureAggregation::DirectionMean),
            ("merged_features", TextureAggregation::MergeMatrices),
        ] {
            config.aggregation = aggregation;
            let actual = features::extract(&image, &mask, None, 1, &config).unwrap();
            for (family, expected) in reference["features"][id][mode].as_object().unwrap() {
                assert_eq!(expected["status"], "ok");
                for (name, value) in expected["value"].as_object().unwrap() {
                    let key = format!("original_{family}_{name}");
                    match actual.get(&key) {
                        Some(&found) if scalar_matches(found, value, 1e-10, 1e-8) => (),
                        found => failures.push(format!(
                            "{id}/{mode}/{key}: native={found:?}, reference={value}"
                        )),
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn discrete_gray_values_and_raw_counts_match_public_301() {
    let inputs = fixture("cases.json");
    let reference = fixture("reference.json");
    for case in inputs["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let (image, mask, config) = input(case);
        if id == "empty_roi" {
            assert!(features::discretize(&image, &mask, 1, &config.discretization).is_err());
            continue;
        }
        let discrete = features::discretize(&image, &mask, 1, &config.discretization).unwrap();
        fn flatten(value: &Value, output: &mut Vec<u32>) {
            if let Some(items) = value.as_array() {
                for item in items {
                    flatten(item, output);
                }
            } else {
                output.push(value.as_u64().unwrap() as u32);
            }
        }
        let mut expected_discrete = vec![];
        flatten(
            &reference["features"][id]["discretized"]["value"],
            &mut expected_discrete,
        );
        assert_eq!(discrete.values, expected_discrete, "{id} discretization");
        let native = features::texture_matrices(&discrete, 0).unwrap();
        let raw = &reference["features"][id]["raw_matrices"];
        for (name, directional) in [("glcm", &native.glcm), ("glrlm", &native.glrlm)] {
            let expected = &raw[name]["value"];
            let angles = expected["angles_xyz"].as_array().unwrap();
            assert_eq!(directional.len(), angles.len(), "{id}/{name}: angle count");
            for (a, angle) in angles.iter().enumerate() {
                let mut xyz = [0; 3];
                for (d, value) in angle.as_array().unwrap().iter().enumerate() {
                    xyz[d] = value.as_i64().unwrap() as i32;
                }
                let actual = directional.iter().find(|v| v.offset == xyz).unwrap();
                assert_eq!(
                    actual.matrix.row_values, discrete.gray_levels,
                    "{id}/{name}: true gray labels"
                );
                assert_eq!(
                    count_entries(&actual.matrix),
                    reference_counts(&expected["counts"], Some(a)),
                    "{id}/{name}/{xyz:?}"
                );
            }
        }
        for (name, actual) in [("glszm", &native.glszm), ("gldm", &native.gldm)] {
            assert_eq!(
                actual.row_values, discrete.gray_levels,
                "{id}/{name}: true gray labels"
            );
            assert_eq!(
                count_entries(actual),
                reference_counts(&raw[name]["value"], None),
                "{id}/{name}"
            );
        }
        let expected_rows = raw["ngtdm"]["value"].as_array().unwrap();
        for row in &native.ngtdm {
            let expected = &expected_rows[row.gray_level as usize - 1];
            assert_eq!(
                row.count,
                expected[0].as_f64().unwrap() as u64,
                "{id}/ngtdm count"
            );
            assert!(
                scalar_matches(row.difference_sum, &expected[1], 1e-10, 1e-8),
                "{id}/ngtdm difference sum"
            );
            assert_eq!(row.gray_level, expected[2].as_f64().unwrap() as u32);
        }
    }
}

#[test]
fn floating_discretization_edges_match_frozen_arange_digitize() {
    let edge_cases = fixture("binning_edges.json");
    for case in edge_cases["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let (image, mask, config) = input(case);
        let actual = features::discretize(&image, &mask, 1, &config.discretization);
        if case["native_rule"] == "error_invalid_or_unsupported" {
            let error =
                actual.expect_err("Unrepresentable bins cannot become a successful all-zero ROI.");
            assert!(
                matches!(
                    error.code,
                    ov_radiomics_core::errors::ErrorCode::InvalidInput
                        | ov_radiomics_core::errors::ErrorCode::Unsupported
                ),
                "{id}"
            );
        } else {
            let expected: Vec<u32> = case["discretized"]["value"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(actual.unwrap().values, expected, "{id}");
        }
    }
}
