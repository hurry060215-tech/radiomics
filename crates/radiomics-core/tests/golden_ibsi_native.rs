//! Independent IBSI track: public integer matrices, Fraction first-order and
//! reported source precision. PythonGolden tolerances are never changed here.
use ov_radiomics_core::contracts::{Discretization, ExtractionConfig, TextureAggregation};
use ov_radiomics_core::{features, imaging};
use serde_json::Value;
use std::collections::BTreeMap;

fn directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/ibsi_digital_v1")
}
fn read(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(directory().join(name)).unwrap()).unwrap()
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * b.abs(),
        "{a:.17e} != {b:.17e}"
    )
}
fn expected_counts(table: &Value) -> BTreeMap<(u32, u32), u64> {
    table["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let v = &r["values"];
            (
                (v[0].as_f64().unwrap() as u32, v[1].as_f64().unwrap() as u32),
                v[2].as_f64().unwrap() as u64,
            )
        })
        .collect()
}
fn add_counts(output: &mut BTreeMap<(u32, u32), u64>, matrix: &features::CountMatrix) {
    for (r, &rv) in matrix.row_values.iter().enumerate() {
        for (c, &cv) in matrix.column_values.iter().enumerate() {
            let count = matrix.counts[r * matrix.column_values.len() + c];
            if count > 0 {
                *output.entry((rv, cv)).or_insert(0) += count;
            }
        }
    }
}

#[test]
fn independent_ibsi_original_phantom_counts_and_fullprecision_firstorder() {
    let image = imaging::read_nifti(directory().join("image.nii.gz")).unwrap();
    let mask = imaging::read_mask(directory().join("mask.nii.gz")).unwrap();
    let published = read("published_reference.json");
    let full = read("firstorder_fullprecision.json");
    let mut config = ExtractionConfig {
        profile: "python_golden_v1".into(),
        discretization: Discretization::FixedWidth { width: 1. },
        aggregation: TextureAggregation::DirectionMean,
        gldm_alpha: 0,
        voxel_array_shift: 0.,
    };
    let discrete = features::discretize(&image, &mask, 1, &config.discretization).unwrap();
    assert_eq!(discrete.gray_levels, vec![1, 3, 4, 6]);
    let matrices = features::texture_matrices(&discrete, 0).unwrap();
    let mut glcm = BTreeMap::new();
    for direction in &matrices.glcm {
        add_counts(&mut glcm, &direction.matrix);
    }
    let mut glrlm = BTreeMap::new();
    for direction in &matrices.glrlm {
        add_counts(&mut glrlm, &direction.matrix);
    }
    assert_eq!(glcm, expected_counts(&published["matrices"]["glcm_merged"]));
    assert_eq!(
        glrlm,
        expected_counts(&published["matrices"]["glrlm_merged"])
    );
    for (name, matrix) in [("glszm", &matrices.glszm), ("gldm", &matrices.gldm)] {
        let mut found = BTreeMap::new();
        add_counts(&mut found, matrix);
        assert_eq!(found, expected_counts(&published["matrices"][name]));
    }
    for row in &matrices.ngtdm {
        let expected = published["matrices"]["ngtdm"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["values"][0].as_f64().unwrap() as u32 == row.gray_level)
            .unwrap();
        assert_eq!(row.count, expected["values"][2].as_f64().unwrap() as u64);
        let printed = expected["printed"][1].as_str().unwrap();
        let places = printed.split('.').nth(1).map_or(0, str::len);
        let half = 0.5 * 10_f64.powi(-(places as i32));
        assert!(
            (row.difference_sum - expected["values"][1].as_f64().unwrap()).abs() <= half,
            "NGTDM s_i exceeds published six-decimal precision"
        );
    }
    for (mode, aggregation) in [
        ("3D, averaged", TextureAggregation::DirectionMean),
        ("3D, merged", TextureAggregation::MergeMatrices),
    ] {
        config.aggregation = aggregation;
        let values = features::extract(&image, &mask, None, 1, &config).unwrap();
        println!(
            "IBSI_NATIVE_ACTUAL_JSON:{}",
            serde_json::json!({"mode":mode,"features":values})
        );
        assert_eq!(values["original_shape_basic_VoxelCount"], 74.);
        assert_eq!(values["original_shape_basic_Volume_mm3"], 592.);
        for (name, expected) in full["ibsi_features"].as_object().unwrap() {
            let actual = values[&format!("original_firstorder_{name}")];
            if name == "Kurtosis" {
                close(actual - 3., expected.as_f64().unwrap());
                close(
                    actual,
                    full["native_raw_kurtosis_definition"].as_f64().unwrap(),
                );
            } else if full["exact_integer_features"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == name)
            {
                assert_eq!(actual, expected.as_f64().unwrap(), "{name}");
            } else {
                close(actual, expected.as_f64().unwrap());
            }
        }
        for feature in published["features"].as_array().unwrap() {
            let key = feature["feature"].as_str().unwrap();
            let actual = values[key];
            for expected in feature["rows"].as_array().unwrap() {
                let method = expected["aggregation"].as_str().unwrap();
                if method != "3D" && method != mode {
                    continue;
                }
                let compared = if key == "original_firstorder_Kurtosis" {
                    actual - 3.
                } else {
                    actual
                };
                let precision = expected["published_tolerance"]
                    .as_f64()
                    .unwrap_or_else(|| expected["printed_half_unit"].as_f64().unwrap());
                assert!(
                    (compared - expected["value"].as_f64().unwrap()).abs() <= precision,
                    "{key}/{mode}: native={actual} compared={compared} published={expected:?}"
                );
            }
        }
    }
    // Passing the explicit subtraction check proves the declared deviation;
    // the raw native Kurtosis feature itself is NOT marked IBSI conformant.
}
