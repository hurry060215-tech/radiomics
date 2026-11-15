//! Independent pixel/mask/coordinate acceptance, with no reference regeneration.
use ov_radiomics_core::contracts::{
    Discretization, ExtractionConfig, RasterImage, RasterPixels, TextureAggregation, TileRecord,
};
use ov_radiomics_core::imaging::Connectivity;
use ov_radiomics_core::pathology_pixels::{
    self as h, ColorConfig, EdgePolicy, MaskKind, QcConfig, SlideMetadata, TileSpec,
};
use serde_json::Value;
use std::collections::BTreeMap;

fn read(name: &str) -> Value {
    let p = std::path::Path::new(env!("RAD_GOLDEN_DIR"))
        .join("python_golden_v1")
        .join(name);
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn compare(actual: &Value, expected: &Value, path: &str, color: bool) {
    match expected {
        Value::Object(map) => {
            for (key, value) in map {
                assert!(actual.get(key).is_some(), "Missing {path}/{key}");
                compare(&actual[key], value, &format!("{path}/{key}"), color);
            }
        }
        Value::Array(values) => {
            let a = actual.as_array().unwrap();
            assert_eq!(a.len(), values.len(), "{path}");
            for (i, (a, b)) in a.iter().zip(values).enumerate() {
                compare(a, b, &format!("{path}/{i}"), color);
            }
        }
        Value::Number(value) => {
            let a = actual.as_f64().unwrap();
            let b = value.as_f64().unwrap();
            let (atol, rtol) = if color { (1e-6, 1e-5) } else { (1e-10, 1e-8) };
            let error = if path.ends_with("mean_degrees") {
                (a - b)
                    .abs()
                    .min((a - b + 360.).abs())
                    .min((a - b - 360.).abs())
            } else {
                (a - b).abs()
            };
            assert!(
                a.is_finite() && error <= atol + rtol * b.abs(),
                "{path}: {a} != {b}"
            );
            if path.contains("/counts/")
                || path.ends_with("/count")
                || path.ends_with("denominator")
            {
                assert_eq!(a, b, "{path}");
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
fn image(patch: &Value) -> RasterImage {
    RasterImage {
        width: patch["width"].as_u64().unwrap() as usize,
        height: patch["height"].as_u64().unwrap() as usize,
        pixels: RasterPixels::Rgb8(
            patch["pixels"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| {
                    p.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u8)
                })
                .collect(),
        ),
    }
}
fn rect(value: &Value) -> [usize; 4] {
    std::array::from_fn(|i| value[i].as_u64().unwrap() as usize)
}

#[test]
fn raw_qc_color_histograms_hue_and_tissue_missing_states_match_independent_spec() {
    let inputs = read("pathology_inputs.json");
    let reference = read("pathology_reference.json");
    let config: QcConfig = serde_json::from_value(inputs["qc_config"].clone()).unwrap();
    let color = ColorConfig {
        fixed_edges: serde_json::from_value(inputs["fixed_edges"].clone()).unwrap(),
    };
    for patch in inputs["patches"].as_array().unwrap() {
        let id = patch["id"].as_str().unwrap();
        let raster = image(patch);
        let valid = rect(&patch["valid_pixels"]);
        let expected = &reference["patches"][id];
        let qc = h::qc_patch(&raster, valid, &config).unwrap();
        compare(
            &serde_json::to_value(&qc).unwrap(),
            &expected["qc"],
            id,
            false,
        );
        let raw = h::qc_patch(&raster, valid, &QcConfig::default()).unwrap();
        compare(
            &serde_json::to_value(raw).unwrap(),
            &expected["default_no_exclusion"],
            id,
            false,
        );
        let found = h::color_features(&raster, valid, &color, None).unwrap();
        compare(
            &serde_json::to_value(found).unwrap(),
            &expected["colors"],
            id,
            true,
        );
        let tissue = h::color_features(&raster, valid, &color, Some(&qc.tissue_mask)).unwrap();
        compare(
            &serde_json::to_value(tissue).unwrap(),
            &expected["colors_tissue_only"],
            id,
            true,
        );
    }
    let gray = RasterImage {
        width: 3,
        height: 3,
        pixels: RasterPixels::Gray16(vec![1000; 9]),
    };
    assert!(h::qc_patch(&gray, [0, 0, 3, 3], &QcConfig::default()).is_err());
    assert!(h::color_features(&gray, [0, 0, 3, 3], &color, None).is_err());
}

#[test]
fn streamed_tiles_masks_patch_units_and_padding_are_exact() {
    let inputs = read("pathology_inputs.json");
    let reference = read("pathology_reference.json");
    for (i, tiling) in inputs["tilings"].as_array().unwrap().iter().enumerate() {
        let width = tiling["width"].as_u64().unwrap() as usize;
        let height = tiling["height"].as_u64().unwrap() as usize;
        let source = RasterImage {
            width,
            height,
            pixels: RasterPixels::Rgb8(vec![0; width * height * 3]),
        };
        let spec = TileSpec {
            patch_size: std::array::from_fn(|j| tiling["patch_size"][j].as_u64().unwrap() as usize),
            stride: std::array::from_fn(|j| tiling["stride"][j].as_u64().unwrap() as usize),
            edge: if tiling["edge"] == "drop" {
                EdgePolicy::Drop
            } else {
                EdgePolicy::Pad
            },
            pad_value: 255,
        };
        let metadata = SlideMetadata {
            patient_id: "P".into(),
            slide_id: "S".into(),
            source: "synthetic".into(),
            origin0: [0., 0.],
            pixel_size0: std::array::from_fn(|j| tiling["downsample"][j].as_f64().unwrap()),
            mpp_x_um: None,
            mpp_y_um: None,
        };
        let tiles: Vec<_> = h::tile_image(&source, &spec, &metadata)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        let expected = reference["tilings"][i].as_array().unwrap();
        assert_eq!(tiles.len(), expected.len());
        for (tile, expected) in tiles.iter().zip(expected) {
            compare(
                &serde_json::to_value(&tile.record).unwrap(),
                expected,
                "tile",
                false,
            );
            tile.record.validate().unwrap();
        }
    }
    for mask in inputs["masks"].as_array().unwrap() {
        let id = mask["id"].as_str().unwrap();
        let width = mask["width"].as_u64().unwrap() as usize;
        let height = mask["height"].as_u64().unwrap() as usize;
        let valid = rect(&mask["valid_pixels"]);
        let labels: Vec<_> = mask["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let classes: Vec<_> = mask["classes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let coverage = h::annotation_coverage(width, height, &labels, valid, &classes).unwrap();
        compare(
            &serde_json::to_value(coverage).unwrap(),
            &reference["masks"][id]["coverage"],
            id,
            false,
        );
        let kind = if mask["mask_kind"] == "instance" {
            MaskKind::Instance
        } else {
            MaskKind::Semantic
        };
        let connectivity = if mask["connectivity"] == 4 {
            Connectivity::Four
        } else {
            Connectivity::Eight
        };
        let mpp = mask["mpp"]
            .as_array()
            .map(|v| [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]);
        let objects = h::object_features(
            width,
            height,
            &labels,
            valid,
            kind,
            connectivity,
            mpp,
            mask["overlapping_tiles"].as_bool().unwrap(),
        )
        .unwrap();
        compare(
            &serde_json::to_value(objects).unwrap(),
            &reference["masks"][id]["objects"],
            id,
            false,
        );
    }
    let tile = TileRecord {
        patient_id: "P".into(),
        slide_id: "S".into(),
        tile_id: "T".into(),
        x0: 10.,
        y0: 20.,
        width0: 8.,
        height0: 12.,
        patch_width: 4,
        patch_height: 4,
        valid_pixels: [0, 0, 4, 4],
        mpp_x_um: Some(0.25),
        mpp_y_um: Some(0.5),
        source: "test".into(),
        qc: BTreeMap::new(),
        exclusion_reasons: vec![],
    };
    assert_eq!(h::patch_pixel_mpp(&tile).unwrap(), Some([0.5, 1.5]));
    let mut labels = vec![0; 16];
    labels[0] = 1;
    labels[1] = 1;
    let objects =
        h::object_features_for_tile(&labels, &tile, MaskKind::Instance, Connectivity::Four, true)
            .unwrap();
    assert_eq!(objects.objects[0].area_um2, Some(1.5));
}

#[test]
fn grayscale_pathology_reuses_frozen_radiomics_values_without_8bit_mapping() {
    let reference = read("reference.json");
    let expected = &reference["features"]["hand_2d"]["features"];
    let image = RasterImage {
        width: 3,
        height: 2,
        pixels: RasterPixels::Gray8(vec![1, 1, 2, 2, 3, 3]),
    };
    let mut config = ExtractionConfig {
        profile: "python_golden_v1".into(),
        discretization: Discretization::FixedWidth { width: 1. },
        aggregation: TextureAggregation::DirectionMean,
        gldm_alpha: 0,
        voxel_array_shift: 0.,
    };
    let values = h::texture_features(&image, [0, 0, 3, 2], None, &config).unwrap();
    assert_eq!(values.len(), 55);
    for (family, result) in expected.as_object().unwrap() {
        for (name, value) in result["value"].as_object().unwrap() {
            let found = values[&format!("original_{family}_{name}")];
            let expected = value.as_f64().unwrap();
            assert!((found - expected).abs() <= 1e-10 + 1e-8 * expected.abs());
        }
    }
    let image = RasterImage {
        width: 3,
        height: 2,
        pixels: RasterPixels::Gray16(vec![1000, 1000, 2000, 2000, 3000, 3000]),
    };
    config.discretization = Discretization::FixedWidth { width: 1000. };
    let values = h::texture_features(&image, [0, 0, 3, 2], None, &config).unwrap();
    assert_eq!(values["original_firstorder_Mean"], 2000.);
    assert_eq!(
        values["original_glcm_Contrast"],
        reference["features"]["hand_2d"]["features"]["glcm"]["value"]["Contrast"]
            .as_f64()
            .unwrap()
    );
}
