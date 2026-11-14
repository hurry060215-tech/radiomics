use super::*;
use serde_json::{json, Value};
fn fixture() -> (Value, Value) {
    (
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/pathology_inputs.json"
        )))
        .unwrap(),
        serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/pathology_reference.json"
        )))
        .unwrap(),
    )
}
fn image_from_case(case: &Value) -> RasterImage {
    RasterImage {
        width: case["width"].as_u64().unwrap() as usize,
        height: case["height"].as_u64().unwrap() as usize,
        pixels: RasterPixels::Rgb8(
            case["pixels"]
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
fn rect(case: &Value) -> [usize; 4] {
    std::array::from_fn(|j| case["valid_pixels"][j].as_u64().unwrap() as usize)
}
fn compare(actual: &Value, expected: &Value, path: &str, color: bool) {
    match (actual, expected) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "keys: {path}"
            );
            for (k, v) in b {
                compare(&a[k], v, &format!("{path}.{k}"), color);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "length {path}");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                compare(x, y, &format!("{path}[{i}]"), color);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            let exact = path.contains("counts")
                || path.contains("denominator")
                || path.contains("pixels")
                || path.contains("count")
                || path.contains("object_id")
                || path.contains("source_label")
                || path.contains("area_px")
                || path.contains("bbox_xywh");
            let (atol, rtol) = if exact {
                (0., 0.)
            } else if color {
                (1e-6, 1e-5)
            } else {
                (1e-10, 1e-8)
            };
            assert!((a - b).abs() <= atol + rtol * b.abs(), "{path}: {a} != {b}");
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
fn metadata() -> SlideMetadata {
    SlideMetadata {
        patient_id: "patient".into(),
        slide_id: "slide".into(),
        source: "decoded_srgb.png".into(),
        origin0: [0., 0.],
        pixel_size0: [1., 1.],
        mpp_x_um: None,
        mpp_y_um: None,
    }
}

#[test]
fn frozen_rgb_qc_colors_missing_states_and_exact_histograms() {
    let (inputs, expected) = fixture();
    let config: QcConfig = serde_json::from_value(inputs["qc_config"].clone()).unwrap();
    let colors: ColorConfig =
        serde_json::from_value(json!({"fixed_edges":inputs["fixed_edges"]})).unwrap();
    for case in inputs["patches"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let image = image_from_case(case);
        let rect = rect(case);
        let qc = qc_patch(&image, rect, &config).unwrap();
        compare(
            &serde_json::to_value(&qc).unwrap(),
            &expected["patches"][id]["qc"],
            &format!("{id}.qc"),
            false,
        );
        let default = QcConfig {
            max_white_fraction: None,
            max_low_saturation_fraction: None,
            max_black_border_fraction: None,
            min_tissue_fraction: None,
            min_laplacian_variance: None,
            ..config.clone()
        };
        let qc_default = qc_patch(&image, rect, &default).unwrap();
        compare(
            &serde_json::to_value(qc_default).unwrap(),
            &expected["patches"][id]["default_no_exclusion"],
            &format!("{id}.default"),
            false,
        );
        let actual = color_features(&image, rect, &colors, None).unwrap();
        compare(
            &serde_json::to_value(actual).unwrap(),
            &expected["patches"][id]["colors"],
            &format!("{id}.colors"),
            true,
        );
        let actual = color_features(&image, rect, &colors, Some(&qc.tissue_mask)).unwrap();
        compare(
            &serde_json::to_value(actual).unwrap(),
            &expected["patches"][id]["colors_tissue_only"],
            &format!("{id}.tissue_colors"),
            true,
        );
    }
}
#[test]
fn frozen_streaming_tile_geometry_and_padding_pixels() {
    let (inputs, expected) = fixture();
    for (i, case) in inputs["tilings"].as_array().unwrap().iter().enumerate() {
        let width = case["width"].as_u64().unwrap() as usize;
        let height = case["height"].as_u64().unwrap() as usize;
        let image = RasterImage {
            width,
            height,
            pixels: RasterPixels::Gray16((0..width * height).map(|i| 1000 + i as u16).collect()),
        };
        let spec = TileSpec {
            patch_size: std::array::from_fn(|j| case["patch_size"][j].as_u64().unwrap() as usize),
            stride: std::array::from_fn(|j| case["stride"][j].as_u64().unwrap() as usize),
            edge: serde_json::from_value(case["edge"].clone()).unwrap(),
            pad_value: 60000,
        };
        let mut meta = metadata();
        meta.pixel_size0 = std::array::from_fn(|j| case["downsample"][j].as_f64().unwrap());
        let tiles = tile_image(&image, &spec, &meta)
            .unwrap()
            .collect::<RadResult<Vec<_>>>()
            .unwrap();
        assert_eq!(
            tiles.len(),
            expected["tilings"][i].as_array().unwrap().len()
        );
        for (tile, reference) in tiles.iter().zip(expected["tilings"][i].as_array().unwrap()) {
            let record = serde_json::to_value(&tile.record).unwrap();
            for key in [
                "x0",
                "y0",
                "width0",
                "height0",
                "patch_width",
                "patch_height",
                "valid_pixels",
            ] {
                compare(&record[key], &reference[key], key, false);
            }
            let source_x = (tile.record.x0 / meta.pixel_size0[0]) as usize;
            let source_y = (tile.record.y0 / meta.pixel_size0[1]) as usize;
            if let RasterPixels::Gray16(values) = &tile.image.pixels {
                for y in 0..tile.image.height {
                    for x in 0..tile.image.width {
                        let want =
                            if x < tile.record.valid_pixels[2] && y < tile.record.valid_pixels[3] {
                                1000 + ((source_y + y) * width + source_x + x) as u16
                            } else {
                                60000
                            };
                        assert_eq!(values[y * tile.image.width + x], want);
                    }
                }
            } else {
                panic!("Gray16 was not preserved");
            }
        }
        let repeated = tile_image(&image, &spec, &meta)
            .unwrap()
            .map(|t| t.unwrap().record.tile_id)
            .collect::<Vec<_>>();
        assert_eq!(
            repeated,
            tiles
                .iter()
                .map(|t| t.record.tile_id.clone())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn frozen_mask_coverage_semantic_components_and_instance_ids() {
    let (inputs, expected) = fixture();
    for case in inputs["masks"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let width = case["width"].as_u64().unwrap() as usize;
        let height = case["height"].as_u64().unwrap() as usize;
        let labels = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let classes = case["classes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let coverage = annotation_coverage(width, height, &labels, rect(case), &classes).unwrap();
        compare(
            &serde_json::to_value(coverage).unwrap(),
            &expected["masks"][id]["coverage"],
            &format!("{id}.coverage"),
            false,
        );
        let kind: MaskKind = serde_json::from_value(case["mask_kind"].clone()).unwrap();
        let connectivity = if case["connectivity"] == 4 {
            imaging::Connectivity::Four
        } else {
            imaging::Connectivity::Eight
        };
        let mpp = case["mpp"]
            .as_array()
            .map(|v| [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]);
        let actual = object_features(
            width,
            height,
            &labels,
            rect(case),
            kind,
            connectivity,
            mpp,
            case["overlapping_tiles"].as_bool().unwrap(),
        )
        .unwrap();
        compare(
            &serde_json::to_value(actual).unwrap(),
            &expected["masks"][id]["objects"],
            &format!("{id}.objects"),
            false,
        );
    }
}
#[test]
fn patch_scale_mpp_preserves_level0_meaning() {
    let image = RasterImage {
        width: 2,
        height: 1,
        pixels: RasterPixels::Gray8(vec![1, 1]),
    };
    let spec = TileSpec {
        patch_size: [2, 1],
        stride: [2, 1],
        edge: EdgePolicy::Drop,
        pad_value: 0,
    };
    let mut meta = metadata();
    meta.pixel_size0 = [2., 3.];
    meta.mpp_x_um = Some(0.25);
    meta.mpp_y_um = Some(0.5);
    meta.origin0 = [100., 200.];
    let tile = tile_image(&image, &spec, &meta)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(tile.record.x0, 100.);
    assert_eq!(tile.record.y0, 200.);
    assert_eq!(patch_pixel_mpp(&tile.record).unwrap(), Some([0.5, 1.5]));
    let objects = object_features_for_tile(
        &[1, 1],
        &tile.record,
        MaskKind::Instance,
        imaging::Connectivity::Four,
        true,
    )
    .unwrap();
    assert_eq!(objects.objects[0].area_um2, Some(1.5));
    assert_eq!(objects.count_scope, "patch_local");
    assert!(!objects.warnings.is_empty());
}
#[test]
fn invalid_parameters_grayscale_and_empty_regions_are_explicit() {
    let rgb = RasterImage {
        width: 2,
        height: 2,
        pixels: RasterPixels::Rgb8(vec![128; 12]),
    };
    let qc = QcConfig {
        border_band_px: 0,
        ..QcConfig::default()
    };
    assert!(qc_patch(&rgb, [0, 0, 2, 2], &qc).is_err());
    let qc = QcConfig {
        min_laplacian_variance: Some(0.),
        ..QcConfig::default()
    };
    let result = qc_patch(&rgb, [0, 0, 2, 2], &qc).unwrap();
    assert_eq!(result.scores["laplacian_variance"], None);
    assert_eq!(
        result.exclusion_reasons,
        vec!["laplacian_variance_undefined"]
    );
    assert!(histogram(&[1.], &[0., f64::NAN, 2.]).is_err());
    assert!(histogram(&[3.], &[0., 2.]).is_err());
    let gray = RasterImage {
        width: 2,
        height: 2,
        pixels: RasterPixels::Gray16(vec![60000; 4]),
    };
    assert_eq!(
        qc_patch(&gray, [0, 0, 2, 2], &QcConfig::default())
            .unwrap_err()
            .code,
        ErrorCode::Unsupported
    );
    let spec = TileSpec {
        patch_size: [2, 2],
        stride: [1, 1],
        edge: EdgePolicy::Pad,
        pad_value: 256,
    };
    assert!(tile_image(&rgb, &spec, &metadata()).is_err());
    let spec = TileSpec {
        pad_value: 0,
        ..spec
    };
    let mut meta = metadata();
    meta.pixel_size0 = [0., 1.];
    assert!(tile_image(&rgb, &spec, &meta).is_err());
    assert!(object_features(
        2,
        2,
        &[1, 0, 0, 0],
        [0, 0, 2, 2],
        MaskKind::Instance,
        imaging::Connectivity::Four,
        Some([-1., 1.]),
        false
    )
    .is_err());
    let empty = object_features(
        2,
        2,
        &[0; 4],
        [0, 0, 2, 2],
        MaskKind::Semantic,
        imaging::Connectivity::Four,
        None,
        false,
    )
    .unwrap();
    assert_eq!(empty.status, "empty_foreground");
    assert_eq!(empty.size_distribution, None);
}
#[test]
fn gray16_texture_uses_shared_engine_without_8bit_mapping() {
    let image = RasterImage {
        width: 3,
        height: 3,
        pixels: RasterPixels::Gray16(vec![0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 65535]),
    };
    let config = ExtractionConfig {
        profile: "python_golden_v1".into(),
        discretization: crate::contracts::Discretization::FixedWidth { width: 1000. },
        aggregation: crate::contracts::TextureAggregation::DirectionMean,
        gldm_alpha: 0,
        voxel_array_shift: 0.,
    };
    let output = texture_features(&image, [0, 0, 3, 3], None, &config).unwrap();
    let geometry = pixel_geometry(3, 3);
    let expected = features::extract(
        &ImageVolume {
            geometry: geometry.clone(),
            values: vec![0., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 65535.],
        },
        &MaskVolume {
            geometry,
            labels: vec![1; 9],
        },
        None,
        1,
        &config,
    )
    .unwrap();
    for (key, value) in &output {
        assert!(!key.starts_with("original_shape_basic_"));
        let reference = expected[key];
        assert!(value == &reference || (value.is_nan() && reference.is_nan()));
    }
    assert!(output["original_firstorder_Mean"] > 255.);
    assert!(texture_features(&image, [0, 0, 3, 3], Some(&[false; 9]), &config).is_err());
}
#[test]
fn missing_qc_export_is_preserved_with_finite_tile_projection() {
    let image = RasterImage {
        width: 2,
        height: 2,
        pixels: RasterPixels::Rgb8(vec![128; 12]),
    };
    let spec = TileSpec {
        patch_size: [2, 2],
        stride: [2, 2],
        edge: EdgePolicy::Drop,
        pad_value: 0,
    };
    let mut tile = tile_image(&image, &spec, &metadata())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let qc = qc_patch(&tile.image, tile.record.valid_pixels, &QcConfig::default()).unwrap();
    qc.apply_to_record(&mut tile.record).unwrap();
    assert!(!tile.record.qc.contains_key("laplacian_variance"));
    assert_eq!(
        serde_json::to_value(qc).unwrap()["scores"]["laplacian_variance"],
        Value::Null
    );
}

#[test]
fn tile_ids_distinguish_decoded_regions_scales_and_patch_extents() {
    let image = RasterImage {
        width: 3,
        height: 3,
        pixels: RasterPixels::Gray8(vec![7; 9]),
    };
    let spec = TileSpec {
        patch_size: [2, 2],
        stride: [2, 2],
        edge: EdgePolicy::Drop,
        pad_value: 0,
    };
    let original = tile_image(&image, &spec, &metadata())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .record
        .tile_id;
    for meta in [
        SlideMetadata {
            origin0: [1., 0.],
            ..metadata()
        },
        SlideMetadata {
            pixel_size0: [2., 1.],
            ..metadata()
        },
        SlideMetadata {
            source: "another.png".into(),
            ..metadata()
        },
    ] {
        let id = tile_image(&image, &spec, &meta)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .record
            .tile_id;
        assert_ne!(id, original);
    }
    let changed = TileSpec {
        patch_size: [3, 3],
        ..spec
    };
    let id = tile_image(&image, &changed, &metadata())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .record
        .tile_id;
    assert_ne!(id, original);
}
