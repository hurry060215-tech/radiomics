use super::*;
use std::{io::Write, path::PathBuf};

fn numbers(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn affine_from_json(value: &serde_json::Value, ndim: usize) -> [[f64; 4]; 4] {
    let mut affine = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    for i in 0..ndim {
        for j in 0..ndim {
            affine[i][j] = value[i][j].as_f64().unwrap();
        }
        affine[i][3] = value[i][ndim].as_f64().unwrap();
    }
    affine
}
fn assert_close(actual: &[f64], reference: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), reference.len());
    for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        assert!((a - b).abs() <= atol + rtol * b.abs(), "[{i}] {a} != {b}");
    }
}
#[test]
fn frozen_independent_simpleitk_geometry_reference() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../golden/python_golden_v1/geometry.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let dimensions = numbers(&case["shape_xyz"]);
        let ndim = dimensions.len();
        let mut shape = [1; 3];
        for j in 0..ndim {
            shape[j] = dimensions[j] as usize;
        }
        let source = ImageVolume {
            geometry: Geometry {
                shape,
                ndim: ndim as u8,
                affine: affine_from_json(&case["affine_ras"], ndim),
                unit: SpatialUnit::Millimeter,
                diagnostics: vec![],
            },
            values: numbers(&case["values_x_fastest"]),
        };
        let mask = MaskVolume {
            geometry: source.geometry.clone(),
            labels: numbers(&case["mask_x_fastest"])
                .iter()
                .map(|v| *v as u32)
                .collect(),
        };
        let mut spacing = [1.; 3];
        for (j, v) in numbers(&case["resample"]["spacing_mm"]).iter().enumerate() {
            spacing[j] = *v;
        }
        let grid = spacing_grid(&source.geometry, spacing).unwrap();
        assert_close(
            &grid.affine.concat(),
            &affine_from_json(&case["resample"]["affine_ras"], ndim).concat(),
            1e-6,
            0.,
        );
        let expected_shape = numbers(&case["resample"]["shape_xyz"]);
        for j in 0..ndim {
            assert_eq!(grid.shape[j], expected_shape[j] as usize);
        }
        for (mode, key) in [
            (Interpolation::Linear, "linear"),
            (Interpolation::Nearest, "nearest"),
        ] {
            let actual = resample_like(&source, &grid, mode, -999.).unwrap();
            assert_close(
                &actual.values,
                &numbers(&case["resample"]["outputs_x_fastest"][key]),
                1e-6,
                1e-5,
            );
        }
        assert_eq!(
            resample_mask_like(&mask, &grid, 0).unwrap().labels,
            numbers(&case["resample"]["outputs_x_fastest"]["mask_nearest"])
                .iter()
                .map(|v| *v as u32)
                .collect::<Vec<_>>()
        );
        let mut start = [0; 3];
        let mut end = [1; 3];
        let crop_index = numbers(&case["crop"]["index_xyz"]);
        let crop_shape = numbers(&case["crop"]["shape_xyz"]);
        for j in 0..ndim {
            start[j] = crop_index[j] as usize;
            end[j] = start[j] + crop_shape[j] as usize;
        }
        let cropped = crop_image(&source, BoundingBox { start, end }).unwrap();
        assert_eq!(cropped.values, numbers(&case["crop"]["values_x_fastest"]));
        assert_close(
            &cropped.geometry.affine.concat(),
            &affine_from_json(&case["crop"]["affine_ras"], ndim).concat(),
            1e-6,
            0.,
        );
        for landmark in case["world_landmarks_ras_mm"].as_array().unwrap() {
            let mut p = [0.; 3];
            for (j, v) in numbers(&landmark["index_xyz"]).iter().enumerate() {
                p[j] = *v;
            }
            assert_close(
                &source.geometry.world(p)[..ndim],
                &numbers(&landmark["world"]),
                1e-6,
                0.,
            );
        }
    }
    let probe = &fixture["boundary_probe"];
    for expected in probe["outputs"].as_array().unwrap() {
        let p = numbers(&expected["source_continuous_index_xy"]);
        for (mode, key) in [
            (Interpolation::Linear, "linear"),
            (Interpolation::Nearest, "nearest"),
        ] {
            let actual = interpolate(
                &numbers(&probe["source_values_x_fastest"]),
                [3, 2, 1],
                2,
                [p[0], p[1], 0.],
                mode,
                -999.,
            );
            assert_close(&[actual], &[expected[key].as_f64().unwrap()], 1e-6, 1e-5);
        }
    }
}

fn geometry(shape: [usize; 3], ndim: u8) -> Geometry {
    Geometry {
        shape,
        ndim,
        affine: [
            [2., 0., 0., 10.],
            [0., 3., 0., -20.],
            [0., 0., 4., 30.],
            [0., 0., 0., 1.],
        ],
        unit: SpatialUnit::Millimeter,
        diagnostics: vec![],
    }
}
fn image(shape: [usize; 3], ndim: u8) -> ImageVolume {
    ImageVolume {
        geometry: geometry(shape, ndim),
        values: (0..shape.iter().product()).map(|x| x as f64).collect(),
    }
}
struct TestFile(PathBuf);
impl TestFile {
    fn new(extension: &str) -> Self {
        let token = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "rad-image-{}-{token}.{extension}",
            std::process::id()
        )))
    }
    fn write(&self, bytes: &[u8]) {
        File::options()
            .write(true)
            .create_new(true)
            .open(&self.0)
            .unwrap()
            .write_all(bytes)
            .unwrap();
    }
}
impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn nifti_bytes(ndim: i16, datatype: i16, unit: u8, sform: bool, qform: bool) -> Vec<u8> {
    let mut bytes = vec![0u8; 352 + 3 * 2 * 8];
    bytes[0..4].copy_from_slice(&348i32.to_le_bytes());
    for (j, value) in [ndim, 3, 2, 1, 1, 1, 1, 1].iter().enumerate() {
        bytes[40 + j * 2..42 + j * 2].copy_from_slice(&value.to_le_bytes());
    }
    bytes[70..72].copy_from_slice(&datatype.to_le_bytes());
    bytes[72..74].copy_from_slice(&64i16.to_le_bytes());
    for (j, value) in [1f32, 2., 3., 4., 1., 1., 1., 1.].iter().enumerate() {
        bytes[76 + j * 4..80 + j * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[108..112].copy_from_slice(&352f32.to_le_bytes());
    bytes[112..116].copy_from_slice(&2f32.to_le_bytes());
    bytes[116..120].copy_from_slice(&1f32.to_le_bytes());
    bytes[123] = unit;
    bytes[252..254].copy_from_slice(&i16::from(qform).to_le_bytes());
    bytes[254..256].copy_from_slice(&i16::from(sform).to_le_bytes());
    for (j, value) in [-2f32, 0., 0., 10., 0., 3., 0., -20., 0., 0., 4., 30.]
        .iter()
        .enumerate()
    {
        bytes[280 + j * 4..284 + j * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[344..348].copy_from_slice(b"n+1\0");
    for j in 0..6 {
        bytes[352 + j * 8..360 + j * 8].copy_from_slice(&(j as f64).to_le_bytes());
    }
    bytes
}
#[test]
fn nifti_preserves_buffer_affine_units_scaling_and_conflict() {
    let file = TestFile::new("nii");
    file.write(&nifti_bytes(2, 64, 2, true, true));
    let loaded = read_nifti(&file.0).unwrap();
    assert_eq!(loaded.values, vec![1., 3., 5., 7., 9., 11.]);
    assert_eq!(loaded.geometry.shape, [3, 2, 1]);
    assert_eq!(loaded.geometry.ndim, 2);
    assert_eq!(loaded.geometry.world([2., 1., 0.]), [6., -17., 30.]);
    assert_eq!(loaded.geometry.spacing_mm().unwrap(), [2., 3., 4.]);
    assert!(loaded
        .geometry
        .diagnostics
        .iter()
        .any(|d| d.code == "nifti_sform_qform_conflict"));
    assert_eq!(read_mask(&file.0).unwrap().labels, vec![1, 3, 5, 7, 9, 11]);
}
#[test]
fn nifti_absent_transforms_unknown_units_invalid_dimensions() {
    let file = TestFile::new("nii");
    file.write(&nifti_bytes(3, 64, 0, false, false));
    let loaded = read_nifti(&file.0).unwrap();
    assert_eq!(loaded.geometry.ndim, 3);
    assert_eq!(loaded.geometry.affine[0], [2., 0., 0., 0.]);
    assert!(loaded.geometry.spacing_mm().is_err());
    assert!(loaded
        .geometry
        .diagnostics
        .iter()
        .any(|d| d.code == "nifti_no_world_transform"));
    let file4 = TestFile::new("nii");
    file4.write(&nifti_bytes(4, 64, 2, true, false));
    assert_eq!(
        read_nifti(&file4.0).unwrap_err().code,
        ErrorCode::Unsupported
    );
    let rgb = TestFile::new("nii");
    rgb.write(&nifti_bytes(3, 128, 2, true, false));
    assert!(read_nifti(&rgb.0).is_err());
}
#[test]
fn qform_rotation_reflection_and_nonfinite_input() {
    let h = nifti::NiftiHeader {
        qform_code: 1,
        pixdim: [-1., 2., 3., 4., 1., 1., 1., 1.],
        quatern_d: (0.5f32).sqrt(),
        quatern_x: 10.,
        quatern_y: 20.,
        quatern_z: 30.,
        ..Default::default()
    };
    let a = qform_affine(&h).unwrap();
    assert!(a[0][0].abs() < 1e-6);
    assert!((a[1][0] - 2.).abs() < 1e-6);
    assert!((a[0][1] + 3.).abs() < 1e-6);
    assert_eq!(a[2][2], -4.);
    let mut invalid_h = h;
    invalid_h.quatern_b = f32::NAN;
    assert!(qform_affine(&invalid_h).is_err());
}
#[test]
fn crop_world_coordinates_margin_spacing_and_empty_roi() {
    let source = image([6, 5, 3], 3);
    let mut labels = vec![0; 90];
    labels[index([6, 5, 3], [2, 1, 1])] = 7;
    labels[index([6, 5, 3], [3, 2, 1])] = 7;
    let mask = MaskVolume {
        geometry: source.geometry.clone(),
        labels,
    };
    let selected = select_labels(&mask, &[7]).unwrap();
    let bounds = bounding_box(&selected).unwrap();
    assert_eq!(
        bounds,
        BoundingBox {
            start: [2, 1, 1],
            end: [4, 3, 2]
        }
    );
    let out = crop_image(&source, bounds).unwrap();
    assert_eq!(out.geometry.shape, [2, 2, 1]);
    assert_eq!(out.geometry.ndim, 3);
    assert_eq!(
        out.geometry.world([0., 0., 0.]),
        source.geometry.world([2., 1., 1.])
    );
    assert_eq!(out.values[0], source.values[index([6, 5, 3], [2, 1, 1])]);
    assert_eq!(
        crop_mask(&selected, bounds).unwrap().labels,
        vec![1, 0, 0, 1]
    );
    assert_eq!(
        roi_bounds_margin_mm(&selected, [1., 3.1, 4.]).unwrap(),
        BoundingBox {
            start: [1, 0, 0],
            end: [5, 5, 3]
        }
    );
    assert_eq!(maximum_roi_section(&selected, 2).unwrap(), bounds);
    assert!(select_labels(&mask, &[0]).is_err());
    assert!(select_labels(&mask, &[9]).is_err());
    let empty = MaskVolume {
        geometry: source.geometry.clone(),
        labels: vec![0; 90],
    };
    assert_eq!(
        bounding_box(&empty).unwrap_err().code,
        ErrorCode::EmptyResult
    );
}
#[test]
fn resampling_half_voxel_boundary_units_and_interpolator() {
    let mut source = image([2, 1, 1], 2);
    source.values = vec![0., 10.];
    source.geometry.affine = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    let mut target = source.geometry.clone();
    target.shape = [5, 1, 1];
    target.affine[0] = [0.5, 0., 0., -0.5];
    assert_eq!(
        resample_like(&source, &target, Interpolation::Linear, -99.)
            .unwrap()
            .values,
        vec![0., 0., 5., 10., -99.]
    );
    assert_eq!(
        resample_like(&source, &target, Interpolation::Nearest, -99.)
            .unwrap()
            .values,
        vec![0., 0., 10., 10., -99.]
    );
    let mask = MaskVolume {
        geometry: source.geometry.clone(),
        labels: vec![7, 11],
    };
    assert_eq!(
        resample_mask_like(&mask, &target, 0).unwrap().labels,
        vec![7, 7, 11, 11, 0]
    );
    let grid = spacing_grid(&source.geometry, [0.5, 1., 1.]).unwrap();
    assert_eq!(grid.shape, [4, 1, 1]);
    assert_eq!(grid.world([0., 0., 0.]), [-0.25, 0., 0.]);
    let mut meter = source.clone();
    meter.geometry.unit = SpatialUnit::Meter;
    for row in meter.geometry.affine.iter_mut().take(3) {
        for value in row.iter_mut() {
            *value /= 1000.;
        }
    }
    assert_eq!(
        resample_like(&meter, &source.geometry, Interpolation::Nearest, -99.)
            .unwrap()
            .values,
        source.values
    );
}
#[test]
fn intensity_fit_resegment_and_constant_statistics() {
    let source = image([2, 2, 1], 2);
    let mask = MaskVolume {
        geometry: source.geometry.clone(),
        labels: vec![0, 1, 1, 0],
    };
    let transform = fit_intensity_transform(&source, Normalization::ZScore, Some(&mask)).unwrap();
    assert_eq!(
        (
            transform.offset,
            transform.scale,
            transform.statistical_voxels
        ),
        (1.5, 0.5, 2)
    );
    assert_eq!(
        transform.apply(&source).unwrap().values,
        vec![-3., -1., 1., 3.]
    );
    assert_eq!(
        clip_percentile(&source, 25., 75., None).unwrap().values,
        vec![0.75, 1., 2., 2.25]
    );
    assert_eq!(
        resegment(&source, &mask, 1.5, 3.).unwrap().labels,
        vec![0, 0, 1, 0]
    );
    assert_eq!(mask.labels, vec![0, 1, 1, 0]);
    let constant = ImageVolume {
        geometry: source.geometry.clone(),
        values: vec![1.; 4],
    };
    assert!(fit_intensity_transform(&constant, Normalization::ZScore, None).is_err());
    assert!(resegment(&source, &mask, 9., 10.).is_err());
}
#[test]
fn connectivity_labels_ties_dilation_and_invalid_dimensions() {
    let mask = MaskVolume {
        geometry: geometry([3, 3, 1], 2),
        labels: vec![1, 0, 0, 0, 2, 0, 0, 0, 1],
    };
    let four = connected_components(&mask, Connectivity::Four, false).unwrap();
    assert_eq!(four.components.len(), 3);
    assert_eq!(four.components[0].bounds.start, [0, 0, 0]);
    assert_eq!(
        connected_components(&mask, Connectivity::Eight, false)
            .unwrap()
            .components
            .len(),
        1
    );
    assert_eq!(
        connected_components(&mask, Connectivity::Eight, true)
            .unwrap()
            .components
            .len(),
        3
    );
    assert_eq!(
        largest_component(&mask, Connectivity::Four).unwrap().labels,
        vec![1, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert!(connected_components(&mask, Connectivity::Six, false).is_err());
    let center = MaskVolume {
        geometry: mask.geometry.clone(),
        labels: vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
    };
    assert_eq!(
        dilate_voxels(&center, 1, Connectivity::Four)
            .unwrap()
            .labels,
        vec![0, 1, 0, 1, 1, 1, 0, 1, 0]
    );
}
#[test]
fn anisotropic_3d_distance_transform_matches_brute_force() {
    let shape = [5, 4, 3];
    let mut labels = vec![0; 60];
    labels[0] = 1;
    labels[14] = 1;
    labels[55] = 1;
    let mask = MaskVolume {
        geometry: geometry(shape, 3),
        labels,
    };
    let spacing = [1.2, 3.1, 4.5];
    let distance = squared_distance_to(&mask, true, spacing);
    for (i, &actual) in distance.iter().enumerate() {
        let p = position(shape, i);
        let expected = mask
            .labels
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(seed, _)| {
                let q = position(shape, seed);
                (0..3)
                    .map(|j| ((p[j] as f64 - q[j] as f64) * spacing[j]).powi(2))
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            (actual - expected).abs() < 1e-10,
            "{i}: {actual} vs {expected}"
        );
    }
}
#[test]
fn physical_ring_distance_organ_intersection_and_geometry_errors() {
    let mut mask = MaskVolume {
        geometry: geometry([5, 5, 1], 2),
        labels: vec![0; 25],
    };
    mask.labels[12] = 1;
    let ring = peritumoral_ring(&mask, 0., 2.1, None).unwrap();
    let active: Vec<usize> = ring
        .labels
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != 0).then_some(i))
        .collect();
    assert_eq!(active, vec![11, 13]);
    let organ = MaskVolume {
        geometry: mask.geometry.clone(),
        labels: (0..25).map(|i| u32::from(i == 11)).collect(),
    };
    assert_eq!(
        peritumoral_ring(&mask, 0., 2.1, Some(&organ))
            .unwrap()
            .labels,
        organ.labels
    );
    assert_eq!(
        peritumoral_ring(&mask, -2.1, 0., None).unwrap().labels,
        mask.labels
    );
    mask.geometry.affine[0][1] = 0.25;
    assert!(peritumoral_ring(&mask, 0., 3., None).is_err());
    mask.geometry = geometry([5, 5, 1], 2);
    mask.geometry.unit = SpatialUnit::Unknown;
    assert!(peritumoral_ring(&mask, 0., 3., None).is_err());
}
#[test]
fn raster_types_alpha_and_multipage_tiff() {
    let png = TestFile::new("png");
    image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 0, 255, 0])
        .unwrap()
        .save(&png.0)
        .unwrap();
    assert_eq!(
        read_raster(&png.0).unwrap().pixels,
        RasterPixels::Rgb8(vec![255, 0, 0, 0, 255, 0])
    );
    let alpha = TestFile::new("png");
    image::RgbaImage::from_raw(1, 1, vec![0, 0, 0, 255])
        .unwrap()
        .save(&alpha.0)
        .unwrap();
    assert!(read_raster(&alpha.0).is_err());
    let gray = TestFile::new("tiff");
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(File::create(&gray.0).unwrap()).unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray16>(2, 1, &[0, 65535])
            .unwrap();
    }
    assert_eq!(
        read_raster(&gray.0).unwrap().pixels,
        RasterPixels::Gray16(vec![0, 65535])
    );
    let multi = TestFile::new("tif");
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(File::create(&multi.0).unwrap()).unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(1, 1, &[1])
            .unwrap();
        encoder
            .write_image::<tiff::encoder::colortype::Gray8>(1, 1, &[2])
            .unwrap();
    }
    assert_eq!(
        read_raster(&multi.0).unwrap_err().code,
        ErrorCode::Unsupported
    );
}
