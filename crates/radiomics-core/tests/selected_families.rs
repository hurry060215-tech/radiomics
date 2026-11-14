use ov_radiomics_core::{
    contracts::*,
    features::{self, FeatureFamily, ALL_FAMILIES},
};

#[test]
fn selected_families_are_exact_projections_of_the_single_original_engine() {
    let config = ExtractionConfig {
        profile: "python_golden_v1".into(),
        discretization: Discretization::FixedWidth { width: 1. },
        aggregation: TextureAggregation::DirectionMean,
        gldm_alpha: 0,
        voxel_array_shift: 0.,
    };
    for (shape, ndim) in [([3, 2, 1], 2), ([3, 2, 2], 3)] {
        let geometry = Geometry {
            shape,
            ndim,
            affine: [
                [1., 0., 0., 0.],
                [0., 2., 0., 0.],
                [0., 0., 3., 0.],
                [0., 0., 0., 1.],
            ],
            unit: SpatialUnit::Millimeter,
            diagnostics: vec![],
        };
        let n = shape.iter().product::<usize>();
        let image = ImageVolume {
            geometry: geometry.clone(),
            values: (0..n).map(|i| (i % 4) as f64 + 1.).collect(),
        };
        let mask = MaskVolume {
            geometry,
            labels: vec![1; n],
        };
        let full = features::extract(&image, &mask, None, 1, &config).unwrap();
        let mut union = std::collections::BTreeMap::new();
        for (family, prefix, count) in [
            (FeatureFamily::FirstOrder, "firstorder", 16),
            (FeatureFamily::Glcm, "glcm", 10),
            (FeatureFamily::Glrlm, "glrlm", 8),
            (FeatureFamily::Glszm, "glszm", 8),
            (FeatureFamily::Gldm, "gldm", 8),
            (FeatureFamily::Ngtdm, "ngtdm", 5),
            (FeatureFamily::ShapeBasic, "shape_basic", 11),
        ] {
            let selected =
                features::extract_families(&image, &mask, None, 1, &config, &[family]).unwrap();
            assert_eq!(selected.len(), count);
            for (key, value) in selected {
                assert!(key.starts_with(&format!("original_{prefix}_")));
                assert!(
                    value.is_nan() && full[&key].is_nan()
                        || value.to_bits() == full[&key].to_bits()
                );
                assert!(union.insert(key, value).is_none());
            }
        }
        assert_eq!(union.len(), full.len());
        assert!(features::extract_families(&image, &mask, None, 1, &config, &[]).is_err());
        assert!(
            features::extract_families(&image, &mask, None, 1, &config, &[ALL_FAMILIES[0]; 2])
                .is_err()
        );
    }
}
