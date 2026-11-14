use ov_radiomics_core::contracts::*;

fn geometry() -> Geometry {
    Geometry {
        shape: [1, 1, 1],
        ndim: 3,
        unit: SpatialUnit::Millimeter,
        affine: [
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.],
            [0., 0., 0., 1.],
        ],
        diagnostics: vec![],
    }
}
#[test]
fn singleton_axes_still_require_complete_affine_alignment() {
    let original = geometry();
    let mut other = original.clone();
    other.affine[0][0] = -1.;
    assert!(original.aligned_with(&other).is_err());
    other = original.clone();
    other.affine[2][2] = 4.;
    assert!(original.aligned_with(&other).is_err());
    other = original.clone();
    other.affine[0][3] = 0.5e-6;
    original.aligned_with(&other).unwrap();
}
#[test]
fn explicit_missing_flags_and_units_cannot_be_fabricated() {
    let mut data = NumericBuffer {
        rows: 1,
        cols: 2,
        values: vec![2., 0.],
        missing: vec![false, true],
    };
    data.validate().unwrap();
    data.values[1] = 3.;
    assert!(data.validate().is_err());
    data.values[1] = f64::NAN;
    assert!(data.validate().is_err());
    let mut g = geometry();
    g.unit = SpatialUnit::Unknown;
    assert!(g.spacing_mm().is_err());
    g.affine[0][1] = 0.5;
    assert!(g.require_orthogonal().is_err());
}
fn prediction() -> PredictionTable {
    PredictionTable {
        task: Task::Binary,
        unit: PredictionUnit::Patient,
        classes: vec!["N".into(), "P".into()],
        rows: vec![PredictionRow {
            sample_id: "s1".into(),
            patient_id: "p1".into(),
            slide_id: Some("slide1".into()),
            tile_id: Some("tile1".into()),
            cohort: "test".into(),
            fold: None,
            training_source: "fit-sha256".into(),
            probabilities: vec![0.25, 0.75],
            value: None,
        }],
    }
}
#[test]
fn actual_prediction_units_are_unique_and_hierarchy_is_consistent() {
    let mut p = prediction();
    p.validate().unwrap();
    let mut second = p.rows[0].clone();
    second.sample_id = "s2".into();
    p.rows.push(second);
    assert!(p.validate().is_err());
    p.unit = PredictionUnit::Tile;
    assert!(p.validate().is_err());
    p.rows[1].tile_id = Some("tile2".into());
    p.rows[1].patient_id = "p2".into();
    assert!(p.validate().is_err());
    p.rows[1].slide_id = Some("slide2".into());
    p.validate().unwrap();
    p.rows[1].probabilities = vec![0., 0.];
    assert!(p.validate().is_err());
}

#[test]
fn empty_slide_statuses_keep_the_patient_parent() {
    let mut table = TileTable {
        tiles: vec![],
        features: None,
        feature_names: vec![],
        slide_status: std::collections::BTreeMap::from([("empty-slide".into(), "no_tiles".into())]),
        slide_patients: std::collections::BTreeMap::from([(
            "empty-slide".into(),
            "patient".into(),
        )]),
    };
    table.validate().unwrap();
    table.slide_patients.clear();
    assert!(table.validate().is_err());
}
