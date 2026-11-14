use ov_radiomics_core::{contracts::*, workflow::*};
use std::{
    fs,
    path::{Path, PathBuf},
};
fn temp() -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("radiomics-workflow-{}-{nonce}", std::process::id()));
    fs::create_dir(&path).unwrap();
    path
}
fn nifti(path: &Path, values: &[f64]) {
    assert_eq!(values.len(), 6);
    let mut bytes = vec![0u8; 352];
    bytes[0..4].copy_from_slice(&348i32.to_le_bytes());
    for (i, value) in [2i16, 3, 2, 1, 1, 1, 1, 1].iter().enumerate() {
        bytes[40 + i * 2..42 + i * 2].copy_from_slice(&value.to_le_bytes());
    }
    bytes[70..72].copy_from_slice(&64i16.to_le_bytes());
    bytes[72..74].copy_from_slice(&64i16.to_le_bytes());
    for (i, value) in [1f32, 2., 3., 4., 1., 1., 1., 1.].iter().enumerate() {
        bytes[76 + i * 4..80 + i * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[108..112].copy_from_slice(&352f32.to_le_bytes());
    bytes[112..116].copy_from_slice(&1f32.to_le_bytes());
    bytes[123] = 2;
    bytes[254..256].copy_from_slice(&1i16.to_le_bytes());
    for (offset, value) in [(280, 2f32), (300, 3.), (320, 4.)] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[344..348].copy_from_slice(b"n+1\0");
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}
fn cohort(root: &Path) -> Vec<CohortSample> {
    nifti(&root.join("image.nii"), &[1., 1., 2., 2., 3., 3.]);
    nifti(&root.join("mask.nii"), &[1.; 6]);
    fs::write(root.join("cohort.csv"),"sample_id,patient_id,image,mask,region\nB,P1,image.nii,mask.nii,tumor\nA,P1,image.nii,mask.nii,tumor\n").unwrap();
    read_cohort(root.join("cohort.csv")).unwrap()
}
fn config() -> ExtractConfig {
    ExtractConfig {
        extraction: ExtractionConfig {
            profile: "python_golden_v1".into(),
            discretization: Discretization::FixedWidth { width: 1. },
            aggregation: TextureAggregation::DirectionMean,
            gldm_alpha: 0,
            voxel_array_shift: 0.,
        },
        labels: vec![1],
        preparation: vec![],
    }
}
#[test]
fn threads_cache_content_changes_and_export_preserve_identity() {
    let root = temp();
    let samples = cohort(&root);
    let cfg = config();
    let first = extract_batch(&samples, &cfg, 1, Some(&root.join("cache"))).unwrap();
    let second = extract_batch(&samples, &cfg, 3, Some(&root.join("cache"))).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.records[0].metadata.sample_id, "A");
    assert_eq!(
        first.records[0].features["original_firstorder_Mean"],
        Some(2.)
    );
    assert_eq!(first.table.as_ref().unwrap().samples[0].patient_id, "P1");
    let old = first.records[0].cache_key.clone();
    nifti(&root.join("image.nii"), &[2., 2., 4., 4., 6., 6.]);
    let changed = extract_batch(&samples, &cfg, 2, Some(&root.join("cache"))).unwrap();
    assert_ne!(old, changed.records[0].cache_key);
    assert_eq!(
        changed.records[0].features["original_firstorder_Mean"],
        Some(4.)
    );
    let output = root.join("output");
    export_extraction(&changed, &output, false).unwrap();
    assert!(export_extraction(&changed, &output, false).is_err());
    assert!(output.join("features.csv").is_file());
    assert!(output.join("metadata.csv").is_file());
    let raw: ExtractionBatch =
        serde_json::from_slice(&fs::read(output.join("extraction.json")).unwrap()).unwrap();
    assert_eq!(raw, changed);
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn failed_samples_and_undefined_features_remain_explicit() {
    let root = temp();
    let mut samples = cohort(&root);
    samples[0].image = root.join("absent.nii");
    let batch = extract_batch(&samples, &config(), 2, None).unwrap();
    assert_eq!(batch.status, BatchStatus::Partial);
    assert_eq!(batch.failures.len(), 1);
    assert_eq!(batch.records.len(), 1);
    nifti(&root.join("mask.nii"), &[1., 0., 0., 0., 0., 0.]);
    let batch = extract_batch(&samples[1..], &config(), 1, None).unwrap();
    assert_eq!(batch.records[0].features["original_glcm_Contrast"], None);
    let table = batch.table.unwrap();
    table.validate().unwrap();
    let column = table
        .features
        .iter()
        .position(|f| f.name == "original_glcm_Contrast")
        .unwrap();
    assert!(table.data.missing[column]);
    let all_failed = extract_batch(&samples[..1], &config(), 1, None).unwrap();
    assert_eq!(all_failed.status, BatchStatus::Empty);
    assert!(all_failed.table.is_none());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn corrupt_cache_is_diagnosed_and_recomputed() {
    let root = temp();
    let samples = cohort(&root);
    let cache = root.join("cache");
    let batch = extract_batch(&samples, &config(), 1, Some(&cache)).unwrap();
    let key = &batch.records[0].cache_key;
    fs::write(cache.join(format!("{key}.json")), b"{}").unwrap();
    let recomputed = extract_batch(&samples, &config(), 1, Some(&cache)).unwrap();
    assert_eq!(recomputed.records[0].features, batch.records[0].features);
    assert!(recomputed.records[0]
        .quality
        .iter()
        .any(|q| q.code == "cache_rejected"));
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn malformed_configuration_and_duplicate_ids_fail_before_processing() {
    let root = temp();
    let mut samples = cohort(&root);
    samples[0].metadata.sample_id = samples[1].metadata.sample_id.clone();
    assert!(extract_batch(&samples, &config(), 1, None).is_err());
    let mut cfg = config();
    cfg.preparation = vec![
        PrepareStep::Resegment {
            lower: 1.,
            upper: 2.,
        },
        PrepareStep::ClipRange {
            lower: 0.,
            upper: 1.,
        },
    ];
    assert!(cfg.validate().is_err());
    let raw = serde_json::json!({"extraction":config().extraction,"labels":[1],"unsupported_filter":"wavelet"});
    assert!(serde_json::from_value::<ExtractConfig>(raw).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_per_sample_roi_labels_change_features_and_cache_identity() {
    let root = temp();
    let mut samples = cohort(&root);
    nifti(&root.join("mask.nii"), &[1., 1., 2., 2., 0., 0.]);
    samples[0].roi_label = Some(1);
    let first = extract_batch(&samples[..1], &config(), 1, Some(&root.join("cache"))).unwrap();
    samples[0].roi_label = Some(2);
    let second = extract_batch(&samples[..1], &config(), 1, Some(&root.join("cache"))).unwrap();
    assert_eq!(
        first.records[0].features["original_firstorder_Mean"],
        Some(1.)
    );
    assert_eq!(
        second.records[0].features["original_firstorder_Mean"],
        Some(2.)
    );
    assert_ne!(first.records[0].cache_key, second.records[0].cache_key);
    assert!(second.records[0]
        .quality
        .iter()
        .any(|q| q.code == "effective_roi_labels" && q.message == "[2]"));
    fs::remove_dir_all(root).unwrap();
}
