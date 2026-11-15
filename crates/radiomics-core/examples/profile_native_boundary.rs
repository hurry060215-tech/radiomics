//! Diagnostic only: same frozen typed image/config through core and original gateway.
//! No algorithm, contract, tolerance or official benchmark is changed.
#[allow(dead_code)]
#[path = "../../radiomics-python/src/gateway.rs"]
mod gateway;

use ov_radiomics_core::{contracts::*, features, imaging, workflow};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, hint::black_box, path::PathBuf, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema_version: u32,
    request: ArrayRequest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArrayRequest {
    operation: String,
    image: ImageVolume,
    morphology_mask: MaskVolume,
    intensity_mask: Option<MaskVolume>,
    roi_label: u32,
    config: ExtractionConfig,
    families: Option<Vec<features::FeatureFamily>>,
}

fn extract(request: &ArrayRequest) -> BTreeMap<String, f64> {
    features::extract_families(
        &request.image, &request.morphology_mask, request.intensity_mask.as_ref(),
        request.roi_label, &request.config,
        request.families.as_deref().unwrap_or(&features::ALL_FAMILIES),
    ).expect("Same public extraction core")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("Usage: profile_native_boundary REQUEST_JSON SOURCE_DIR OUTPUT_JSON REPEATS WARMUPS".into());
    }
    let payload = fs::read_to_string(&args[1])?;
    let envelope: Envelope = serde_json::from_str(&payload)?;
    assert_eq!(envelope.schema_version, 1);
    let request = &envelope.request;
    assert_eq!(request.operation, "array_features");
    assert!(request.intensity_mask.is_none() && request.families.is_none());
    let source = PathBuf::from(&args[2]);
    let repeats: usize = args[4].parse()?;
    let warmups: usize = args[5].parse()?;
    assert!(repeats > 0);
    let source_image = imaging::read_nifti(source.join("image.nii.gz"))?;
    let source_mask = imaging::read_mask(source.join("mask.nii.gz"))?;
    let bounds = imaging::roi_bounds_margin_voxels(&source_mask, [0; 3])?;
    let cropped_image = imaging::crop_image(&source_image, bounds)?;
    let cropped_mask = imaging::crop_mask(&source_mask, bounds)?;
    // Exact bytes/affine, not a similar synthetic image or a different algorithm.
    assert_eq!(cropped_image.values, request.image.values);
    assert_eq!(cropped_mask.labels, request.morphology_mask.labels);
    assert_eq!(cropped_image.geometry.shape, request.image.geometry.shape);
    assert_eq!(cropped_image.geometry.affine, request.image.geometry.affine);
    let features = extract(request);
    assert_eq!(features.len(), 66);
    let expected = serde_json::to_value(&features)?;
    assert_eq!(serde_json::from_str::<Value>(&gateway::execute(&payload)?)?, expected);
    let stages = [
        ("typed_serde_parse", 1usize),
        ("image_mask_validate_geometry", 100),
        ("image_mask_clone", 10),
        ("image_mask_serialize", 1),
        ("core_extract_preloaded", 1),
        ("original_gateway_preencoded", 1),
        ("nifti_read_pair", 1),
        ("crop_preloaded_source_pair", 1),
        ("nifti_read_crop_core", 1),
    ];
    let mut raw = Vec::new();
    use std::io::Write;
    let mut raw_file = fs::OpenOptions::new().write(true).create_new(true)
        .open(PathBuf::from(&args[3]).with_extension("raw.jsonl"))?;
    // Rotate stage order across rounds. These are independent probes; their
    // medians must not be subtracted/added to infer an official L2/L3 stage.
    for round in 0..warmups + repeats {
        for offset in 0..stages.len() {
            let (stage, iterations) = stages[(offset + round) % stages.len()];
            let started = Instant::now();
            for _ in 0..iterations {
                match stage {
                    "typed_serde_parse" => { black_box(serde_json::from_str::<Envelope>(black_box(&payload))?); }
                    "image_mask_validate_geometry" => {
                        request.image.validate()?;
                        request.morphology_mask.validate()?;
                        request.image.geometry.aligned_with(&request.morphology_mask.geometry)?;
                        black_box(());
                    }
                    "image_mask_clone" => {
                        black_box((request.image.clone(), request.morphology_mask.clone()));
                    }
                    "image_mask_serialize" => {
                        black_box(serde_json::to_string(&(&request.image, &request.morphology_mask))?);
                    }
                    "core_extract_preloaded" => { black_box(extract(black_box(request))); }
                    "original_gateway_preencoded" => { black_box(gateway::execute(black_box(&payload))?); }
                    "nifti_read_pair" => {
                        black_box((imaging::read_nifti(source.join("image.nii.gz"))?,
                            imaging::read_mask(source.join("mask.nii.gz"))?));
                    }
                    "crop_preloaded_source_pair" => {
                        black_box((imaging::crop_image(&source_image, bounds)?,
                            imaging::crop_mask(&source_mask, bounds)?));
                    }
                    "nifti_read_crop_core" => {
                        let image = imaging::read_nifti(source.join("image.nii.gz"))?;
                        let mask = imaging::read_mask(source.join("mask.nii.gz"))?;
                        let bounds = imaging::roi_bounds_margin_voxels(&mask, [0; 3])?;
                        let image = imaging::crop_image(&image, bounds)?;
                        let mask = imaging::crop_mask(&mask, bounds)?;
                        black_box(features::extract_families(&image, &mask, None,
                            request.roi_label, &request.config, &features::ALL_FAMILIES)?);
                    }
                    _ => unreachable!(),
                }
            }
            let total_ns = started.elapsed().as_nanos();
            if round >= warmups {
                let observation = json!({"round":round-warmups,"stage":stage,"iterations":iterations,
                    "total_nanoseconds":total_ns as u64,
                    "nanoseconds_per_call":total_ns as f64/iterations as f64});
                writeln!(raw_file, "{}", observation)?;
                raw_file.flush()?;
                raw.push(observation);
            }
        }
    }
    let report = json!({"kind":"diagnostic_not_formal_benchmark", "repeats":repeats,
        "warmups_per_stage":warmups,"implementation_sha256":workflow::implementation_sha256(),
        "schema":"radiomics.boundary-core-diagnostic.v1","payload_bytes":payload.len(),
        "prepared_shape_xyz":request.image.geometry.shape,
        "same_nifti_crop_values_labels_affine_exact":true,
        "features":features,"raw":raw,
        "scope":"preloaded core includes original core validation/discretization/all matrices; gateway compiled verbatim from reviewed source; NIfTI probes exclude workflow SHA/cache/export"});
    let mut output = fs::OpenOptions::new().write(true).create_new(true).open(&args[3])?;
    output.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    Ok(())
}
