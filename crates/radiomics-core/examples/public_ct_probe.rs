//! Independent public-CT acceptance probe: only calls existing native core APIs.
//! Original/reference CT data is supplied externally and never embedded here.
use ov_radiomics_core::{contracts::Discretization, features, imaging, workflow};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};

fn values_sha(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
fn labels_sha(values: &[u32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
fn triplet(value: &Value) -> [usize; 3] {
    std::array::from_fn(|i| value[i].as_u64().unwrap().try_into().unwrap())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 5 {
        return Err("Usage: public_ct_probe IMAGE MASK PROTOCOL OUTPUT EXPECTED_IMPL_SHA".into());
    }
    let expected = args[4]
        .to_str()
        .ok_or("Expected ASCII implementation SHA256")?;
    if workflow::implementation_sha256() != expected {
        return Err(format!(
            "Core implementation differs from immutable acceptance build: {} != {expected}",
            workflow::implementation_sha256()
        )
        .into());
    }
    let protocol: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let image = imaging::read_nifti(&args[0])?;
    let mask = imaging::read_mask(&args[1])?;
    image.geometry.aligned_with(&mask.geometry)?;
    let crop = imaging::BoundingBox {
        start: triplet(&protocol["matrix_case"]["start_xyz"]),
        end: triplet(&protocol["matrix_case"]["end_xyz_exclusive"]),
    };
    let cropped_image = imaging::crop_image(&image, crop)?;
    let cropped_mask = imaging::crop_mask(&mask, crop)?;
    let discrete = features::discretize(
        &cropped_image,
        &cropped_mask,
        1,
        &Discretization::FixedWidth {
            width: protocol["settings"]["binWidth"]
                .as_f64()
                .ok_or("Explicit bin width required")?,
        },
    )?;
    let matrices = features::texture_matrices(
        &discrete,
        protocol["settings"]["gldm_a"]
            .as_u64()
            .ok_or("Explicit GLDM alpha required")?
            .try_into()?,
    )?;
    let output = json!({
        "implementation_sha256":workflow::implementation_sha256(),
        "source_geometry":image.geometry,
        "source_image_f64_le_sha256":values_sha(&image.values),
        "source_mask_u32_le_sha256":labels_sha(&mask.labels),
        "crop_geometry":cropped_image.geometry,
        "crop_image_f64_le_sha256":values_sha(&cropped_image.values),
        "crop_mask_u32_le_sha256":labels_sha(&cropped_mask.labels),
        "discrete_u32_le_sha256":labels_sha(&discrete.values),
        "matrices":matrices
    });
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    file.write_all(&serde_json::to_vec_pretty(&output)?)?;
    file.sync_all()?;
    Ok(())
}
