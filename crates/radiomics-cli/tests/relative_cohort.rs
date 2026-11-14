use std::{fs, process::Command};

#[test]
fn bare_relative_cohort_resolves_without_global_working_directory_changes() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "radiomics-cli-relative-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&directory).unwrap();
    fs::write(
        directory.join("cohort.csv"),
        "sample_id,patient_id,image,mask\n001,P001,missing-image.nii,missing-mask.nii\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_radiomics"))
        .current_dir(&directory)
        .args(["check", "--cohort", "cohort.csv"])
        .output()
        .unwrap();
    // The row-level missing image is diagnosed (exit 2), rather than failing to
    // open the valid cohort itself (exit 1) because parent(\"cohort.csv\") is empty.
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report[0]["sample_id"], "001");
    assert_eq!(report[0]["ok"], false);
    fs::remove_file(directory.join("cohort.csv")).unwrap();
    fs::remove_dir(directory).unwrap();
}
