//! Compile-time fixture discovery for workspace and standalone archive tests.
use std::{env, fs, path::PathBuf};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = root.ancestors().find_map(|ancestor| {
        let manifest = fs::read_to_string(ancestor.join("Cargo.toml")).ok()?;
        let declares_workspace = manifest
            .lines()
            .any(|line| line.split('#').next().unwrap_or("").trim() == "[workspace]");
        let golden = ancestor.join("golden");
        (declares_workspace && golden.is_dir()).then_some(golden)
    });
    let golden = workspace.unwrap_or_else(|| root.join("package-assets/golden"));
    let golden = golden.canonicalize().unwrap_or(golden);
    println!("cargo:rustc-env=RAD_GOLDEN_DIR={}", golden.display());
    for name in [
        "python_golden_v1/manifest.json",
        "ibsi_digital_v1/manifest.json",
    ] {
        let manifest = golden.join(name);
        if manifest.is_file() {
            println!("cargo:rerun-if-changed={}", manifest.display());
        }
    }
}
