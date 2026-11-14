use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn sources(path: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(path).expect("Read core source directory") {
        let entry = entry.expect("Read source entry");
        let path = entry.path();
        if path.is_dir() {
            sources(&path, output)
        } else if path.extension().is_some_and(|e| e == "rs") {
            output.push(path);
        }
    }
}

// Compile-time test-data path only, without an external environment override.
// The original digest algorithm is unchanged; build.rs remains one of its inputs.
fn golden_directory(root: &Path) -> PathBuf {
    for ancestor in root.ancestors() {
        let manifest = ancestor.join("Cargo.toml");
        let workspace = fs::read_to_string(&manifest).is_ok_and(|text| {
            text.lines()
                .any(|line| line.split('#').next().unwrap_or("").trim() == "[workspace]")
        });
        let golden = ancestor.join("golden");
        if workspace && golden.is_dir() {
            return golden.canonicalize().unwrap_or(golden);
        }
    }
    let local = root.join("package-assets/golden");
    local.canonicalize().unwrap_or(local)
}
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let golden = golden_directory(&root);
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
    let mut files = vec![root.join("Cargo.toml"), root.join("build.rs")];
    sources(&root.join("src"), &mut files);
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        let bytes = fs::read(&path).expect("Read implementation source");
        // Git text checkouts differ in line endings, not algorithm identity.
        digest.update(
            String::from_utf8(bytes)
                .expect("UTF-8 source")
                .replace("\r\n", "\n")
                .as_bytes(),
        );
    }
    if let Some(lock) = root
        .ancestors()
        .map(|p| p.join("Cargo.lock"))
        .find(|p| p.is_file())
    {
        println!("cargo:rerun-if-changed={}", lock.display());
        digest.update(fs::read(lock).expect("Read dependency lock"));
    }
    println!(
        "cargo:rustc-env=RAD_IMPLEMENTATION_SHA256={:x}",
        digest.finalize()
    );
}
