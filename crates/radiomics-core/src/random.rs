//! Shared, versioned seed derivation for new native components.
//! Legacy bootstrap compatibility uses its separately frozen index sequences.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

pub const RNG_SCHEME: &str = "sha256-chacha8-rand0.8.5-rand_chacha0.3.1-v1";
pub fn derive_seed(seed: u64, namespace: &str, identifier: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(seed.to_le_bytes());
    digest.update((namespace.len() as u64).to_le_bytes());
    digest.update(namespace.as_bytes());
    digest.update(identifier);
    digest.finalize().into()
}
pub fn component_rng(seed: u64, namespace: &str, identifier: &[u8]) -> ChaCha8Rng {
    ChaCha8Rng::from_seed(derive_seed(seed, namespace, identifier))
}
