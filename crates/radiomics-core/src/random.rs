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

pub const LEGACY_BOOTSTRAP_SCHEME: &str = "numpy-randomstate-mt19937-rk_interval-rand_mt4.2.2-v1";
pub struct LegacyBinaryBootstrap {
    rng: rand_mt::Mt,
    positive: Vec<usize>,
    negative: Vec<usize>,
    remaining: usize,
}
fn interval(rng: &mut rand_mt::Mt, maximum: u32) -> usize {
    if maximum == 0 {
        return 0;
    }
    let mut mask = maximum;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    loop {
        let value = rng.next_u32() & mask;
        if value <= maximum {
            return value as usize;
        }
    }
}
impl Iterator for LegacyBinaryBootstrap {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let mut indices = Vec::with_capacity(self.positive.len() + self.negative.len());
        for _ in 0..self.positive.len() {
            indices.push(self.positive[interval(&mut self.rng, (self.positive.len() - 1) as u32)]);
        }
        for _ in 0..self.negative.len() {
            indices.push(self.negative[interval(&mut self.rng, (self.negative.len() - 1) as u32)]);
        }
        Some(indices)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for LegacyBinaryBootstrap {}
/// Streaming NumPy RandomState-compatible stratified binary draws: positives first.
pub fn legacy_binary_bootstrap(
    y: &[u8],
    replicates: usize,
    seed: u32,
) -> crate::errors::RadResult<LegacyBinaryBootstrap> {
    use crate::errors::{ErrorCode, RadiomicsError};
    let error = || {
        RadiomicsError::new(ErrorCode::InvalidInput,"Legacy binary bootstrap needs both classes, positive replicate count, and group sizes fitting uint32.")
    };
    if replicates == 0 || y.iter().any(|&v| v > 1) {
        return Err(error());
    }
    let positive: Vec<_> = y
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 1)
        .map(|(i, _)| i)
        .collect();
    let negative: Vec<_> = y
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 0)
        .map(|(i, _)| i)
        .collect();
    if [&positive, &negative]
        .iter()
        .any(|v| v.is_empty() || u32::try_from(v.len() - 1).is_err())
    {
        return Err(error());
    }
    Ok(LegacyBinaryBootstrap {
        rng: rand_mt::Mt::new(seed),
        positive,
        negative,
        remaining: replicates,
    })
}
pub fn legacy_bootstrap_indices(
    y: &[u8],
    replicates: usize,
    seed: u32,
) -> crate::errors::RadResult<Vec<Vec<usize>>> {
    Ok(legacy_binary_bootstrap(y, replicates, seed)?.collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_draws_match_both_independent_frozen_numpy_sequences() {
        let baseline: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/reference.json"
        )))
        .unwrap();
        let reference = &baseline["baseline_metrics"];
        let y: Vec<u8> = serde_json::from_value(reference["inputs"]["y"].clone()).unwrap();
        let expected: Vec<Vec<usize>> =
            serde_json::from_value(reference["bootstrap_indices"].clone()).unwrap();
        assert_eq!(
            legacy_bootstrap_indices(&y, expected.len(), 42).unwrap(),
            expected
        );
        let r: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("RAD_GOLDEN_DIR"),
            "/python_golden_v1/evaluation_reference.json"
        )))
        .unwrap();
        let b = &r["patient_binary_bootstrap"];
        let y: Vec<u8> = serde_json::from_value(r["cases"]["binary"]["y"].clone()).unwrap();
        let expected: Vec<Vec<usize>> = serde_json::from_value(b["indices"].clone()).unwrap();
        let seed = b["seed"].as_u64().unwrap() as u32;
        assert_eq!(
            legacy_binary_bootstrap(&y, expected.len(), seed)
                .unwrap()
                .collect::<Vec<_>>(),
            expected
        );
    }
}
