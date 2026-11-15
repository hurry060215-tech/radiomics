//! Patient-grouped splits with explicit training provenance for every nested fold.
use crate::contracts::{unique_nonempty, Task};
use crate::errors::{ErrorCode, RadResult, RadiomicsError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn error(code: ErrorCode, message: &str) -> RadiomicsError {
    RadiomicsError::new(code, message)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitObservation {
    pub sample_id: String,
    pub patient_id: String,
    pub cohort: String,
    /// Required for training classification rows only. External labels never affect splits.
    pub class_label: Option<String>,
    pub center: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldPartition {
    pub training_patients: BTreeSet<String>,
    pub validation_patients: BTreeSet<String>,
}
impl FoldPartition {
    pub fn validate(&self, population: &BTreeSet<String>) -> RadResult<()> {
        if self.training_patients.is_empty()
            || self.validation_patients.is_empty()
            || !self
                .training_patients
                .is_disjoint(&self.validation_patients)
            || self
                .training_patients
                .union(&self.validation_patients)
                .cloned()
                .collect::<BTreeSet<_>>()
                != *population
        {
            return Err(error(
                ErrorCode::Leakage,
                "Fold must partition its full population into nonempty disjoint patient sets.",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedFold {
    pub outer: FoldPartition,
    pub inner: Vec<FoldPartition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitPlan {
    pub schema_version: u32,
    pub task: Task,
    pub seed: u64,
    /// BTree containers make fingerprints independent of incoming row order.
    pub sample_patients: BTreeMap<String, String>,
    pub sample_classes: BTreeMap<String, String>,
    pub sample_centers: BTreeMap<String, Option<String>>,
    pub cohorts: BTreeMap<String, BTreeSet<String>>,
    pub classes: Vec<String>,
    pub patient_classes: BTreeMap<String, BTreeSet<String>>,
    pub outer_folds: Vec<NestedFold>,
    pub final_inner_folds: Vec<FoldPartition>,
    pub rng_scheme: String,
}

fn ordering(seed: u64, domain: &str, patient: &str) -> [u8; 32] {
    crate::random::derive_seed(seed, domain, patient.as_bytes())
}

fn check_classes(
    part: &FoldPartition,
    labels: &BTreeMap<String, BTreeSet<String>>,
    classes: &[String],
) -> RadResult<()> {
    for patients in [&part.training_patients, &part.validation_patients] {
        for class in classes {
            if !patients
                .iter()
                .any(|p| labels.get(p).is_some_and(|c| c.contains(class)))
            {
                return Err(error(ErrorCode::InvalidInput,"A training or validation fold is missing a class; change the explicit split or fold count."));
            }
        }
    }
    Ok(())
}

fn folds(
    population: &BTreeSet<String>,
    labels: &BTreeMap<String, BTreeSet<String>>,
    classes: &[String],
    k: usize,
    seed: u64,
    domain: &str,
) -> RadResult<Vec<FoldPartition>> {
    if k < 2 || population.len() < k {
        return Err(error(
            ErrorCode::InvalidInput,
            "CV needs at least two folds and at least one patient per fold.",
        ));
    }
    let totals: Vec<usize> = classes
        .iter()
        .map(|c| {
            population
                .iter()
                .filter(|p| labels.get(*p).is_some_and(|v| v.contains(c)))
                .count()
        })
        .collect();
    if totals.iter().any(|&n| n < k) {
        return Err(error(
            ErrorCode::InvalidInput,
            "Each class needs at least k distinct training patients, including in inner CV.",
        ));
    }
    let mut patients: Vec<_> = population.iter().cloned().collect();
    // Rare-class patients first; hash is a versioned deterministic seed-dependent tie breaker.
    patients.sort_by_key(|p| {
        let rarity = classes
            .iter()
            .enumerate()
            .filter(|(_, c)| labels.get(p).is_some_and(|v| v.contains(*c)))
            .map(|(j, _)| totals[j])
            .min()
            .unwrap_or(usize::MAX);
        (rarity, ordering(seed, domain, p))
    });
    let mut buckets = vec![BTreeSet::new(); k];
    let mut counts = vec![vec![0usize; classes.len()]; k];
    for patient in patients {
        let has: Vec<_> = classes
            .iter()
            .map(|c| labels.get(&patient).is_some_and(|s| s.contains(c)))
            .collect();
        let mut best = 0;
        let mut best_score = f64::INFINITY;
        for candidate in 0..k {
            // Minimize the total normalized imbalance after adding this patient.
            let mut score = 0.0;
            for (class, &total) in totals.iter().enumerate() {
                let expected = total as f64 / k as f64;
                for (fold, counter) in counts.iter().enumerate() {
                    let value = counter[class] as f64 + f64::from(candidate == fold && has[class]);
                    score += ((value - expected) / total as f64).powi(2);
                }
            }
            // With equal class balance, favor the smallest bucket, then stable fold index.
            let tie = score.total_cmp(&best_score).is_eq();
            if score < best_score || (tie && buckets[candidate].len() < buckets[best].len()) {
                best = candidate;
                best_score = score;
            }
        }
        for (j, &present) in has.iter().enumerate() {
            counts[best][j] += usize::from(present);
        }
        buckets[best].insert(patient);
    }
    let mut result = Vec::new();
    for validation in buckets {
        let partition = FoldPartition {
            training_patients: population.difference(&validation).cloned().collect(),
            validation_patients: validation,
        };
        partition.validate(population)?;
        check_classes(&partition, labels, classes)?;
        result.push(partition);
    }
    Ok(result)
}

fn validate_collection(
    parts: &[FoldPartition],
    population: &BTreeSet<String>,
    labels: &BTreeMap<String, BTreeSet<String>>,
    classes: &[String],
) -> RadResult<()> {
    if parts.len() < 2 {
        return Err(error(
            ErrorCode::InvalidInput,
            "CV requires at least two folds.",
        ));
    }
    let mut seen = BTreeSet::new();
    for part in parts {
        part.validate(population)?;
        check_classes(part, labels, classes)?;
        for patient in &part.validation_patients {
            if !seen.insert(patient.clone()) {
                return Err(error(
                    ErrorCode::Leakage,
                    "A patient is validated in more than one CV fold.",
                ));
            }
        }
    }
    if seen != *population {
        return Err(error(
            ErrorCode::Alignment,
            "CV does not validate every training patient exactly once.",
        ));
    }
    Ok(())
}

impl SplitPlan {
    pub fn from_cohorts(
        observations: &[SplitObservation],
        task: Task,
        classes: Vec<String>,
        outer_k: usize,
        inner_k: usize,
        seed: u64,
    ) -> RadResult<Self> {
        unique_nonempty(&classes, "Class labels")?;
        if (task == Task::Regression && !classes.is_empty())
            || (task == Task::Binary && classes.len() != 2)
            || (task == Task::Multiclass && classes.len() < 3)
        {
            return Err(error(
                ErrorCode::InvalidInput,
                "Class order does not match the requested task.",
            ));
        }
        unique_nonempty(
            &observations
                .iter()
                .map(|o| o.sample_id.clone())
                .collect::<Vec<_>>(),
            "Sample IDs",
        )?;
        let mut result = Self {
            schema_version: 1,
            task,
            seed,
            sample_patients: BTreeMap::new(),
            sample_classes: BTreeMap::new(),
            sample_centers: BTreeMap::new(),
            cohorts: BTreeMap::new(),
            classes,
            patient_classes: BTreeMap::new(),
            outer_folds: Vec::new(),
            final_inner_folds: Vec::new(),
            rng_scheme: "sha256-group-stratification-v1".into(),
        };
        let mut membership = BTreeMap::new();
        for row in observations {
            if row.patient_id.trim().is_empty() || row.cohort.trim().is_empty() {
                return Err(error(
                    ErrorCode::InvalidInput,
                    "Patient IDs and cohort names must be nonempty.",
                ));
            }
            if membership
                .insert(row.patient_id.clone(), row.cohort.clone())
                .is_some_and(|c| c != row.cohort)
            {
                return Err(error(
                    ErrorCode::Leakage,
                    "A patient occurs in multiple cohorts.",
                ));
            }
            result
                .sample_patients
                .insert(row.sample_id.clone(), row.patient_id.clone());
            if row.center.as_ref().is_some_and(|s| s.trim().is_empty()) {
                return Err(error(
                    ErrorCode::InvalidInput,
                    "Center IDs must be nonempty when supplied.",
                ));
            }
            result
                .sample_centers
                .insert(row.sample_id.clone(), row.center.clone());
            result
                .cohorts
                .entry(row.cohort.clone())
                .or_default()
                .insert(row.patient_id.clone());
            if row.cohort == "train" && task != Task::Regression {
                let class = row
                    .class_label
                    .as_ref()
                    .filter(|s| result.classes.contains(s))
                    .ok_or_else(|| {
                        error(
                            ErrorCode::InvalidInput,
                            "Training rows require a known class label.",
                        )
                    })?;
                result
                    .patient_classes
                    .entry(row.patient_id.clone())
                    .or_default()
                    .insert(class.clone());
                result
                    .sample_classes
                    .insert(row.sample_id.clone(), class.clone());
            }
        }
        let training = result
            .cohorts
            .get("train")
            .ok_or_else(|| error(ErrorCode::InvalidInput, "Missing training cohort."))?;
        for (i, outer) in folds(
            training,
            &result.patient_classes,
            &result.classes,
            outer_k,
            seed,
            "outer",
        )?
        .into_iter()
        .enumerate()
        {
            let inner = folds(
                &outer.training_patients,
                &result.patient_classes,
                &result.classes,
                inner_k,
                seed,
                &format!("outer-{i}-inner"),
            )?;
            result.outer_folds.push(NestedFold { outer, inner });
        }
        result.final_inner_folds = folds(
            training,
            &result.patient_classes,
            &result.classes,
            inner_k,
            seed,
            "final-inner",
        )?;
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> RadResult<()> {
        if self.schema_version != 1
            || !matches!(
                self.rng_scheme.as_str(),
                "sha256-group-stratification-v1" | "explicit-patient-partitions-v1"
            )
        {
            return Err(error(
                ErrorCode::Schema,
                "Unknown split schema or randomization scheme.",
            ));
        }
        unique_nonempty(&self.classes, "Class labels")?;
        if (self.task == Task::Regression && !self.classes.is_empty())
            || (self.task == Task::Binary && self.classes.len() != 2)
            || (self.task == Task::Multiclass && self.classes.len() < 3)
        {
            return Err(error(
                ErrorCode::Schema,
                "Split class order does not match task.",
            ));
        }
        let training = self
            .cohorts
            .get("train")
            .ok_or_else(|| error(ErrorCode::InvalidInput, "Missing training cohort."))?;
        let mut all = BTreeSet::new();
        for (cohort, patients) in &self.cohorts {
            if cohort.trim().is_empty() || patients.is_empty() {
                return Err(error(
                    ErrorCode::InvalidInput,
                    "Empty cohort name or patient set.",
                ));
            }
            for patient in patients {
                if patient.trim().is_empty() || !all.insert(patient.clone()) {
                    return Err(error(
                        ErrorCode::Leakage,
                        "Empty patient ID or patient shared across cohorts.",
                    ));
                }
            }
        }
        if self.sample_patients.is_empty()
            || self.sample_patients.keys().any(|s| s.trim().is_empty())
            || self
                .sample_patients
                .values()
                .cloned()
                .collect::<BTreeSet<_>>()
                != all
        {
            return Err(error(
                ErrorCode::Alignment,
                "Sample-patient map differs from the cohort population.",
            ));
        }
        if self.sample_centers.keys().collect::<BTreeSet<_>>()
            != self.sample_patients.keys().collect::<BTreeSet<_>>()
            || self
                .sample_centers
                .values()
                .flatten()
                .any(|s| s.trim().is_empty())
        {
            return Err(error(
                ErrorCode::Alignment,
                "Sample center provenance is incomplete or invalid.",
            ));
        }
        if self.task == Task::Regression {
            if !self.patient_classes.is_empty() || !self.sample_classes.is_empty() {
                return Err(error(
                    ErrorCode::Schema,
                    "Regression splits cannot contain classification targets.",
                ));
            }
        } else if self
            .patient_classes
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != *training
            || self
                .patient_classes
                .values()
                .any(|s| s.is_empty() || s.iter().any(|c| !self.classes.contains(c)))
        {
            return Err(error(
                ErrorCode::Alignment,
                "Training patient classes are incomplete or unknown.",
            ));
        }
        let mut reconstructed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        if self.task != Task::Regression {
            let expected: BTreeSet<_> = self
                .sample_patients
                .iter()
                .filter(|(_, p)| training.contains(*p))
                .map(|(s, _)| s.clone())
                .collect();
            if self.sample_classes.keys().cloned().collect::<BTreeSet<_>>() != expected {
                return Err(error(
                    ErrorCode::Alignment,
                    "Training sample class map is incomplete.",
                ));
            }
            for (sample, class) in &self.sample_classes {
                reconstructed
                    .entry(self.sample_patients[sample].clone())
                    .or_default()
                    .insert(class.clone());
            }
            if reconstructed != self.patient_classes {
                return Err(error(
                    ErrorCode::Alignment,
                    "Sample and patient class maps disagree.",
                ));
            }
        }
        let outer: Vec<_> = self.outer_folds.iter().map(|f| f.outer.clone()).collect();
        validate_collection(&outer, training, &self.patient_classes, &self.classes)?;
        for f in &self.outer_folds {
            validate_collection(
                &f.inner,
                &f.outer.training_patients,
                &self.patient_classes,
                &self.classes,
            )?;
        }
        validate_collection(
            &self.final_inner_folds,
            training,
            &self.patient_classes,
            &self.classes,
        )
    }
    pub fn fingerprint(&self) -> RadResult<String> {
        self.validate()?;
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
    pub fn assert_shared(&self, other: &Self) -> RadResult<()> {
        self.validate()?;
        other.validate()?;
        if self != other {
            return Err(error(ErrorCode::Alignment,"Fusion requires the same SplitPlan, including actual training and validation patient sets."));
        }
        Ok(())
    }
    /// Validate an incoming component by identity, with labels checked only for training rows.
    pub fn validate_observations(&self, rows: &[SplitObservation]) -> RadResult<()> {
        self.validate()?;
        unique_nonempty(
            &rows.iter().map(|o| o.sample_id.clone()).collect::<Vec<_>>(),
            "Sample IDs",
        )?;
        if rows.len() != self.sample_patients.len() {
            return Err(error(
                ErrorCode::Alignment,
                "SplitPlan sample population differs.",
            ));
        }
        for row in rows {
            if self.sample_patients.get(&row.sample_id) != Some(&row.patient_id)
                || !self
                    .cohorts
                    .get(&row.cohort)
                    .is_some_and(|s| s.contains(&row.patient_id))
            {
                return Err(error(
                    ErrorCode::Alignment,
                    "Sample, patient or cohort membership differs from SplitPlan.",
                ));
            }
            if row.cohort == "train"
                && self.task != Task::Regression
                && !row
                    .class_label
                    .as_ref()
                    .is_some_and(|c| self.sample_classes.get(&row.sample_id) == Some(c))
            {
                return Err(error(
                    ErrorCode::Alignment,
                    "Training class differs from SplitPlan.",
                ));
            }
            if self.sample_centers.get(&row.sample_id) != Some(&row.center) {
                return Err(error(
                    ErrorCode::Alignment,
                    "Sample center differs from SplitPlan.",
                ));
            }
        }
        Ok(())
    }
}

/// Set one explicit center holdout without dividing any patient across centers/cohorts.
pub fn center_holdout(
    rows: &[SplitObservation],
    held_out: &BTreeSet<String>,
) -> RadResult<Vec<SplitObservation>> {
    if held_out.is_empty() {
        return Err(error(
            ErrorCode::InvalidInput,
            "Specify at least one held-out center.",
        ));
    }
    let mut centers = BTreeMap::new();
    let mut memberships = BTreeMap::new();
    let mut observed = BTreeSet::new();
    for row in rows {
        let center = row
            .center
            .as_ref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                error(
                    ErrorCode::InvalidInput,
                    "Center holdout requires complete center identifiers.",
                )
            })?;
        if matches!(row.cohort.as_str(), "train" | "test" | "unassigned") {
            observed.insert(center.clone());
        }
        if memberships
            .insert(&row.patient_id, &row.cohort)
            .is_some_and(|c| c != &row.cohort)
        {
            return Err(error(
                ErrorCode::Leakage,
                "A patient occurs in multiple cohorts before center holdout.",
            ));
        }
        if centers
            .insert(&row.patient_id, center)
            .is_some_and(|c| c != center)
        {
            return Err(error(
                ErrorCode::Leakage,
                "One patient occurs in multiple centers; assign that patient explicitly.",
            ));
        }
    }
    if !held_out.is_subset(&observed) || held_out == &observed {
        return Err(error(
            ErrorCode::InvalidInput,
            "Held-out centers must exist and leave a nonempty training population.",
        ));
    }
    Ok(rows
        .iter()
        .cloned()
        .map(|mut r| {
            if !matches!(r.cohort.as_str(), "train" | "test" | "unassigned") {
                return r;
            }
            r.cohort = if held_out.contains(r.center.as_ref().unwrap()) {
                "test"
            } else {
                "train"
            }
            .into();
            r
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows() -> Vec<SplitObservation> {
        (0..36)
            .flat_map(|i| {
                (0..2).map(move |lesion| SplitObservation {
                    sample_id: format!("P{i:02}-L{lesion}"),
                    patient_id: format!("P{i:02}"),
                    cohort: if i < 30 { "train" } else { "test" }.into(),
                    class_label: Some((i % 3).to_string()),
                    center: Some(if i < 30 { "A" } else { "B" }.into()),
                })
            })
            .collect()
    }
    fn plan(rows: &[SplitObservation]) -> SplitPlan {
        SplitPlan::from_cohorts(
            rows,
            Task::Multiclass,
            vec!["0".into(), "1".into(), "2".into()],
            3,
            2,
            42,
        )
        .unwrap()
    }
    #[test]
    fn repeated_patients_are_grouped_and_nested() {
        let p = plan(&rows());
        p.validate().unwrap();
        for fold in &p.outer_folds {
            for inner in &fold.inner {
                assert!(inner
                    .training_patients
                    .is_disjoint(&fold.outer.validation_patients));
                assert!(inner
                    .validation_patients
                    .is_subset(&fold.outer.training_patients));
            }
        }
    }
    #[test]
    fn split_is_row_order_and_external_target_independent() {
        let original = rows();
        let p = plan(&original);
        let mut changed = original.clone();
        changed.reverse();
        for row in &mut changed {
            if row.cohort != "train" {
                row.class_label = None;
            }
        }
        let q = plan(&changed);
        assert_eq!(p.fingerprint().unwrap(), q.fingerprint().unwrap());
        p.validate_observations(&changed).unwrap();
    }
    #[test]
    fn same_fold_numbers_do_not_prove_same_training_patients() {
        let p = plan(&rows());
        let mut q = p.clone();
        q.outer_folds.swap(0, 1);
        assert!(p.assert_shared(&q).is_err());
        let mut leak = p.clone();
        let patient = leak.outer_folds[0]
            .outer
            .validation_patients
            .first()
            .unwrap()
            .clone();
        leak.outer_folds[0].inner[0]
            .training_patients
            .insert(patient);
        assert!(leak.validate().is_err());
    }
    #[test]
    fn overlaps_missing_classes_and_unknown_schema_fail() {
        let mut r = rows();
        r.last_mut().unwrap().patient_id = "P00".into();
        assert!(SplitPlan::from_cohorts(
            &r,
            Task::Multiclass,
            vec!["0".into(), "1".into(), "2".into()],
            3,
            2,
            42
        )
        .is_err());
        let mut p = plan(&rows());
        p.schema_version = 2;
        assert!(p.validate().is_err());
        let mut p = plan(&rows());
        p.outer_folds[0].inner[0].validation_patients.clear();
        assert!(p.validate().is_err());
    }
    #[test]
    fn explicit_center_holdout() {
        let r = rows();
        assert_eq!(
            center_holdout(&r, &BTreeSet::from(["B".into()])).unwrap(),
            r
        );
    }
    #[test]
    fn center_holdout_preserves_external_cohorts() {
        let mut r = rows();
        r.push(SplitObservation {
            sample_id: "external-sample".into(),
            patient_id: "external-patient".into(),
            cohort: "external".into(),
            class_label: None,
            center: Some("A".into()),
        });
        let assigned = center_holdout(&r, &BTreeSet::from(["B".into()])).unwrap();
        assert_eq!(assigned.last().unwrap().cohort, "external");
    }
}
