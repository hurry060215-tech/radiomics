//! Preserve saved cohort identity at standalone prediction boundaries.
use crate::{
    errors::{ErrorCode, RadResult, RadiomicsError},
    split::SplitPlan,
};

pub(super) fn validate_cohort(plan: &SplitPlan, patient: &str, cohort: &str) -> RadResult<()> {
    let original = plan
        .cohorts
        .iter()
        .find_map(|(name, patients)| patients.contains(patient).then_some(name.as_str()));
    if original.is_some_and(|name| name != cohort)
        || original.is_none() && matches!(cohort, "train" | "test")
    {
        return Err(RadiomicsError::new(ErrorCode::Leakage,"Prediction patient cohort differs from the saved SplitPlan; new patients require an explicitly named external cohort."));
    }
    Ok(())
}
