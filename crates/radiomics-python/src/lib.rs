//! Minimal owned-vector boundary. Core algorithms never call Python.
use ov_radiomics_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
mod gateway;
mod pathology;
mod typed_arrays;

fn input_error(error: core::InputError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
mod _native {
    use super::*;

    #[pymodule_export]
    use crate::typed_arrays::{array_features_typed, OwnedArraySample};

    #[pyfunction]
    fn request_json(py: Python<'_>, payload: String) -> PyResult<String> {
        py.detach(|| crate::gateway::execute(&payload))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn pathology_json(py: Python<'_>, payload: String) -> PyResult<String> {
        py.detach(|| crate::pathology::execute(&payload))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn pathology_tiles(
        py: Python<'_>,
        payload: String,
    ) -> PyResult<crate::pathology::PatchIterator> {
        py.detach(|| crate::pathology::tiles(&payload))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn pathology_overlay(
        py: Python<'_>,
        payload: String,
    ) -> PyResult<crate::pathology::OverlayIterator> {
        py.detach(|| crate::pathology::overlay(&payload))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn wsidata_snapshot(py: Python<'_>, payload: String) -> PyResult<String> {
        py.detach(|| {
            let input: serde_json::Value = serde_json::from_str(&payload)?;
            Ok(serde_json::to_string(&core::wsidata_contract::snapshot(
                &input,
            )?)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn from_wsidata(py: Python<'_>, payload: String, buffer: String) -> PyResult<String> {
        py.detach(|| {
            let input: core::wsidata_contract::WsiInput = serde_json::from_str(&payload)?;
            let buffer: core::contracts::NumericBuffer = serde_json::from_str(&buffer)?;
            Ok(serde_json::to_string(&core::wsidata_contract::adapt(
                &input, &buffer,
            )?)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn validate_tile_table(py: Python<'_>, table: String) -> PyResult<String> {
        py.detach(|| {
            let table: core::contracts::TileTable = serde_json::from_str(&table)?;
            table.validate()?;
            Ok(serde_json::to_string(&table)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn align_tile_embeddings(
        py: Python<'_>,
        table: String,
        ids: Vec<String>,
        buffer: String,
        feature_names: Vec<String>,
    ) -> PyResult<String> {
        py.detach(|| {
            let table: core::contracts::TileTable = serde_json::from_str(&table)?;
            let buffer: core::contracts::NumericBuffer = serde_json::from_str(&buffer)?;
            let aligned =
                core::pathology_aggregate::align_embeddings(&table, &ids, &buffer, &feature_names)?;
            Ok(serde_json::to_string(&aligned)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    #[pyo3(signature=(observations, task, classes, outer_k, inner_k, seed, held_out_centers=None))]
    #[allow(clippy::too_many_arguments)] // Preserve the existing positional Python boundary.
    fn create_split_plan(
        py: Python<'_>,
        observations: String,
        task: String,
        classes: Vec<String>,
        outer_k: usize,
        inner_k: usize,
        seed: u64,
        held_out_centers: Option<Vec<String>>,
    ) -> PyResult<String> {
        py.detach(|| {
            let rows: Vec<core::split::SplitObservation> = serde_json::from_str(&observations)?;
            let rows = match held_out_centers {
                Some(centers) => {
                    core::split::center_holdout(&rows, &centers.into_iter().collect())?
                }
                None => rows,
            };
            let task: core::contracts::Task =
                serde_json::from_value(serde_json::Value::String(task))?;
            let plan =
                core::split::SplitPlan::from_cohorts(&rows, task, classes, outer_k, inner_k, seed)?;
            Ok(serde_json::to_string(&plan)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn validate_split_plan(py: Python<'_>, plan: String) -> PyResult<String> {
        py.detach(|| {
            let plan: core::split::SplitPlan = serde_json::from_str(&plan)?;
            plan.fingerprint()
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn validate_split_observations(
        py: Python<'_>,
        plan: String,
        observations: String,
    ) -> PyResult<()> {
        py.detach(|| {
            let plan: core::split::SplitPlan = serde_json::from_str(&plan)?;
            let rows: Vec<core::split::SplitObservation> = serde_json::from_str(&observations)?;
            plan.validate_observations(&rows)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn assert_shared_split_plan(py: Python<'_>, first: String, second: String) -> PyResult<()> {
        py.detach(|| {
            let first: core::split::SplitPlan = serde_json::from_str(&first)?;
            let second: core::split::SplitPlan = serde_json::from_str(&second)?;
            first.assert_shared(&second)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pymodule_export]
    const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    const IMPLEMENTATION_SHA256: &str = core::workflow::implementation_sha256();

    #[pyfunction]
    fn read_cohort(py: Python<'_>, path: String) -> PyResult<String> {
        py.detach(|| {
            let samples = core::workflow::read_cohort(path)?;
            Ok(serde_json::to_string(&samples)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    #[pyo3(signature=(samples, config, n_jobs=1, cache=None, threads=1))]
    fn extract_images(
        py: Python<'_>,
        samples: String,
        config: String,
        n_jobs: usize,
        cache: Option<String>,
        threads: usize,
    ) -> PyResult<String> {
        py.detach(|| {
            let samples: Vec<core::workflow::CohortSample> = serde_json::from_str(&samples)?;
            let config: core::workflow::ExtractConfig = serde_json::from_str(&config)?;
            let batch = core::workflow::extract_batch_with_threads(
                &samples,
                &config,
                n_jobs,
                cache.as_deref().map(std::path::Path::new),
                threads,
            )?;
            Ok(serde_json::to_string(&batch)?)
        })
        .map_err(|e: core::errors::RadiomicsError| PyValueError::new_err(e.to_string()))
    }

    #[pyfunction]
    fn auc(py: Python<'_>, y: Vec<u8>, scores: Vec<f64>) -> PyResult<f64> {
        py.detach(|| core::auc(&y, &scores)).map_err(input_error)
    }

    #[pyfunction]
    fn binary_summary(
        py: Python<'_>,
        y: Vec<u8>,
        scores: Vec<f64>,
        threshold: f64,
    ) -> PyResult<(f64, u64, u64, u64, u64, f64)> {
        let s = py
            .detach(|| core::binary_summary(&y, &scores, threshold))
            .map_err(input_error)?;
        Ok((s.auc, s.tn, s.fp, s.fn_count, s.tp, s.brier))
    }

    #[pyfunction]
    fn delong_summary(py: Python<'_>, y: Vec<u8>, scores: Vec<f64>) -> PyResult<(f64, f64)> {
        py.detach(|| core::delong_summary(&y, &scores))
            .map_err(input_error)
    }

    #[pyfunction]
    fn paired_delong_summary(
        py: Python<'_>,
        y: Vec<u8>,
        a: Vec<f64>,
        b: Vec<f64>,
    ) -> PyResult<(f64, f64, f64)> {
        py.detach(|| core::paired_delong_summary(&y, &a, &b))
            .map_err(input_error)
    }

    #[pyfunction]
    fn net_benefit(
        py: Python<'_>,
        y: Vec<u8>,
        scores: Vec<f64>,
        thresholds: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| core::net_benefit(&y, &scores, &thresholds))
            .map_err(input_error)
    }

    #[pyfunction]
    fn hl_statistic(
        py: Python<'_>,
        counts: Vec<u64>,
        observed: Vec<f64>,
        expected: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| core::hl_statistic(&counts, &observed, &expected))
            .map_err(input_error)
    }
}
