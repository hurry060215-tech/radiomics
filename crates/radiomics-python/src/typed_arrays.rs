//! Typed, explicitly owned array boundary. The feature formulas remain in core.
use core::contracts::{ExtractionConfig, Geometry, MaskVolume, SampleMetadata};
use core::features::{FeatureFamily, ImageView, ALL_FAMILIES};
use numpy::{ndarray::ArrayView3, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods};
use ov_radiomics_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    metadata: SampleMetadata,
    image_geometry: Geometry,
    mask_geometry: Geometry,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    config: ExtractionConfig,
    roi_label: u32,
    families: Option<Vec<FeatureFamily>>,
    #[serde(default)]
    intensity_mask: Option<MaskVolume>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedRequest {
    image_geometry: Geometry,
    mask_geometry: Geometry,
    #[serde(flatten)]
    parameters: Parameters,
}

#[derive(Serialize)]
struct ImageSnapshot<'a> {
    geometry: &'a Geometry,
    #[serde(with = "core::statistics::float_vec_state")]
    values: &'a [f64],
}

fn check_shape(shape: &[usize], geometry: &Geometry) -> PyResult<()> {
    geometry.validate().map_err(error)?;
    if shape != [geometry.shape[2], geometry.shape[1], geometry.shape[0]] {
        return Err(error(
            "Typed array shape differs from the declared geometry.",
        ));
    }
    Ok(())
}

fn encoded_features(
    image: ImageView<'_>,
    mask: &MaskVolume,
    parameters: Parameters,
) -> PyResult<String> {
    let result = core::features::extract_families_view(
        image,
        mask,
        parameters.intensity_mask.as_ref(),
        parameters.roi_label,
        &parameters.config,
        parameters.families.as_deref().unwrap_or(&ALL_FAMILIES),
    )
    .map_err(error)?;
    let result: BTreeMap<_, _> = result
        .into_iter()
        .map(|(name, value)| {
            let value = if value.is_nan() {
                serde_json::Value::String("NaN".into())
            } else if value == f64::INFINITY {
                serde_json::Value::String("Inf".into())
            } else if value == f64::NEG_INFINITY {
                serde_json::Value::String("-Inf".into())
            } else {
                serde_json::json!(value)
            };
            (name, value)
        })
        .collect();
    serde_json::to_string(&result).map_err(error)
}

/// Private standalone typed entry: incompatible dtype is rejected by PyO3 and
/// non-C layout by Rust. The Python caller can then choose the original JSON path.
#[pyfunction]
pub fn array_features_typed(
    py: Python<'_>,
    image_buffer: PyReadonlyArray3<'_, f64>,
    mask_buffer: PyReadonlyArray3<'_, u32>,
    config_json: String,
) -> PyResult<String> {
    if !image_buffer.is_c_contiguous() || !mask_buffer.is_c_contiguous() {
        return Err(error("Typed arrays require C-contiguous layout."));
    }
    let request: TypedRequest = serde_json::from_str(&config_json).map_err(error)?;
    check_shape(image_buffer.shape(), &request.image_geometry)?;
    check_shape(mask_buffer.shape(), &request.mask_geometry)?;
    // Python can mutate its ndarray after we release the GIL. Take the explicit
    // owned snapshot while holding it, as the public sample constructor does.
    let values: Arc<[f64]> = Arc::from(image_buffer.as_slice().map_err(error)?);
    let mask = MaskVolume {
        geometry: request.mask_geometry,
        labels: mask_buffer.as_slice().map_err(error)?.to_vec(),
    };
    drop(image_buffer);
    drop(mask_buffer);
    py.detach(|| {
        encoded_features(
            ImageView {
                geometry: &request.image_geometry,
                values: &values,
            },
            &mask,
            request.parameters,
        )
    })
}

/// One owned immutable normalization at ImageSample construction. Cloned Python
/// views retain this object, so the Arc storage is valid through every extraction.
#[pyclass(frozen, module = "ov_radiomics._native")]
pub struct OwnedArraySample {
    metadata: SampleMetadata,
    geometry: Geometry,
    values: Arc<[f64]>,
    mask: Arc<MaskVolume>,
}

#[pymethods]
impl OwnedArraySample {
    #[new]
    fn new(
        image_buffer: PyReadonlyArray3<'_, f64>,
        mask_buffer: PyReadonlyArray3<'_, u32>,
        header_json: String,
    ) -> PyResult<Self> {
        if !image_buffer.is_c_contiguous() || !mask_buffer.is_c_contiguous() {
            return Err(error("Typed arrays require C-contiguous layout."));
        }
        let header: Header = serde_json::from_str(&header_json).map_err(error)?;
        header.metadata.validate().map_err(error)?;
        check_shape(image_buffer.shape(), &header.image_geometry)?;
        check_shape(mask_buffer.shape(), &header.mask_geometry)?;
        header
            .image_geometry
            .aligned_with(&header.mask_geometry)
            .map_err(error)?;
        Ok(Self {
            metadata: header.metadata,
            geometry: header.image_geometry,
            values: Arc::from(image_buffer.as_slice().map_err(error)?),
            mask: Arc::new(MaskVolume {
                geometry: header.mask_geometry,
                labels: mask_buffer.as_slice().map_err(error)?.to_vec(),
            }),
        })
    }

    fn extract(&self, py: Python<'_>, config_json: String) -> PyResult<String> {
        let parameters: Parameters = serde_json::from_str(&config_json).map_err(error)?;
        py.detach(|| {
            encoded_features(
                ImageView {
                    geometry: &self.geometry,
                    values: &self.values,
                },
                &self.mask,
                parameters,
            )
        })
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.metadata).map_err(error)
    }

    fn image_geometry_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.geometry).map_err(error)
    }

    fn mask_geometry_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.mask.geometry).map_err(error)
    }

    fn image_json(&self) -> PyResult<String> {
        serde_json::to_string(&ImageSnapshot {
            geometry: &self.geometry,
            values: &self.values,
        })
        .map_err(error)
    }

    fn mask_json(&self) -> PyResult<String> {
        serde_json::to_string(&*self.mask).map_err(error)
    }

    fn image_array<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let owned = this.borrow();
        let shape = owned.geometry.shape;
        let view = ArrayView3::from_shape((shape[2], shape[1], shape[0]), &owned.values)
            .map_err(error)?;
        // SAFETY: the immutable Arc cannot reallocate or mutate, and NumPy keeps
        // `this` alive as its base. The returned array is made read-only before
        // it is exposed to Python; it does not own this Rust allocation.
        let array = unsafe { PyArray3::borrow_from_array(&view, this.clone().into_any()) };
        array.call_method1("setflags", (false,))?;
        Ok(array)
    }

    fn mask_array<'py>(this: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray3<u32>>> {
        let owned = this.borrow();
        let shape = owned.mask.geometry.shape;
        let view = ArrayView3::from_shape((shape[2], shape[1], shape[0]), &owned.mask.labels[..])
            .map_err(error)?;
        // SAFETY: as above; the mask Vec lives behind an immutable retained Arc.
        let array = unsafe { PyArray3::borrow_from_array(&view, this.clone().into_any()) };
        array.call_method1("setflags", (false,))?;
        Ok(array)
    }
}
