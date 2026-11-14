//! Serialization/conversion boundary only. All numerical work calls the Rust core.
use core::{
    contracts::*,
    errors::{ErrorCode, RadResult, RadiomicsError},
    learning::*,
    models::{ModelPrediction, ModelTarget},
    preprocessing::*,
};
use ov_radiomics_core as core;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema_version: u32,
    request: Request,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedTraining {
    name: String,
    training: TrainingResult,
    model_name: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectorState {
    preprocessing: FittedPreprocessor,
    selection: FittedSelector,
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    ValidateImage {
        image: ImageVolume,
    },
    ValidateMask {
        mask: MaskVolume,
    },
    ValidateImageSample {
        sample: ImageSample,
    },
    ValidateFeatureTable {
        table: FeatureTable,
    },
    ValidatePredictionTable {
        table: PredictionTable,
    },
    Train {
        study: StudyData,
        plan: core::split::SplitPlan,
        config: TrainingConfig,
        n_jobs: usize,
    },
    PredictPipeline {
        model: Box<core::bundle::ModelBundle<FittedPipeline>>,
        study: StudyData,
    },
    ValidatePipeline {
        model: Box<core::bundle::ModelBundle<FittedPipeline>>,
    },
    SavePipeline {
        model: Box<core::bundle::ModelBundle<FittedPipeline>>,
        path: String,
        overwrite: bool,
    },
    PredictMatrix {
        model: Box<core::bundle::ModelBundle<FittedPipeline>>,
        features: TabularData,
    },
    FitFusion {
        inputs: Vec<NamedTraining>,
        config: FusionConfig,
    },
    PredictFusion {
        model: Box<core::bundle::ModelBundle<FittedFusion>>,
        studies: BTreeMap<String, StudyData>,
    },
    PredictFusionTables {
        model: Box<core::bundle::ModelBundle<FittedFusion>>,
        tables: BTreeMap<String, PredictionTable>,
    },
    ValidateFusion {
        model: Box<core::bundle::ModelBundle<FittedFusion>>,
    },
    SaveFusion {
        model: Box<core::bundle::ModelBundle<FittedFusion>>,
        path: String,
        overwrite: bool,
    },
    PreprocessFit {
        data: TabularData,
        config: PreprocessConfig,
    },
    PreprocessTransform {
        state: FittedPreprocessor,
        data: TabularData,
    },
    SelectorFit {
        data: TabularData,
        target: ModelTarget,
        preprocessing: PreprocessConfig,
        selection: SelectorConfig,
    },
    SelectorTransform {
        state: SelectorState,
        data: TabularData,
    },
    AggregatePatientPredictions {
        predictions: PredictionTable,
    },
    ClassificationMetrics {
        task: Task,
        target: ModelTarget,
        prediction: ModelPrediction,
        threshold: f64,
    },
    RegressionMetrics {
        y: Vec<f64>,
        prediction: Vec<f64>,
    },
    BinaryMetrics {
        y: Vec<u8>,
        scores: Vec<f64>,
        threshold: f64,
    },
    RocCurve {
        y: Vec<u8>,
        scores: Vec<f64>,
    },
    PrecisionRecallCurve {
        y: Vec<u8>,
        scores: Vec<f64>,
    },
    CalibrationCurve {
        y: Vec<u8>,
        scores: Vec<f64>,
        n_bins: usize,
        strategy: core::evaluation::CalibrationStrategy,
    },
    DelongInterval {
        y: Vec<u8>,
        scores: Vec<f64>,
        confidence: f64,
    },
    DelongTest {
        y: Vec<u8>,
        a: Vec<f64>,
        b: Vec<f64>,
    },
    DecisionCurve {
        y: Vec<u8>,
        scores: Vec<f64>,
        thresholds: Vec<f64>,
    },
    HosmerLemeshow {
        y: Vec<u8>,
        scores: Vec<f64>,
        n_bins: usize,
    },
    BootstrapBinary {
        y: Vec<u8>,
        scores: Vec<f64>,
        replicates: usize,
        seed: u32,
        confidence: f64,
    },
    BootstrapMulticlass {
        target: ModelTarget,
        prediction: ModelPrediction,
        replicates: usize,
        seed: u64,
        confidence: f64,
    },
    BootstrapRegression {
        target: ModelTarget,
        prediction: ModelPrediction,
        replicates: usize,
        seed: u64,
        confidence: f64,
    },
    ClinicalLogistic {
        data: TabularData,
        target: Vec<Option<u8>>,
        features: Vec<String>,
        reference: BTreeMap<String, String>,
        config: core::clinical_native::ClinicalConfig,
    },
    Baseline {
        data: TabularData,
        classes: Vec<String>,
        labels: Vec<Option<usize>>,
        features: Vec<String>,
        cohorts: Option<Vec<String>>,
        config: core::evaluation::baseline::BaselineConfig,
    },
    Icc {
        data: NumericBuffer,
        kind: core::reliability::IccType,
        confidence: f64,
    },
    ReadStudy {
        spec: StudyTableSpec,
        task: Task,
        base: String,
    },
    ReadFeatureTable {
        path: String,
        manifest: Option<String>,
    },
    WriteFeatureTable {
        table: FeatureTable,
        path: String,
        overwrite: bool,
        null_value: String,
    },
    ReadNifti {
        path: String,
    },
    ReadMask {
        path: String,
    },
    ImageResampleSpacing {
        image: ImageVolume,
        spacing_mm: [f64; 3],
        interpolation: core::imaging::Interpolation,
        outside: f64,
    },
    ImageResampleLike {
        image: ImageVolume,
        reference: Geometry,
        interpolation: core::imaging::Interpolation,
        outside: f64,
    },
    MaskResampleLike {
        mask: MaskVolume,
        reference: Geometry,
        outside: u32,
    },
    MaskPeritumoralRing {
        mask: MaskVolume,
        inner_mm: f64,
        outer_mm: f64,
        organ_mask: Option<MaskVolume>,
    },
    MaskSelectLabels {
        mask: MaskVolume,
        labels: Vec<u32>,
    },
    MaskLargestComponent {
        mask: MaskVolume,
        connectivity: core::imaging::Connectivity,
    },
    MaskDilateVoxels {
        mask: MaskVolume,
        iterations: usize,
        connectivity: core::imaging::Connectivity,
    },
    ImageClipRange {
        image: ImageVolume,
        lower: f64,
        upper: f64,
    },
    ImageClipPercentile {
        image: ImageVolume,
        lower: f64,
        upper: f64,
        statistical_mask: Option<MaskVolume>,
    },
    ImageNormalize {
        image: ImageVolume,
        method: core::imaging::Normalization,
        statistical_mask: Option<MaskVolume>,
    },
    MaskResegment {
        image: ImageVolume,
        mask: MaskVolume,
        lower: f64,
        upper: f64,
    },
    ArrayFeatures {
        image: ImageVolume,
        morphology_mask: MaskVolume,
        intensity_mask: Option<MaskVolume>,
        roi_label: u32,
        config: ExtractionConfig,
    },
}
fn encoded<T: Serialize>(value: &T) -> RadResult<String> {
    Ok(serde_json::to_string(value)?)
}
pub fn execute(payload: &str) -> RadResult<String> {
    let envelope: Envelope = serde_json::from_str(payload)?;
    if envelope.schema_version != 1 {
        return Err(RadiomicsError::new(
            ErrorCode::Schema,
            "Unknown native request schema version.",
        ));
    }
    match envelope.request {
        Request::ValidateImage { image } => {
            image.validate()?;
            encoded(&image)
        }
        Request::ValidateMask { mask } => {
            mask.validate()?;
            encoded(&mask)
        }
        Request::ValidateImageSample { sample } => {
            sample.validate()?;
            encoded(&sample)
        }
        Request::ValidateFeatureTable { table } => {
            table.validate()?;
            encoded(&table)
        }
        Request::ValidatePredictionTable { table } => {
            table.validate()?;
            encoded(&table)
        }
        Request::Train {
            study,
            plan,
            config,
            n_jobs,
        } => encoded(&core::learning::train(&study, &plan, &config, n_jobs)?),
        Request::PredictPipeline { model, study } => {
            encoded(&core::learning::predict(&model, &study)?)
        }
        Request::ValidatePipeline { model } => {
            model.validate()?;
            encoded(&model)
        }
        Request::SavePipeline {
            model,
            path,
            overwrite,
        } => {
            model.save(&path, overwrite)?;
            encoded(&path)
        }
        Request::PredictMatrix { model, features } => {
            model.validate()?;
            encoded(&model.fitted.predict(&features)?)
        }
        Request::FitFusion { inputs, config } => {
            let borrowed: Vec<_> = inputs
                .iter()
                .map(|i| FusionInput {
                    name: &i.name,
                    training: &i.training,
                    model_name: &i.model_name,
                })
                .collect();
            encoded(&core::learning::fit_fusion(&borrowed, &config)?)
        }
        Request::PredictFusion { model, studies } => {
            encoded(&core::learning::predict_fusion(&model, &studies)?)
        }
        Request::PredictFusionTables { model, tables } => {
            encoded(&core::learning::predict_fusion_tables(&model, &tables)?)
        }
        Request::ValidateFusion { model } => {
            model.validate()?;
            encoded(&model)
        }
        Request::SaveFusion {
            model,
            path,
            overwrite,
        } => {
            model.save(&path, overwrite)?;
            encoded(&path)
        }
        Request::PreprocessFit { data, config } => encoded(&fit_preprocessor(&data, &config)?),
        Request::PreprocessTransform { state, data } => encoded(&state.transform(&data)?),
        Request::SelectorFit {
            data,
            target,
            preprocessing,
            selection,
        } => {
            let preprocessing = fit_preprocessor(&data, &preprocessing)?;
            let x = preprocessing.transform(&data)?;
            let selection = fit_selector(
                &x,
                &preprocessing.output_names,
                &target,
                &selection,
                Some(&preprocessing.unscaled_variances()?),
            )?;
            encoded(&SelectorState {
                preprocessing,
                selection,
            })
        }
        Request::SelectorTransform { state, data } => {
            state.preprocessing.validate()?;
            state.selection.validate()?;
            if state.selection.feature_names != state.preprocessing.output_names {
                return Err(RadiomicsError::new(
                    ErrorCode::Alignment,
                    "Saved selector input schema differs from preprocessing.",
                ));
            }
            let x = state.preprocessing.transform(&data)?;
            encoded(
                &state
                    .selection
                    .transform(&x, &state.preprocessing.output_names)?,
            )
        }
        Request::AggregatePatientPredictions { predictions } => {
            encoded(&aggregate_patient_predictions(&predictions)?)
        }
        Request::ClassificationMetrics {
            task,
            target,
            prediction,
            threshold,
        } => encoded(&core::evaluation::classification_metrics(
            task,
            &target,
            &prediction,
            threshold,
        )?),
        Request::RegressionMetrics { y, prediction } => {
            encoded(&core::evaluation::regression_metrics(&y, &prediction)?)
        }
        Request::BinaryMetrics {
            y,
            scores,
            threshold,
        } => encoded(&core::evaluation::binary_metrics(&y, &scores, threshold)?),
        Request::RocCurve { y, scores } => encoded(&core::evaluation::roc_curve(&y, &scores)?),
        Request::PrecisionRecallCurve { y, scores } => {
            encoded(&core::evaluation::precision_recall_curve(&y, &scores)?)
        }
        Request::CalibrationCurve {
            y,
            scores,
            n_bins,
            strategy,
        } => encoded(&core::evaluation::calibration_curve(
            &y, &scores, n_bins, strategy,
        )?),
        Request::DelongInterval {
            y,
            scores,
            confidence,
        } => encoded(&core::evaluation::delong_interval(&y, &scores, confidence)?),
        Request::DelongTest { y, a, b } => encoded(&core::evaluation::paired_delong(&y, &a, &b)?),
        Request::DecisionCurve {
            y,
            scores,
            thresholds,
        } => encoded(&core::evaluation::decision_curve(&y, &scores, &thresholds)?),
        Request::HosmerLemeshow { y, scores, n_bins } => {
            encoded(&core::evaluation::hosmer_lemeshow(&y, &scores, n_bins)?)
        }
        Request::BootstrapBinary {
            y,
            scores,
            replicates,
            seed,
            confidence,
        } => encoded(&core::evaluation::bootstrap_binary_seeded(
            &y, &scores, replicates, seed, confidence,
        )?),
        Request::BootstrapMulticlass {
            target,
            prediction,
            replicates,
            seed,
            confidence,
        } => encoded(&core::evaluation::bootstrap_multiclass_seeded(
            &target,
            &prediction,
            replicates,
            seed,
            confidence,
        )?),
        Request::BootstrapRegression {
            target,
            prediction,
            replicates,
            seed,
            confidence,
        } => encoded(&core::evaluation::bootstrap_regression_seeded(
            &target,
            &prediction,
            replicates,
            seed,
            confidence,
        )?),
        Request::ClinicalLogistic {
            data,
            target,
            features,
            reference,
            config,
        } => encoded(&core::clinical_native::logistic_inference(
            &data, &target, &features, &reference, &config,
        )?),
        Request::Baseline {
            data,
            classes,
            labels,
            features,
            cohorts,
            config,
        } => encoded(&core::evaluation::baseline::baseline_table(
            &data,
            &classes,
            &labels,
            &features,
            cohorts.as_deref(),
            &config,
        )?),
        Request::Icc {
            data,
            kind,
            confidence,
        } => encoded(&core::reliability::icc(&data, kind, confidence)?),
        Request::ReadStudy { spec, task, base } => encoded(&core::table_io::read_study(
            &spec,
            task,
            std::path::Path::new(&base),
        )?),
        Request::ReadFeatureTable { path, manifest } => {
            encoded(&core::table_io::read_feature_table(
                path,
                manifest.as_deref().map(std::path::Path::new),
            )?)
        }
        Request::WriteFeatureTable {
            table,
            path,
            overwrite,
            null_value,
        } => encoded(&core::table_io::write_feature_table(
            &table,
            path,
            overwrite,
            &null_value,
        )?),
        Request::ReadNifti { path } => encoded(&core::imaging::read_nifti(path)?),
        Request::ReadMask { path } => encoded(&core::imaging::read_mask(path)?),
        Request::ImageResampleSpacing {
            image,
            spacing_mm,
            interpolation,
            outside,
        } => encoded(&core::imaging::resample_spacing(
            &image,
            spacing_mm,
            interpolation,
            outside,
        )?),
        Request::ImageResampleLike {
            image,
            reference,
            interpolation,
            outside,
        } => encoded(&core::imaging::resample_like(
            &image,
            &reference,
            interpolation,
            outside,
        )?),
        Request::MaskResampleLike {
            mask,
            reference,
            outside,
        } => encoded(&core::imaging::resample_mask_like(
            &mask, &reference, outside,
        )?),
        Request::MaskPeritumoralRing {
            mask,
            inner_mm,
            outer_mm,
            organ_mask,
        } => encoded(&core::imaging::peritumoral_ring(
            &mask,
            inner_mm,
            outer_mm,
            organ_mask.as_ref(),
        )?),
        Request::MaskSelectLabels { mask, labels } => {
            encoded(&core::imaging::select_labels(&mask, &labels)?)
        }
        Request::MaskLargestComponent { mask, connectivity } => {
            encoded(&core::imaging::largest_component(&mask, connectivity)?)
        }
        Request::MaskDilateVoxels {
            mask,
            iterations,
            connectivity,
        } => encoded(&core::imaging::dilate_voxels(
            &mask,
            iterations,
            connectivity,
        )?),
        Request::ImageClipRange {
            image,
            lower,
            upper,
        } => encoded(&core::imaging::clip_range(&image, lower, upper)?),
        Request::ImageClipPercentile {
            image,
            lower,
            upper,
            statistical_mask,
        } => encoded(&core::imaging::clip_percentile(
            &image,
            lower,
            upper,
            statistical_mask.as_ref(),
        )?),
        Request::ImageNormalize {
            image,
            method,
            statistical_mask,
        } => {
            let fitted =
                core::imaging::fit_intensity_transform(&image, method, statistical_mask.as_ref())?;
            encoded(&serde_json::json!({"image":fitted.apply(&image)?,"fitted":fitted}))
        }
        Request::MaskResegment {
            image,
            mask,
            lower,
            upper,
        } => encoded(&core::imaging::resegment(&image, &mask, lower, upper)?),
        Request::ArrayFeatures {
            image,
            morphology_mask,
            intensity_mask,
            roi_label,
            config,
        } => {
            let output = core::features::extract(
                &image,
                &morphology_mask,
                intensity_mask.as_ref(),
                roi_label,
                &config,
            )?;
            let mapped: BTreeMap<_, _> = output
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        if v.is_nan() {
                            serde_json::Value::String("NaN".into())
                        } else if v == f64::INFINITY {
                            serde_json::Value::String("Inf".into())
                        } else if v == f64::NEG_INFINITY {
                            serde_json::Value::String("-Inf".into())
                        } else {
                            serde_json::json!(v)
                        },
                    )
                })
                .collect();
            encoded(&mapped)
        }
    }
}
