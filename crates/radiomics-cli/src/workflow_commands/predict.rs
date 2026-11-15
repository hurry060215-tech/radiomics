use super::common::*;
use clap::Args;
use ov_radiomics_core::{
    bundle::{ModelBundle, ModelManifest},
    errors::RadResult,
    learning::{self, FittedFusion, FittedPipeline},
    table_io::LoadReport,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

#[derive(Debug, Args, Serialize)]
pub struct PredictCommand {
    #[arg(long)]
    pub model: PathBuf,
    /// Pipeline: StudyTableSpec JSON. Fusion: schema-one named component input map.
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long, default_value = "radiomics-predictions")]
    pub output: PathBuf,
    #[arg(long)]
    pub overwrite: bool,
}
#[derive(Deserialize)]
struct Header {
    manifest: ModelManifest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FusionInputs {
    schema_version: u32,
    #[serde(deserialize_with = "unique_paths")]
    components: BTreeMap<String, PathBuf>,
}
fn unique_paths<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, PathBuf>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, PathBuf>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique component names mapped to StudyTableSpec JSON paths")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut values = BTreeMap::new();
            while let Some((name, path)) = map.next_entry::<String, PathBuf>()? {
                if name.trim().is_empty() || values.insert(name, path).is_some() {
                    return Err(serde::de::Error::custom(
                        "Fusion component names must be nonempty and unique.",
                    ));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(Visitor)
}
struct RoutedPrediction {
    table: ov_radiomics_core::contracts::PredictionTable,
    reports: BTreeMap<String, LoadReport>,
    specifications: BTreeMap<String, serde_json::Value>,
    manifest: ModelManifest,
}
pub fn predict(args: PredictCommand) -> RadResult<u8> {
    let header: Header = read_json(&args.model)?;
    header.manifest.validate()?;
    let kind = header
        .manifest
        .training_configuration
        .get("model_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| schema("Model manifest has no explicit model_kind."))?;
    let output = Output::preflight(
        &args.output,
        args.overwrite,
        [
            "predictions.json",
            "predictions.csv",
            "load_report.json",
            "prediction_metadata.json",
        ]
        .map(PathBuf::from),
    )?;
    let routed = match kind {
        "pipeline" => {
            let model = ModelBundle::<FittedPipeline>::load(&args.model)?;
            let (spec, loaded) = load_study_record(&args.data, model.manifest.task)?;
            let predictions = learning::predict(&model, &loaded.data)?;
            RoutedPrediction {
                table: predictions,
                reports: BTreeMap::from([("pipeline".into(), loaded.report)]),
                specifications: BTreeMap::from([(
                    "pipeline".into(),
                    json!({"specification":spec,"spec_directory":std::fs::canonicalize(parent(&args.data))?}),
                )]),
                manifest: model.manifest.clone(),
            }
        }
        "fusion" => {
            let model = ModelBundle::<FittedFusion>::load(&args.model)?;
            let inputs: FusionInputs = read_json(&args.data)?;
            if inputs.schema_version != 1 {
                return Err(schema("Unknown fusion input schema version."));
            }
            let expected: BTreeSet<_> = model
                .fitted
                .components
                .iter()
                .map(|c| c.name.clone())
                .collect();
            if inputs.components.keys().cloned().collect::<BTreeSet<_>>() != expected {
                return Err(schema(
                    "Fusion inputs must exactly match the fitted named components.",
                ));
            }
            let mut data = BTreeMap::new();
            let mut reports = BTreeMap::new();
            let mut specifications = BTreeMap::new();
            for (name, path) in inputs.components {
                let path = resolve(parent(&args.data), &path);
                let (spec, loaded) = load_study_record(&path, model.manifest.task)?;
                data.insert(name.clone(), loaded.data);
                specifications.insert(name.clone(),json!({"specification":spec,"spec_directory":std::fs::canonicalize(parent(&path))?}));
                reports.insert(name, loaded.report);
            }
            let predictions = learning::predict_fusion(&model, &data)?;
            RoutedPrediction {
                table: predictions,
                reports,
                specifications,
                manifest: model.manifest.clone(),
            }
        }
        _ => {
            return Err(schema(format!(
                "Unknown model_kind '{kind}'; expected pipeline or fusion."
            )))
        }
    };
    let RoutedPrediction {
        table: predictions,
        reports,
        specifications,
        manifest,
    } = routed;
    output.begin()?;
    output.predictions("predictions.json", "predictions.csv", &predictions)?;
    output.json("load_report.json", &reports)?;
    output.json("prediction_metadata.json",&json!({"format":"radiomics.cli.prediction","schema_version":1,"task":predictions.task,"classes":predictions.classes,"unit":predictions.unit,"model_kind":kind,"model_schema_version":manifest.schema_version,"model_threshold":manifest.threshold,"input_specifications":specifications,"rows":predictions.rows.len(),"cohorts":predictions.rows.iter().map(|r|r.cohort.as_str()).collect::<BTreeSet<_>>(),"runtime":"rust","fit_performed":false}))?;
    println!(
        "{}",
        json!({"status":"complete","command":"predict","rows":predictions.rows.len(),"unit":predictions.unit,"output":args.output})
    );
    Ok(0)
}
