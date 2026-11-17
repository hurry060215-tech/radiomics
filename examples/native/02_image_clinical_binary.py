"""NIfTI+ROI+clinical data → grouped native binary pipelines and OOF fusion."""

from _shared import image_cohort, output_arguments, provenance, save_patient_roc

import ov_radiomics as rad


def main():
    output = output_arguments(__doc__)
    extracted, identities, classes = image_cohort(output)
    features = [
        "original_firstorder_Mean",
        "original_firstorder_Variance",
        "original_firstorder_90Percentile",
    ]
    # Each clinical row represents one patient. Joining onto lesions is explicitly
    # many-to-one; patient grouping is retained in every nested fit and evaluation.
    clinical = identities.drop_duplicates("patient_id")[
        ["patient_id", "signal_b", "noise_a", "target", "group"]
    ].copy()
    clinical = clinical.rename(columns={"signal_b": "clinical_a", "noise_a": "clinical_b"})
    clinical["category"] = ["A" if i % 2 else "B" for i in range(len(clinical))]
    clinical.to_csv(output / "clinical.csv", index=False)
    data = extracted.features.merge(
        identities[["ID", "patient_id"]], on="ID", validate="one_to_one"
    ).merge(clinical, on="patient_id", validate="many_to_one")
    data.to_csv(output / "study.csv", index=False)
    selector = rad.FeatureSelector(backend="rust", univariate=None, lasso=False)
    image = rad.train_models(
        data,
        "target",
        features,
        backend="rust",
        classes=classes,
        positive_label=classes[1],
        selector=selector,
        cv=3,
        inner_cv=2,
        patient_id="patient_id",
    )
    clinic = rad.train_models(
        data,
        "target",
        ["clinical_a", "clinical_b", "category"],
        categorical=["category"],
        backend="rust",
        classes=classes,
        positive_label=classes[1],
        split_plan=image.split_plan,
        scale="standard",
    )
    image.save(output / "image-model")
    clinic.save(output / "clinical-model")
    fusion = rad.fit_fusion(rad.fusion_data({"Image": image, "Clinical": clinic}), backend="rust")
    fusion.save(output / "fusion.rad")
    for name, result in [("image", image), ("clinical", clinic), ("fusion", fusion)]:
        rad.evaluate(result, backend="rust", prediction_unit="patient", ci_method=None).to_csv(
            output / (name + "-metrics.csv"), index=False
        )
    external = data.loc[data.group != "train"].drop(columns="target")
    before = image.predict(external)
    restored = rad.load_model(output / "image-model/model-000.rad")
    assert before.to_dict() == restored.predict(external).to_dict()
    before.dataframe.to_csv(output / "unlabeled-predictions.csv", index=False)
    save_patient_roc(image, data, output / "heldout-patient-roc.png")
    provenance(
        output,
        "02_image_clinical_binary",
        patients=data.patient_id.nunique(),
        samples=len(data),
        split_plan_sha256=image.split_plan.fingerprint,
        evaluation_unit="patient",
        fusion_training="verified base OOF; meta training predictions apparent",
    )
    print(rad.evaluate(image, ci_method=None).to_string(index=False))
    print("Outputs:", output)


if __name__ == "__main__":
    main()
