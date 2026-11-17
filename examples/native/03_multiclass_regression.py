"""Three-class softmax and continuous Ridge using the same grouped native core."""

from _shared import output_arguments, provenance, task_data

import ov_radiomics as rad


def main():
    output = output_arguments(__doc__)
    for task in ("multiclass", "regression"):
        data, features, classes = task_data(task)
        selection = rad.FeatureSelector(task=task, backend="rust", univariate=None, lasso=False)
        result = rad.train_models(
            data,
            "target",
            features,
            task=task,
            classes=classes,
            backend="rust",
            selector=selection,
            patient_id="patient_id",
            cv=3,
            inner_cv=2,
        )
        result.save(output / task)
        report = rad.evaluate(
            result,
            backend="rust",
            prediction_unit="patient",
            ci_method="bootstrap",
            n_bootstrap=200,
            random_state=20261114,
        )
        report.to_csv(output / (task + "-metrics.csv"), index=False)
        result.predict(data.loc[data.group != "train"].drop(columns="target")).dataframe.to_csv(
            output / (task + "-unlabeled-predictions.csv"), index=False
        )
        print(task, report.to_string(index=False), sep="\n")
    provenance(
        output,
        "03_multiclass_regression",
        models=dict(multiclass="multinomial softmax LR with L2", regression="Ridge"),
        evaluation_unit="patient",
        bootstrap="200 replicates for demonstration; use predeclared study replication count",
    )
    print("Outputs:", output)


if __name__ == "__main__":
    main()
