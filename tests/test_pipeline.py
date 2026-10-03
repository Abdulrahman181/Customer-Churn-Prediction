from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import pytest

from churn_prediction.pipeline import prepare_features, train_and_evaluate, train_from_csv


def sample_data(rows_per_class: int = 50) -> pd.DataFrame:
    labels = ["No"] * rows_per_class + ["Yes"] * rows_per_class
    rows = len(labels)
    return pd.DataFrame(
        {
            "customerID": [f"PRIVATE-CUSTOMER-{index:04d}" for index in range(rows)],
            "Churn": labels,
            # Known outcome proxies must never become predictors.
            "Churn Value": [int(value == "Yes") for value in labels],
            "tenure": [index % 48 for index in range(rows)],
            "TotalCharges": [" " if index % 19 == 0 else str(index * 2.5) for index in range(rows)],
            "Contract": ["month-to-month" if value == "Yes" else "two year" for value in labels],
            "Service": [None if index % 17 == 0 else ("fiber" if index % 2 else "dsl") for index in range(rows)],
        }
    )


def test_prepare_features_normalizes_target_and_excludes_ids_and_target_proxies():
    X, y = prepare_features(sample_data())
    assert y.tolist() == [0] * 50 + [1] * 50
    assert "customerID" not in X
    assert "Churn Value" not in X
    assert "Churn" not in X
    assert pd.api.types.is_numeric_dtype(X["TotalCharges"])
    assert X["TotalCharges"].isna().any()
    custom = sample_data().rename(columns={"Churn": "Outcome"})
    custom["Churn Reason"] = ["customer left" if label == "Yes" else "" for label in custom["Outcome"]]
    custom_X, _ = prepare_features(custom, target_column="Outcome")
    assert "Outcome" not in custom_X
    assert "Churn Reason" not in custom_X


def test_input_validation_rejects_missing_and_nonbinary_targets():
    data = sample_data()
    with pytest.raises(ValueError, match="both churn classes"):
        prepare_features(data.assign(Churn="Yes"))
    with pytest.raises(ValueError, match="only Yes/No or 1/0"):
        prepare_features(data.assign(Churn=["invalid"] + ["No"] * 49 + ["Yes"] * 50))
    with pytest.raises(ValueError, match="missing values"):
        prepare_features(data.assign(Churn=[None] + ["No"] * 49 + ["Yes"] * 50))


def test_training_writes_aggregate_metrics_schema_and_private_model_artifact(tmp_path):
    metrics = train_and_evaluate(sample_data(), output_dir=tmp_path)
    assert set(metrics["test"]) == {
        "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix_labels_0_1"
    }
    assert sum(map(sum, metrics["test"]["confusion_matrix_labels_0_1"])) == 20

    report = json.loads((tmp_path / "metrics.json").read_text())
    schema = json.loads((tmp_path / "schema.json").read_text())
    assert report == metrics
    assert "customerID" not in schema["feature_columns"]
    assert "Churn Value" not in schema["feature_columns"]
    assert not any(f"PRIVATE-CUSTOMER-{index:04d}" in (tmp_path / "metrics.json").read_text() for index in range(100))
    assert (tmp_path / "model.joblib").is_file()
    bundle = joblib.load(tmp_path / "model.joblib")
    assert "customerID" not in bundle["feature_columns"]
    assert "Churn Value" not in bundle["feature_columns"]
    if os.name == "posix":
        assert (tmp_path / "model.joblib").stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "metrics.json").stat().st_mode & 0o777 == 0o600


def test_csv_entrypoint_reads_local_file_and_rejects_missing_path(tmp_path):
    csv_path = tmp_path / "telco.csv"
    sample_data().to_csv(csv_path, index=False)
    metrics = train_from_csv(csv_path, output_dir=tmp_path / "artifacts")
    assert "f1" in metrics["test"]
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        train_from_csv(tmp_path / "absent.csv", output_dir=tmp_path / "other")


def test_training_rejects_too_small_classes():
    with pytest.raises(ValueError, match="At least five rows"):
        train_and_evaluate(sample_data(rows_per_class=4))
