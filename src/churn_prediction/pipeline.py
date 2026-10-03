"""Leakage-aware local training for the Telco churn example.

No row-level records or predictions are persisted. The test partition is not
used for fitting or threshold selection.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ID_COLUMNS = {"customerid", "customer_id", "id"}
# Exclude common duplicate/derived target columns even if one is the requested label.
TARGET_PROXY_COLUMNS = {
    "churn", "churn value", "churn label", "churn score", "churn reason",
    "churn category", "churned", "churn probability",
}
SCHEMA_VERSION = 1


def default_data_path() -> Path:
    """Return the documented project-relative default without OS-specific paths."""
    return Path("data") / DEFAULT_FILENAME


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place {DEFAULT_FILENAME} in data/ "
            "or pass --data / set CHURN_DATA_PATH to a local CSV."
        )
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV header: {exc}") from exc
    if not header:
        raise ValueError("Dataset is empty or has no header row.")
    if any(not name.strip() for name in header):
        raise ValueError("Dataset contains a blank column name.")
    if len({name.strip().casefold() for name in header}) != len(header):
        raise ValueError("Dataset contains duplicate column names.")
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not parse dataset CSV: {exc}") from exc


def _normalise_target(target: pd.Series, target_column: str) -> pd.Series:
    if target.isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")
    values = target.astype("string").str.strip().str.casefold()
    mapping = {
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "0": 0,
        "1": 1,
    }
    labels = values.map(mapping)
    if labels.isna().any():
        unexpected = sorted(set(values[labels.isna()].astype(str)))
        raise ValueError(
            f"Target {target_column!r} must contain only Yes/No or 1/0 labels; "
            f"found {unexpected[:5]}."
        )
    if labels.nunique() != 2:
        raise ValueError("Target must contain both churn classes (Yes and No).")
    return labels.astype("int8")


def prepare_features(
    frame: pd.DataFrame, target_column: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the input and return features with identifiers/proxies excluded."""
    if frame.empty:
        raise ValueError("Dataset has no data rows.")
    if not frame.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")
    if not all(isinstance(name, str) and name.strip() for name in frame.columns):
        raise ValueError("Every dataset column must have a non-empty string name.")
    columns_by_casefold = {name.strip().casefold(): name for name in frame.columns}
    requested = target_column.strip().casefold()
    if requested not in columns_by_casefold:
        raise ValueError(f"Required target column {target_column!r} is missing.")
    actual_target = columns_by_casefold[requested]
    y = _normalise_target(frame[actual_target], actual_target)

    excluded = set(ID_COLUMNS) | set(TARGET_PROXY_COLUMNS) | {actual_target.strip().casefold()}
    feature_columns = [
        name
        for name in frame.columns
        if name.strip().casefold() not in excluded
    ]
    if not feature_columns:
        raise ValueError("No eligible feature columns remain after excluding identifiers and targets.")

    X = frame.loc[:, feature_columns].copy()
    # Convert numeric-looking text (notably whitespace-filled TotalCharges) to
    # numeric; otherwise preserve it as a categorical feature. Empty strings are
    # treated as missing and imputed inside the training-only pipeline.
    for name in X.columns:
        series = X[name]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cleaned = series.map(lambda value: value.strip() if isinstance(value, str) else value)
            nonblank = cleaned.notna() & cleaned.astype("string").str.strip().ne("")
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if bool((numeric[nonblank].notna()).all()):
                X[name] = numeric
            else:
                X[name] = cleaned.mask(cleaned.astype("string").str.strip().eq(""), np.nan)
    return X, y


def _make_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [name for name in X.columns if name not in numeric_columns]
    transformers = []
    if numeric_columns:
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, numeric_columns))
    if categorical_columns:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical, categorical_columns))
    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns were found.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimator = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", estimator)])


def _choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose an F1 threshold on validation rows only, with deterministic ties."""
    candidates = np.unique(np.concatenate(([0.5], probabilities)))
    scored = [
        (f1_score(y_true, probabilities >= threshold, zero_division=0), float(threshold))
        for threshold in candidates
    ]
    best_f1 = max(score for score, _ in scored)
    tied_thresholds = [threshold for score, threshold in scored if score == best_f1]
    return min(tied_thresholds, key=lambda threshold: (abs(threshold - 0.5), -threshold))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def train_and_evaluate(
    frame: pd.DataFrame,
    output_dir: str | Path = "artifacts",
    target_column: str = "Churn",
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit on train, select threshold on validation, and report test metrics once."""
    X, y = prepare_features(frame, target_column=target_column)
    if y.value_counts().min() < 5:
        raise ValueError("At least five rows from each target class are required for stratified splits.")

    # 60/20/20 three-way split, stratified at each boundary. The test partition
    # is held back from all fitting and threshold selection.
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_valid,
    )
    model = _make_pipeline(X_train, random_state)
    model.fit(X_train, y_train)
    validation_probabilities = model.predict_proba(X_valid)[:, 1]
    threshold = _choose_threshold(y_valid, validation_probabilities)

    # The held-out test set is touched once, only after model and threshold are fixed.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype("int8")
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": target_column,
        "validation_selected_threshold": threshold,
        "test": {
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y_test, test_predictions, labels=[0, 1]
            ).tolist(),
        },
    }

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Model bundles are executable pickle-based artifacts: only load files from
    # trusted sources. They contain the fitted estimator and feature names, not rows.
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "excluded_identifier_and_target_columns": sorted(ID_COLUMNS | TARGET_PROXY_COLUMNS),
        "positive_label": 1,
        "decision_threshold": threshold,
        "random_state": random_state,
        "estimator": model,
    }
    fd, temp_name = tempfile.mkstemp(prefix=".model.", suffix=".tmp", dir=output_path)
    try:
        with os.fdopen(fd, "wb") as stream:
            joblib.dump(bundle, stream, compress=3)
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, output_path / "model.joblib")
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    _atomic_json(output_path / "metrics.json", metrics)
    _atomic_json(
        output_path / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "target_column": target_column,
            "feature_columns": X.columns.tolist(),
            "excluded_identifier_and_target_columns": sorted(ID_COLUMNS | TARGET_PROXY_COLUMNS),
        },
    )
    return metrics


def train_from_csv(
    data_path: str | Path,
    output_dir: str | Path = "artifacts",
    target_column: str = "Churn",
    random_state: int = 42,
) -> dict[str, Any]:
    """Read a local CSV, validate it, and run the three-way training workflow."""
    return train_and_evaluate(
        _read_csv(data_path),
        output_dir=output_dir,
        target_column=target_column,
        random_state=random_state,
    )
