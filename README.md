# Customer Churn Prediction

A reproducible scikit-learn baseline for the Telco Customer Churn classification dataset.

## Scope

The notebook compares **logistic regression**, **random forest**, and **support-vector machine** pipelines. Preprocessing is leakage-safe: the train/test split happens before imputation, scaling, one-hot encoding, or model fitting, and every transformer is fitted only on the training partition. Model selection uses stratified cross-validation on the training partition; the holdout is not used to choose a model.

This repository does **not** include the source CSV. No accuracy or model-ranking claim is made here until the notebook is run against a real copy of the data.

## Data setup

The expected schema includes `Churn`, `tenure`, `MonthlyCharges`, and `TotalCharges` (the commonly used Telco Customer Churn CSV also includes `customerID`). Put the file at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

or point to another location:

```bash
export CUSTOMER_CHURN_DATA=/absolute/path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If the file is missing, the notebook raises a clear `FileNotFoundError` with these instructions. The notebook validates required columns, duplicate headers, usable numeric values, both target classes, and enough class examples for the split and cross-validation. `TotalCharges`, `tenure`, and `MonthlyCharges` are parsed numerically; remaining missing feature values and unknown categorical values are handled inside the pipeline.

## Install and run

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python validate_project.py
jupyter lab "Customer Churn prediction.ipynb"
```

The static validator checks notebook JSON, AST syntax, stale outputs and execution counts, notebook-only magics, portable data configuration, and the presence of a stratified split, pipeline, and training-only cross-validation. It does not train models. Run the regression tests with `pytest`.

## Evaluation notes

The notebook uses a fixed `RANDOM_STATE = 42`, a stratified 80/20 holdout, and up to five shuffled stratified cross-validation folds on the training set. It reports cross-validated ROC-AUC for model selection plus held-out accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and a classification report. The holdout is evaluated only after selection and is not used for tuning.

The former notebook applied SMOTE, scaling, feature selection, and derived features before the split, which could leak information from the eventual test set. Those steps were removed in favor of a single `ColumnTransformer`/`Pipeline` workflow. Deep-learning cells were also removed: they were not reproducible without a heavy optional dependency and incorrectly passed the test set as validation data.

## Repository files

- `Customer Churn prediction.ipynb` — cleaned, output-free analysis notebook.
- `validate_project.py` — dependency-free static validation.
- `tests/test_validate_project.py` — regression tests for notebook validation.
- `requirements.txt` — runtime, notebook, and test dependencies.
- `ENGINEERING_REVIEW.md` — audit findings, changes, and validation evidence.
