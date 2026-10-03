# Customer Churn Prediction

This repository contains an exploratory Jupyter notebook and a separate, reproducible Python training workflow for the Telco Customer Churn example. The CSV is **not included**. No performance metrics or external validation are claimed by this repository.

## Dataset and privacy

Obtain the Telco Customer Churn CSV separately from a source whose terms permit your use. The expected filename is `WA_Fn-UseC_-Telco-Customer-Churn.csv`; place it under `data/` in the repository, or set `CHURN_DATA_PATH` to the local CSV path. Do not commit private, licensed, or otherwise restricted data. Local datasets, model artifacts, and generated CSVs are ignored by Git.

The training workflow requires a binary `Churn` column with `Yes`/`No` or `1`/`0` labels. It excludes common customer ID fields and known duplicate/derived churn columns from predictors, converts numeric-looking text such as whitespace-containing `TotalCharges`, and imputes missing feature values using training-fold statistics. It validates the target and reports actionable errors for missing files, malformed headers, missing labels, and invalid classes.

## Install

Use Python 3.11 in an isolated environment. The pinned full requirements include dependencies used by the notebook (including TensorFlow); the new training package has a smaller pinned dependency set and does not require TensorFlow.

```bash
python3.11 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Package workflow and tests
python -m pip install -e ".[test]"
# Install this as well only when using the full exploratory notebook
python -m pip install -r requirements.txt
```

## Reproducible training workflow

Run from the repository root. `--data` overrides the default path; otherwise `CHURN_DATA_PATH` is used when set, and `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is the default.

```bash
churn-train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir artifacts
# Or set CHURN_DATA_PATH=/path/to/file.csv and run: churn-train
```

This workflow creates a deterministic stratified 60/20/20 train/validation/test split. Imputation, scaling, and categorical encoding are fitted only on training rows in an sklearn pipeline. A class-balanced logistic-regression baseline is fitted on training rows; its decision threshold is selected on validation data; test metrics are computed only after those choices are fixed. The held-out test data are not used for training or threshold tuning. This implementation does not oversample the data.

The ignored `artifacts/` directory contains a fitted `model.joblib`, a feature/target `schema.json`, and aggregate `metrics.json`. It does not save customer-level predictions or input records. The report includes aggregate accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and the validation-selected threshold; these are outputs of your local run, **not precomputed or independently validated repository results**. Joblib files use Python pickle internally: load only artifacts from sources you trust, and protect them like other model files.

## Exploratory notebook

Open `Customer Churn prediction.ipynb` in Jupyter. Install `requirements.txt` first. Notebook outputs are cleared from version control to avoid retaining customer-level samples or implying fresh results. The notebook is exploratory and not an alternative to the cleaner package workflow above: its hand-selected features and some feature-selection/model-fitting/EDA cells are fit or examined before its holdout split, so its reported comparisons can be optimistic. Neural-network fitting now uses a validation split from training data instead of using the test partition as validation, but the notebook as a whole still has pre-split leakage. The LSTM/GRU sections reshape each tabular row to one timestep and do not model longitudinal histories.

## Tests and limitations

Run automated checks with:

```bash
python -m pytest
python -m compileall -q src tests
```

Tests use generated in-memory fixtures only; they do not represent the real dataset or model performance. This is an educational example, not a production churn system. It has not been externally validated and should not be used to make high-impact customer decisions without independent review of data quality, privacy, fairness, security, and deployment requirements.
