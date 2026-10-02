# Customer Churn Prediction

This repository contains one Jupyter notebook, `Customer Churn prediction.ipynb`, that explores the Telco Customer Churn CSV and compares several classification approaches. The dataset is **not included** in this repository, so the notebook cannot run end to end until you obtain the data separately.

## Requirements

Use Python 3.11 and install the pinned dependencies in an isolated environment:

```bash
python3.11 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Jupyter with `jupyter lab` and open the notebook.

## Dataset

Obtain the Telco Customer Churn CSV from its source under the applicable license/terms. The expected filename is `WA_Fn-UseC_-Telco-Customer-Churn.csv`; place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, or set `CHURN_DATA_PATH` to the CSV's location before starting Jupyter. The notebook raises an actionable `FileNotFoundError` if the file cannot be found. Do not commit private or licensed data to this repository.

The CSV is expected to contain the columns used by the notebook, including `Churn`, `customerID`, `tenure`, `MonthlyCharges`, and the Telco service/payment fields. No Kaggle API credential is needed or should be added to the notebook.

## Running the notebook

Run cells from top to bottom. The notebook contains exploratory plots, feature engineering, train/test comparison of traditional classifiers, and experimental ANN/LSTM/GRU sections. Scaling is fitted using training rows, SMOTE is now applied only to the training partition for the basic comparisons, and prior saved outputs have been removed so they are not mistaken for fresh results.

## Limitations

- No dataset is checked in, and the notebook has **not been executed or model performance validated** as part of this maintenance change.
- The feature list is manually specified, and some exploratory feature-selection/model-fitting cells run before the final holdout split. Treat reported metrics as exploratory; for publication or deployment, move all feature selection inside a cross-validation pipeline and evaluate on a genuinely untouched holdout.
- The LSTM/GRU sections reshape each customer's tabular row to a sequence with one timestep. They are included as experiments, but do not model longitudinal customer histories.
- Hyperparameter search uses an imbalanced-learn pipeline so scaling and oversampling occur within each CV training fold. The notebook still requires review and execution with the intended dataset before its findings can be relied on.
- This is an educational analysis, not a production churn system. Validate data quality, fairness, privacy, and deployment requirements independently.
