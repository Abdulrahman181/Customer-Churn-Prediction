# Engineering Review — Customer Churn Prediction

**Repository:** `Abdulrahman181/Customer-Churn-Prediction`  
**Local path:** `/home/ubuntu/github_priority/Customer-Churn-Prediction`  
**Review date:** 2026-10-13  
**Scope:** all files present at review start (`README.md` and `Customer Churn prediction.ipynb`), plus the repaired notebook and supporting project files.

## Executive summary

The original repository contained only a README and one executed notebook. The notebook was not portable or safely reproducible: it referenced a Windows-only CSV path, contained a Python syntax error caused by `%matplotlib inline`, had a missing `plotly.express` import as a consequence of that parse failure, performed resampling and transformations before the train/test split, and used the test set as validation data for neural networks. The dataset was not present in the repository, so a complete training run could not be honestly performed.

The notebook was replaced with a short, output-free scikit-learn baseline that uses a configurable data path, explicit schema checks, a stratified holdout, and leakage-safe `ColumnTransformer`/`Pipeline` preprocessing. The README now documents the missing-data condition and does not claim model performance that was not rerun. A dependency manifest, `.gitignore`, and dependency-free static validator were added.

## Findings in the original notebook

### Critical / high priority

1. **Non-portable data path:** `D:\desctop\WA_Fn-UseC_-Telco-Customer-Churn.csv` could not work outside the original Windows machine and the referenced CSV was not committed.
2. **Syntax error:** `%matplotlib inline` was embedded in a Python code cell. Python AST parsing failed at that cell; therefore the imports in that cell were not valid Python execution. The notebook also used `px` later without a reliably executed import.
3. **Data leakage from resampling:** `SMOTE.fit_resample(X, y)` was called before the train/test split. Synthetic observations derived from the eventual test distribution could influence training.
4. **Data leakage from preprocessing/feature selection:** scaling and `SelectKBest.fit_transform(X, y)` were performed before the split. The original notebook also created engineered fields on a resampled full dataset, outside a training-only pipeline.
5. **Test-set contamination in deep learning:** the original ANN/LSTM/GRU calls passed `validation_data=(X_test, y_test)` while training. This makes the held-out test set part of model-selection feedback and invalidates the reported final test metrics.
6. **Unreproducible / misleading comparison:** deep-learning training had no explicit seed and the README asserted that Random Forest was best without a fresh, valid, leakage-free run. The repository had no dependency manifest, so the original notebook could not be recreated reliably.

### Medium / maintainability

- `TotalCharges` was dropped instead of being parsed numerically, losing a potentially useful feature.
- The original notebook mutated dataframes in place across many cells, making reruns order-dependent.
- `df_resampled.astype(int)` risked inappropriate coercion after one-hot encoding and before scaling.
- Old outputs and execution counts were embedded throughout the notebook, including training logs and plots from an unavailable environment.
- The README described directories (`data`, `notebooks`) that did not exist and described deep-learning models that were not documented with reproducible setup.

## Changes made

- Rebuilt `Customer Churn prediction.ipynb` as an 8-cell, output-free, AST-valid notebook.
- Added `CUSTOMER_CHURN_DATA` environment-variable support with a relative default under `data/`.
- Added actionable `FileNotFoundError` guidance when the dataset is absent.
- Added required-column validation and robust numeric parsing for `TotalCharges`.
- Removed customer identifier from features while retaining other available columns.
- Split data with `test_size=0.20`, `random_state=42`, and `stratify=y` before all learned transformations.
- Added training-only median imputation, standardization, one-hot encoding with `handle_unknown="ignore"`, and model fitting inside scikit-learn pipelines.
- Added balanced accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report. No numeric results are recorded because the real dataset was unavailable.
- Removed the prior SMOTE-before-split flow and the ANN/LSTM/GRU sections that used the test set as validation data. This keeps the committed workflow lightweight and avoids an unsupported heavy dependency.
- Added `requirements.txt` with NumPy, pandas, scikit-learn, JupyterLab, and nbformat.
- Added `.gitignore` for environments, caches, checkpoints, and uncommitted dataset files.
- Added `validate_project.py`, which checks notebook JSON, code-cell AST syntax, stale outputs, notebook magics/shell commands, portable data configuration, and required split/pipeline markers.
- Updated `README.md` with setup, data schema, run instructions, leakage safeguards, metric definitions, and limitations.

## Validation performed

| Check | Result |
|---|---|
| Notebook JSON parse (`python3 -m json.tool`) | Passed |
| AST parse of every notebook code cell | Passed; 5 code cells |
| Stale notebook outputs | Passed; 0 outputs |
| `python3 validate_project.py` | Passed |
| `ruff check validate_project.py` | Passed after automatic import sorting |
| `python3 -m py_compile validate_project.py` | Passed |
| `git diff --check` | Passed |
| Full model execution | **Not run**: dataset absent and `scikit-learn`/JupyterLab are not installed in this sandbox |

The full notebook was intentionally not executed. Running it requires installing `requirements.txt` and supplying the real dataset. No fabricated data or performance results were used.

## Remaining risks / next steps

- Obtain the licensed/source CSV and run the notebook in a clean virtual environment.
- Add training-set cross-validation for model selection and confidence intervals or repeated splits before making a production claim.
- Confirm the exact dataset schema and label distribution; the notebook intentionally fails fast on unsupported `Churn` labels.
- Consider a calibration/threshold policy and business-cost metric before deployment; accuracy alone is insufficient for churn intervention decisions.

## Second Senior Review

**Review date:** 2026-10-15
**Scope:** every tracked file, the complete notebook, repository history/diff, configuration, and newly added regression tests.

### Findings

1. **Test-set model-selection leakage remained.** The first repair evaluated all models on the holdout and then selected the model with the highest holdout ROC-AUC. That makes the reported holdout comparison optimistic when the selected model is treated as the final choice.
2. **Pipeline transformer state was shared across models.** `make_model` reused one `ColumnTransformer` instance in three pipelines. The current data happened to use identical training rows, but shared fitted state is an unsafe object contract and can silently contaminate future model-specific changes or fitting order.
3. **The notebook lacked defensive data validation.** Empty files, duplicate headers, unusable/all-missing required numeric columns, infinite numeric values, one-class targets, and too-small classes produced obscure downstream errors or invalid evaluation plans.
4. **Static validation did not check clean execution state or all IPython commands.** It checked outputs and a narrow subset of magics, but not non-null execution counts, `get_ipython`, shell/magic variants, or training-only model-selection markers. There were also no regression tests for the validator.

### Fixes completed

- Added up-to-five-fold shuffled `StratifiedKFold`/`cross_validate` on `X_train`; model selection now uses training-only mean ROC-AUC, followed by one held-out evaluation.
- Changed preprocessing construction to a factory so every model owns a fresh `ColumnTransformer` and fitted transformer state.
- Added explicit CSV-empty, empty-dataframe, duplicate-column, numeric coercion/finite-value, target-label/class-count, feature-count, and cross-validation-size checks. The notebook now fails fast with actionable messages rather than claiming metrics for invalid inputs.
- Expanded `validate_project.py` with reusable `validate_notebook`, malformed-cell checks, stale execution-count checks, broad IPython magic/shell detection, and cross-validation markers.
- Added `tests/test_validate_project.py` regression tests; documented the revised evaluation contract and test command; declared pytest in `requirements.txt`.
- Confirmed the notebook remains output-free and does not contain secrets, serialized-artifact loading, API/UI code, or machine-specific paths. No artifact loader or API/UI file was missing because the repository documents and implements a notebook-only baseline.

### Second-review validation

- `python3 validate_project.py` passed: 5 code cells, valid AST, clean execution state, and portable data configuration.
- `ruff check validate_project.py tests/test_validate_project.py`, `python3 -m compileall -q validate_project.py tests`, `pytest -q` (2 passed), and `git diff --check` passed.
- Every notebook code cell was executed in order against a temporary schema-valid synthetic CSV; the runtime smoke test passed. The temporary file and its metrics were not committed.
- The source CSV is still absent from the repository, so a real-data end-to-end training run and genuine performance metrics remain intentionally unclaimed.

### Remaining risks

- Run the notebook in a clean environment with the licensed/source CSV and verify the exact schema and label distribution.
- Cross-validation and the holdout remain a baseline evaluation, not production validation; add repeated CV/confidence intervals, calibration, threshold policy, and business-cost evaluation before deployment.
- Dependency ranges are minimum-version ranges rather than a lockfile; capture a tested lock/constraints file if byte-for-byte environment reproducibility is required.
