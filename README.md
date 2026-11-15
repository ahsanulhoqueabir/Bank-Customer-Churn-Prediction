# Bank Customer Churn Prediction (CSE 4101 — Lab Assignment 10)

Supervised binary classification: predict whether a bank customer exits (`Exited = 1`) using Logistic Regression, Random Forest and Gradient Boosting inside leakage-safe scikit-learn pipelines. Implementation spec: [index.md](index.md).

## Contents
```
notebooks/bank_customer_churn_prediction.ipynb   main Colab notebook (all 10 tasks)
docs/en/, docs/bn/                              7 detailed guides in English / বাংলা (below)
data/                                            local copies of train.csv, test.csv (git-ignored)
README.md, CLAUDE.md, index.md, .gitignore
```
The layout follows the structure in `index.md` with no deviations; `data/` also holds `test.csv` and `sample_submission.csv` (the latter is unused).

## Dataset
Kaggle *Bank Customer Churn Prediction (2026)*: <https://www.kaggle.com/competitions/bank-customer-churn-prediction-2026/data>.
Drive path used by the notebook: `/content/drive/MyDrive/Customer Churn Prediction/train.csv` (optional unlabeled `test.csv` in the same folder). Outside Colab the notebook falls back to `data/train.csv` for development only.

## Run (Google Colab)
1. Put `train.csv` (and optionally `test.csv`) in `My Drive/Customer Churn Prediction/`.
2. Open the notebook in Colab → *Runtime → Run all* → authorise Drive.
Details: [docs/en/07](docs/en/07_colab_and_google_drive_guide.md) · [বাংলা](docs/bn/07_colab_and_google_drive_guide.md). Local run: `jupyter nbconvert --to notebook --execute notebooks/bank_customer_churn_prediction.ipynb` from the project root (needs pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter).

## Workflow
inspect/EDA → drop identifiers (`id`, `CustomerId`, `Surname`) → stratified 70/15/15 split → preprocessing in pipelines (median/most-frequent imputation, one-hot, scaling for LR only) → compare 3 models on validation → `GridSearchCV` (5-fold, ROC-AUC, training split only) → choose model + F1 threshold on validation → train-vs-validation diagnostics → **single** internal-test evaluation → predictions for five unlabeled rows.

## Documentation (English · বাংলা)
1. [Assignment and workflow](docs/en/01_assignment_and_workflow.md) · [অ্যাসাইনমেন্ট ও ওয়ার্কফ্লো](docs/bn/01_assignment_and_workflow.md)
2. [Dataset and EDA](docs/en/02_dataset_and_eda.md) · [ডেটাসেট ও EDA](docs/bn/02_dataset_and_eda.md)
3. [Preprocessing and data leakage](docs/en/03_preprocessing_and_data_leakage.md) · [প্রি-প্রসেসিং ও ডেটা লিকেজ](docs/bn/03_preprocessing_and_data_leakage.md)
4. [Models and hyperparameter tuning](docs/en/04_models_and_hyperparameter_tuning.md) · [মডেল ও হাইপারপ্যারামিটার টিউনিং](docs/bn/04_models_and_hyperparameter_tuning.md)
5. [Evaluation and interpretation](docs/en/05_evaluation_and_interpretation.md) · [মূল্যায়ন ও ব্যাখ্যা](docs/bn/05_evaluation_and_interpretation.md)
6. [Internal implementation guide](docs/en/06_internal_implementation_guide.md) · [অভ্যন্তরীণ বাস্তবায়ন নির্দেশিকা](docs/bn/06_internal_implementation_guide.md)
7. [Colab and Google Drive guide](docs/en/07_colab_and_google_drive_guide.md) · [Google Colab ও Google Drive নির্দেশিকা](docs/bn/07_colab_and_google_drive_guide.md)

## Verification status
* **Executed:** the full notebook ran top-to-bottom without errors on the local copies `data/train.csv` and `data/test.csv` (Python 3.13.5, scikit-learn 1.6.1) via the local fallback path; the saved notebook contains those outputs. Docs 02–05 quote those outputs.
* **Not executed:** the Colab-only `drive.mount` branch and Drive path (no Colab/Drive access during development); Colab runtime/version behaviour.

Metrics depend on the actual data and execution (library versions, seed); do not treat the numbers in the docs as guaranteed for another file or environment.
