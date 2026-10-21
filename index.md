# Bank Customer Churn Prediction

> **Purpose:** Use this document as the source of truth for implementing the complete CSE 4101 Machine Learning Practical Lab Assignment 10. Read this file fully before coding. Implement the work in a beginner-friendly, reproducible way that runs in **Google Colab** and uses the dataset stored in Google Drive.

---

## 1. Project Overview

### Assignment details

- **Course:** CSE 4101 — Introduction to Machine Learning
- **Assignment:** Practical Lab Assignment 10
- **Title:** Complete Machine Learning Workflow for Bank Customer Churn Prediction
- **Type:** Individual practical
- **Language:** Python
- **Primary environment:** Google Colab
- **Problem type:** Supervised binary classification
- **Goal:** Predict whether a bank customer is likely to leave/close their account, using the labeled `Exited` outcome.

### Dataset

- **Kaggle competition:** Bank Customer Churn Prediction (2026)
- **Dataset page:** https://www.kaggle.com/competitions/bank-customer-churn-prediction-2026/data
- **Labeled training file:** `train.csv`
- **Target column specified by the assignment:** `Exited` (0/1)
- **Google Drive dataset folder:** `Customer Churn Prediction`
- **Expected Drive path:** `/content/drive/MyDrive/Customer Churn Prediction/`
- **Local development copy, if available:** `data/train.csv`

The local `data/train.csv` is a convenient inspection/development copy, but Google Colab must work using the Google Drive folder. Do not hard-code assumptions about the dataset's complete column list. Inspect the actual CSV headers and data types at runtime, and explicitly report any differences from the assignment's expected fields.

**Important:** The assignment PDF is the authoritative task specification. Do not silently omit any of its ten practical tasks.

---

## 2. How Claude Code Should Work

Act as a careful machine-learning engineer and a patient technical instructor.

1. Inspect the existing repository before creating or modifying files.
2. Inspect `data/train.csv` if it exists in the current project. Use it to understand the real schema, data types, missing values, target distribution, and identifier columns. Do not assume this local file will exist in Colab.
3. Build the primary workflow as a clean, sequential Google Colab notebook.
4. Put detailed implementation and conceptual documentation in a root-level `docs/` directory.
5. Keep notebook cells readable, logically grouped, and suitable for a student explaining the work in a lab viva or report.
6. Use small, meaningful functions where they improve clarity; avoid unnecessarily complex abstractions.
7. Add comments explaining important decisions, not comments that merely repeat every line of code.
8. Execute/test the code where the available environment permits. If Google Drive or Colab is unavailable in the local environment, explain that limitation and make the notebook self-checking when run in Colab.
9. Do not fabricate outputs, metrics, observations, column names, or conclusions. Data-driven claims must be generated from the actual dataset and model results.
10. Do not claim a cell has been executed successfully unless it has actually been executed.

### Do not overengineer

This is a university practical assignment, not a production ML platform. Prefer understandable scikit-learn pipelines and explicit evaluation steps over elaborate frameworks.

---

## 3. Required Project Structure

Inspect the existing repository first and preserve useful files. Adapt paths if the repository already has an established structure, but maintain the following logical organization:

```text
project-root/
├── README.md
├── CLAUDE.md                     # concise project instructions for Claude Code
├── notebooks/
│   └── bank_customer_churn_prediction.ipynb
├── docs/
│   ├── 01_assignment_and_workflow.md
│   ├── 02_dataset_and_eda.md
│   ├── 03_preprocessing_and_data_leakage.md
│   ├── 04_models_and_hyperparameter_tuning.md
│   ├── 05_evaluation_and_interpretation.md
│   ├── 06_internal_implementation_guide.md
│   └── 07_colab_and_google_drive_guide.md
├── data/
│   └── train.csv                  # existing local copy, if available; do not require it in Colab
└── .gitignore
```

Do not copy large/private dataset files into documentation or commit dataset files unintentionally. Add appropriate data, notebook checkpoint, cache, and generated-artifact entries to `.gitignore` where relevant. Keep the notebook and docs tracked.

If the repository already uses a different sensible layout, do not destructively reorganize it merely to match this example. Explain any structural differences in the README.

---

## 4. Google Colab and Google Drive Requirements

The notebook must run from top to bottom in Google Colab after the user grants Drive access.

### Data loading

- Mount Google Drive using `google.colab.drive.mount('/content/drive')` in a Colab-specific setup cell.
- Default expected file:
  `/content/drive/MyDrive/Customer Churn Prediction/train.csv`
- Check whether the expected file exists before attempting to load it.
- If it is missing, show a clear, actionable error listing the expected path and explaining where to place `train.csv`.
- Use `pathlib.Path` for readable path handling.
- Read the CSV with pandas.
- Print the loaded file path, dataset shape, column names, and a small preview.
- Validate that the `Exited` target exists before supervised modeling. If it does not, stop with a clear message rather than continuing incorrectly.
- Check for duplicate column names, unexpected target values, empty data, and obvious schema problems.
- Do not silently switch to a different file or download a dataset from an unrelated source.
- Optionally provide a clearly labeled local-development fallback for `data/train.csv`, but keep the Drive path as the default in Colab.
- If the assignment's Kaggle competition test file is available, locate it explicitly and load it separately. Its exact filename and columns must be inspected rather than assumed.

### Reproducibility

- Define one documented random seed (for example, `RANDOM_STATE = 42`) and use it consistently where applicable.
- Display relevant package versions.
- Record important model settings and preprocessing choices.
- Avoid nondeterministic behavior where practical, while explaining that some library/compute operations may still vary.
- Notebook execution order must not depend on hidden variables left over from a previous session.

### Dependencies

Use the assignment's required stack:

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Use standard Colab-installed packages where possible. Install packages only if necessary and explain why. No text-vectorization workflow is needed unless the actual dataset contains a text feature that requires it; the assignment's bank-customer task is not an SMS classification task.

---

## 5. Required Notebook Structure

Create a polished, fully runnable notebook with Markdown explanations before important code sections. Each major section should explain **what is being done, why it is needed, how it works internally at a useful beginner level, and how to interpret the output**.

Recommended order:

1. Title, assignment metadata, and objectives
2. Problem definition and end-to-end ML workflow
3. Environment setup and reproducibility
4. Google Drive mounting and data loading
5. Initial data inspection and Task 1
6. Exploratory data analysis and Task 2
7. Identifier/leakage review and feature selection — Task 3
8. Stratified 70/15/15 split — Task 4
9. Training-only preprocessing pipeline — Task 5
10. Baseline model training and comparison — Task 6
11. Hyperparameter tuning using validation data — Task 7
12. Training-versus-validation generalization analysis — Task 8
13. One-time evaluation on the untouched internal test set — Task 9
14. Predictions for five unlabeled competition-test rows, if that file is available — Task 10
15. Final conclusions, unexpected findings, limitations, and reproducibility notes

Do not skip a task if a dataset detail is different from what the assignment anticipates. Instead, explain the discrepancy and implement the closest methodologically valid approach.

---

## 6. Detailed Assignment Requirements

### Task 1 — Load and inspect the labeled training data

Required:

- Load `train.csv`.
- Show its shape and a preview.
- List actual column names and data types.
- Inspect the `Exited` distribution using counts and percentages.
- Provide relevant descriptive output.
- State at least one observation based on the actual results.

Also inspect:

- Missing values per column
- Duplicate rows
- Unique values of the target
- Numerical and categorical columns
- Potential identifier columns

Do not write a fixed observation in advance. Generate the observation from the real output, for example whether the target appears imbalanced, only if the actual counts support it.

**Expected deliverable:** Relevant printed output/table/figure and a short evidence-based observation.

### Task 2 — Exploratory Data Analysis (EDA)

Explore churn patterns across geography, age, account-related features, and tenure **where those fields actually exist**.

Required:

- Descriptive statistics.
- Missing-value counts.
- Target distribution.
- At least **three informative visualizations**.
- Clear labels, readable titles, legends where needed, and sensible axis formatting.
- At least **three observations supported by the plots**.

Useful plots, selected according to actual columns:

- Count/bar plot of `Exited` distribution.
- Churn rate by geography/country/region.
- Churn rate by age groups.
- Churn rate by tenure or another account-related feature.
- Churn rate by product/activity/balance features, if present.
- Numerical feature distributions grouped by `Exited`.

Guidelines:

- For churn rates, compute the mean of the binary `Exited` target within each group and display it as a percentage.
- Show group sample sizes where useful; tiny groups can make rates misleading.
- Use sensible age bins only if an age column exists, and document the bin boundaries.
- Avoid plotting every column just to increase plot count. Choose informative visuals.
- Do not infer causation from correlations or group-level differences.
- Mention any class imbalance, missingness, skew, outliers, or group-size limitations supported by the data.

**Expected deliverable:** EDA tables, at least three labeled visualizations, and three data-supported insights.

### Task 3 — Exclude identifiers and investigate target leakage

Required:

- Identify and exclude identifier columns from model features.
- Exclude `Exited` from `X`; use it only as `y`.
- Review remaining columns for target leakage.
- Document each removal and the reason.
- Display the resulting feature list or schema and explain its significance.

Guidelines:

- Never assume every column named `id` is automatically harmless or that every unusual column leaks the target. Inspect actual semantics and dataset documentation if available.
- Identifiers may include unique customer/row IDs that do not represent a meaningful generalizable customer attribute.
- Target leakage means a feature exposes the target directly or contains information that would not legitimately be available at prediction time.
- Make the exclusion decision explicit in a table, such as `column`, `decision`, `reason`.
- Do not remove legitimate predictive features solely because they correlate with `Exited`.
- Preserve the original DataFrame and create a separate modeling feature set.
- If leakage cannot be established from the available evidence, describe it as a risk/uncertainty instead of stating it as fact.

**Expected deliverable:** Documented feature audit, resulting features, and an explanation.

### Task 4 — Stratified 70/15/15 split of labeled training data

Create three subsets from the labeled `train.csv`:

- **70% training**
- **15% validation**
- **15% internal test**

Use stratification based on `Exited` for both split stages so that class proportions are approximately preserved.

Recommended two-stage approach:

1. Split 70% training from 30% temporary data using `train_test_split(..., test_size=0.30, stratify=y, random_state=RANDOM_STATE)`.
2. Split the temporary 30% equally into validation and internal test, using stratification on the temporary labels and a fixed seed. This produces 15% + 15% of the full dataset.

Requirements:

- Split rows before fitting any imputers, encoders, scalers, feature selectors, or models.
- Keep the same row alignment between features and labels.
- Report each split's row count and percentage.
- Display `Exited` counts/rates per split in a table or graph.
- Explain why stratification is useful, especially when the target classes are imbalanced.
- State that the internal test split remains untouched during model selection and tuning.

**Expected deliverable:** Split-size/class-distribution output and a brief methodological justification.

### Task 5 — Missing-value imputation and categorical encoding without leakage

Implement preprocessing with scikit-learn pipelines/`ColumnTransformer`.

Required:

- Determine numerical and categorical columns from the training features.
- Numerical imputation: use a documented strategy, such as median imputation.
- Categorical imputation: use a documented strategy, such as most-frequent imputation or a suitable constant category.
- One-hot encode categorical columns.
- Use `handle_unknown='ignore'` for one-hot encoding so unseen validation/test categories do not crash the workflow.
- Use a model-appropriate numerical scaling strategy where needed. Logistic Regression generally benefits from scaling; tree-based models usually do not require it. Keep the design understandable and avoid applying unnecessary transformations without explanation.
- Fit preprocessing **only on the training split** through a Pipeline. Transform validation and internal test using the already-fitted pipeline; never fit preprocessing separately on validation/test.
- Ensure the same pipeline pattern is used during cross-validation/hyperparameter tuning so preprocessing is fitted inside each training fold.
- Record methods, settings, and package/library choices.
- Demonstrate leakage prevention in Markdown and/or a small verification output.

Explain internally:

- Why imputation is required.
- Why categorical values need encoding for these estimators.
- What one-hot encoding creates.
- Why the encoder must tolerate unseen categories.
- The difference between `fit`, `transform`, and `fit_transform`.
- Why fitting preprocessing before splitting leaks information.

If the actual dataset has no missing values, still implement a robust pipeline or report the finding and explain how the configured imputer would handle missing values if encountered.

**Expected deliverable:** Documented preprocessing pipeline and evidence that held-out rows did not influence preprocessing fitting.

### Task 6 — Train and compare three classifiers

Train these required algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

Use a preprocessing-plus-model Pipeline for each model so each algorithm receives correctly transformed data.

Requirements:

- Record model configurations and key settings.
- Train on the training split only.
- Evaluate and compare on the validation split.
- Include, at minimum, **Recall, F1-score, and ROC-AUC** as appropriate. Include precision and accuracy as additional context, but do not rely on accuracy alone.
- Calculate ROC-AUC using predicted probabilities or decision scores, not hard class predictions.
- Use an explicit, documented decision threshold for recall/F1 predictions (default 0.5 is acceptable as a starting point). Do not silently tune a threshold on the internal test set.
- Handle ROC-AUC safely if an evaluation set unexpectedly contains only one target class; report it as undefined rather than crashing.
- Produce a comparative results table and an interpretation grounded in actual results.
- Explain the high-level differences between a linear model and the two tree-based ensemble methods.
- If using `class_weight`, sampling, or any imbalance-specific method, explain why and ensure it is used only in the training process. Do not apply multiple imbalance remedies automatically without evidence.

**Expected deliverable:** Comparison table, recorded configurations, and interpretation.

### Task 7 — Tune at least two hyperparameters using validation data

Tune at least two meaningful hyperparameters. The experiment must use training/validation data only; do not use the internal test set to choose settings.

Possible examples:

- Logistic Regression: `C`, `class_weight` (where appropriate), or solver-compatible settings.
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, or `max_features`.
- Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth` through its base estimator where supported, or `min_samples_leaf` where supported by the chosen implementation.

Requirements:

- Define a small, computationally sensible search space for Colab.
- Tune at least two hyperparameters in a documented experiment. Prefer a principled search using training-only pipelines and cross-validation within the 70% training split, then compare/select using validation evidence. A manual validation experiment is acceptable if implemented rigorously and clearly documented.
- Use stratified CV for classification when feasible.
- Ensure preprocessing is inside the pipeline used for CV.
- Select the scoring metric deliberately; recall, F1, or ROC-AUC may be used depending on the stated objective. If using multiple metrics, explain the refit/selection rule.
- Record each tested setting and corresponding score in a table or plot.
- State the selected configuration and why it was chosen.
- Do not use internal-test performance to pick hyperparameters, models, or thresholds.

**Expected deliverable:** Experiment table/plot, selected settings, and evidence-based explanation.

### Task 8 — Compare training and validation Recall, F1, and ROC-AUC

For the selected/tuned model, compare:

- Training Recall
- Validation Recall
- Training F1-score
- Validation F1-score
- Training ROC-AUC
- Validation ROC-AUC

Requirements:

- Put the comparison in a clear table and preferably a grouped bar chart or another readable visualization.
- Calculate metrics consistently using the same threshold and positive-class definition.
- ROC-AUC should use probabilities/scores.
- Interpret train–validation gaps as possible evidence of overfitting/underfitting, not absolute proof.
- Discuss class imbalance and the trade-offs between false positives and false negatives where relevant.
- Acknowledge uncertainty, data representativeness, and metric limitations.

**Expected deliverable:** Diagnostic table/figure and a careful interpretation.

### Task 9 — Evaluate once on the untouched internal test split

This is the final internal generalization check.

Required:

- Finalize model/configuration and any threshold using training/validation data first.
- Evaluate the chosen model **once** on the internal test split for the final reported assessment.
- Report Recall, F1-score, ROC-AUC, and other useful metrics such as precision and accuracy.
- Show a clearly labeled confusion matrix with class names/labels.
- Explain the confusion-matrix cells (true positives, true negatives, false positives, false negatives) in the context of customer churn.
- Interpret results and discuss limitations.
- Do not go back and tune the model based on internal-test results. If a change is made after viewing the test metrics, the test set is no longer a clean untouched final assessment and this must be acknowledged.

**Expected deliverable:** Final results table, confusion matrix, visualization, and interpretation.

### Task 10 — Predict churn risk for five unlabeled competition-test rows

- If the competition provides an unlabeled `test.csv`, load it separately and inspect its actual columns.
- Do not attempt to calculate supervised evaluation metrics on an unlabeled file.
- Apply the already-fitted final pipeline/model; do not refit preprocessing or the model on the competition-test rows.
- Align feature columns to the model's expected inputs and handle missing/unseen categories through the fitted pipeline.
- Remove identifiers and other excluded columns from model inputs, but retain identifiers separately for reporting/submission if present.
- Predict churn labels and probabilities for the first five valid rows (or five representative rows if fewer than five are available).
- Show a small table containing an identifier when available, predicted label, and estimated churn probability/risk.
- Explain that a probability is a model estimate, not a guarantee or a calibrated real-world certainty.
- Do not invent the competition-test file or pretend it exists. If it is unavailable, clearly mark this task as blocked by the missing file, explain how to supply it, and keep the rest of the notebook runnable.
- Summarize key findings, unexpected outcomes, and at least **two real-world limitations**. Consider dataset representativeness, changing customer behavior, fairness, probability calibration, missing business costs, and the difference between prediction and causal explanation.

**Expected deliverable:** Five-row prediction table if the unlabeled file exists, otherwise a clear limitation note; plus final conclusion and two limitations.

---

## 7. Evaluation and Modeling Rules

### Leakage prevention

This workflow must maintain strict boundaries:

- **Training split:** fit imputation, encoding, scaling, and model parameters.
- **Validation split:** compare models, assess training/validation generalization, and select hyperparameters or thresholds if the chosen procedure requires it.
- **Internal test split:** one-time final assessment after choices are finalized.
- **Competition test data:** inference only when labels are absent.

Do not:

- Fit preprocessing on the full labeled dataset before splitting.
- Select features or thresholds using internal-test labels.
- Use internal-test scores to select the best model.
- Report validation results as final unbiased test results.
- Treat unlabeled competition-test predictions as accuracy or evaluation results.

### Metrics

Explain each metric in beginner-friendly language:

- **Accuracy:** fraction of all predictions that are correct; may be misleading with class imbalance.
- **Precision:** among customers predicted to churn, the fraction that actually churned.
- **Recall:** among customers who actually churned, the fraction identified by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** how well the model ranks positive cases above negative cases across thresholds; it is not tied to one fixed threshold.
- **Confusion matrix:** counts of correct/incorrect predictions by class.

Make clear which class is positive (`Exited = 1`) and state the prediction threshold used for threshold-dependent metrics.

### Model-selection discipline

- Choose the selection metric before comparing results, with a short reason.
- Keep validation and test roles distinct.
- If tuning via cross-validation, CV folds must be drawn only from the training split.
- Use the validation split to compare candidates/finalize decisions.
- Evaluate the internal test split only after the model is selected.
- Do not promise a specific score or claim one algorithm will definitely be best.

---

## 8. Required `docs/` Documentation

The `docs/` folder is a major deliverable, not an optional afterthought. Write detailed, coherent Markdown documents that explain both **what the implementation does** and **what happens internally**. Assume the reader is a beginner who understands basic Python but is still learning machine learning and scikit-learn.

### `docs/01_assignment_and_workflow.md`

Include:

- Assignment objective and problem statement.
- Dataset and target definition.
- The ten tasks mapped to the notebook sections.
- An end-to-end workflow diagram using Mermaid or plain Markdown.
- Why the order of operations matters.
- A checklist mapping each assignment requirement to a notebook section and its output.

### `docs/02_dataset_and_eda.md`

Include:

- How the CSV is loaded and inspected.
- How to interpret rows, columns, data types, descriptive statistics, and missing-value summaries.
- How the `Exited` distribution is calculated.
- The actual discovered dataset schema, generated from inspection where possible.
- Explanation of each EDA chart: question, calculation, axes, interpretation, and caveats.
- The three evidence-based observations from the actual plots.
- Any schema discrepancies or unexpected findings.
- Do not invent feature descriptions if the column meaning is unknown.

### `docs/03_preprocessing_and_data_leakage.md`

Include:

- Feature/target separation.
- Identifier and leakage review, with a decision table.
- Stratified 70/15/15 split mathematics and implementation.
- Numerical/categorical feature detection.
- Imputation, one-hot encoding, scaling where relevant, `ColumnTransformer`, and `Pipeline`.
- Internal behavior of `fit`, `transform`, and `fit_transform`.
- A concrete explanation of how preprocessing before splitting leaks information.
- How unseen categories are handled.
- How to verify that held-out data does not participate in fitting.

### `docs/04_models_and_hyperparameter_tuning.md`

Include:

- Beginner-friendly explanation of Logistic Regression, Random Forest, and Gradient Boosting.
- What each model learns, how predictions are generated at a high level, strengths, limitations, and important settings.
- Why pipelines are used with each model.
- Hyperparameter versus model parameter.
- The selected tuning strategy, search space, scoring metric, cross-validation, and refit behavior.
- Actual experiment results and the reason for the chosen configuration.
- Avoid generic claims that are contradicted by the measured results.

### `docs/05_evaluation_and_interpretation.md`

Include:

- Definitions and interpretation of Recall, Precision, F1, Accuracy, ROC-AUC, and confusion matrix.
- Why accuracy can be misleading under class imbalance.
- Actual training/validation diagnostic comparison.
- Actual internal-test results and confusion matrix interpretation.
- False-positive and false-negative implications for a bank.
- The distinction between association, prediction, causation, and calibrated risk.
- Limitations and potential improvements.
- Clearly distinguish observations from speculation.

### `docs/06_internal_implementation_guide.md`

This is the most important conceptual guide. Explain the actual code and its internal behavior in detail, step by step.

For each major notebook stage, explain:

1. What the code is trying to achieve.
2. Why that step is necessary.
3. Which inputs it receives and outputs it creates.
4. How the relevant pandas/scikit-learn functions work internally at a useful conceptual level.
5. What is learned from data during `fit`.
6. What is merely applied during `transform`/`predict`.
7. What can go wrong and how the implementation avoids it.
8. How to read the output and debug common errors.

Cover at least:

- Drive mounting and file-path checks.
- `pandas.read_csv`, DataFrame shape, dtypes, `describe`, `isna`, `value_counts`, and groupby-based churn rates.
- Matplotlib/Seaborn plotting logic.
- `train_test_split` and stratification.
- `SimpleImputer`, `OneHotEncoder`, `StandardScaler` where used.
- `ColumnTransformer` and `Pipeline`, including how the components are chained.
- Logistic Regression, Random Forest, Gradient Boosting.
- Model fitting, prediction, predicted probabilities, and decision thresholds.
- Recall/F1/ROC-AUC calculations and the confusion matrix.
- Hyperparameter search and cross-validation if used.
- How the untouched internal test set is protected.
- How predictions are made for the unlabeled competition-test file.
- Common errors: missing file, absent target column, unexpected categories, missing values, invalid parameter combinations, class-imbalance pitfalls, and ROC-AUC errors when only one class is present.

Use small code excerpts from the actual implementation where they make the explanation clearer. Keep code snippets synchronized with the final notebook; do not document functions or parameters that the project does not use as if they were implemented.

### `docs/07_colab_and_google_drive_guide.md`

Include:

- How to open the notebook in Colab.
- How to authorize/mount Drive.
- Exact expected dataset path.
- How to fix a missing-file error.
- How to run all cells from top to bottom.
- How to confirm the correct CSV loaded.
- What outputs and figures should appear.
- How to handle a missing optional competition `test.csv`.
- How to download/export the notebook and save outputs.
- Reproducibility notes and troubleshooting.

### Documentation quality rules

- Write substantive explanations, not a table of contents disguised as documentation.
- Explain concepts in clear, correct language suitable for a beginner.
- Use consistent names for data splits, target, features, metrics, and models.
- Include Mermaid diagrams where useful, but provide enough prose that the documents remain understandable if Mermaid does not render.
- Include actual results only after running the code on the real data. If execution is not possible, write clearly labeled placeholders/instructions rather than fabricating results.
- Ensure all claims, code examples, file paths, and model configurations match the notebook.

---

## 9. README and `CLAUDE.md`

### `README.md`

Provide:

- Project overview.
- Dataset source and Drive path.
- Notebook location.
- Brief setup/run instructions.
- A summary of the ML workflow.
- Links to all detailed docs.
- A note that generated metrics depend on the actual dataset and execution.

### `CLAUDE.md`

Keep it concise and operational. Tell future Claude Code sessions to:

- Read this implementation brief and the docs before making major changes.
- Preserve assignment requirements.
- Inspect the real dataset schema instead of assuming column names.
- Keep the workflow Colab-compatible and Drive-first.
- Prevent data leakage.
- Never use the internal test set for tuning.
- Keep documentation synchronized with code.
- Never invent execution results.
- Run available tests/checks and report what was and was not executed.

This current file is the primary implementation specification; `CLAUDE.md` should point back to it rather than duplicate every detail.

---

## 10. Code Quality and Presentation

- Use clear variable names such as `X`, `y`, `X_train`, `X_val`, `X_test`, and matching label variables consistently.
- Use a consistent naming scheme for the validation and internal test splits. Avoid calling both of them simply `test`.
- Use functions for repeated tasks such as metric calculation or plotting when that improves readability.
- Use formatted tables for comparisons.
- Every chart must have a title, labeled axes, and readable tick labels.
- Keep charts visually clean and appropriate for a lab report.
- Avoid excessive output, repeated plots, decorative charts without analytical value, and unexplained magic numbers.
- Set seeds and document model configurations.
- Make notebook cells safe to rerun where practical.
- Handle edge cases gracefully and show actionable error messages.
- Avoid suppressing important warnings indiscriminately.
- Do not use test-set outcomes to make modeling decisions.

---

## 11. Final Verification Checklist

Before declaring the task complete, verify the following.

### Assignment coverage

- [ ] Task 1: actual dataset loaded and inspected; target distribution shown; at least one evidence-based observation.
- [ ] Task 2: descriptive statistics, missing-value counts, target distribution, at least three visualizations, and at least three plot-supported observations.
- [ ] Task 3: identifiers/leakage reviewed; excluded columns and reasons documented.
- [ ] Task 4: stratified 70/15/15 split; split sizes and target distributions shown.
- [ ] Task 5: imputation and one-hot encoding in training-only pipelines; leakage prevention explained.
- [ ] Task 6: Logistic Regression, Random Forest, and Gradient Boosting trained and compared.
- [ ] Task 7: at least two hyperparameters tuned using training/validation evidence, not internal-test results.
- [ ] Task 8: training-versus-validation Recall, F1, and ROC-AUC compared and interpreted.
- [ ] Task 9: selected model evaluated once on untouched internal test data; confusion matrix shown.
- [ ] Task 10: predictions for five unlabeled competition-test rows if the file exists; otherwise clear explanation; conclusion and at least two real-world limitations.

### Technical verification

- [ ] Notebook can run top-to-bottom in Colab after Drive authorization.
- [ ] Default Drive path is `/content/drive/MyDrive/Customer Churn Prediction/train.csv`.
- [ ] Actual columns and target are validated at runtime.
- [ ] Preprocessing is fitted only on the appropriate training data.
- [ ] CV, if used, includes preprocessing inside the pipeline.
- [ ] No internal-test leakage into tuning, model selection, or threshold selection.
- [ ] ROC-AUC uses scores/probabilities and handles single-class edge cases.
- [ ] All required plots have titles and labeled axes.
- [ ] Random seed and key configurations are recorded.
- [ ] Missing optional competition-test file does not break earlier tasks.

### Documentation verification

- [ ] All seven requested docs exist and contain substantive explanations.
- [ ] README has accurate run instructions and doc links.
- [ ] `CLAUDE.md` contains concise persistent agent rules.
- [ ] Notebook and documentation agree on code, model settings, and metric names.
- [ ] No fabricated results or unsupported dataset claims.
- [ ] Data files are not accidentally committed.

---

## 12. Final Deliverables

The completed project should include:

1. A fully organized Google Colab-compatible notebook implementing all feasible assignment tasks.
2. A root `README.md` explaining the project and how to run it.
3. A root `CLAUDE.md` with concise instructions for future Claude Code sessions.
4. The seven detailed Markdown documents listed under `docs/`.
5. Clean, labeled tables and visualizations generated by actual execution.
6. A clear final conclusion with actual findings and at least two real-world limitations.
7. No fabricated metrics, data observations, or claims of successful execution that have not been verified.

## 13. Final Instruction to the Agent

Start by inspecting the existing project files and the actual `data/train.csv` if available. Then implement the notebook and documentation systematically. Follow the assignment's ten tasks in order, preserve the training/validation/internal-test boundaries, and make Google Drive the default data source in Colab. The work is complete only when the notebook, documentation, and README are consistent and the final verification checklist has been addressed. If a requirement cannot be verified or a dataset file is unavailable, state that explicitly and provide the next action needed—never fabricate a result.
