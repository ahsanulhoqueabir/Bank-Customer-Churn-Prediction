# 06 — Internal Implementation Guide

A stage-by-stage walkthrough of `notebooks/bank_customer_churn_prediction.ipynb`. Code excerpts are copied from the notebook.

## Stage 1 — Setup and reproducibility (§1)
Imports, then constants: `RANDOM_STATE = 42`, `THRESHOLD_DEFAULT = 0.5`, `TARGET = "Exited"`. The seed is passed explicitly to every random component (`train_test_split`, `StratifiedKFold`, models), so reruns on the same library versions give the same splits and models (verified: two executions produced identical results). Versions are printed (local run: Python 3.13.5, pandas 2.2.3, scikit-learn 1.6.1); Colab versions may differ slightly and can shift results a little.

## Stage 2 — Drive mount and file checks (§2)
```python
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
if IN_COLAB:
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/Customer Churn Prediction")
else:
    DATA_DIR = root / "data"       # local development fallback only
```
`drive.mount` opens an OAuth consent flow and exposes your Drive as a FUSE file system at `/content/drive`, so ordinary file functions work. `Path(...) / "train.csv"` builds a path portably; `.exists()` is checked **before** reading, raising `FileNotFoundError` with the expected path and instructions if missing. After `read_csv`, the asserts check: non-empty, unique column names, target present, no missing target, values ⊆ {0,1}. Each assert message tells you what to fix. The local fallback is chosen only when *not* in Colab; Colab never silently falls back.
*Not executed here:* the `drive.mount` branch (no Colab in the dev environment).

## Stage 3 — Inspection (§3)
* `df.dtypes`, `df.nunique()`, `df.isna().sum()` return Series indexed by column name; `join` combines them into one table.
* `value_counts()` tallies unique values; `sort_index()` orders 0 then 1.
* Identifier heuristic: a column is "identifier-like" if its name contains "id" or more than 30% of its values are distinct. It is a screening heuristic (it also flagged `EstimatedSalary`, a legitimate feature), not a decision; Stage 5 decides.
* The printed observation is built from the counts (f-string), so it changes if the data change.

## Stage 4 — EDA (§4)
`frame.groupby(by).agg(n="size", churn_rate="mean")` splits rows into groups, then counts rows and averages the 0/1 target inside each group (= churn rate). `pd.cut(Age, bins=[17,30,40,50,60,100])` assigns each age to a right-closed interval. Matplotlib/Seaborn: `plt.subplots` creates a grid of axes; `sns.barplot` draws one bar per category (bar height = the y value given); `ax.text` annotates bars; `ax.axhline` draws the overall rate; `sns.histplot(..., stat="density", common_norm=False)` normalises each class separately so unequal class sizes do not hide differences. Every axis has a title and axis labels. If a plot errors with a categorical-axis message, convert the group labels to strings (done in `rate_bar`).

## Stage 5 — Feature audit (§5)
Builds a decision DataFrame, prints evidence (CustomerId duplicates, id sequential, correlations from `corrwith`), then:
```python
EXCLUDE_COLS = ["id", "CustomerId", "Surname"]
X = df.drop(columns=EXCLUDE_COLS + [TARGET]); y = df[TARGET]
```
`drop` returns a copy; `df` remains complete.

## Stage 6 — Split (§6)
```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
```
Internally the function takes row positions, shuffles them with a `RandomState(42)`, and with `stratify` allocates each class's rows proportionally. Features and labels are indexed together, so alignment is preserved; the asserts confirm it. **Protecting the test set:** `X_test`/`y_test` are not referenced by any code between this cell and §11 (Task 9); the CV in §9 sees only `X_train`.

## Stage 7 — Preprocessing (§7)
`make_preprocessor(scale_numeric)` returns a `ColumnTransformer` with a numeric sub-pipeline (`SimpleImputer(median)`, optional `StandardScaler`) and a categorical one (`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`). `make_pipeline(model, scale_numeric)` prepends it to a model.
* **fit learns:** medians (`statistics_`), means/stds (`mean_`, `scale_`), category lists (`categories_`).
* **transform applies:** the same numbers to any rows. `ColumnTransformer.transform` routes each column list through its sub-pipeline and horizontally stacks results (13 output columns; names via `get_feature_names_out()`).
* The verification block fits `make_preprocessor(True).fit(X_train)` and asserts the learned numbers equal the training-split statistics; it transforms `X_val` (transform only) and an `"Atlantis"` row (all-zero one-hot).
Pitfalls: column lists are computed from `X_train`; if you add a column later, rebuild. Scaling placed outside the pipeline would leak.

## Stage 8 — Models and metrics (§8)
`pipe.fit(X_train, y_train)`: preprocessing `fit_transform`, then the model's `fit`. `pipe.predict_proba(X)[:, 1]` returns P(class 1) (column 0 is class 0; classes are ordered 0,1). Hard predictions: `(proba >= threshold).astype(int)` — making the threshold explicit instead of relying on `predict` (which uses 0.5 for these models). `evaluate()` returns Accuracy, Precision, Recall, F1 (with `zero_division=0` so an all-negative prediction gives 0 rather than a warning/error) and ROC-AUC from probabilities through `safe_roc_auc`, which returns NaN if `len(np.unique(y_true)) < 2`. `roc_auc_score` raises `ValueError` otherwise.
How the models compute (see doc 04): LR = sigmoid of a weighted sum; RF = average of tree probabilities; GB = sum of scaled tree outputs → sigmoid.

## Stage 9 — Tuning (§9)
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(base, param_grids[name], scoring="roc_auc", cv=cv, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
```
For each of the grid combinations × 5 folds: clone the unfitted pipeline, set parameters (`model__C` addresses the `model` step), fit on 4 folds, compute ROC-AUC on the 5th. `cv_results_` holds mean/std per setting; `best_params_` and `best_score_` the winner; `refit=True` fits `best_estimator_` on all of `X_train`. Because the *pipeline* is cloned, imputer/scaler/encoder are re-learned inside each fold — no fold leakage. Then validation scores of the three tuned models choose `final_name`; the validation F1 sweep over thresholds 0.10–0.90 picks `FINAL_THRESHOLD`. `ValueError: Invalid parameter` means a grid key does not match a step name (`model__...`). Large grids are slow; reduce them.

## Stage 10 — Train vs validation (§10)
Calls `evaluate(final_model, X_train, y_train, FINAL_THRESHOLD)` and the same on validation, builds a table and grouped bar chart (`DataFrame.plot(kind="bar")`, `bar_label` for values). The gap column is a diagnostic only.

## Stage 11 — Internal test (§11)
One `predict_proba` on `X_test`, metrics, and `confusion_matrix(y_test, test_pred, labels=[0, 1])` → `ravel()` yields `tn, fp, fn, tp` in that order. `ConfusionMatrixDisplay` draws it with class names. The cell comment states that changes after this point invalidate the test set's status.

## Stage 12 — Competition rows (§12)
If `test.csv` is absent: print an actionable message and move on. Otherwise: read it, check it has all columns in `X_train.columns`, assert there is no `Exited` column, keep `id` aside, select `comp[X_train.columns]` (same order), take the first five rows and call `final_model.predict_proba` — **no fit/refit** of anything. Missing values would be imputed with the stored training medians/modes; unseen categories become all-zero one-hot. The first five ids in the local `data/test.csv` are 15000–15004. No metrics are computed because there are no labels. (The first-five rows are used as written in the assignment; "valid" rows = no missing required columns, which holds for the local file.)

## Stage 13 — Conclusions (§13)
Printed summary is assembled from variables created earlier, so it cannot drift from results. Limitations are written text.

## Common errors cheat-sheet
| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: train.csv not found` | wrong Drive folder/name | create `MyDrive/Customer Churn Prediction/` and put `train.csv` there |
| `AssertionError: Target column 'Exited' missing` | wrong file | check you loaded the labeled train file |
| `ValueError: Input contains NaN` | NaNs in a column not covered by the pipeline | add the column to the pipeline lists |
| `ValueError: Invalid parameter ...` | bad `param_grid` key | use `model__<param>` |
| `ValueError: Only one class present` | ROC-AUC on a single-class set | handled by `safe_roc_auc` (NaN) |
| unseen category | new value at inference | handled by `handle_unknown="ignore"` |
| Recall very low with high accuracy | class imbalance | look at the confusion matrix; consider threshold / class weights using validation data |
