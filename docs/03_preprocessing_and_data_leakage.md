# 03 — Preprocessing and Data Leakage

## Feature/target separation and the leakage audit
`X` = input columns, `y = df['Exited']`. `X` is built with `df.drop(columns=[...])`, which returns a **new** DataFrame, so the original `df` is preserved.

| Column | Decision | Reason |
|---|---|---|
| id | EXCLUDE | sequential row index (verified: consecutive integers); no generalisable meaning, lets flexible models memorise rows |
| CustomerId | EXCLUDE | identifier; arbitrary number. 6,349 distinct values over 15,000 rows, so it is *not* a unique row key |
| Surname | EXCLUDE | 778 distinct text values; an arbitrary label and a fairness/privacy concern |
| Exited | TARGET | outcome; only used as `y` |
| CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary | KEEP | no evidence of leakage |

Evidence for "no leakage": the largest |correlation| between a kept numeric feature and the target is Age at 0.452 (then NumOfProducts 0.311, IsActiveMember 0.194); none is near 1, i.e. no feature is a copy of the label. **Limits of this evidence:** leakage can also be *temporal* (a feature recorded after the customer left, e.g. a final balance of 0 or "inactive" status). The CSV cannot prove when each field was recorded, so this is an **assumption/risk** to confirm with the bank. Features are not dropped merely for being predictive (Age, NumOfProducts).

**Duplicate-customer risk.** Because `CustomerId` repeats, the same person can appear in training and validation/test rows. If that person's rows are similar, held-out scores can be slightly optimistic. A grouped split (e.g. `GroupShuffleSplit` on `CustomerId`) would address it but the assignment specifies a stratified split, and rows with the same ID may not truly be the same person (the dataset looks synthetic), so it is documented rather than altered.

## The stratified 70/15/15 split
`train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)` shuffles row indices with a seeded random generator and, for stratification, samples within each class so each side gets (almost) the same class share. Stage 2 splits the 30% temporary part 50/50 with `stratify=y_temp`, giving 15% + 15% of the full data.

Arithmetic: 15,000 × 0.70 = 10,500; 4,500 × 0.5 = 2,250 each. Actual result:

| Split | Rows | Share | Churners | Churn rate |
|---|---|---|---|---|
| train | 10,500 | 70% | 2,132 | 20.30% |
| validation | 2,250 | 15% | 457 | 20.31% |
| internal test | 2,250 | 15% | 457 | 20.31% |

(full data: 20.31%). Without stratification, a random split of a 20/80 target could by chance put noticeably more or fewer churners in one subset, making scores incomparable. The notebook asserts the three index sets are disjoint, add up to all rows, and that `X`/`y` indices align. The variable names `X_val`/`y_val` (validation) and `X_test`/`y_test` (internal test) are never interchanged.

## Feature types
`select_dtypes(include='number')` gives numeric columns; the rest are categorical. Result: numeric = CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary; categorical = Geography, Gender. The 0/1 flags are numeric floats and are kept numeric (already a valid binary encoding).

## The preprocessing components
* **`SimpleImputer(strategy='median')`** (numeric) / **`most_frequent`** (categorical): `fit` learns one fill value per column from the data it is given (`statistics_`); `transform` replaces NaNs with that value. Logistic Regression raises an error on NaN (tree ensembles have partial NaN support depending on the scikit-learn version), and the scaler/encoder assume complete data, so an explicit strategy is the safe, uniform choice. This dataset has **0 missing values**, so the imputers currently change nothing; they are kept so the pipeline would still work for data with gaps (e.g. a competition file with NaNs).
* **`OneHotEncoder(handle_unknown='ignore')`**: `fit` learns the list of categories per column (`categories_`); `transform` outputs one 0/1 column per category (Geography → France/Germany/Spain, Gender → Female/Male; 5 columns). Text cannot be fed to the models, and plain integer codes (France=0, Germany=1, Spain=2) would impose a false ordering on linear models. `handle_unknown='ignore'` encodes a never-seen category as all zeros instead of raising an error; the notebook proves this with a probe row `Geography='Atlantis'`.
* **`StandardScaler`** (Logistic Regression pipeline only): `fit` learns each column's mean and standard deviation (`mean_`, `scale_`); `transform` outputs `(x − mean)/std`. Logistic Regression's regularised optimisation is sensitive to feature magnitudes (Balance ~10⁵ vs HasCrCard 0/1). Trees split on thresholds, which are unaffected by monotone rescaling, so Random Forest and Gradient Boosting skip it.
* **`ColumnTransformer`**: applies different sub-pipelines to different column lists and concatenates the outputs (13 columns here: 8 numeric + 5 one-hot).
* **`Pipeline`**: chains `preprocess → model`. `pipe.fit(X, y)` calls `fit_transform` on the preprocessor then `fit` on the model; `pipe.predict_proba(X_new)` calls only `transform` then the model's `predict_proba`.

## fit vs transform vs fit_transform
* `fit(X)` — *learn* statistics (medians, means, categories) from X.
* `transform(X)` — *apply* already-learned statistics; learns nothing.
* `fit_transform(X)` — `fit` then `transform` on the same X; only legitimate on training data.

## How preprocessing before splitting leaks information
Example from the actual data: the median `EstimatedSalary` is 122,929.6 in the training split but 122,964.2 on all rows; the mean `Balance` is 42,506 (train) vs 42,625 (all). The "all rows" numbers contain information about validation/test rows. The differences here are small, but the principle matters more with outliers, rare categories or heavy tuning: the model would be evaluated on data that helped shape its inputs, so scores overestimate real-world performance. In a pipeline, the statistics come only from the rows passed to `fit`.

Inside cross-validation, `GridSearchCV` clones the whole pipeline for each fold and fits it on the fold's training part, so the imputer/scaler/encoder never see the fold's held-out part. This is why preprocessing is part of the pipeline rather than done once up front.

## Verifying held-out data does not participate in fitting
Notebook §7 fits the preprocessor on `X_train` only and checks with `np.allclose` that the learned `imputer.statistics_` and `scaler.mean_` equal the training-split medians/means, and displays them next to the (different) full-data values. It then transforms validation rows (transform only) and shows the unseen-category probe encodes without error.

## Troubleshooting
* `ValueError: Input contains NaN` → a column escaped the imputers (e.g. a new column not in the lists).
* `ValueError: columns are missing` → inference data lacks a column used in training; select `X_train.columns` in order.
* Different `X_train` columns after changing `EXCLUDE_COLS` change everything downstream; re-run the notebook from the top.
