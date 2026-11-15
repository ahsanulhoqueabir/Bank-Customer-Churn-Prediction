# 06 — Internal Implementation Guide | অভ্যন্তরীণ বাস্তবায়ন নির্দেশিকা

> **📌 এক নজরে (At a glance)**
> `notebooks/bank_customer_churn_prediction.ipynb`-এর ১৩টি ধাপের পরপর ব্যাখ্যা। কোড-অংশগুলো নোটবুক থেকে হুবহু নেওয়া; প্রতিটি ধাপের সঙ্গে বাংলা ও English ব্যাখ্যা আছে।
> *A stage-by-stage walkthrough of the notebook. Code excerpts are copied from the notebook.*

## Stage 1 — Setup and reproducibility (§1) | সেটআপ ও পুনরুৎপাদনযোগ্যতা

**বাংলা:** import-এর পর ধ্রুবকগুলো: `RANDOM_STATE = 42`, `THRESHOLD_DEFAULT = 0.5`, `TARGET = "Exited"`। seed প্রতিটি random উপাদানে (`train_test_split`, `StratifiedKFold`, মডেল) স্পষ্টভাবে দেওয়া হয়, ফলে একই library সংস্করণে পুনরায় চালালে একই split ও মডেল মেলে (যাচাইকৃত: দুটি execution অভিন্ন ফল দিয়েছে)। সংস্করণ প্রিন্ট হয় (লোকাল রান: Python 3.13.5, pandas 2.2.3, scikit-learn 1.6.1); Colab-এর সংস্করণ কিছুটা আলাদা হতে পারে এবং ফলে সামান্য পার্থক্য আনতে পারে।
**English:** Imports, then constants: `RANDOM_STATE = 42`, `THRESHOLD_DEFAULT = 0.5`, `TARGET = "Exited"`. The seed is passed explicitly to every random component (`train_test_split`, `StratifiedKFold`, models), so reruns on the same library versions give the same splits and models (verified: two executions produced identical results). Versions are printed (local run: Python 3.13.5, pandas 2.2.3, scikit-learn 1.6.1); Colab versions may differ slightly and can shift results a little.

## Stage 2 — Drive mount and file checks (§2) | Drive মাউন্ট ও ফাইল যাচাই

```python
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
if IN_COLAB:
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/Customer Churn Prediction")
else:
    DATA_DIR = root / "data"       # local development fallback only
```

**বাংলা:** `drive.mount` একটি OAuth সম্মতি-প্রবাহ খোলে এবং আপনার Drive-কে `/content/drive`-এ FUSE ফাইল-সিস্টেম হিসেবে প্রকাশ করে, ফলে সাধারণ ফাইল-ফাংশন কাজ করে। `Path(...) / "train.csv"` বহনযোগ্যভাবে পথ বানায়; পড়ার **আগে** `.exists()` দেখা হয় এবং ফাইল না থাকলে প্রত্যাশিত পথ ও নির্দেশনাসহ `FileNotFoundError` ওঠে। `read_csv`-এর পর assert যাচাই করে: ডেটা খালি নয়, কলামের নাম অনন্য, টার্গেট আছে, টার্গেটে missing নেই, মান ⊆ {0,1}। প্রতিটি assert বার্তা বলে দেয় কী ঠিক করতে হবে। লোকাল fallback কেবল Colab-এ *না* থাকলে বাছা হয়; Colab কখনো নীরবে fallback করে না।
*এখানে চালানো হয়নি:* `drive.mount` শাখা (ডেভেলপমেন্ট পরিবেশে Colab নেই)।
**English:** `drive.mount` opens an OAuth consent flow and exposes your Drive as a FUSE file system at `/content/drive`, so ordinary file functions work. `Path(...) / "train.csv"` builds a path portably; `.exists()` is checked **before** reading, raising `FileNotFoundError` with the expected path and instructions if missing. After `read_csv`, the asserts check: non-empty, unique column names, target present, no missing target, values ⊆ {0,1}. Each assert message tells you what to fix. The local fallback is chosen only when *not* in Colab; Colab never silently falls back.
*Not executed here:* the `drive.mount` branch (no Colab in the dev environment).

## Stage 3 — Inspection (§3) | পরিদর্শন

**বাংলা:**
* `df.dtypes`, `df.nunique()`, `df.isna().sum()` কলামের নামে সূচিত Series ফেরত দেয়; `join` সেগুলোকে একটি টেবিলে জোড়ে।
* `value_counts()` স্বতন্ত্র মান গোনে; `sort_index()` ০ তারপর ১ ক্রমে সাজায়।
* Identifier heuristic: কলামের নামে "id" থাকলে বা মানের ৩০%-এর বেশি স্বতন্ত্র হলে সেটিকে "identifier-সদৃশ" ধরা হয়। এটি স্ক্রিনিং হিউরিস্টিক (এটি `EstimatedSalary`-কেও চিহ্নিত করেছে, যা বৈধ ফিচার), সিদ্ধান্ত নয়; সিদ্ধান্ত Stage 5-এ।
* প্রিন্ট-করা পর্যবেক্ষণ গণনা থেকে f-string দিয়ে তৈরি, তাই ডেটা বদলালে তা-ও বদলায়।

**English:**
* `df.dtypes`, `df.nunique()`, `df.isna().sum()` return Series indexed by column name; `join` combines them into one table.
* `value_counts()` tallies unique values; `sort_index()` orders 0 then 1.
* Identifier heuristic: a column is "identifier-like" if its name contains "id" or more than 30% of its values are distinct. It is a screening heuristic (it also flagged `EstimatedSalary`, a legitimate feature), not a decision; Stage 5 decides.
* The printed observation is built from the counts (f-string), so it changes if the data change.

## Stage 4 — EDA (§4) | এক্সপ্লোরেটরি বিশ্লেষণ

**বাংলা:** `frame.groupby(by).agg(n="size", churn_rate="mean")` সারিগুলোকে গ্রুপে ভাগ করে, গ্রুপের আকার গোনে এবং ০/১ টার্গেটের গড় (= churn rate) বের করে। `pd.cut(Age, bins=[17,30,40,50,60,100])` প্রতিটি বয়সকে right-closed ব্যবধানে বসায়। Matplotlib/Seaborn: `plt.subplots` অক্ষের গ্রিড বানায়; `sns.barplot` প্রতি category-র জন্য একটি বার আঁকে (বারের উচ্চতা = দেওয়া y মান); `ax.text` বার চিহ্নিত করে; `ax.axhline` সামগ্রিক হার আঁকে; `sns.histplot(..., stat="density", common_norm=False)` প্রতিটি ক্লাসকে আলাদা normalise করে যাতে অসম ক্লাস-আকার পার্থক্য লুকিয়ে না ফেলে। প্রতিটি অক্ষে title ও axis label আছে। categorical-অক্ষ সংক্রান্ত error এলে গ্রুপ-লেবেলকে string-এ রূপান্তর করুন (`rate_bar`-এ করা আছে)।
**English:** `frame.groupby(by).agg(n="size", churn_rate="mean")` splits rows into groups, then counts rows and averages the 0/1 target inside each group (= churn rate). `pd.cut(Age, bins=[17,30,40,50,60,100])` assigns each age to a right-closed interval. Matplotlib/Seaborn: `plt.subplots` creates a grid of axes; `sns.barplot` draws one bar per category (bar height = the y value given); `ax.text` annotates bars; `ax.axhline` draws the overall rate; `sns.histplot(..., stat="density", common_norm=False)` normalises each class separately so unequal class sizes do not hide differences. Every axis has a title and axis labels. If a plot errors with a categorical-axis message, convert the group labels to strings (done in `rate_bar`).

## Stage 5 — Feature audit (§5) | ফিচার অডিট

**বাংলা:** একটি সিদ্ধান্ত-DataFrame বানায়, প্রমাণ প্রিন্ট করে (`CustomerId`-এর duplicate, `id` ক্রমিক কি না, `corrwith` থেকে correlation), তারপর:
**English:** Builds a decision DataFrame, prints evidence (CustomerId duplicates, id sequential, correlations from `corrwith`), then:

```python
EXCLUDE_COLS = ["id", "CustomerId", "Surname"]
X = df.drop(columns=EXCLUDE_COLS + [TARGET]); y = df[TARGET]
```

**বাংলা:** `drop` একটি কপি ফেরত দেয়; `df` সম্পূর্ণ থাকে।
**English:** `drop` returns a copy; `df` remains complete.

## Stage 6 — Split (§6) | বিভাজন

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
```

**বাংলা:** ভেতরে ফাংশনটি সারির অবস্থান নেয়, `RandomState(42)` দিয়ে এলোমেলো করে, এবং `stratify` থাকলে প্রতিটি ক্লাসের সারি অনুপাতে বরাদ্দ করে। ফিচার ও লেবেল একসঙ্গে index হয়, তাই সারিবদ্ধতা বজায় থাকে; assert তা নিশ্চিত করে। **টেস্ট সেট সুরক্ষা:** এই সেল থেকে §11 (Task 9) পর্যন্ত কোনো কোড `X_test`/`y_test` উল্লেখ করে না; §9-এর CV কেবল `X_train` দেখে।
**English:** Internally the function takes row positions, shuffles them with a `RandomState(42)`, and with `stratify` allocates each class's rows proportionally. Features and labels are indexed together, so alignment is preserved; the asserts confirm it. **Protecting the test set:** `X_test`/`y_test` are not referenced by any code between this cell and §11 (Task 9); the CV in §9 sees only `X_train`.

## Stage 7 — Preprocessing (§7) | প্রি-প্রসেসিং

**বাংলা:** `make_preprocessor(scale_numeric)` একটি `ColumnTransformer` ফেরত দেয়: numeric sub-pipeline (`SimpleImputer(median)`, ঐচ্ছিক `StandardScaler`) এবং categorical sub-pipeline (`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`)। `make_pipeline(model, scale_numeric)` মডেলের আগে এটি বসায়।
* **fit যা শেখে:** median (`statistics_`), mean/std (`mean_`, `scale_`), category-তালিকা (`categories_`)।
* **transform যা প্রয়োগ করে:** একই সংখ্যাগুলো যেকোনো সারিতে। `ColumnTransformer.transform` প্রতিটি কলাম-তালিকাকে নিজ sub-pipeline-এ পাঠায় এবং ফল অনুভূমিকভাবে জোড়ে (১৩টি আউটপুট কলাম; নাম `get_feature_names_out()` থেকে)।
* যাচাই-ব্লক `make_preprocessor(True).fit(X_train)` ফিট করে এবং assert করে যে শেখা সংখ্যা training-split পরিসংখ্যানের সমান; `X_val` শুধু transform করে এবং একটি `"Atlantis"` সারিকে (সব-শূন্য one-hot) পরীক্ষা করে।
* সতর্কতা: কলাম-তালিকা `X_train` থেকে হিসাব হয়; পরে কলাম যোগ করলে পুনরায় বানান। pipeline-এর বাইরে scaling করলে leakage ঘটবে।

**English:** `make_preprocessor(scale_numeric)` returns a `ColumnTransformer` with a numeric sub-pipeline (`SimpleImputer(median)`, optional `StandardScaler`) and a categorical one (`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`). `make_pipeline(model, scale_numeric)` prepends it to a model.
* **fit learns:** medians (`statistics_`), means/stds (`mean_`, `scale_`), category lists (`categories_`).
* **transform applies:** the same numbers to any rows. `ColumnTransformer.transform` routes each column list through its sub-pipeline and horizontally stacks results (13 output columns; names via `get_feature_names_out()`).
* The verification block fits `make_preprocessor(True).fit(X_train)` and asserts the learned numbers equal the training-split statistics; it transforms `X_val` (transform only) and an `"Atlantis"` row (all-zero one-hot).
* Pitfalls: column lists are computed from `X_train`; if you add a column later, rebuild. Scaling placed outside the pipeline would leak.

## Stage 8 — Models and metrics (§8) | মডেল ও metric

**বাংলা:** `pipe.fit(X_train, y_train)`: প্রি-প্রসেসিংয়ে `fit_transform`, তারপর মডেলের `fit`। `pipe.predict_proba(X)[:, 1]` P(ক্লাস ১) ফেরত দেয় (কলাম ০ = ক্লাস ০; ক্লাস ০,১ ক্রমে)। Hard অনুমান: `(proba >= threshold).astype(int)` — `predict`-এর (যা এই মডেলে ০.৫ ব্যবহার করে) ওপর নির্ভর না করে threshold স্পষ্ট করা হয়েছে। `evaluate()` Accuracy, Precision, Recall, F1 (`zero_division=0` সহ, যাতে সব-negative অনুমানে warning/error না দিয়ে ০ আসে) এবং `safe_roc_auc` দিয়ে সম্ভাবনা থেকে ROC-AUC ফেরত দেয়; `safe_roc_auc` `len(np.unique(y_true)) < 2` হলে NaN দেয়, অথচ `roc_auc_score` তখন `ValueError` তোলে। মডেল কীভাবে হিসাব করে তা [doc 04](04_models_and_hyperparameter_tuning.md)-এ: LR = weighted sum-এর sigmoid; RF = tree-সম্ভাবনার গড়; GB = স্কেল-করা tree-আউটপুটের যোগফল → sigmoid।
**English:** `pipe.fit(X_train, y_train)`: preprocessing `fit_transform`, then the model's `fit`. `pipe.predict_proba(X)[:, 1]` returns P(class 1) (column 0 is class 0; classes are ordered 0,1). Hard predictions: `(proba >= threshold).astype(int)` — making the threshold explicit instead of relying on `predict` (which uses 0.5 for these models). `evaluate()` returns Accuracy, Precision, Recall, F1 (with `zero_division=0` so an all-negative prediction gives 0 rather than a warning/error) and ROC-AUC from probabilities through `safe_roc_auc`, which returns NaN if `len(np.unique(y_true)) < 2`. `roc_auc_score` raises `ValueError` otherwise. How the models compute (see doc 04): LR = sigmoid of a weighted sum; RF = average of tree probabilities; GB = sum of scaled tree outputs → sigmoid.

## Stage 9 — Tuning (§9) | টিউনিং

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(base, param_grids[name], scoring="roc_auc", cv=cv, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
```

**বাংলা:** প্রতিটি grid সমন্বয় × ৫ fold-এর জন্য: ফিট-না-করা (unfitted) pipeline ক্লোন করা, প্যারামিটার সেট করা (`model__C` `model` ধাপকে ঠিকানা করে), ৪ fold-এ ফিট করা, ৫ম fold-এ ROC-AUC হিসাব করা। `cv_results_` প্রতি সেটিংয়ের mean/std ধরে; `best_params_` ও `best_score_` বিজয়ীকে; `refit=True` পুরো `X_train`-এ `best_estimator_` ফিট করে। যেহেতু *pipeline* ক্লোন হয়, imputer/scaler/encoder প্রতি fold-এ নতুন করে শেখা হয় — fold-leakage নেই। তারপর টিউন-করা তিন মডেলের validation স্কোর `final_name` বাছে; 0.10–0.90 threshold-এ validation F1 sweep `FINAL_THRESHOLD` বাছে। `ValueError: Invalid parameter` মানে grid key কোনো ধাপের নামের সঙ্গে মিলছে না (`model__...`)। বড় grid ধীর; ছোট করুন।
**English:** For each of the grid combinations × 5 folds: clone the unfitted pipeline, set parameters (`model__C` addresses the `model` step), fit on 4 folds, compute ROC-AUC on the 5th. `cv_results_` holds mean/std per setting; `best_params_` and `best_score_` the winner; `refit=True` fits `best_estimator_` on all of `X_train`. Because the *pipeline* is cloned, imputer/scaler/encoder are re-learned inside each fold — no fold leakage. Then validation scores of the three tuned models choose `final_name`; the validation F1 sweep over thresholds 0.10–0.90 picks `FINAL_THRESHOLD`. `ValueError: Invalid parameter` means a grid key does not match a step name (`model__...`). Large grids are slow; reduce them.

## Stage 10 — Train vs validation (§10) | ট্রেইন বনাম ভ্যালিডেশন

**বাংলা:** `evaluate(final_model, X_train, y_train, FINAL_THRESHOLD)` এবং validation-এ একই কল ডাকে, টেবিল ও গ্রুপ-করা বার চার্ট (`DataFrame.plot(kind="bar")`, `bar_label`) বানায়। gap কলামটি কেবল একটি ডায়াগনস্টিক।
**English:** Calls `evaluate(final_model, X_train, y_train, FINAL_THRESHOLD)` and the same on validation, builds a table and grouped bar chart (`DataFrame.plot(kind="bar")`, `bar_label` for values). The gap column is a diagnostic only.

## Stage 11 — Internal test (§11) | অভ্যন্তরীণ টেস্ট

**বাংলা:** `X_test`-এ একবার `predict_proba`, metric হিসাব, এবং `confusion_matrix(y_test, test_pred, labels=[0, 1])` → `ravel()` ক্রমে `tn, fp, fn, tp` দেয়। `ConfusionMatrixDisplay` ক্লাসের নামসহ আঁকে। সেলের মন্তব্য জানায়: এই ফল দেখার পর কিছু বদলালে টেস্ট সেটের মর্যাদা নষ্ট হয়।
**English:** One `predict_proba` on `X_test`, metrics, and `confusion_matrix(y_test, test_pred, labels=[0, 1])` → `ravel()` yields `tn, fp, fn, tp` in that order. `ConfusionMatrixDisplay` draws it with class names. The cell comment states that changes after this point invalidate the test set's status.

## Stage 12 — Competition rows (§12) | প্রতিযোগিতার সারি

**বাংলা:** `test.csv` না থাকলে একটি কার্যকর বার্তা প্রিন্ট করে এগিয়ে যায়। থাকলে: ফাইল পড়ে, `X_train.columns`-এর সব কলাম আছে কি না যাচাই করে, `Exited` কলাম নেই তা assert করে, `id` আলাদা রাখে, `comp[X_train.columns]` (একই ক্রম) নেয়, প্রথম পাঁচ সারি নিয়ে `final_model.predict_proba` ডাকে — **কিছুই fit/refit হয় না**। Missing মান থাকলে সংরক্ষিত training median/mode দিয়ে পূরণ হতো; অদেখা category সব-শূন্য one-hot হতো। কোনো লেবেল নেই বলে কোনো metric হিসাব হয় না। লোকাল `data/test.csv`-এর প্রথম পাঁচ `id` হলো 15000–15004। (অ্যাসাইনমেন্টে যেমন লেখা, প্রথম পাঁচ সারি নেওয়া হয়েছে; "বৈধ" সারি = প্রয়োজনীয় কলামে missing নেই, যা লোকাল ফাইলে সত্য।)
**English:** If `test.csv` is absent: print an actionable message and move on. Otherwise: read it, check it has all columns in `X_train.columns`, assert there is no `Exited` column, keep `id` aside, select `comp[X_train.columns]` (same order), take the first five rows and call `final_model.predict_proba` — **no fit/refit** of anything. Missing values would be imputed with the stored training medians/modes; unseen categories become all-zero one-hot. The first five ids in the local `data/test.csv` are 15000–15004. No metrics are computed because there are no labels. (The first-five rows are used as written in the assignment; "valid" rows = no missing required columns, which holds for the local file.)

## Stage 13 — Conclusions (§13) | উপসংহার

**বাংলা:** প্রিন্ট-করা সারসংক্ষেপ আগের সেলে তৈরি ভ্যারিয়েবল থেকে জোড়া হয়, তাই ফল থেকে সরে যেতে পারে না। সীমাবদ্ধতাগুলো লিখিত পাঠ্য।
**English:** Printed summary is assembled from variables created earlier, so it cannot drift from results. Limitations are written text.

## Common errors cheat-sheet | সাধারণ error-এর দ্রুত তালিকা

| Symptom / লক্ষণ | Cause / কারণ | Fix / সমাধান |
|---|---|---|
| `FileNotFoundError: train.csv not found` | wrong Drive folder/name / ভুল Drive ফোল্ডার/নাম | create `MyDrive/Customer Churn Prediction/` and put `train.csv` there |
| `AssertionError: Target column 'Exited' missing` | wrong file / ভুল ফাইল | check you loaded the labeled train file |
| `ValueError: Input contains NaN` | NaNs in a column not covered by the pipeline | add the column to the pipeline lists |
| `ValueError: Invalid parameter ...` | bad `param_grid` key | use `model__<param>` |
| `ValueError: Only one class present` | ROC-AUC on a single-class set / এক-ক্লাস সেটে ROC-AUC | handled by `safe_roc_auc` (NaN) |
| unseen category / অদেখা category | new value at inference | handled by `handle_unknown="ignore"` |
| Recall very low with high accuracy / উচ্চ accuracy সত্ত্বেও কম recall | class imbalance / ক্লাস-অসমতা | look at the confusion matrix; consider threshold / class weights using validation data |
