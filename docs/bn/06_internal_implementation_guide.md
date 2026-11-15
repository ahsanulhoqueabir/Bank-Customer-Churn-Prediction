# 06 — অভ্যন্তরীণ বাস্তবায়ন নির্দেশিকা

> **📌 এক নজরে**
> `notebooks/bank_customer_churn_prediction.ipynb`-এর ১৩টি ধাপের পরপর ব্যাখ্যা। কোড-অংশগুলো নোটবুক থেকে হুবহু নেওয়া; প্রতিটি ধাপের সঙ্গে বাংলা ব্যাখ্যা আছে।

## ধাপ 1 — সেটআপ ও পুনরুৎপাদনযোগ্যতা

import-এর পর ধ্রুবকগুলো: `RANDOM_STATE = 42`, `THRESHOLD_DEFAULT = 0.5`, `TARGET = "Exited"`। seed প্রতিটি random উপাদানে (`train_test_split`, `StratifiedKFold`, মডেল) স্পষ্টভাবে দেওয়া হয়, ফলে একই library সংস্করণে পুনরায় চালালে একই split ও মডেল মেলে (যাচাইকৃত: দুটি execution অভিন্ন ফল দিয়েছে)। সংস্করণ প্রিন্ট হয় (লোকাল রান: Python 3.13.5, pandas 2.2.3, scikit-learn 1.6.1); Colab-এর সংস্করণ কিছুটা আলাদা হতে পারে এবং ফলে সামান্য পার্থক্য আনতে পারে।

## ধাপ 2 — Drive মাউন্ট ও ফাইল যাচাই

```python
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
if IN_COLAB:
    drive.mount("/content/drive")
    DATA_DIR = Path("/content/drive/MyDrive/Customer Churn Prediction")
else:
    DATA_DIR = root / "data"       # local development fallback only
```

`drive.mount` একটি OAuth সম্মতি-প্রবাহ খোলে এবং আপনার Drive-কে `/content/drive`-এ FUSE ফাইল-সিস্টেম হিসেবে প্রকাশ করে, ফলে সাধারণ ফাইল-ফাংশন কাজ করে। `Path(...) / "train.csv"` বহনযোগ্যভাবে পথ বানায়; পড়ার **আগে** `.exists()` দেখা হয় এবং ফাইল না থাকলে প্রত্যাশিত পথ ও নির্দেশনাসহ `FileNotFoundError` ওঠে। `read_csv`-এর পর assert যাচাই করে: ডেটা খালি নয়, কলামের নাম অনন্য, টার্গেট আছে, টার্গেটে missing নেই, মান ⊆ {0,1}। প্রতিটি assert বার্তা বলে দেয় কী ঠিক করতে হবে। লোকাল fallback কেবল Colab-এ *না* থাকলে বাছা হয়; Colab কখনো নীরবে fallback করে না।
*এখানে চালানো হয়নি:* `drive.mount` শাখা (ডেভেলপমেন্ট পরিবেশে Colab নেই)।

## ধাপ 3 — পরিদর্শন

* `df.dtypes`, `df.nunique()`, `df.isna().sum()` কলামের নামে সূচিত Series ফেরত দেয়; `join` সেগুলোকে একটি টেবিলে জোড়ে।
* `value_counts()` স্বতন্ত্র মান গোনে; `sort_index()` ০ তারপর ১ ক্রমে সাজায়।
* Identifier heuristic: কলামের নামে "id" থাকলে বা মানের ৩০%-এর বেশি স্বতন্ত্র হলে সেটিকে "identifier-সদৃশ" ধরা হয়। এটি স্ক্রিনিং হিউরিস্টিক (এটি `EstimatedSalary`-কেও চিহ্নিত করেছে, যা বৈধ ফিচার), সিদ্ধান্ত নয়; সিদ্ধান্ত Stage 5-এ।
* প্রিন্ট-করা পর্যবেক্ষণ গণনা থেকে f-string দিয়ে তৈরি, তাই ডেটা বদলালে তা-ও বদলায়।

## ধাপ 4 — এক্সপ্লোরেটরি বিশ্লেষণ

`frame.groupby(by).agg(n="size", churn_rate="mean")` সারিগুলোকে গ্রুপে ভাগ করে, গ্রুপের আকার গোনে এবং ০/১ টার্গেটের গড় (= churn rate) বের করে। `pd.cut(Age, bins=[17,30,40,50,60,100])` প্রতিটি বয়সকে right-closed ব্যবধানে বসায়। Matplotlib/Seaborn: `plt.subplots` অক্ষের গ্রিড বানায়; `sns.barplot` প্রতি category-র জন্য একটি বার আঁকে (বারের উচ্চতা = দেওয়া y মান); `ax.text` বার চিহ্নিত করে; `ax.axhline` সামগ্রিক হার আঁকে; `sns.histplot(..., stat="density", common_norm=False)` প্রতিটি ক্লাসকে আলাদা normalise করে যাতে অসম ক্লাস-আকার পার্থক্য লুকিয়ে না ফেলে। প্রতিটি অক্ষে title ও axis label আছে। categorical-অক্ষ সংক্রান্ত error এলে গ্রুপ-লেবেলকে string-এ রূপান্তর করুন (`rate_bar`-এ করা আছে)।

## ধাপ 5 — ফিচার অডিট

একটি সিদ্ধান্ত-DataFrame বানায়, প্রমাণ প্রিন্ট করে (`CustomerId`-এর duplicate, `id` ক্রমিক কি না, `corrwith` থেকে correlation), তারপর:

```python
EXCLUDE_COLS = ["id", "CustomerId", "Surname"]
X = df.drop(columns=EXCLUDE_COLS + [TARGET]); y = df[TARGET]
```

`drop` একটি কপি ফেরত দেয়; `df` সম্পূর্ণ থাকে।

## ধাপ 6 — বিভাজন

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
```

ভেতরে ফাংশনটি সারির অবস্থান নেয়, `RandomState(42)` দিয়ে এলোমেলো করে, এবং `stratify` থাকলে প্রতিটি ক্লাসের সারি অনুপাতে বরাদ্দ করে। ফিচার ও লেবেল একসঙ্গে index হয়, তাই সারিবদ্ধতা বজায় থাকে; assert তা নিশ্চিত করে। **টেস্ট সেট সুরক্ষা:** এই সেল থেকে §11 (Task 9) পর্যন্ত কোনো কোড `X_test`/`y_test` উল্লেখ করে না; §9-এর CV কেবল `X_train` দেখে।

## ধাপ 7 — প্রি-প্রসেসিং

`make_preprocessor(scale_numeric)` একটি `ColumnTransformer` ফেরত দেয়: numeric sub-pipeline (`SimpleImputer(median)`, ঐচ্ছিক `StandardScaler`) এবং categorical sub-pipeline (`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`)। `make_pipeline(model, scale_numeric)` মডেলের আগে এটি বসায়।
* **fit যা শেখে:** median (`statistics_`), mean/std (`mean_`, `scale_`), category-তালিকা (`categories_`)।
* **transform যা প্রয়োগ করে:** একই সংখ্যাগুলো যেকোনো সারিতে। `ColumnTransformer.transform` প্রতিটি কলাম-তালিকাকে নিজ sub-pipeline-এ পাঠায় এবং ফল অনুভূমিকভাবে জোড়ে (১৩টি আউটপুট কলাম; নাম `get_feature_names_out()` থেকে)।
* যাচাই-ব্লক `make_preprocessor(True).fit(X_train)` ফিট করে এবং assert করে যে শেখা সংখ্যা training-split পরিসংখ্যানের সমান; `X_val` শুধু transform করে এবং একটি `"Atlantis"` সারিকে (সব-শূন্য one-hot) পরীক্ষা করে।
* সতর্কতা: কলাম-তালিকা `X_train` থেকে হিসাব হয়; পরে কলাম যোগ করলে পুনরায় বানান। pipeline-এর বাইরে scaling করলে leakage ঘটবে।

## ধাপ 8 — মডেল ও metric

`pipe.fit(X_train, y_train)`: প্রি-প্রসেসিংয়ে `fit_transform`, তারপর মডেলের `fit`। `pipe.predict_proba(X)[:, 1]` P(ক্লাস ১) ফেরত দেয় (কলাম ০ = ক্লাস ০; ক্লাস ০,১ ক্রমে)। Hard অনুমান: `(proba >= threshold).astype(int)` — `predict`-এর (যা এই মডেলে ০.৫ ব্যবহার করে) ওপর নির্ভর না করে threshold স্পষ্ট করা হয়েছে। `evaluate()` Accuracy, Precision, Recall, F1 (`zero_division=0` সহ, যাতে সব-negative অনুমানে warning/error না দিয়ে ০ আসে) এবং `safe_roc_auc` দিয়ে সম্ভাবনা থেকে ROC-AUC ফেরত দেয়; `safe_roc_auc` `len(np.unique(y_true)) < 2` হলে NaN দেয়, অথচ `roc_auc_score` তখন `ValueError` তোলে। মডেল কীভাবে হিসাব করে তা [doc 04](04_models_and_hyperparameter_tuning.md)-এ: LR = weighted sum-এর sigmoid; RF = tree-সম্ভাবনার গড়; GB = স্কেল-করা tree-আউটপুটের যোগফল → sigmoid।

## ধাপ 9 — টিউনিং

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(base, param_grids[name], scoring="roc_auc", cv=cv, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
```

প্রতিটি grid সমন্বয় × ৫ fold-এর জন্য: ফিট-না-করা (unfitted) pipeline ক্লোন করা, প্যারামিটার সেট করা (`model__C` `model` ধাপকে ঠিকানা করে), ৪ fold-এ ফিট করা, ৫ম fold-এ ROC-AUC হিসাব করা। `cv_results_` প্রতি সেটিংয়ের mean/std ধরে; `best_params_` ও `best_score_` বিজয়ীকে; `refit=True` পুরো `X_train`-এ `best_estimator_` ফিট করে। যেহেতু *pipeline* ক্লোন হয়, imputer/scaler/encoder প্রতি fold-এ নতুন করে শেখা হয় — fold-leakage নেই। তারপর টিউন-করা তিন মডেলের validation স্কোর `final_name` বাছে; 0.10–0.90 threshold-এ validation F1 sweep `FINAL_THRESHOLD` বাছে। `ValueError: Invalid parameter` মানে grid key কোনো ধাপের নামের সঙ্গে মিলছে না (`model__...`)। বড় grid ধীর; ছোট করুন।

## ধাপ 10 — ট্রেইন বনাম ভ্যালিডেশন

`evaluate(final_model, X_train, y_train, FINAL_THRESHOLD)` এবং validation-এ একই কল ডাকে, টেবিল ও গ্রুপ-করা বার চার্ট (`DataFrame.plot(kind="bar")`, `bar_label`) বানায়। gap কলামটি কেবল একটি ডায়াগনস্টিক।

## ধাপ 11 — অভ্যন্তরীণ টেস্ট

`X_test`-এ একবার `predict_proba`, metric হিসাব, এবং `confusion_matrix(y_test, test_pred, labels=[0, 1])` → `ravel()` ক্রমে `tn, fp, fn, tp` দেয়। `ConfusionMatrixDisplay` ক্লাসের নামসহ আঁকে। সেলের মন্তব্য জানায়: এই ফল দেখার পর কিছু বদলালে টেস্ট সেটের মর্যাদা নষ্ট হয়।

## ধাপ 12 — প্রতিযোগিতার সারি

`test.csv` না থাকলে একটি কার্যকর বার্তা প্রিন্ট করে এগিয়ে যায়। থাকলে: ফাইল পড়ে, `X_train.columns`-এর সব কলাম আছে কি না যাচাই করে, `Exited` কলাম নেই তা assert করে, `id` আলাদা রাখে, `comp[X_train.columns]` (একই ক্রম) নেয়, প্রথম পাঁচ সারি নিয়ে `final_model.predict_proba` ডাকে — **কিছুই fit/refit হয় না**। Missing মান থাকলে সংরক্ষিত training median/mode দিয়ে পূরণ হতো; অদেখা category সব-শূন্য one-hot হতো। কোনো লেবেল নেই বলে কোনো metric হিসাব হয় না। লোকাল `data/test.csv`-এর প্রথম পাঁচ `id` হলো 15000–15004। (অ্যাসাইনমেন্টে যেমন লেখা, প্রথম পাঁচ সারি নেওয়া হয়েছে; "বৈধ" সারি = প্রয়োজনীয় কলামে missing নেই, যা লোকাল ফাইলে সত্য।)

## ধাপ 13 — উপসংহার

প্রিন্ট-করা সারসংক্ষেপ আগের সেলে তৈরি ভ্যারিয়েবল থেকে জোড়া হয়, তাই ফল থেকে সরে যেতে পারে না। সীমাবদ্ধতাগুলো লিখিত পাঠ্য।

## সাধারণ error-এর দ্রুত তালিকা

| লক্ষণ | কারণ | সমাধান |
|---|---|---|
| `FileNotFoundError: train.csv not found` | ভুল Drive ফোল্ডার/নাম | `MyDrive/Customer Churn Prediction/` বানিয়ে সেখানে `train.csv` রাখুন |
| `AssertionError: Target column 'Exited' missing` | ভুল ফাইল | লেবেলযুক্ত train ফাইল লোড হয়েছে কি না দেখুন |
| `ValueError: Input contains NaN` | pipeline-এর আওতার বাইরের কলামে NaN | কলামটি pipeline-এর তালিকায় যোগ করুন |
| `ValueError: Invalid parameter ...` | ভুল `param_grid` key | `model__<param>` ব্যবহার করুন |
| `ValueError: Only one class present` | এক-ক্লাস সেটে ROC-AUC | `safe_roc_auc` সামলায় (NaN) |
| অদেখা category | inference-এ নতুন মান | `handle_unknown="ignore"` সামলায় |
| উচ্চ accuracy সত্ত্বেও কম recall | ক্লাস-অসমতা | confusion matrix দেখুন; validation ডেটায় threshold / class weight বিবেচনা করুন |
