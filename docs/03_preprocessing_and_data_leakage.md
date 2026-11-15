# 03 — Preprocessing and Data Leakage | প্রি-প্রসেসিং ও ডেটা লিকেজ

> **📌 এক নজরে (At a glance)**
> `id`, `CustomerId`, `Surname` বাদ দিয়ে ১০টি ফিচার রাখা হয়েছে। ডেটা আগে stratified ৭০/১৫/১৫ ভাগ করা হয়, তারপর সব প্রি-প্রসেসিং (imputer, one-hot, scaler) `Pipeline`-এর ভেতরে শুধু training অংশ থেকে শেখা হয়, যাতে leakage না ঘটে।
> *Identifiers are dropped, the data is split first, and all preprocessing is learned from training rows only inside a `Pipeline`.*

## 1. Feature/target separation and the leakage audit | ফিচার/টার্গেট পৃথকীকরণ ও লিকেজ অডিট

**বাংলা:** `X` = ইনপুট কলাম, `y = df['Exited']`। `X` তৈরি হয় `df.drop(columns=[...])` দিয়ে, যা একটি **নতুন** DataFrame ফেরত দেয়; ফলে মূল `df` অক্ষত থাকে।
**English:** `X` = input columns, `y = df['Exited']`. `X` is built with `df.drop(columns=[...])`, which returns a **new** DataFrame, so the original `df` is preserved.

| Column | Decision / সিদ্ধান্ত | Reason / কারণ |
|---|---|---|
| id | EXCLUDE | sequential row index (verified: consecutive integers); no generalisable meaning, lets flexible models memorise rows / ক্রমিক সারি-সূচক; কোনো সাধারণীকরণযোগ্য অর্থ নেই |
| CustomerId | EXCLUDE | identifier; arbitrary number. 6,349 distinct values over 15,000 rows, so it is *not* a unique row key / স্বেচ্ছাচারী সংখ্যা; প্রতিটি সারির অনন্য কী নয় |
| Surname | EXCLUDE | 778 distinct text values; an arbitrary label and a fairness/privacy concern / স্বেচ্ছাচারী লেবেল; ন্যায্যতা ও গোপনীয়তার উদ্বেগ |
| Exited | TARGET | outcome; only used as `y` / কেবল `y` হিসেবে ব্যবহৃত |
| CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary | KEEP | no evidence of leakage / লিকেজের প্রমাণ নেই |

**বাংলা:** "লিকেজ নেই"-র প্রমাণ: রাখা সংখ্যাসূচক ফিচারগুলোর মধ্যে টার্গেটের সঙ্গে সর্বোচ্চ |correlation| হলো Age ০.৪৫২ (তারপর NumOfProducts ০.৩১১, IsActiveMember ০.১৯৪); কোনোটিই ১-এর কাছাকাছি নয়, অর্থাৎ কোনো ফিচার লেবেলের প্রতিলিপি নয়। **এই প্রমাণের সীমা:** লিকেজ *সময়ভিত্তিক* (temporal)ও হতে পারে — যেমন গ্রাহক চলে যাওয়ার *পরে* রেকর্ড হওয়া কোনো ফিল্ড (শেষ balance ০ বা "inactive" অবস্থা)। CSV থেকে প্রতিটি ফিল্ড কখন রেকর্ড হয়েছে তা প্রমাণ করা যায় না, তাই এটি একটি **অনুমান/ঝুঁকি**, যা ব্যাংকের সঙ্গে নিশ্চিত করা উচিত। কেবল পূর্বাভাসে কার্যকর (Age, NumOfProducts) বলেই কোনো ফিচার বাদ দেওয়া হয়নি।

**English:** Evidence for "no leakage": the largest |correlation| between a kept numeric feature and the target is Age at 0.452 (then NumOfProducts 0.311, IsActiveMember 0.194); none is near 1, i.e. no feature is a copy of the label. **Limits of this evidence:** leakage can also be *temporal* (a feature recorded after the customer left, e.g. a final balance of 0 or "inactive" status). The CSV cannot prove when each field was recorded, so this is an **assumption/risk** to confirm with the bank. Features are not dropped merely for being predictive (Age, NumOfProducts).

**Duplicate-customer risk | ডুপ্লিকেট-গ্রাহক ঝুঁকি।**
**বাংলা:** `CustomerId` পুনরাবৃত্ত হওয়ায় একই ব্যক্তি training এবং validation/test উভয় অংশে থাকতে পারেন। তাঁর সারিগুলো একই রকম হলে held-out স্কোর সামান্য আশাবাদী হয়ে যায়। `CustomerId`-ভিত্তিক grouped split (যেমন `GroupShuffleSplit`) এর সমাধান হতো, কিন্তু অ্যাসাইনমেন্টে stratified split নির্দিষ্ট করা আছে, আর একই ID-র সারিগুলো সত্যিই একই ব্যক্তির কি না তাও নিশ্চিত নয় (ডেটাসেটটি সিনথেটিক মনে হয়); তাই বিষয়টি পরিবর্তন না করে নথিভুক্ত করা হয়েছে।
**English:** Because `CustomerId` repeats, the same person can appear in training and validation/test rows. If that person's rows are similar, held-out scores can be slightly optimistic. A grouped split (e.g. `GroupShuffleSplit` on `CustomerId`) would address it but the assignment specifies a stratified split, and rows with the same ID may not truly be the same person (the dataset looks synthetic), so it is documented rather than altered.

## 2. The stratified 70/15/15 split | স্ট্র্যাটিফাইড ৭০/১৫/১৫ বিভাজন

**বাংলা:** `train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)` একটি seed-যুক্ত random generator দিয়ে সারির সূচক এলোমেলো করে এবং stratification-এর জন্য প্রতিটি ক্লাসের ভেতর থেকে নমুনা নেয়, ফলে দুই পাশেই ক্লাসের অনুপাত (প্রায়) সমান থাকে। দ্বিতীয় ধাপে ৩০% অস্থায়ী অংশকে `stratify=y_temp` দিয়ে ৫০/৫০ ভাগ করা হয়, যা মোট ডেটার ১৫% + ১৫%।
**English:** `train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)` shuffles row indices with a seeded random generator and, for stratification, samples within each class so each side gets (almost) the same class share. Stage 2 splits the 30% temporary part 50/50 with `stratify=y_temp`, giving 15% + 15% of the full data.

Arithmetic / হিসাব: 15,000 × 0.70 = 10,500; 4,500 × 0.5 = 2,250 each. Actual result / প্রকৃত ফলাফল:

| Split | Rows | Share | Churners | Churn rate |
|---|---|---|---|---|
| train | 10,500 | 70% | 2,132 | 20.30% |
| validation | 2,250 | 15% | 457 | 20.31% |
| internal test | 2,250 | 15% | 457 | 20.31% |

(full data / সম্পূর্ণ ডেটা: 20.31%).

**বাংলা:** Stratification ছাড়া ২০/৮০ টার্গেটে এলোমেলো বিভাজনে কোনো অংশে কাকতালীয়ভাবে লক্ষণীয় বেশি বা কম চার্নার পড়তে পারে, ফলে স্কোরগুলো তুলনাযোগ্য থাকে না। নোটবুক assert করে যে তিনটি index-সেট পরস্পর বিচ্ছিন্ন (disjoint), মিলে সব সারি হয়, এবং `X`/`y`-এর index মিলে যায়। `X_val`/`y_val` (validation) এবং `X_test`/`y_test` (internal test) নাম দুটি কখনো অদলবদল হয় না।
**English:** Without stratification, a random split of a 20/80 target could by chance put noticeably more or fewer churners in one subset, making scores incomparable. The notebook asserts the three index sets are disjoint, add up to all rows, and that `X`/`y` indices align. The variable names `X_val`/`y_val` (validation) and `X_test`/`y_test` (internal test) are never interchanged.

## 3. Feature types | ফিচারের ধরন

**বাংলা:** `select_dtypes(include='number')` সংখ্যাসূচক কলাম দেয়; বাকিগুলো categorical। ফল: numeric = CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary; categorical = Geography, Gender। ০/১ ফ্ল্যাগগুলো numeric float হিসেবেই রাখা হয়েছে (ইতোমধ্যে বৈধ binary encoding)।
**English:** `select_dtypes(include='number')` gives numeric columns; the rest are categorical. Result: numeric = CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary; categorical = Geography, Gender. The 0/1 flags are numeric floats and are kept numeric (already a valid binary encoding).

## 4. The preprocessing components | প্রি-প্রসেসিংয়ের উপাদান

**বাংলা:**
* **`SimpleImputer(strategy='median')`** (numeric) / **`most_frequent`** (categorical): `fit` প্রতি কলামের জন্য একটি পূরণ-মান শেখে (`statistics_`); `transform` NaN-কে সেই মান দিয়ে বদলে দেয়। Logistic Regression NaN পেলে error দেয় (tree ensemble-এর ক্ষেত্রে scikit-learn-এর সংস্করণভেদে আংশিক NaN সমর্থন আছে), আর scaler/encoder পূর্ণ ডেটা ধরে নেয় — তাই স্পষ্ট কৌশল নিরাপদ ও সুসংগত। এই ডেটাসেটে **০টি missing value** থাকায় imputer এখন কিছু বদলায় না; তবে ফাঁকযুক্ত ডেটা (যেমন NaN-সহ প্রতিযোগিতার ফাইল) এলেও pipeline যাতে কাজ করে, সেজন্য এগুলো রাখা হয়েছে।
* **`OneHotEncoder(handle_unknown='ignore')`**: `fit` প্রতি কলামের category-তালিকা শেখে (`categories_`); `transform` প্রতিটি category-র জন্য একটি ০/১ কলাম বানায় (Geography → France/Germany/Spain, Gender → Female/Male; মোট ৫টি কলাম)। মডেলে টেক্সট দেওয়া যায় না, আর সরল integer কোড (France=0, Germany=1, Spain=2) linear মডেলে ভুল ক্রম আরোপ করত। `handle_unknown='ignore'` আগে-না-দেখা category-কে error না দিয়ে সব-শূন্য হিসেবে encode করে; নোটবুক `Geography='Atlantis'` প্রোব সারি দিয়ে তা প্রমাণ করে।
* **`StandardScaler`** (শুধু Logistic Regression pipeline-এ): `fit` প্রতি কলামের mean ও standard deviation শেখে (`mean_`, `scale_`); `transform` আউটপুট দেয় `(x − mean)/std`। Logistic Regression-এর regularised optimisation ফিচারের মানের পরিমাপের প্রতি সংবেদনশীল (Balance ~10⁵ বনাম HasCrCard 0/1)। Tree মডেল threshold দিয়ে ভাগ করে, যা monotone rescaling-এ বদলায় না — তাই Random Forest ও Gradient Boosting এটি বাদ দেয়।
* **`ColumnTransformer`**: ভিন্ন কলাম-তালিকায় ভিন্ন sub-pipeline প্রয়োগ করে ফলাফল জুড়ে দেয় (এখানে ১৩টি কলাম: ৮ numeric + ৫ one-hot)।
* **`Pipeline`**: `preprocess → model` শৃঙ্খলিত করে। `pipe.fit(X, y)` প্রথমে preprocessor-এ `fit_transform`, তারপর মডেলে `fit` ডাকে; `pipe.predict_proba(X_new)` শুধু `transform` করে এবং মডেলের `predict_proba` ডাকে।

**English:**
* **`SimpleImputer(strategy='median')`** (numeric) / **`most_frequent`** (categorical): `fit` learns one fill value per column from the data it is given (`statistics_`); `transform` replaces NaNs with that value. Logistic Regression raises an error on NaN (tree ensembles have partial NaN support depending on the scikit-learn version), and the scaler/encoder assume complete data, so an explicit strategy is the safe, uniform choice. This dataset has **0 missing values**, so the imputers currently change nothing; they are kept so the pipeline would still work for data with gaps (e.g. a competition file with NaNs).
* **`OneHotEncoder(handle_unknown='ignore')`**: `fit` learns the list of categories per column (`categories_`); `transform` outputs one 0/1 column per category (Geography → France/Germany/Spain, Gender → Female/Male; 5 columns). Text cannot be fed to the models, and plain integer codes (France=0, Germany=1, Spain=2) would impose a false ordering on linear models. `handle_unknown='ignore'` encodes a never-seen category as all zeros instead of raising an error; the notebook proves this with a probe row `Geography='Atlantis'`.
* **`StandardScaler`** (Logistic Regression pipeline only): `fit` learns each column's mean and standard deviation (`mean_`, `scale_`); `transform` outputs `(x − mean)/std`. Logistic Regression's regularised optimisation is sensitive to feature magnitudes (Balance ~10⁵ vs HasCrCard 0/1). Trees split on thresholds, which are unaffected by monotone rescaling, so Random Forest and Gradient Boosting skip it.
* **`ColumnTransformer`**: applies different sub-pipelines to different column lists and concatenates the outputs (13 columns here: 8 numeric + 5 one-hot).
* **`Pipeline`**: chains `preprocess → model`. `pipe.fit(X, y)` calls `fit_transform` on the preprocessor then `fit` on the model; `pipe.predict_proba(X_new)` calls only `transform` then the model's `predict_proba`.

## 5. fit vs transform vs fit_transform | fit, transform ও fit_transform

**বাংলা:**
* `fit(X)` — X থেকে পরিসংখ্যান (median, mean, category) **শেখে**।
* `transform(X)` — ইতোমধ্যে শেখা পরিসংখ্যান **প্রয়োগ** করে; নতুন কিছু শেখে না।
* `fit_transform(X)` — একই X-এ `fit`, তারপর `transform`; শুধু training ডেটাতেই বৈধ।

**English:**
* `fit(X)` — *learn* statistics (medians, means, categories) from X.
* `transform(X)` — *apply* already-learned statistics; learns nothing.
* `fit_transform(X)` — `fit` then `transform` on the same X; only legitimate on training data.

## 6. How preprocessing before splitting leaks information | ভাগ করার আগে প্রি-প্রসেসিং করলে তথ্য ফাঁস যেভাবে ঘটে

**বাংলা:** প্রকৃত ডেটার উদাহরণ: `EstimatedSalary`-র median training split-এ 122,929.6 কিন্তু সব সারিতে 122,964.2; `Balance`-এর mean 42,506 (train) বনাম 42,625 (সব সারি)। "সব সারি"-র সংখ্যায় validation/test সারির তথ্য মিশে আছে। এখানে পার্থক্য ছোট, কিন্তু outlier, বিরল category বা ব্যাপক টিউনিং থাকলে নীতিটি আরও গুরুত্বপূর্ণ: মডেল এমন ডেটায় মূল্যায়িত হয় যা তার ইনপুট গঠনেই ভূমিকা রেখেছে, ফলে স্কোর বাস্তবের চেয়ে বেশি দেখায়। Pipeline-এ পরিসংখ্যান শুধু `fit`-এ দেওয়া সারি থেকে আসে।

Cross-validation-এর ভেতরে `GridSearchCV` প্রতি fold-এর জন্য পুরো pipeline ক্লোন করে এবং fold-এর training অংশে ফিট করে, তাই imputer/scaler/encoder কখনো fold-এর held-out অংশ দেখে না। এ কারণেই প্রি-প্রসেসিং একবারে আগে না করে pipeline-এর অংশ করা হয়েছে।

**English:** Example from the actual data: the median `EstimatedSalary` is 122,929.6 in the training split but 122,964.2 on all rows; the mean `Balance` is 42,506 (train) vs 42,625 (all). The "all rows" numbers contain information about validation/test rows. The differences here are small, but the principle matters more with outliers, rare categories or heavy tuning: the model would be evaluated on data that helped shape its inputs, so scores overestimate real-world performance. In a pipeline, the statistics come only from the rows passed to `fit`.

Inside cross-validation, `GridSearchCV` clones the whole pipeline for each fold and fits it on the fold's training part, so the imputer/scaler/encoder never see the fold's held-out part. This is why preprocessing is part of the pipeline rather than done once up front.

## 7. Verifying held-out data does not participate in fitting | held-out ডেটা ফিটিংয়ে অংশ নেয় না — যাচাই

**বাংলা:** নোটবুকের §7-এ preprocessor শুধু `X_train`-এ ফিট করা হয় এবং `np.allclose` দিয়ে যাচাই করা হয় যে শেখা `imputer.statistics_` ও `scaler.mean_` training-split-এর median/mean-এর সমান; পাশে (ভিন্ন) সম্পূর্ণ-ডেটার মানও দেখানো হয়। তারপর validation সারিতে শুধু transform করা হয় এবং অদেখা-category প্রোব error ছাড়াই encode হয়।
**English:** Notebook §7 fits the preprocessor on `X_train` only and checks with `np.allclose` that the learned `imputer.statistics_` and `scaler.mean_` equal the training-split medians/means, and displays them next to the (different) full-data values. It then transforms validation rows (transform only) and shows the unseen-category probe encodes without error.

## 8. Troubleshooting | সমস্যা সমাধান

**বাংলা:**
* `ValueError: Input contains NaN` → কোনো কলাম imputer-এর তালিকার বাইরে রয়ে গেছে (যেমন তালিকাভুক্ত নয় এমন নতুন কলাম)।
* `ValueError: columns are missing` → inference ডেটায় training-এ ব্যবহৃত কলাম নেই; `X_train.columns` অনুযায়ী ক্রমে কলাম নির্বাচন করুন।
* `EXCLUDE_COLS` বদলালে `X_train`-এর কলাম বদলায় এবং পরবর্তী সবকিছু প্রভাবিত হয়; নোটবুক শুরু থেকে আবার চালান।

**English:**
* `ValueError: Input contains NaN` → a column escaped the imputers (e.g. a new column not in the lists).
* `ValueError: columns are missing` → inference data lacks a column used in training; select `X_train.columns` in order.
* Different `X_train` columns after changing `EXCLUDE_COLS` change everything downstream; re-run the notebook from the top.
