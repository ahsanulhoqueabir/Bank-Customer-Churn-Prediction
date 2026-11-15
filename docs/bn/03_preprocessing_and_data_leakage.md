# 03 — প্রি-প্রসেসিং ও ডেটা লিকেজ

> **📌 এক নজরে**
> `id`, `CustomerId`, `Surname` বাদ দিয়ে ১০টি ফিচার রাখা হয়েছে। ডেটা আগে stratified ৭০/১৫/১৫ ভাগ করা হয়, তারপর সব প্রি-প্রসেসিং (imputer, one-hot, scaler) `Pipeline`-এর ভেতরে শুধু training অংশ থেকে শেখা হয়, যাতে leakage না ঘটে।

## 1. ফিচার/টার্গেট পৃথকীকরণ ও লিকেজ অডিট

`X` = ইনপুট কলাম, `y = df['Exited']`। `X` তৈরি হয় `df.drop(columns=[...])` দিয়ে, যা একটি **নতুন** DataFrame ফেরত দেয়; ফলে মূল `df` অক্ষত থাকে।

| কলাম | সিদ্ধান্ত | কারণ |
|---|---|---|
| id | EXCLUDE | ক্রমিক সারি-সূচক (যাচাইকৃত: পরপর পূর্ণসংখ্যা); কোনো সাধারণীকরণযোগ্য অর্থ নেই, নমনীয় মডেলকে সারি মুখস্থ করতে দেয় |
| CustomerId | EXCLUDE | identifier; স্বেচ্ছাচারী সংখ্যা। ১৫,০০০ সারিতে ৬,৩৪৯টি স্বতন্ত্র মান, তাই এটি সারির অনন্য কী *নয়* |
| Surname | EXCLUDE | ৭৭৮টি স্বতন্ত্র টেক্সট মান; স্বেচ্ছাচারী লেবেল এবং ন্যায্যতা/গোপনীয়তার উদ্বেগ |
| Exited | TARGET | ফলাফল; কেবল `y` হিসেবে ব্যবহৃত |
| CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary | KEEP | লিকেজের প্রমাণ নেই |

"লিকেজ নেই"-র প্রমাণ: রাখা সংখ্যাসূচক ফিচারগুলোর মধ্যে টার্গেটের সঙ্গে সর্বোচ্চ |correlation| হলো Age ০.৪৫২ (তারপর NumOfProducts ০.৩১১, IsActiveMember ০.১৯৪); কোনোটিই ১-এর কাছাকাছি নয়, অর্থাৎ কোনো ফিচার লেবেলের প্রতিলিপি নয়। **এই প্রমাণের সীমা:** লিকেজ *সময়ভিত্তিক* (temporal)ও হতে পারে — যেমন গ্রাহক চলে যাওয়ার *পরে* রেকর্ড হওয়া কোনো ফিল্ড (শেষ balance ০ বা "inactive" অবস্থা)। CSV থেকে প্রতিটি ফিল্ড কখন রেকর্ড হয়েছে তা প্রমাণ করা যায় না, তাই এটি একটি **অনুমান/ঝুঁকি**, যা ব্যাংকের সঙ্গে নিশ্চিত করা উচিত। কেবল পূর্বাভাসে কার্যকর (Age, NumOfProducts) বলেই কোনো ফিচার বাদ দেওয়া হয়নি।

**ডুপ্লিকেট-গ্রাহক ঝুঁকি (Duplicate-customer risk)।**
`CustomerId` পুনরাবৃত্ত হওয়ায় একই ব্যক্তি training এবং validation/test উভয় অংশে থাকতে পারেন। তাঁর সারিগুলো একই রকম হলে held-out স্কোর সামান্য আশাবাদী হয়ে যায়। `CustomerId`-ভিত্তিক grouped split (যেমন `GroupShuffleSplit`) এর সমাধান হতো, কিন্তু অ্যাসাইনমেন্টে stratified split নির্দিষ্ট করা আছে, আর একই ID-র সারিগুলো সত্যিই একই ব্যক্তির কি না তাও নিশ্চিত নয় (ডেটাসেটটি সিনথেটিক মনে হয়); তাই বিষয়টি পরিবর্তন না করে নথিভুক্ত করা হয়েছে।

## 2. স্ট্র্যাটিফাইড ৭০/১৫/১৫ বিভাজন

`train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)` একটি seed-যুক্ত random generator দিয়ে সারির সূচক এলোমেলো করে এবং stratification-এর জন্য প্রতিটি ক্লাসের ভেতর থেকে নমুনা নেয়, ফলে দুই পাশেই ক্লাসের অনুপাত (প্রায়) সমান থাকে। দ্বিতীয় ধাপে ৩০% অস্থায়ী অংশকে `stratify=y_temp` দিয়ে ৫০/৫০ ভাগ করা হয়, যা মোট ডেটার ১৫% + ১৫%।

হিসাব: 15,000 × 0.70 = 10,500; 4,500 × 0.5 = 2,250 করে। প্রকৃত ফলাফল:

| Split | সারি | অংশ | Churner | Churn rate |
|---|---|---|---|---|
| train | 10,500 | 70% | 2,132 | 20.30% |
| validation | 2,250 | 15% | 457 | 20.31% |
| internal test | 2,250 | 15% | 457 | 20.31% |

(সম্পূর্ণ ডেটা: 20.31%).

Stratification ছাড়া ২০/৮০ টার্গেটে এলোমেলো বিভাজনে কোনো অংশে কাকতালীয়ভাবে লক্ষণীয় বেশি বা কম চার্নার পড়তে পারে, ফলে স্কোরগুলো তুলনাযোগ্য থাকে না। নোটবুক assert করে যে তিনটি index-সেট পরস্পর বিচ্ছিন্ন (disjoint), মিলে সব সারি হয়, এবং `X`/`y`-এর index মিলে যায়। `X_val`/`y_val` (validation) এবং `X_test`/`y_test` (internal test) নাম দুটি কখনো অদলবদল হয় না।

## 3. ফিচারের ধরন

`select_dtypes(include='number')` সংখ্যাসূচক কলাম দেয়; বাকিগুলো categorical। ফল: numeric = CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary; categorical = Geography, Gender। ০/১ ফ্ল্যাগগুলো numeric float হিসেবেই রাখা হয়েছে (ইতোমধ্যে বৈধ binary encoding)।

## 4. প্রি-প্রসেসিংয়ের উপাদান

* **`SimpleImputer(strategy='median')`** (numeric) / **`most_frequent`** (categorical): `fit` প্রতি কলামের জন্য একটি পূরণ-মান শেখে (`statistics_`); `transform` NaN-কে সেই মান দিয়ে বদলে দেয়। Logistic Regression NaN পেলে error দেয় (tree ensemble-এর ক্ষেত্রে scikit-learn-এর সংস্করণভেদে আংশিক NaN সমর্থন আছে), আর scaler/encoder পূর্ণ ডেটা ধরে নেয় — তাই স্পষ্ট কৌশল নিরাপদ ও সুসংগত। এই ডেটাসেটে **০টি missing value** থাকায় imputer এখন কিছু বদলায় না; তবে ফাঁকযুক্ত ডেটা (যেমন NaN-সহ প্রতিযোগিতার ফাইল) এলেও pipeline যাতে কাজ করে, সেজন্য এগুলো রাখা হয়েছে।
* **`OneHotEncoder(handle_unknown='ignore')`**: `fit` প্রতি কলামের category-তালিকা শেখে (`categories_`); `transform` প্রতিটি category-র জন্য একটি ০/১ কলাম বানায় (Geography → France/Germany/Spain, Gender → Female/Male; মোট ৫টি কলাম)। মডেলে টেক্সট দেওয়া যায় না, আর সরল integer কোড (France=0, Germany=1, Spain=2) linear মডেলে ভুল ক্রম আরোপ করত। `handle_unknown='ignore'` আগে-না-দেখা category-কে error না দিয়ে সব-শূন্য হিসেবে encode করে; নোটবুক `Geography='Atlantis'` প্রোব সারি দিয়ে তা প্রমাণ করে।
* **`StandardScaler`** (শুধু Logistic Regression pipeline-এ): `fit` প্রতি কলামের mean ও standard deviation শেখে (`mean_`, `scale_`); `transform` আউটপুট দেয় `(x − mean)/std`। Logistic Regression-এর regularised optimisation ফিচারের মানের পরিমাপের প্রতি সংবেদনশীল (Balance ~10⁵ বনাম HasCrCard 0/1)। Tree মডেল threshold দিয়ে ভাগ করে, যা monotone rescaling-এ বদলায় না — তাই Random Forest ও Gradient Boosting এটি বাদ দেয়।
* **`ColumnTransformer`**: ভিন্ন কলাম-তালিকায় ভিন্ন sub-pipeline প্রয়োগ করে ফলাফল জুড়ে দেয় (এখানে ১৩টি কলাম: ৮ numeric + ৫ one-hot)।
* **`Pipeline`**: `preprocess → model` শৃঙ্খলিত করে। `pipe.fit(X, y)` প্রথমে preprocessor-এ `fit_transform`, তারপর মডেলে `fit` ডাকে; `pipe.predict_proba(X_new)` শুধু `transform` করে এবং মডেলের `predict_proba` ডাকে।

## 5. fit, transform ও fit_transform

* `fit(X)` — X থেকে পরিসংখ্যান (median, mean, category) **শেখে**।
* `transform(X)` — ইতোমধ্যে শেখা পরিসংখ্যান **প্রয়োগ** করে; নতুন কিছু শেখে না।
* `fit_transform(X)` — একই X-এ `fit`, তারপর `transform`; শুধু training ডেটাতেই বৈধ।

## 6. ভাগ করার আগে প্রি-প্রসেসিং করলে তথ্য ফাঁস যেভাবে ঘটে

প্রকৃত ডেটার উদাহরণ: `EstimatedSalary`-র median training split-এ 122,929.6 কিন্তু সব সারিতে 122,964.2; `Balance`-এর mean 42,506 (train) বনাম 42,625 (সব সারি)। "সব সারি"-র সংখ্যায় validation/test সারির তথ্য মিশে আছে। এখানে পার্থক্য ছোট, কিন্তু outlier, বিরল category বা ব্যাপক টিউনিং থাকলে নীতিটি আরও গুরুত্বপূর্ণ: মডেল এমন ডেটায় মূল্যায়িত হয় যা তার ইনপুট গঠনেই ভূমিকা রেখেছে, ফলে স্কোর বাস্তবের চেয়ে বেশি দেখায়। Pipeline-এ পরিসংখ্যান শুধু `fit`-এ দেওয়া সারি থেকে আসে।

Cross-validation-এর ভেতরে `GridSearchCV` প্রতি fold-এর জন্য পুরো pipeline ক্লোন করে এবং fold-এর training অংশে ফিট করে, তাই imputer/scaler/encoder কখনো fold-এর held-out অংশ দেখে না। এ কারণেই প্রি-প্রসেসিং একবারে আগে না করে pipeline-এর অংশ করা হয়েছে।

## 7. held-out ডেটা ফিটিংয়ে অংশ নেয় না — যাচাই

নোটবুকের §7-এ preprocessor শুধু `X_train`-এ ফিট করা হয় এবং `np.allclose` দিয়ে যাচাই করা হয় যে শেখা `imputer.statistics_` ও `scaler.mean_` training-split-এর median/mean-এর সমান; পাশে (ভিন্ন) সম্পূর্ণ-ডেটার মানও দেখানো হয়। তারপর validation সারিতে শুধু transform করা হয় এবং অদেখা-category প্রোব error ছাড়াই encode হয়।

## 8. সমস্যা সমাধান

* `ValueError: Input contains NaN` → কোনো কলাম imputer-এর তালিকার বাইরে রয়ে গেছে (যেমন তালিকাভুক্ত নয় এমন নতুন কলাম)।
* `ValueError: columns are missing` → inference ডেটায় training-এ ব্যবহৃত কলাম নেই; `X_train.columns` অনুযায়ী ক্রমে কলাম নির্বাচন করুন।
* `EXCLUDE_COLS` বদলালে `X_train`-এর কলাম বদলায় এবং পরবর্তী সবকিছু প্রভাবিত হয়; নোটবুক শুরু থেকে আবার চালান।

