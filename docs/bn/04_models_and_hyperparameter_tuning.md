# 04 — মডেল ও হাইপারপ্যারামিটার টিউনিং

> **📌 এক নজরে**
> তিনটি মডেল — Logistic Regression, Random Forest, Gradient Boosting — প্রতিটি `Pipeline`-এ মোড়ানো। Baseline-এ Gradient Boosting সেরা (validation ROC-AUC 0.9288)। `GridSearchCV` (5-fold, ROC-AUC) চালানোর পর নির্বাচিত মডেল: **Gradient Boosting** (`learning_rate=0.05, max_depth=3, n_estimators=200`); validation F1 সর্বাধিক করে threshold **0.39**।

## 1. তিনটি মডেল

প্রতিটি মডেল সবসময় `Pipeline([("preprocess", ColumnTransformer), ("model", estimator)])` হিসেবে মোড়ানো। কারণ: (১) training ও prediction-এ একই রূপান্তর প্রয়োগ হয়; (২) cross-validation প্রতিটি fold-এর ভেতরে প্রি-প্রসেসিং নতুন করে ফিট করে (লিকেজ নেই); (৩) প্রতিযোগিতার ফাইলসহ ভবিষ্যদ্বাণীর জন্য একটিমাত্র অবজেক্ট (`fit`, `predict_proba`) যথেষ্ট।

### 1.1 লজিস্টিক রিগ্রেশন

*কী শেখে:* প্রতিটি ইনপুট কলামের জন্য একটি coefficient এবং একটি intercept। *কীভাবে অনুমান করে:* `z = intercept + Σ wᵢxᵢ` হিসাব করে, তারপর sigmoid `p = 1/(1+e^−z)` দিয়ে P(`Exited=1`) পায়। সিদ্ধান্ত-সীমা (decision boundary) (scaled, one-hot) ফিচারে রৈখিক। ফিট হয় log-loss + L2 penalty (`penalty='l2'`) ন্যূনতম করে। *শক্তি:* দ্রুত, স্থিতিশীল, ব্যাখ্যাযোগ্য weight। *সীমা:* feature engineering ছাড়া non-linear প্রভাব বা interaction ধরতে পারে না। মূল সেটিং: `C` (regularisation-এর বিপরীত শক্তি; ছোট মান = বেশি shrinkage), `class_weight` (`'balanced'` ফিটিংয়ে সংখ্যালঘু ক্লাসের ওজন বাড়ায়), `max_iter=1000` (optimiser যেন converge করে)। scaled preprocessor ব্যবহার করে।

### 1.2 র‍্যান্ডম ফরেস্ট

*কী শেখে:* বহু decision tree; প্রতিটি training সারির bootstrap নমুনায় তৈরি, এবং প্রতি split-এ ফিচারের একটি এলোমেলো উপসেট বিবেচনা করে (`max_features='sqrt'`)। *কীভাবে অনুমান করে:* প্রতিটি tree ভোট/ক্লাস-সম্ভাবনা দেয়; forest সেগুলোর গড় নেয়। অনেক ভিন্ন গভীর tree-র গড় একক tree-র variance (overfitting) কমায়। *শক্তি:* non-linearity ও interaction ধরে, scaling লাগে না, শক্তপোক্ত। *সীমা:* বড়/ধীর, কম ব্যাখ্যাযোগ্য, সম্ভাবনা কম calibrated হতে পারে। টিউন করা সেটিং: `n_estimators`, `max_depth`, `min_samples_leaf`।

### 1.3 গ্রেডিয়েন্ট বুস্টিং

*কী শেখে:* ছোট (shallow) regression tree-র একটি ক্রম। প্রতিটি নতুন tree বর্তমান ensemble-এর log-loss-এর gradient (মোটামুটি residual error) ফিট করে; তার অবদান `learning_rate` দিয়ে স্কেল করে যোগ করা হয়। *কীভাবে অনুমান করে:* স্কেল-করা tree-আউটপুট যোগ করে log-odds স্কোর বানায়, তারপর sigmoid প্রয়োগ করে। *শক্তি:* ট্যাবুলার ডেটায় প্রায়ই সবচেয়ে নিখুঁত; *সীমা:* হাইপারপ্যারামিটারের প্রতি বেশি সংবেদনশীল, ক্রমিক training (ধীর), অতিরিক্ত/গভীর tree ও উচ্চ learning rate হলে overfit করতে পারে। টিউন করা সেটিং: `n_estimators`, `learning_rate`, `max_depth`।

### 1.4 রৈখিক মডেল বনাম tree ensemble

রৈখিক মডেল: প্রতি ফিচারের বৈশ্বিক, যোগাত্মক প্রভাব। Tree: piecewise-constant অঞ্চল, যা "বয়স ৪৫–৬০ *এবং* একটি product"-এর মতো interaction স্বয়ংক্রিয়ভাবে ধরতে পারে। Random Forest স্বাধীন tree-র গড় নিয়ে variance কমায়; Gradient Boosting ক্রমান্বয়ে ভুল শুধরে bias কমায়।

## 2. প্যারামিটার বনাম হাইপারপ্যারামিটার

*Parameter* ডেটা থেকে `fit`-এর সময় শেখা হয় (logistic coefficient, tree split threshold, imputer median)। *Hyperparameter* অনুশীলনকারী ফিটিংয়ের আগে ঠিক করেন (`C`, `max_depth`, `learning_rate`, fold-সংখ্যা) এবং পরীক্ষার মাধ্যমে নির্বাচন করেন।

## 3. Task 6 — বেসলাইন (validation split, threshold 0.5, কোনো imbalance প্রতিকার ছাড়া)

প্রায়-ডিফল্ট সেটিং (LR `C=1`, `max_iter=1000`; RF ২০০টি tree; GB ডিফল্ট: ১০০টি tree, learning rate 0.1, depth 3); সবগুলোয় `random_state=42`।

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.8978 | 0.7963 | 0.6674 | 0.7262 | 0.9288 |
| Random Forest | 0.8831 | 0.7487 | 0.6389 | 0.6895 | 0.9137 |
| Logistic Regression | 0.8729 | 0.7568 | 0.5514 | 0.6380 | 0.8760 |

এই validation split-এ দেখানো সব metric-এ Gradient Boosting প্রথম এবং Logistic Regression শেষ। যেহেতু churner ২০.৩%, threshold 0.5-এ তিনটি মডেলই উল্লেখযোগ্য সংখ্যক churner মিস করে (recall ০.৫৫–০.৬৭)।

## 4. Task 7 — টিউনিং কৌশল

* **নির্বাচনী metric (তুলনার আগেই ঠিক করা): ROC-AUC।** এটি threshold-নিরপেক্ষ ranking মান মাপে, ফলে threshold পরে আলাদাভাবে বাছা যায়।
* **পদ্ধতি:** `GridSearchCV(pipeline, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True)` শুধু `X_train, y_train`-এ চালানো। প্রতিটি grid বিন্দু ও প্রতিটি ৫ fold-এর জন্য scikit-learn pipeline ক্লোন করে training সারির ৪/৫ অংশে (প্রি-প্রসেসিংসহ) ফিট করে, বাকি ১/৫-এ স্কোর করে এবং গড় নেয়। Stratified fold প্রতিটিতে ২০% churn অনুপাত বজায় রাখে। অনুসন্ধানের পর সেরা সেটিং পুরো training split-এ পুনরায় ফিট হয় (`refit=True`); validation-এর ভূমিকা অক্ষত রাখতে train+validation একত্র করা **হয় না**।
* **Search space (অনুসন্ধান-পরিসর):**
  * Logistic Regression: `C ∈ {0.01, 0.1, 1, 10}` × `class_weight ∈ {None, 'balanced'}` = ৮টি সেটিং
  * Random Forest: `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 8, 16}` × `min_samples_leaf ∈ {1, 5}` = ১২টি সেটিং
  * Gradient Boosting: `n_estimators ∈ {100, 200}` × `learning_rate ∈ {0.05, 0.1}` × `max_depth ∈ {2, 3, 4}` = ১২টি সেটিং

  Colab-এ যুক্তিসঙ্গত সময়ে চলার জন্য grid ইচ্ছাকৃতভাবে ছোট (৩২টি সেটিং × ৫ fold = ১৬০টি fit)।
* **`class_weight`** শুধু Logistic Regression-এর হাইপারপ্যারামিটার হিসেবে আছে এবং CV প্রমাণে বিচার করা হয়; কোনো resampling বা অন্য imbalance প্রতিকার ব্যবহার হয়নি, এবং tree মডেলে কোনোটিই নেই।

### 4.1 ফলাফল (প্রতি মডেলের সেরা; mean ± std CV ROC-AUC-সহ ৩২টি সেটিংয়ের পূর্ণ টেবিল নোটবুক §9-এ প্রিন্ট হয়)

| Model | Best CV ROC-AUC (train folds) | Best setting |
|---|---|---|
| Logistic Regression | 0.8783 | `C=1`, `class_weight='balanced'` |
| Random Forest | 0.9292 | `max_depth=8`, `min_samples_leaf=1`, `n_estimators=300` |
| Gradient Boosting | 0.9325 | `learning_rate=0.05`, `max_depth=3`, `n_estimators=200` |

Logistic Regression-এর কয়েকটি সেটিং শুধু চতুর্থ দশমিকে ভিন্ন (যেমন 0.8774 বনাম 0.8783), অথচ fold-এর standard deviation ≈ 0.01; তাই ওই পার্থক্যগুলো noise-এর মধ্যে।

### 4.2 টিউন-করা মডেলের validation তুলনা (threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | baseline ROC-AUC |
|---|---|---|---|---|---|---|
| Gradient Boosting | 0.8960 | 0.7911 | 0.6630 | 0.7214 | 0.9283 | 0.9288 |
| Random Forest | 0.8938 | 0.8187 | 0.6127 | 0.7009 | 0.9256 | 0.9137 |
| Logistic Regression | 0.8133 | 0.5268 | 0.7965 | 0.6341 | 0.8783 | 0.8760 |

**পর্যবেক্ষণ:** টিউনিং Random Forest-এর validation ROC-AUC লক্ষণীয়ভাবে বাড়িয়েছে (0.9137 → 0.9256), Logistic Regression-এর সামান্য, আর Gradient Boosting-এর মোটেই নয় (0.9288 → 0.9283; কার্যত অপরিবর্তিত, অর্থাৎ তার ডিফল্টই grid-এর সর্বোত্তমের কাছাকাছি ছিল)। Logistic Regression-এ `class_weight='balanced'` recall বাড়িয়েছে (0.55 → 0.80) কিন্তু precision কমিয়েছে (0.76 → 0.53) — এটি ranking উন্নত করে না, বরং operating point সরায় (ROC-AUC প্রায় অপরিবর্তিত)।

**নির্বাচিত মডেল: Gradient Boosting** (`learning_rate=0.05, max_depth=3, n_estimators=200`) — সর্বোচ্চ validation ROC-AUC (0.9283) ও সর্বোচ্চ CV ROC-AUC (0.9325)। টিউন-করা Random Forest-এর চেয়ে validation-এ ব্যবধান ছোট (0.0027), তাই পছন্দটি সমর্থিত কিন্তু চূড়ান্ত নয়।

## 5. সিদ্ধান্ত-threshold

শুধু **validation** split-এ 0.10–0.90 grid (ধাপ 0.01) থেকে বাছা হয়েছে, validation F1 সর্বাধিক করে: **0.39** (validation F1 0.7410, যেখানে 0.5-এ 0.7214)। যে ডেটায় পরে ফলাফল জানানো হয়, সেই ডেটাতেই threshold বাছলে validation সংখ্যা সামান্য আশাবাদী হয়; এজন্যই চূড়ান্ত দাবি অস্পৃষ্ট internal test সেটের ওপর নির্ভর করে ([doc 05](05_evaluation_and_interpretation.md))।

## 6. সাধারণ সমস্যা

* অবৈধ grid key: pipeline-এর ভেতরের ধাপের প্যারামিটারে `stepname__param` রূপ ব্যবহার করুন (`model__C`)।
* `class_weight='balanced'`-এর সঙ্গে অন্য imbalance সমাধান মেশালে সংশোধন দ্বিগুণ গণনা হয়।
* অনুসন্ধান ধীর হলে grid ছোট করুন, `n_jobs=-1` দিন, বা `RandomizedSearchCV` ব্যবহার করুন।

