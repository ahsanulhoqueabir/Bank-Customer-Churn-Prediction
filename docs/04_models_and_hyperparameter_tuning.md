# 04 — Models and Hyperparameter Tuning | মডেল ও হাইপারপ্যারামিটার টিউনিং

> **📌 এক নজরে (At a glance)**
> তিনটি মডেল — Logistic Regression, Random Forest, Gradient Boosting — প্রতিটি `Pipeline`-এ মোড়ানো। Baseline-এ Gradient Boosting সেরা (validation ROC-AUC 0.9288)। `GridSearchCV` (5-fold, ROC-AUC) চালানোর পর নির্বাচিত মডেল: **Gradient Boosting** (`learning_rate=0.05, max_depth=3, n_estimators=200`); validation F1 সর্বাধিক করে threshold **0.39**।
> *Three pipelines compared; Gradient Boosting selected; threshold 0.39 chosen on validation F1.*

## 1. The three models | তিনটি মডেল

**বাংলা:** প্রতিটি মডেল সবসময় `Pipeline([("preprocess", ColumnTransformer), ("model", estimator)])` হিসেবে মোড়ানো। কারণ: (১) training ও prediction-এ একই রূপান্তর প্রয়োগ হয়; (২) cross-validation প্রতিটি fold-এর ভেতরে প্রি-প্রসেসিং নতুন করে ফিট করে (লিকেজ নেই); (৩) প্রতিযোগিতার ফাইলসহ ভবিষ্যদ্বাণীর জন্য একটিমাত্র অবজেক্ট (`fit`, `predict_proba`) যথেষ্ট।
**English:** Each model is always wrapped as `Pipeline([("preprocess", ColumnTransformer), ("model", estimator)])`. Reasons: (1) the exact same transformation is applied at training and prediction time; (2) cross-validation refits preprocessing inside each fold (no leakage); (3) one object (`fit`, `predict_proba`) is all that is needed to deploy/predict, including for the competition file.

### 1.1 Logistic Regression | লজিস্টিক রিগ্রেশন

**বাংলা:** *কী শেখে:* প্রতিটি ইনপুট কলামের জন্য একটি coefficient এবং একটি intercept। *কীভাবে অনুমান করে:* `z = intercept + Σ wᵢxᵢ` হিসাব করে, তারপর sigmoid `p = 1/(1+e^−z)` দিয়ে P(`Exited=1`) পায়। সিদ্ধান্ত-সীমা (decision boundary) (scaled, one-hot) ফিচারে রৈখিক। ফিট হয় log-loss + L2 penalty (`penalty='l2'`) ন্যূনতম করে। *শক্তি:* দ্রুত, স্থিতিশীল, ব্যাখ্যাযোগ্য weight। *সীমা:* feature engineering ছাড়া non-linear প্রভাব বা interaction ধরতে পারে না। মূল সেটিং: `C` (regularisation-এর বিপরীত শক্তি; ছোট মান = বেশি shrinkage), `class_weight` (`'balanced'` ফিটিংয়ে সংখ্যালঘু ক্লাসের ওজন বাড়ায়), `max_iter=1000` (optimiser যেন converge করে)। scaled preprocessor ব্যবহার করে।
**English:** *Learns:* one coefficient per input column plus an intercept. *Predicts:* computes `z = intercept + Σ wᵢxᵢ`, then the sigmoid `p = 1/(1+e^−z)` gives P(`Exited=1`). The decision boundary is linear in the (scaled, one-hot) features. Fitted by minimising log-loss plus an L2 penalty (`penalty='l2'`). *Strengths:* fast, stable, interpretable weights. *Limits:* cannot capture non-linear effects or interactions unless features are engineered. Key settings: `C` (inverse regularisation strength; smaller = stronger shrinkage), `class_weight` (`'balanced'` up-weights the minority class during fitting), `max_iter=1000` (set so the optimiser converges). Uses the scaled preprocessor.

### 1.2 Random Forest | র‍্যান্ডম ফরেস্ট

**বাংলা:** *কী শেখে:* বহু decision tree; প্রতিটি training সারির bootstrap নমুনায় তৈরি, এবং প্রতি split-এ ফিচারের একটি এলোমেলো উপসেট বিবেচনা করে (`max_features='sqrt'`)। *কীভাবে অনুমান করে:* প্রতিটি tree ভোট/ক্লাস-সম্ভাবনা দেয়; forest সেগুলোর গড় নেয়। অনেক ভিন্ন গভীর tree-র গড় একক tree-র variance (overfitting) কমায়। *শক্তি:* non-linearity ও interaction ধরে, scaling লাগে না, শক্তপোক্ত। *সীমা:* বড়/ধীর, কম ব্যাখ্যাযোগ্য, সম্ভাবনা কম calibrated হতে পারে। টিউন করা সেটিং: `n_estimators`, `max_depth`, `min_samples_leaf`।
**English:** *Learns:* many decision trees, each grown on a bootstrap resample of the training rows, considering a random subset of features at each split (`max_features='sqrt'`). *Predicts:* each tree votes/outputs a class probability; the forest averages them. Averaging many different deep trees reduces variance (overfitting of a single tree). *Strengths:* captures non-linearities and interactions, needs no scaling, robust. *Limits:* larger/slower, less interpretable, probabilities can be less well calibrated. Settings tuned: `n_estimators`, `max_depth`, `min_samples_leaf`.

### 1.3 Gradient Boosting (`GradientBoostingClassifier`) | গ্রেডিয়েন্ট বুস্টিং

**বাংলা:** *কী শেখে:* ছোট (shallow) regression tree-র একটি ক্রম। প্রতিটি নতুন tree বর্তমান ensemble-এর log-loss-এর gradient (মোটামুটি residual error) ফিট করে; তার অবদান `learning_rate` দিয়ে স্কেল করে যোগ করা হয়। *কীভাবে অনুমান করে:* স্কেল-করা tree-আউটপুট যোগ করে log-odds স্কোর বানায়, তারপর sigmoid প্রয়োগ করে। *শক্তি:* ট্যাবুলার ডেটায় প্রায়ই সবচেয়ে নিখুঁত; *সীমা:* হাইপারপ্যারামিটারের প্রতি বেশি সংবেদনশীল, ক্রমিক training (ধীর), অতিরিক্ত/গভীর tree ও উচ্চ learning rate হলে overfit করতে পারে। টিউন করা সেটিং: `n_estimators`, `learning_rate`, `max_depth`।
**English:** *Learns:* a sequence of shallow regression trees. Each new tree fits the gradient of the log-loss (roughly, the residual errors) of the current ensemble; its contribution is scaled by `learning_rate` and added. *Predicts:* sums the scaled tree outputs into a log-odds score, then applies a sigmoid. *Strengths:* often the most accurate on tabular data; *limits:* more hyperparameter-sensitive, sequential training (slower), can overfit with too many/too deep trees and a high learning rate. Settings tuned: `n_estimators`, `learning_rate`, `max_depth`.

### 1.4 Linear vs tree ensembles | রৈখিক মডেল বনাম tree ensemble

**বাংলা:** রৈখিক মডেল: প্রতি ফিচারের বৈশ্বিক, যোগাত্মক প্রভাব। Tree: piecewise-constant অঞ্চল, যা "বয়স ৪৫–৬০ *এবং* একটি product"-এর মতো interaction স্বয়ংক্রিয়ভাবে ধরতে পারে। Random Forest স্বাধীন tree-র গড় নিয়ে variance কমায়; Gradient Boosting ক্রমান্বয়ে ভুল শুধরে bias কমায়।
**English:** Linear: global, additive effect per feature. Trees: piecewise-constant regions that can represent "age 45–60 *and* one product" style interactions automatically. Random Forest reduces variance by averaging independent trees; Gradient Boosting reduces bias by sequentially correcting errors.

## 2. Parameters vs hyperparameters | প্যারামিটার বনাম হাইপারপ্যারামিটার

**বাংলা:** *Parameter* ডেটা থেকে `fit`-এর সময় শেখা হয় (logistic coefficient, tree split threshold, imputer median)। *Hyperparameter* অনুশীলনকারী ফিটিংয়ের আগে ঠিক করেন (`C`, `max_depth`, `learning_rate`, fold-সংখ্যা) এবং পরীক্ষার মাধ্যমে নির্বাচন করেন।
**English:** *Parameters* are learned from data by `fit` (logistic coefficients, tree split thresholds, imputer medians). *Hyperparameters* are chosen by the practitioner before fitting (`C`, `max_depth`, `learning_rate`, number of folds) and are selected by experiment.

## 3. Task 6 — baselines | বেসলাইন (validation split, threshold 0.5, no imbalance remedy)

**বাংলা:** প্রায়-ডিফল্ট সেটিং (LR `C=1`, `max_iter=1000`; RF ২০০টি tree; GB ডিফল্ট: ১০০টি tree, learning rate 0.1, depth 3); সবগুলোয় `random_state=42`।
**English:** Default-ish settings (LR `C=1`, `max_iter=1000`; RF 200 trees; GB defaults: 100 trees, learning rate 0.1, depth 3); all with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.8978 | 0.7963 | 0.6674 | 0.7262 | 0.9288 |
| Random Forest | 0.8831 | 0.7487 | 0.6389 | 0.6895 | 0.9137 |
| Logistic Regression | 0.8729 | 0.7568 | 0.5514 | 0.6380 | 0.8760 |

**বাংলা:** এই validation split-এ দেখানো সব metric-এ Gradient Boosting প্রথম এবং Logistic Regression শেষ। যেহেতু churner ২০.৩%, threshold 0.5-এ তিনটি মডেলই উল্লেখযোগ্য সংখ্যক churner মিস করে (recall ০.৫৫–০.৬৭)।
**English:** On this validation split Gradient Boosting ranked first on every metric shown and Logistic Regression last. Because churners are 20.3% of rows, all three models miss a substantial fraction of churners at threshold 0.5 (recall 0.55–0.67).

## 4. Task 7 — tuning strategy | টিউনিং কৌশল

**বাংলা:**
* **নির্বাচনী metric (তুলনার আগেই ঠিক করা): ROC-AUC।** এটি threshold-নিরপেক্ষ ranking মান মাপে, ফলে threshold পরে আলাদাভাবে বাছা যায়।
* **পদ্ধতি:** `GridSearchCV(pipeline, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True)` শুধু `X_train, y_train`-এ চালানো। প্রতিটি grid বিন্দু ও প্রতিটি ৫ fold-এর জন্য scikit-learn pipeline ক্লোন করে training সারির ৪/৫ অংশে (প্রি-প্রসেসিংসহ) ফিট করে, বাকি ১/৫-এ স্কোর করে এবং গড় নেয়। Stratified fold প্রতিটিতে ২০% churn অনুপাত বজায় রাখে। অনুসন্ধানের পর সেরা সেটিং পুরো training split-এ পুনরায় ফিট হয় (`refit=True`); validation-এর ভূমিকা অক্ষত রাখতে train+validation একত্র করা **হয় না**।
* **Search space (অনুসন্ধান-পরিসর):**
  * Logistic Regression: `C ∈ {0.01, 0.1, 1, 10}` × `class_weight ∈ {None, 'balanced'}` = ৮টি সেটিং
  * Random Forest: `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 8, 16}` × `min_samples_leaf ∈ {1, 5}` = ১২টি সেটিং
  * Gradient Boosting: `n_estimators ∈ {100, 200}` × `learning_rate ∈ {0.05, 0.1}` × `max_depth ∈ {2, 3, 4}` = ১২টি সেটিং

  Colab-এ যুক্তিসঙ্গত সময়ে চলার জন্য grid ইচ্ছাকৃতভাবে ছোট (৩২টি সেটিং × ৫ fold = ১৬০টি fit)।
* **`class_weight`** শুধু Logistic Regression-এর হাইপারপ্যারামিটার হিসেবে আছে এবং CV প্রমাণে বিচার করা হয়; কোনো resampling বা অন্য imbalance প্রতিকার ব্যবহার হয়নি, এবং tree মডেলে কোনোটিই নেই।

**English:**
* **Selection metric (decided before comparing): ROC-AUC.** It measures ranking quality independent of any threshold, so the threshold can be chosen separately afterwards.
* **Method:** `GridSearchCV(pipeline, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True)` fitted on `X_train, y_train` only. For each grid point and each of 5 folds, scikit-learn clones the pipeline, fits it (preprocessing included) on 4/5 of the training rows, scores on the remaining 1/5, and averages. Stratified folds keep the 20% churn share in every fold. After the search, the best setting is refit on the whole training split (`refit=True`; train+validation is **not** merged, to keep the validation role clean).
* **Search spaces:**
  * Logistic Regression: `C ∈ {0.01, 0.1, 1, 10}` × `class_weight ∈ {None, 'balanced'}` = 8 settings
  * Random Forest: `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 8, 16}` × `min_samples_leaf ∈ {1, 5}` = 12 settings
  * Gradient Boosting: `n_estimators ∈ {100, 200}` × `learning_rate ∈ {0.05, 0.1}` × `max_depth ∈ {2, 3, 4}` = 12 settings

  Grids are intentionally small so the notebook runs in reasonable time on Colab (32 settings × 5 folds = 160 fits).
* **`class_weight`** appears only as a Logistic Regression hyperparameter, judged by CV evidence; no resampling or other imbalance remedy is used, and the tree models use none.

### 4.1 Results | ফলাফল (best per model; the complete table of all 32 settings with mean ± std CV ROC-AUC is printed in notebook §9)

| Model | Best CV ROC-AUC (train folds) | Best setting |
|---|---|---|
| Logistic Regression | 0.8783 | `C=1`, `class_weight='balanced'` |
| Random Forest | 0.9292 | `max_depth=8`, `min_samples_leaf=1`, `n_estimators=300` |
| Gradient Boosting | 0.9325 | `learning_rate=0.05`, `max_depth=3`, `n_estimators=200` |

**বাংলা:** Logistic Regression-এর কয়েকটি সেটিং শুধু চতুর্থ দশমিকে ভিন্ন (যেমন 0.8774 বনাম 0.8783), অথচ fold-এর standard deviation ≈ 0.01; তাই ওই পার্থক্যগুলো noise-এর মধ্যে।
**English:** Several Logistic Regression settings differ only in the 4th decimal (e.g. 0.8774 vs 0.8783) with a fold standard deviation ≈ 0.01, so those differences are within noise.

### 4.2 Validation comparison of the tuned models | টিউন-করা মডেলের validation তুলনা (threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | baseline ROC-AUC |
|---|---|---|---|---|---|---|
| Gradient Boosting | 0.8960 | 0.7911 | 0.6630 | 0.7214 | 0.9283 | 0.9288 |
| Random Forest | 0.8938 | 0.8187 | 0.6127 | 0.7009 | 0.9256 | 0.9137 |
| Logistic Regression | 0.8133 | 0.5268 | 0.7965 | 0.6341 | 0.8783 | 0.8760 |

**বাংলা পর্যবেক্ষণ:** টিউনিং Random Forest-এর validation ROC-AUC লক্ষণীয়ভাবে বাড়িয়েছে (0.9137 → 0.9256), Logistic Regression-এর সামান্য, আর Gradient Boosting-এর মোটেই নয় (0.9288 → 0.9283; কার্যত অপরিবর্তিত, অর্থাৎ তার ডিফল্টই grid-এর সর্বোত্তমের কাছাকাছি ছিল)। Logistic Regression-এ `class_weight='balanced'` recall বাড়িয়েছে (0.55 → 0.80) কিন্তু precision কমিয়েছে (0.76 → 0.53) — এটি ranking উন্নত করে না, বরং operating point সরায় (ROC-AUC প্রায় অপরিবর্তিত)।

**নির্বাচিত মডেল: Gradient Boosting** (`learning_rate=0.05, max_depth=3, n_estimators=200`) — সর্বোচ্চ validation ROC-AUC (0.9283) ও সর্বোচ্চ CV ROC-AUC (0.9325)। টিউন-করা Random Forest-এর চেয়ে validation-এ ব্যবধান ছোট (0.0027), তাই পছন্দটি সমর্থিত কিন্তু চূড়ান্ত নয়।

**English observations:** tuning improved Random Forest validation ROC-AUC noticeably (0.9137 → 0.9256), Logistic Regression marginally, and Gradient Boosting not at all (0.9288 → 0.9283; essentially unchanged, i.e. its defaults were already near the grid optimum). `class_weight='balanced'` in Logistic Regression raised recall (0.55 → 0.80) at the cost of precision (0.76 → 0.53) — it moves the operating point rather than improving ranking (ROC-AUC barely changed).

**Selected model: Gradient Boosting** (`learning_rate=0.05, max_depth=3, n_estimators=200`) — highest validation ROC-AUC (0.9283) and highest CV ROC-AUC (0.9325). The margin over the tuned Random Forest on validation is small (0.0027), so the choice is supported but not decisive.

## 5. Decision threshold | সিদ্ধান্ত-threshold

**বাংলা:** শুধু **validation** split-এ 0.10–0.90 grid (ধাপ 0.01) থেকে বাছা হয়েছে, validation F1 সর্বাধিক করে: **0.39** (validation F1 0.7410, যেখানে 0.5-এ 0.7214)। যে ডেটায় পরে ফলাফল জানানো হয়, সেই ডেটাতেই threshold বাছলে validation সংখ্যা সামান্য আশাবাদী হয়; এজন্যই চূড়ান্ত দাবি অস্পৃষ্ট internal test সেটের ওপর নির্ভর করে ([doc 05](05_evaluation_and_interpretation.md))।
**English:** Chosen on the **validation** split only, from a grid 0.10–0.90 (step 0.01), maximising validation F1: **0.39** (validation F1 0.7410 versus 0.7214 at 0.5). Choosing it on the same data it is later reported on makes the validation numbers slightly optimistic; that is why the final claim rests on the untouched internal test set (doc 05).

## 6. Common problems | সাধারণ সমস্যা

**বাংলা:**
* অবৈধ grid key: pipeline-এর ভেতরের ধাপের প্যারামিটারে `stepname__param` রূপ ব্যবহার করুন (`model__C`)।
* `class_weight='balanced'`-এর সঙ্গে অন্য imbalance সমাধান মেশালে সংশোধন দ্বিগুণ গণনা হয়।
* অনুসন্ধান ধীর হলে grid ছোট করুন, `n_jobs=-1` দিন, বা `RandomizedSearchCV` ব্যবহার করুন।

**English:**
* Invalid grid keys: parameters of a step inside a pipeline use the `stepname__param` form (`model__C`).
* Mixing `class_weight='balanced'` with other imbalance fixes double-counts the correction.
* Slow search → reduce grid size, use `n_jobs=-1`, or switch to `RandomizedSearchCV`.
