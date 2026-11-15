# 05 — Evaluation and Interpretation | মূল্যায়ন ও ব্যাখ্যা

> **📌 এক নজরে (At a glance)**
> চূড়ান্ত মডেল (Gradient Boosting, threshold 0.39) internal test সেটে একবার মূল্যায়িত হয়েছে: ROC-AUC 0.9329, Recall 0.7637, Precision 0.7211, F1 0.7418। Train ও validation স্কোর কাছাকাছি, তাই overfitting-এর শক্তিশালী লক্ষণ নেই।
> *Final internal-test result: ROC-AUC 0.9329, Recall 0.7637, Precision 0.7211, F1 0.7418; train vs validation gap is small.*

**বাংলা:** Positive class = `Exited = 1` (গ্রাহক ছেড়ে গেছেন)। Threshold-নির্ভর metric-এ সম্ভাবনা ≥ threshold হলে churn অনুমান ধরা হয়; চূড়ান্ত threshold **0.39** (validation F1 দিয়ে বাছা)।
**English:** Positive class = `Exited = 1` (customer churned). Threshold-dependent metrics use probability ≥ threshold ⇒ predicted churn; the final threshold is **0.39** (chosen on validation F1).

## 1. Metrics in plain language | সহজ ভাষায় metric

**বাংলা:** ধরা যাক TP = ধরা-পড়া churner, FN = মিস-হওয়া churner, FP = ভুলে চিহ্নিত stayer, TN = সঠিকভাবে ছেড়ে-দেওয়া stayer।
* **Accuracy** = (TP+TN)/সব। সঠিক অনুমানের ভগ্নাংশ। ২০.৩% churner থাকলে "কেউ যাবে না" অনুমানেও ≈ ৭৯.৭% accuracy মেলে, অথচ কোনো কাজে আসে না — তাই শুধু accuracy বিভ্রান্তিকর।
* **Precision** = TP/(TP+FP)। churn বলে চিহ্নিতদের মধ্যে কতজন সত্যিই গেছেন।
* **Recall** = TP/(TP+FN)। যাঁরা সত্যিই গেছেন তাঁদের মধ্যে কতজনকে ধরা গেছে।
* **F1** = 2·P·R/(P+R)। harmonic mean; precision বা recall যেকোনোটি কম হলে F1 কম।
* **ROC-AUC।** প্রতিটি সম্ভাব্য threshold-এ true-positive rate বনাম false-positive rate আঁকা হয়; AUC হলো ওই বক্ররেখার নিচের ক্ষেত্রফল = একজন এলোমেলো churner একজন এলোমেলো stayer-এর চেয়ে বেশি স্কোর পাওয়ার সম্ভাবনা। ০.৫ = কাকতালীয়, ১.০ = নিখুঁত। এটি সম্ভাবনা (`predict_proba`) থেকে হিসাব হয়, hard label থেকে নয়, এবং threshold-নির্ভর নয়। নোটবুকের `safe_roc_auc` কোনো split-এ একটিমাত্র ক্লাস থাকলে (তখন AUC অসংজ্ঞায়িত) crash না করে NaN ফেরত দেয়।
* **Confusion matrix।** ২×২ গণনা-টেবিল; সারি = প্রকৃত ক্লাস, কলাম = অনুমিত ক্লাস।

**English:** Let TP = churners caught, FN = churners missed, FP = stayers wrongly flagged, TN = stayers correctly left alone.
* **Accuracy** = (TP+TN)/all. Fraction of correct predictions. With 20.3% churners, predicting "nobody leaves" gives ≈79.7% accuracy and zero usefulness, so accuracy alone misleads.
* **Precision** = TP/(TP+FP). Of customers flagged as churn, how many really churned.
* **Recall** = TP/(TP+FN). Of customers who really churned, how many were caught.
* **F1** = 2·P·R/(P+R). Harmonic mean; low if either precision or recall is low.
* **ROC-AUC.** Plot true-positive rate against false-positive rate for every possible threshold; AUC is the area under that curve = the probability that a random churner gets a higher score than a random stayer. 0.5 = chance, 1.0 = perfect. It is computed from probabilities (`predict_proba`), not hard labels, and does not depend on a threshold. The notebook's `safe_roc_auc` returns NaN instead of crashing if a split contains only one class (AUC is undefined then).
* **Confusion matrix.** 2×2 table of counts, rows = true class, columns = predicted class.

## 2. Task 8 — training vs validation | ট্রেইন বনাম ভ্যালিডেশন (Gradient Boosting, threshold 0.39)

| Metric | Training | Validation | Gap (train − val) |
|---|---|---|---|
| Recall | 0.7402 | 0.7418 | −0.0016 |
| F1 | 0.7523 | 0.7410 | 0.0113 |
| ROC-AUC | 0.9425 | 0.9283 | 0.0142 |

**বাংলা ব্যাখ্যা:** Training ও validation স্কোর কাছাকাছি; সবচেয়ে বড় ব্যবধান ROC-AUC-তে ০.০১৪। এতে overfitting-এর শক্তিশালী লক্ষণ নেই (training সারি মুখস্থ করা মডেলে সাধারণত training স্কোর অনেক বেশি হয়), আর উচ্চ পরম স্কোর underfitting-এর বিপক্ষে যায়। এটি *প্রমাণ নয়, ইঙ্গিত*: validation-এ churner মাত্র ৪৫৭ জন, তাই প্রতিটি সংখ্যায় কয়েক শতাংশ-বিন্দুর sampling অনিশ্চয়তা আছে, এবং threshold একই validation split-এ বাছা হয়েছে। Validation-এর recall training-এর চেয়ে সামান্য *বেশি*, যা কেবল noise হতে পারে।
**English reading:** training and validation scores are close; the largest gap is 0.014 in ROC-AUC. This gives no strong sign of overfitting (a model that memorised training rows would typically show a much higher training score) and the high absolute scores argue against underfitting. This is evidence, not proof: the validation set has only 457 churners, so each number has sampling uncertainty of roughly a few percentage points, and the threshold was chosen on this same validation split. Recall on validation is slightly *above* training, which can only be noise.

## 3. Task 9 — final one-time internal-test evaluation | চূড়ান্ত একবারের internal-test মূল্যায়ন (2,250 rows, 457 churners)

| Metric | Validation | Internal test |
|---|---|---|
| Accuracy | 0.8947 | 0.8920 |
| Precision | 0.7402 | 0.7211 |
| Recall | 0.7418 | 0.7637 |
| F1 | 0.7410 | 0.7418 |
| ROC-AUC | 0.9283 | 0.9329 |

**Confusion matrix (rows true, columns predicted):** TN = 1658, FP = 135, FN = 108, TP = 349.

**বাংলা:**
* **TN 1658** — থেকে গেছেন, থাকবেন বলেই অনুমিত।
* **FP 135** — থেকে গেছেন কিন্তু চিহ্নিত হয়েছেন: একটি retention অফার/কল নষ্ট (এবং সম্ভাব্য বিরক্তি)।
* **FN 108** — ছেড়ে গেছেন কিন্তু ধরা পড়েননি: গ্রাহক হারানো, হস্তক্ষেপের সুযোগ নেই।
* **TP 349** — ছেড়ে গেছেন এবং ধরা পড়েছেন: retention পদক্ষেপ সম্ভব।

Recall = 349/457 = 0.764; precision = 349/484 = 0.721। Internal-test ফল validation-এর সঙ্গে সামঞ্জস্যপূর্ণ (ROC-AUC +0.0045, F1 +0.0008); এই মাপের পার্থক্য ২,২৫০ সারিতে sampling variation-এর মধ্যেই পড়ে। নোটবুক `X_test`/`y_test` কেবল §11-এ স্কোরিংয়ের জন্য পড়েছে — মডেল, হাইপারপ্যারামিটার ও threshold চূড়ান্ত হওয়ার পর; এরপর কিছুই বদলানো হয়নি।

**English:**
* TN 1658 — stayed, predicted to stay.
* FP 135 — stayed but flagged: a retention offer or call wasted (and a possible annoyance).
* FN 108 — churned but not flagged: a lost customer and no chance to intervene.
* TP 349 — churned and flagged: retention action possible.

Recall = 349/457 = 0.764; precision = 349/484 = 0.721. Internal-test results are consistent with validation (ROC-AUC +0.0045, F1 +0.0008); differences of this size are within what sampling variation on 2,250 rows would produce. The notebook read `X_test`/`y_test` for scoring only in §11, after model, hyperparameters and threshold were fixed; nothing was changed afterwards.

## 4. Cost trade-off for a bank | ব্যাংকের জন্য খরচের ভারসাম্য

**বাংলা:** False negative-এ গ্রাহকের lifetime value হারায়; false positive-এ একটি অফার ও সদিচ্ছার খরচ হয়। Retention-এর ক্ষেত্রে হারানো গ্রাহকের খরচ সাধারণত নষ্ট অফারের চেয়ে বেশি হলে নিচু threshold (বেশি recall, কম precision) বেশি পছন্দনীয়; F1 দুই ধরনের ভুলকে সমান গুরুত্ব দেয়, যা একটি অনুমান, ব্যবসায়িক সত্য নয়। প্রকৃত খরচ জানা ছিল না এবং মডেল করা হয়নি।
**English:** A false negative loses the customer's lifetime value; a false positive costs an offer and goodwill. If lost customers are more expensive than wasted offers (typical in retention), a lower threshold (higher recall, lower precision) would be preferable; F1 treats both errors as equally important, which is an assumption, not a business fact. Real costs were not available and were not modelled.

## 5. Association vs prediction vs causation vs calibration | সম্পর্ক, পূর্বাভাস, কারণ ও ক্যালিব্রেশন

**বাংলা:**
* *Association:* "Germany-তে churn বেশি" ডেটার বর্ণনা। *Prediction:* মডেল এ ধরনের নকশা ব্যবহার করে নতুন গ্রাহকের স্কোর দেয়। *Causation:* এখানে কিছুই প্রমাণ করে না যে কোনো ফিচার বদলালে (যেমন দ্বিতীয় product বিক্রি) churn বদলাবে; হস্তক্ষেপ বুঝতে পরীক্ষণ (experiment) দরকার।
* *Calibration:* ০.৩৯ অনুমিত সম্ভাবনা মানে এমন গ্রাহকদের ৩৯% সত্যিই যাবেন — তা নিশ্চিত নয়। Boosted-tree সম্ভাবনা calibrated হওয়ার নিশ্চয়তা নেই, এবং কোনো calibration পরীক্ষা (reliability curve) করা হয়নি; তাই সম্ভাবনাকে ঝুঁকি-**স্কোর** হিসেবে দেখা উচিত।

**English:**
* *Association:* "Germany has higher churn" describes data. *Prediction:* the model uses such patterns to score new customers. *Causation:* nothing here shows that changing a feature (e.g. selling a second product) would change churn. Interventions need experiments.
* *Calibration:* a predicted 0.39 does not necessarily mean 39% of such customers churn. Boosted-tree probabilities are not guaranteed calibrated, and no calibration check (reliability curve) was done, so probabilities should be treated as risk **scores**.

## 6. Limitations and possible improvements | সীমাবদ্ধতা ও সম্ভাব্য উন্নতি

**বাংলা:** *এই প্রজেক্টে পর্যবেক্ষিত:* `CustomerId` পুনরাবৃত্তির কারণে গ্রাহক split-এর মধ্যে ফাঁস হতে পারে; ফিচার-রেকর্ডের সময় যাচাই করা হয়নি; একটিমাত্র split ও একটিমাত্র seed (পুনরাবৃত্ত-split variance নেই); একটিমাত্র threshold-মানদণ্ড। *সম্ভাব্য উন্নতি (বাস্তবায়িত নয়):* grouped splitting, আত্মবিশ্বাস-ব্যবধানের জন্য repeated CV, probability calibration, cost-sensitive threshold নির্বাচন, geography/gender/age অনুযায়ী fairness বিশ্লেষণ, learning curve ও feature importance।
**English:** Observed in this project: duplicate `CustomerId`s may leak customers across splits; feature timing unverified; one split and one seed (no repeated-split variance); a single threshold criterion. Possible improvements (not implemented): grouped splitting by customer, repeated CV for confidence intervals, probability calibration, cost-sensitive threshold selection, fairness analysis by geography/gender/age, learning curves and feature importance.
