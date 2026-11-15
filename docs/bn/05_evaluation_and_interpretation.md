# 05 — মূল্যায়ন ও ব্যাখ্যা

> **📌 এক নজরে**
> চূড়ান্ত মডেল (Gradient Boosting, threshold 0.39) internal test সেটে একবার মূল্যায়িত হয়েছে: ROC-AUC 0.9329, Recall 0.7637, Precision 0.7211, F1 0.7418। Train ও validation স্কোর কাছাকাছি, তাই overfitting-এর শক্তিশালী লক্ষণ নেই।

Positive class = `Exited = 1` (গ্রাহক ছেড়ে গেছেন)। Threshold-নির্ভর metric-এ সম্ভাবনা ≥ threshold হলে churn অনুমান ধরা হয়; চূড়ান্ত threshold **0.39** (validation F1 দিয়ে বাছা)।

## 1. সহজ ভাষায় metric

ধরা যাক TP = ধরা-পড়া churner, FN = মিস-হওয়া churner, FP = ভুলে চিহ্নিত stayer, TN = সঠিকভাবে ছেড়ে-দেওয়া stayer।
* **Accuracy** = (TP+TN)/সব। সঠিক অনুমানের ভগ্নাংশ। ২০.৩% churner থাকলে "কেউ যাবে না" অনুমানেও ≈ ৭৯.৭% accuracy মেলে, অথচ কোনো কাজে আসে না — তাই শুধু accuracy বিভ্রান্তিকর।
* **Precision** = TP/(TP+FP)। churn বলে চিহ্নিতদের মধ্যে কতজন সত্যিই গেছেন।
* **Recall** = TP/(TP+FN)। যাঁরা সত্যিই গেছেন তাঁদের মধ্যে কতজনকে ধরা গেছে।
* **F1** = 2·P·R/(P+R)। harmonic mean; precision বা recall যেকোনোটি কম হলে F1 কম।
* **ROC-AUC।** প্রতিটি সম্ভাব্য threshold-এ true-positive rate বনাম false-positive rate আঁকা হয়; AUC হলো ওই বক্ররেখার নিচের ক্ষেত্রফল = একজন এলোমেলো churner একজন এলোমেলো stayer-এর চেয়ে বেশি স্কোর পাওয়ার সম্ভাবনা। ০.৫ = কাকতালীয়, ১.০ = নিখুঁত। এটি সম্ভাবনা (`predict_proba`) থেকে হিসাব হয়, hard label থেকে নয়, এবং threshold-নির্ভর নয়। নোটবুকের `safe_roc_auc` কোনো split-এ একটিমাত্র ক্লাস থাকলে (তখন AUC অসংজ্ঞায়িত) crash না করে NaN ফেরত দেয়।
* **Confusion matrix।** ২×২ গণনা-টেবিল; সারি = প্রকৃত ক্লাস, কলাম = অনুমিত ক্লাস।

## 2. Task 8 — ট্রেইন বনাম ভ্যালিডেশন (Gradient Boosting, threshold 0.39)

| Metric | Training | Validation | Gap (train − val) |
|---|---|---|---|
| Recall | 0.7402 | 0.7418 | −0.0016 |
| F1 | 0.7523 | 0.7410 | 0.0113 |
| ROC-AUC | 0.9425 | 0.9283 | 0.0142 |

Training ও validation স্কোর কাছাকাছি; সবচেয়ে বড় ব্যবধান ROC-AUC-তে ০.০১৪। এতে overfitting-এর শক্তিশালী লক্ষণ নেই (training সারি মুখস্থ করা মডেলে সাধারণত training স্কোর অনেক বেশি হয়), আর উচ্চ পরম স্কোর underfitting-এর বিপক্ষে যায়। এটি *প্রমাণ নয়, ইঙ্গিত*: validation-এ churner মাত্র ৪৫৭ জন, তাই প্রতিটি সংখ্যায় কয়েক শতাংশ-বিন্দুর sampling অনিশ্চয়তা আছে, এবং threshold একই validation split-এ বাছা হয়েছে। Validation-এর recall training-এর চেয়ে সামান্য *বেশি*, যা কেবল noise হতে পারে।

## 3. Task 9 — চূড়ান্ত একবারের internal-test মূল্যায়ন (2,250 rows, 457 churners)

| Metric | Validation | Internal test |
|---|---|---|
| Accuracy | 0.8947 | 0.8920 |
| Precision | 0.7402 | 0.7211 |
| Recall | 0.7418 | 0.7637 |
| F1 | 0.7410 | 0.7418 |
| ROC-AUC | 0.9283 | 0.9329 |

**Confusion matrix (সারি = প্রকৃত, কলাম = অনুমিত):** TN = 1658, FP = 135, FN = 108, TP = 349।

* **TN 1658** — থেকে গেছেন, থাকবেন বলেই অনুমিত।
* **FP 135** — থেকে গেছেন কিন্তু চিহ্নিত হয়েছেন: একটি retention অফার/কল নষ্ট (এবং সম্ভাব্য বিরক্তি)।
* **FN 108** — ছেড়ে গেছেন কিন্তু ধরা পড়েননি: গ্রাহক হারানো, হস্তক্ষেপের সুযোগ নেই।
* **TP 349** — ছেড়ে গেছেন এবং ধরা পড়েছেন: retention পদক্ষেপ সম্ভব।

Recall = 349/457 = 0.764; precision = 349/484 = 0.721। Internal-test ফল validation-এর সঙ্গে সামঞ্জস্যপূর্ণ (ROC-AUC +0.0045, F1 +0.0008); এই মাপের পার্থক্য ২,২৫০ সারিতে sampling variation-এর মধ্যেই পড়ে। নোটবুক `X_test`/`y_test` কেবল §11-এ স্কোরিংয়ের জন্য পড়েছে — মডেল, হাইপারপ্যারামিটার ও threshold চূড়ান্ত হওয়ার পর; এরপর কিছুই বদলানো হয়নি।

## 4. ব্যাংকের জন্য খরচের ভারসাম্য

False negative-এ গ্রাহকের lifetime value হারায়; false positive-এ একটি অফার ও সদিচ্ছার খরচ হয়। Retention-এর ক্ষেত্রে হারানো গ্রাহকের খরচ সাধারণত নষ্ট অফারের চেয়ে বেশি হলে নিচু threshold (বেশি recall, কম precision) বেশি পছন্দনীয়; F1 দুই ধরনের ভুলকে সমান গুরুত্ব দেয়, যা একটি অনুমান, ব্যবসায়িক সত্য নয়। প্রকৃত খরচ জানা ছিল না এবং মডেল করা হয়নি।

## 5. সম্পর্ক, পূর্বাভাস, কারণ ও ক্যালিব্রেশন

* *Association:* "Germany-তে churn বেশি" ডেটার বর্ণনা। *Prediction:* মডেল এ ধরনের নকশা ব্যবহার করে নতুন গ্রাহকের স্কোর দেয়। *Causation:* এখানে কিছুই প্রমাণ করে না যে কোনো ফিচার বদলালে (যেমন দ্বিতীয় product বিক্রি) churn বদলাবে; হস্তক্ষেপ বুঝতে পরীক্ষণ (experiment) দরকার।
* *Calibration:* ০.৩৯ অনুমিত সম্ভাবনা মানে এমন গ্রাহকদের ৩৯% সত্যিই যাবেন — তা নিশ্চিত নয়। Boosted-tree সম্ভাবনা calibrated হওয়ার নিশ্চয়তা নেই, এবং কোনো calibration পরীক্ষা (reliability curve) করা হয়নি; তাই সম্ভাবনাকে ঝুঁকি-**স্কোর** হিসেবে দেখা উচিত।

## 6. সীমাবদ্ধতা ও সম্ভাব্য উন্নতি

*এই প্রজেক্টে পর্যবেক্ষিত:* `CustomerId` পুনরাবৃত্তির কারণে গ্রাহক split-এর মধ্যে ফাঁস হতে পারে; ফিচার-রেকর্ডের সময় যাচাই করা হয়নি; একটিমাত্র split ও একটিমাত্র seed (পুনরাবৃত্ত-split variance নেই); একটিমাত্র threshold-মানদণ্ড। *সম্ভাব্য উন্নতি (বাস্তবায়িত নয়):* grouped splitting, আত্মবিশ্বাস-ব্যবধানের জন্য repeated CV, probability calibration, cost-sensitive threshold নির্বাচন, geography/gender/age অনুযায়ী fairness বিশ্লেষণ, learning curve ও feature importance।
