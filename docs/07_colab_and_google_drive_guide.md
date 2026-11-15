# 07 — Google Colab and Google Drive Guide | Google Colab ও Google Drive নির্দেশিকা

> **📌 এক নজরে (At a glance)**
> Drive-এ `My Drive/Customer Churn Prediction/` ফোল্ডার বানিয়ে `train.csv` (ঐচ্ছিক: `test.csv`) রাখুন → নোটবুক Colab-এ খুলুন → *Runtime → Run all* → Drive অনুমতি দিন। CPU runtime যথেষ্ট; GPU লাগে না।
> *Put `train.csv` in `My Drive/Customer Churn Prediction/`, open the notebook in Colab, and use Runtime → Run all.*

## 1. Prepare Drive | Drive প্রস্তুত করুন

**বাংলা:**
1. Google Drive-এ হুবহু `My Drive/Customer Churn Prediction/` নামে ফোল্ডার বানান।
2. সেখানে `train.csv` আপলোড করুন। Task 10-এর জন্য প্রতিযোগিতার `test.csv`-ও আপলোড করুন (ঐচ্ছিক)।
3. ফলে প্রয়োজনীয় পথ: `/content/drive/MyDrive/Customer Churn Prediction/train.csv` (ঐচ্ছিক: `.../test.csv`)।

**English:**
1. In Google Drive create (exactly) `My Drive/Customer Churn Prediction/`.
2. Upload `train.csv` there. For Task 10 also upload the competition's `test.csv` (optional).
3. Resulting required path: `/content/drive/MyDrive/Customer Churn Prediction/train.csv` (optional: `.../test.csv`).

## 2. Open the notebook in Colab | Colab-এ নোটবুক খুলুন

**বাংলা:**
* *colab.research.google.com → File → Upload notebook* দিয়ে `notebooks/bank_customer_churn_prediction.ipynb` আপলোড করুন; অথবা Drive-এ রেখে *Google Colaboratory* দিয়ে খুলুন; অথবা রিপোজিটরি GitHub-এ push করা থাকলে সেখান থেকে খুলুন।
* সাধারণ CPU runtime ব্যবহার করুন; GPU লাগে না। pandas, NumPy, Matplotlib, Seaborn ও scikit-learn আগে থেকেই ইনস্টল করা, তাই কিছু `pip install` করতে হয় না।

**English:**
* Upload `notebooks/bank_customer_churn_prediction.ipynb` via *colab.research.google.com → File → Upload notebook*, or put it in Drive and open it with *Google Colaboratory*, or open it from GitHub if you push the repo.
* Use a standard CPU runtime; no GPU is needed. pandas, NumPy, Matplotlib, Seaborn and scikit-learn are preinstalled, so nothing is `pip install`ed.

## 3. Run | চালানো

**বাংলা:** *Runtime → Run all*। §2-এ একটি পপ-আপ Drive-অনুমতি চাইবে; আপনার Google অ্যাকাউন্ট দিয়ে অনুমোদন করুন। নোটবুক তখন লোড-হওয়া পথ, shape ও কলাম প্রিন্ট করবে। নিশ্চিত করুন প্রিন্ট-করা পথ হুবহু উপরের Drive-পথ (`data/train.csv` নয়, যা কেবল লোকাল fallback) এবং shape আপনার ফাইলের সঙ্গে মিলছে (লোকাল কপি: 15000 × 14)।

**মোট সময়:** hyper-parameter অনুসন্ধানই ধীর অংশ (১৬০টি pipeline fit); লোকালি বা Colab-এ সময় আনুষ্ঠানিকভাবে মাপা হয়নি, তাই লক্ষণীয় সময় লাগবে ধরে নিন এবং §9-এ ধৈর্য রাখুন।

**English:** *Runtime → Run all*. In §2 a pop-up asks you to authorise Drive access; approve it with your Google account. The notebook then prints the loaded path, shape and columns. Confirm the printed path is exactly the Drive path above (not `data/train.csv`, which is only the local fallback) and the shape matches your file (the local copy: 15000 × 14).

Total runtime: the hyper-parameter search is the slow part (160 pipeline fits); runtime was not formally timed locally or on Colab, so expect it to take noticeable time and be patient at §9.

## 4. Expected outputs | প্রত্যাশিত আউটপুট

**বাংলা:** প্রতিটি টাস্কের জন্য টেবিল প্রিন্ট হয়; চিত্র: target distribution + ৫টি churn/age প্যানেল, tenure লাইন + balance box plot, split churn-rate বার চার্ট, CV-স্কোর বার প্লট (প্রতি মডেলে একটি), validation F1-বনাম-threshold বক্ররেখা, train-vs-validation গ্রুপ বার চার্ট, internal-test confusion matrix; পাঁচ-সারির prediction টেবিল (`test.csv` থাকলে)। একই ডেটা, seed ও library সংস্করণে docs 02–05-এ উদ্ধৃত সংখ্যাই পাওয়া উচিত; অন্য সংস্করণে সামান্য পার্থক্য সম্ভব।
**English:** Printed tables for every task; figures: target distribution + 5 churn/age panels, tenure line + balance box plot, split churn-rate bar chart, CV-score bar plots (one per model), validation F1-vs-threshold curve, train-vs-validation grouped bar chart, internal-test confusion matrix; five-row prediction table (if `test.csv` exists). With the same data, seed and library versions you should get the numbers quoted in docs 02–05; small differences are possible on other versions.

## 5. Missing-file handling | ফাইল না থাকলে করণীয়

**বাংলা:**
* `train.csv` নেই → প্রত্যাশিত পথসহ `FileNotFoundError`; সঠিক ফোল্ডার/ফাইলনামে আপলোড করে (বড়/ছোট হাতের অক্ষর এবং ফোল্ডারনামের ফাঁকা স্থান গুরুত্বপূর্ণ) §2 আবার চালান।
* `test.csv` নেই → Task 10 "BLOCKED: competition test file not found…" প্রিন্ট করে এবং নোটবুক তবু শেষ হয়; `train.csv`-র পাশে `test.csv` আপলোড করে §12 আবার চালান (আগের সেলের মডেল-অবজেক্ট থাকতে হবে, না থাকলে Run all)।

**English:**
* `train.csv` missing → `FileNotFoundError` listing the expected path; fix by uploading to the exact folder/filename (case and the space in the folder name matter), then re-run §2.
* `test.csv` missing → Task 10 prints "BLOCKED: competition test file not found…" and the notebook still finishes; upload `test.csv` next to `train.csv` and re-run §12 (the model objects from earlier cells must still exist, otherwise Run all).

## 6. Save / export | সংরক্ষণ / এক্সপোর্ট

**বাংলা:** *File → Save a copy in Drive* বা *Download .ipynb*; রিপোর্টের জন্য *File → Print* → PDF হিসেবে সংরক্ষণ। আউটপুট সংরক্ষিত `.ipynb`-র ভেতরেই থাকে।
**English:** *File → Save a copy in Drive* or *Download .ipynb*; *File → Print* → save as PDF for a report. Outputs are stored inside the saved `.ipynb`.

## 7. Reproducibility and troubleshooting | পুনরুৎপাদনযোগ্যতা ও সমস্যা সমাধান

**বাংলা:**
* সর্বত্র seed 42; Colab-এর library সংস্করণ docs-এ প্রিন্ট-করা সংস্করণের চেয়ে আলাদা হতে পারে।
* "Drive not mounted"/permission error → mount সেল আবার চালিয়ে পুনরায় অনুমোদন দিন; ফাইলের মালিক অ্যাকাউন্টেই সাইন-ইন আছেন কি না দেখুন।
* সেশন রিস্টার্ট হলে → আবার Run all (সেলগুলো আগের ভ্যারিয়েবলের ওপর নির্ভরশীল)।
* ফোল্ডার আপনার সঙ্গে শেয়ার করা কিন্তু *My Drive*-এ নেই → My Drive-এ shortcut যোগ করুন বা ফোল্ডার কপি করুন।
* নোটবুক লোকালি যাচাই করা হয়েছে (কোনটি চালানো হয়েছে আর কোনটি হয়নি তা README-তে দেখুন); এই প্রজেক্টের ডেভেলপমেন্ট পরিবেশে Drive mount পথ **চালানো হয়নি**।

**English:**
* Seed 42 everywhere; Colab's library versions can differ from those printed in the docs.
* "Drive not mounted"/permission error → re-run the mount cell and re-authorise; check you are signed in with the account that owns the file.
* Session restarted → Run all again (cells depend on earlier variables).
* Folder shared with you but not in *My Drive* → add a shortcut to My Drive or copy the folder.
* Notebook was verified locally (see README for what was and was not executed); the Drive mount path was **not** executed in this project's development environment.
