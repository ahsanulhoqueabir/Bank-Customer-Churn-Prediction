# 07 — Google Colab ও Google Drive নির্দেশিকা

> **📌 এক নজরে**
> Drive-এ `My Drive/Customer Churn Prediction/` ফোল্ডার বানিয়ে `train.csv` (ঐচ্ছিক: `test.csv`) রাখুন → নোটবুক Colab-এ খুলুন → *Runtime → Run all* → Drive অনুমতি দিন। CPU runtime যথেষ্ট; GPU লাগে না।

## 1. Drive প্রস্তুত করুন

1. Google Drive-এ হুবহু `My Drive/Customer Churn Prediction/` নামে ফোল্ডার বানান।
2. সেখানে `train.csv` আপলোড করুন। Task 10-এর জন্য প্রতিযোগিতার `test.csv`-ও আপলোড করুন (ঐচ্ছিক)।
3. ফলে প্রয়োজনীয় পথ: `/content/drive/MyDrive/Customer Churn Prediction/train.csv` (ঐচ্ছিক: `.../test.csv`)।

## 2. Colab-এ নোটবুক খুলুন

* *colab.research.google.com → File → Upload notebook* দিয়ে `notebooks/bank_customer_churn_prediction.ipynb` আপলোড করুন; অথবা Drive-এ রেখে *Google Colaboratory* দিয়ে খুলুন; অথবা রিপোজিটরি GitHub-এ push করা থাকলে সেখান থেকে খুলুন।
* সাধারণ CPU runtime ব্যবহার করুন; GPU লাগে না। pandas, NumPy, Matplotlib, Seaborn ও scikit-learn আগে থেকেই ইনস্টল করা, তাই কিছু `pip install` করতে হয় না।

## 3. চালানো

*Runtime → Run all*। §2-এ একটি পপ-আপ Drive-অনুমতি চাইবে; আপনার Google অ্যাকাউন্ট দিয়ে অনুমোদন করুন। নোটবুক তখন লোড-হওয়া পথ, shape ও কলাম প্রিন্ট করবে। নিশ্চিত করুন প্রিন্ট-করা পথ হুবহু উপরের Drive-পথ (`data/train.csv` নয়, যা কেবল লোকাল fallback) এবং shape আপনার ফাইলের সঙ্গে মিলছে (লোকাল কপি: 15000 × 14)।

**মোট সময়:** hyper-parameter অনুসন্ধানই ধীর অংশ (১৬০টি pipeline fit); লোকালি বা Colab-এ সময় আনুষ্ঠানিকভাবে মাপা হয়নি, তাই লক্ষণীয় সময় লাগবে ধরে নিন এবং §9-এ ধৈর্য রাখুন।

## 4. প্রত্যাশিত আউটপুট

প্রতিটি টাস্কের জন্য টেবিল প্রিন্ট হয়; চিত্র: target distribution + ৫টি churn/age প্যানেল, tenure লাইন + balance box plot, split churn-rate বার চার্ট, CV-স্কোর বার প্লট (প্রতি মডেলে একটি), validation F1-বনাম-threshold বক্ররেখা, train-vs-validation গ্রুপ বার চার্ট, internal-test confusion matrix; পাঁচ-সারির prediction টেবিল (`test.csv` থাকলে)। একই ডেটা, seed ও library সংস্করণে docs 02–05-এ উদ্ধৃত সংখ্যাই পাওয়া উচিত; অন্য সংস্করণে সামান্য পার্থক্য সম্ভব।

## 5. ফাইল না থাকলে করণীয়

* `train.csv` নেই → প্রত্যাশিত পথসহ `FileNotFoundError`; সঠিক ফোল্ডার/ফাইলনামে আপলোড করে (বড়/ছোট হাতের অক্ষর এবং ফোল্ডারনামের ফাঁকা স্থান গুরুত্বপূর্ণ) §2 আবার চালান।
* `test.csv` নেই → Task 10 "BLOCKED: competition test file not found…" প্রিন্ট করে এবং নোটবুক তবু শেষ হয়; `train.csv`-র পাশে `test.csv` আপলোড করে §12 আবার চালান (আগের সেলের মডেল-অবজেক্ট থাকতে হবে, না থাকলে Run all)।

## 6. সংরক্ষণ / এক্সপোর্ট

*File → Save a copy in Drive* বা *Download .ipynb*; রিপোর্টের জন্য *File → Print* → PDF হিসেবে সংরক্ষণ। আউটপুট সংরক্ষিত `.ipynb`-র ভেতরেই থাকে।

## 7. পুনরুৎপাদনযোগ্যতা ও সমস্যা সমাধান

* সর্বত্র seed 42; Colab-এর library সংস্করণ docs-এ প্রিন্ট-করা সংস্করণের চেয়ে আলাদা হতে পারে।
* "Drive not mounted"/permission error → mount সেল আবার চালিয়ে পুনরায় অনুমোদন দিন; ফাইলের মালিক অ্যাকাউন্টেই সাইন-ইন আছেন কি না দেখুন।
* সেশন রিস্টার্ট হলে → আবার Run all (সেলগুলো আগের ভ্যারিয়েবলের ওপর নির্ভরশীল)।
* ফোল্ডার আপনার সঙ্গে শেয়ার করা কিন্তু *My Drive*-এ নেই → My Drive-এ shortcut যোগ করুন বা ফোল্ডার কপি করুন।
* নোটবুক লোকালি যাচাই করা হয়েছে (কোনটি চালানো হয়েছে আর কোনটি হয়নি তা README-তে দেখুন); এই প্রজেক্টের ডেভেলপমেন্ট পরিবেশে Drive mount পথ **চালানো হয়নি**।

