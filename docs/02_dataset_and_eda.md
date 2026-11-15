# 02 — Dataset and Exploratory Data Analysis | ডেটাসেট ও এক্সপ্লোরেটরি ডেটা অ্যানালাইসিস

> **📌 এক নজরে (At a glance)**
> ডেটাসেটে ১৫,০০০ সারি ও ১৪ কলাম; কোনো missing value বা duplicate row নেই। টার্গেট `Exited`-এর ২০.৩১% হলো চার্নার, অর্থাৎ ডেটা **imbalanced**। EDA-তে Germany, বয়স (৫১–৬০), কম product সংখ্যা, নিষ্ক্রিয় সদস্য এবং নারী গ্রাহকদের মধ্যে churn বেশি দেখা যায় — তবে এগুলো কেবল *association*, কারণ-ফল (causation) নয়।
> *15,000 rows × 14 columns, no missing values or duplicates; the target is imbalanced (20.31% churn). EDA shows associations, not causes.*

**বাংলা:** নিচের সব সংখ্যা লোকাল `data/train.csv`-এর ওপর নোটবুক চালিয়ে পাওয়া। আপনার Drive-এর কপি আলাদা হলে সংখ্যাও আলাদা হবে; নোটবুক সবকিছু নতুন করে হিসাব করে।
**English:** All numbers below come from running the notebook on the local `data/train.csv`. If your Drive copy differs, your numbers will differ; the notebook recomputes everything.

## 1. Loading and inspecting | ডেটা লোড ও পরিদর্শন

**বাংলা:** `pd.read_csv(path)` টেক্সট ফাইলকে একটি **DataFrame**-এ রূপান্তর করে — এটি একটি টেবিল, যেখানে প্রতিটি সারি একজন গ্রাহকের রেকর্ড এবং প্রতিটি কলামের একটি নির্দিষ্ট dtype থাকে। প্রধান পরিদর্শন-কমান্ড:
* `df.shape` → `(rows, columns)`। পাওয়া গেছে: `(15000, 14)`।
* `df.dtypes` → প্রতিটি কলামের টাইপ; `object` মানে টেক্সট। সংখ্যা টেক্সট হিসেবে পড়া হলে সেটি নোংরা ডেটার আগাম সংকেত।
* `df.isna().sum()` → কলামভিত্তিক missing value। `isna()` True/False টেবিল দেয়, `sum()` প্রতি কলামে True গোনে। পাওয়া গেছে: প্রতিটি কলামে **০টি missing**।
* `df.duplicated().sum()` → হুবহু একই সারির সংখ্যা। পাওয়া গেছে: ০।
* `df.nunique()` → প্রতি কলামে স্বতন্ত্র মানের সংখ্যা; এতে identifier (প্রায় সবই আলাদা) ও categorical কলাম (অল্প মান) চেনা যায়।
* `df.describe()` → count, mean, std, min, quartile, max। Mean ≫ median বা max ≫ 75% quartile হলে skew/outlier বোঝা যায় (যেমন `EstimatedSalary`-র max ≈ 1.39 M বনাম 75% ≈ 157 k)।
* `df['Exited'].value_counts()` প্রতি ক্লাস গোনে; `len(df)` দিয়ে ভাগ করলে শতকরা হার পাওয়া যায়।

**English:** `pd.read_csv(path)` parses the text file into a **DataFrame**: a table where each row is a customer record and each column has one dtype. Key inspection calls:
* `df.shape` → `(rows, columns)`. Found: `(15000, 14)`.
* `df.dtypes` → type per column; `object` means text. Wrong types (numbers read as text) are an early warning of dirty data.
* `df.isna().sum()` → missing values per column. `isna()` returns a True/False table; `sum()` counts the Trues per column. Found: **0 missing** in every column.
* `df.duplicated().sum()` → number of fully identical rows. Found: 0.
* `df.nunique()` → distinct values per column; helps spot identifiers (nearly unique) and categorical columns (few values).
* `df.describe()` → count, mean, std, min, quartiles, max for numeric columns. Mean ≫ median or a max far above the 75% quartile hints at skew/outliers (e.g. `EstimatedSalary` max ≈ 1.39 M vs 75% ≈ 157 k).
* `df['Exited'].value_counts()` counts each class; dividing by `len(df)` gives percentages.

## 2. Discovered schema | আবিষ্কৃত স্কিমা

**বাংলা:** কলামের *অর্থ* শুধু নাম দেখে অনুমান করা; নোটবুক এমন কোনো ডকুমেন্টেশনের দাবি করে না যা যাচাই করা হয়নি। কোনো "classic" ব্যাংক-churn গঠন আগে থেকে ধরে নেওয়া হয়নি; তবে CSV-তে অতিরিক্ত `id` কলাম আছে এবং 0/1 ফ্ল্যাগগুলো float হিসেবে সংরক্ষিত।
**English:** Column *meanings* are inferred from their names only; the notebook does not claim documentation that was not inspected. Differences from a "classic" bank-churn layout: none were assumed in advance, but note the CSV has an extra `id` column and stores 0/1 flags as floats.

| Column | dtype | distinct values | Notes / মন্তব্য |
|---|---|---|---|
| id | int64 | 15000 | sequential 0…14999 (row index) / ক্রমিক সারি-সূচক |
| CustomerId | float64 | 6349 | repeats across rows / একাধিক সারিতে পুনরাবৃত্তি |
| Surname | object | 778 | text / টেক্সট |
| CreditScore | float64 | 371 | range 431–850 |
| Geography | object | 3 | France 8950, Spain 3296, Germany 2754 |
| Gender | object | 2 | Female, Male |
| Age | float64 | 55 | 18–74 |
| Tenure | float64 | 11 | 0–10 |
| Balance | float64 | 3302 | 64.9% of rows are exactly 0 / ৬৪.৯% সারিতে ঠিক ০ |
| NumOfProducts | float64 | 4 | 1–4 |
| HasCrCard | float64 | 2 | 0/1 |
| IsActiveMember | float64 | 2 | 0/1 |
| EstimatedSalary | float64 | 6152 | |
| Exited | int64 (after cast) | 2 | target / টার্গেট |

## 3. Target distribution | টার্গেটের বণ্টন

**বাংলা:** `Exited=0`: ১১,৯৫৪ (৭৯.৬৯%); `Exited=1`: ৩,০৪৬ (২০.৩১%)। পজিটিভ ক্লাস প্রায় প্রতি পাঁচজনে একজন, অর্থাৎ ডেটা **imbalanced**। যে মডেল সবাইকে "থাকবে" বলে, সে ≈ ৭৯.৭% accuracy পাবে অথচ একজন চার্নারও ধরতে পারবে না। এ কারণেই Recall, F1 ও ROC-AUC ব্যবহার করা হয়।
**English:** `Exited=0`: 11,954 (79.69%); `Exited=1`: 3,046 (20.31%). The positive class is about one-in-five, so the data are imbalanced: a model predicting "stays" for everyone would score ≈ 79.7% accuracy while catching no churners. This is why Recall, F1 and ROC-AUC are used.

## 4. EDA charts | EDA চার্ট

**বাংলা:** কোনো গ্রুপের churn rate = `groupby(col)['Exited'].mean()` × ১০০ (০/১ মানের গড় = ১-এর ভগ্নাংশ)। প্রতিটি বার চার্টে গ্রুপের আকার (`n=`) এবং সামগ্রিক হারে (২০.৩%) একটি ড্যাশ-করা লাল রেখা দেখানো হয়।
**English:** Churn rate for a group = `groupby(col)['Exited'].mean()` (the mean of 0/1 values is the fraction of 1s) × 100. Every bar chart shows group sizes (`n=`) and a dashed red line at the overall rate (20.3%).

| Chart | Question / প্রশ্ন | Calculation / axes | Caveat / সতর্কতা |
|---|---|---|---|
| Target distribution | How imbalanced is the target? / টার্গেট কতটা অসম? | count per class | — |
| Churn by geography | Do countries differ? / দেশভেদে পার্থক্য আছে কি? | mean `Exited` per country | countries differ in size and possibly in customer mix |
| Churn by age group | Does age matter? / বয়স কি গুরুত্বপূর্ণ? | `pd.cut(Age, [17,30,40,50,60,100])` → intervals (17,30], (30,40], … | the (60,100] group has only n=217 |
| Churn by number of products | Does product count matter? | mean `Exited` per value | 3 and 4 products have n=186 and n=32 |
| Churn by `IsActiveMember` | Does activity matter? | mean `Exited` per flag | meaning of "active" not documented |
| Age histogram by outcome | Are churners older? / চার্নাররা কি বয়স্ক? | density histograms, each class normalised (`common_norm=False`) so unequal class sizes do not distort the comparison | |
| Churn by tenure (line) | Does tenure matter? | mean per tenure year | ends (0, 10) have fewer rows |
| Balance box plot | Do balances differ? | box plot by outcome | 64.9% zeros compress the boxes |

**বাংলা টীকা:** বয়সের গ্রুপ `pd.cut`-এর right-closed ব্যবধান, তাই ৩০ বছর বয়সী গ্রাহক প্রথম গ্রুপ `(17,30]`-এ পড়ে। `common_norm=False` দেওয়ায় প্রতিটি ক্লাস আলাদাভাবে normalise হয়, ফলে ক্লাসের আকারের পার্থক্য তুলনাকে বিকৃত করে না।

## 5. Evidence-based observations | প্রমাণভিত্তিক পর্যবেক্ষণ

**বাংলা:**
1. **Geography:** Germany-তে churn সর্বোচ্চ (৩৯.৬%, n=2754); France ১৫.৯% ও Spain ১৬.৩%; সামগ্রিক হার ২০.৩%।
2. **Age:** বয়স বাড়লে churn তীব্রভাবে বাড়ে: ১৮–৩০ বছরে ৪.৩%, ৩১–৪০-এ ৯.৬%, ৪১–৫০-এ ৪০.৯%, ৫১–৬০-এ ৭৩.১% (n=947); ৬০-এর বেশি বয়সে নেমে ৩৭.৩% (মাত্র n=217)।
3. **Products / activity:** ২টি product থাকলে churn ৪.৪% (n=8405), ১টি হলে ৩৮.৮%; ৩ ও ৪টি product-এ ৯৪.৬% ও ৯৬.৯%, কিন্তু গ্রুপ খুব ছোট (১৮৬ ও ৩২), তাই চরম মানগুলো নির্ভরযোগ্য নয়। নিষ্ক্রিয় সদস্যদের churn ২৮.০%, সক্রিয়দের ১২.৪%।
4. **অন্যান্য:** লিঙ্গভেদে পার্থক্য (নারী ২৮.৬%, পুরুষ ১৩.৯%); tenure-এ পরিসর সীমিত (১৫.২%–২৯.১%), tenure ০–১ বছরে কিছুটা বেশি; চার্নারদের গড় balance বেশি (≈ 58.6 k বনাম ≈ 38.6 k), কিন্তু উভয় ক্লাসের median ০।

এগুলো এই নমুনায় বর্ণনামূলক সম্পর্ক; এগুলো প্রমাণ করে না যে geography, বয়স বা product *কারণ* হিসেবে churn ঘটায়।

**English:**
1. **Geography:** Germany has the highest churn (39.6%, n=2754) versus France 15.9% and Spain 16.3%; overall is 20.3%.
2. **Age:** churn rises sharply with age: 4.3% for ages 18–30, 9.6% for 31–40, 40.9% for 41–50 and 73.1% for 51–60 (n=947); it falls to 37.3% above 60 (only n=217).
3. **Products / activity:** customers with 2 products churn at 4.4% (n=8405) versus 38.8% with 1 product; 3 and 4 products show 94.6% and 96.9% but from tiny groups (186 and 32), so the extremes are unreliable. Inactive members churn at 28.0% versus 12.4% for active members.
4. **Other:** gender differs (female 28.6%, male 13.9%); tenure shows a modest range (15.2%–29.1%) with higher values for tenure 0–1; churners have a higher mean balance (≈58.6 k vs ≈38.6 k) but both medians are 0.

These are descriptive associations in this sample. They do not show that geography, age or products *cause* churn.

## 6. Unexpected findings | অপ্রত্যাশিত আবিষ্কার

**বাংলা:**
* `CustomerId`-এর স্বতন্ত্র মান মাত্র ৬,৩৪৯টি, অথচ সারি ১৫,০০০টি (৮০.৬% সারি অন্য সারির সঙ্গে একই ID ভাগ করে)। তাই একই গ্রাহক একাধিক সারিতে, এমনকি ভিন্ন ভিন্ন split-এ থাকতে পারেন (দেখুন [doc 03](03_preprocessing_and_data_leakage.md))।
* `Surname`-এ কিছু অদ্ভুত এন্ট্রি আছে (যেমন সারি ৩-এ "L?"), যা সম্ভবত উৎসের encoding ত্রুটি; কারণ অনুসন্ধান করা হয়নি, এবং কলামটি মডেলিং থেকে বাদ দেওয়া হয়েছে।

**English:**
* `CustomerId` has only 6,349 distinct values for 15,000 rows (80.6% of rows share an ID with another row). The same customer may therefore appear in several rows, possibly in different splits (see doc 03).
* `Surname` contains some odd entries (e.g. "L?" in row 3), possibly an encoding artefact of the source; the cause was not investigated, and the column is excluded from modelling anyway.
