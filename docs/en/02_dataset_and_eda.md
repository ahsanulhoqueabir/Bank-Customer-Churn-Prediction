# 02 — Dataset and Exploratory Data Analysis

All numbers below come from running the notebook on the local `data/train.csv`. If your Drive copy differs, your numbers will differ; the notebook recomputes everything.

## Loading and inspecting
`pd.read_csv(path)` parses the text file into a **DataFrame**: a table where each row is a customer record and each column has one dtype. Key inspection calls:
* `df.shape` → `(rows, columns)`. Found: `(15000, 14)`.
* `df.dtypes` → type per column; `object` means text. Wrong types (numbers read as text) are an early warning of dirty data.
* `df.isna().sum()` → missing values per column. `isna()` returns a True/False table; `sum()` counts the Trues per column. Found: **0 missing** in every column.
* `df.duplicated().sum()` → number of fully identical rows. Found: 0.
* `df.nunique()` → distinct values per column; helps spot identifiers (nearly unique) and categorical columns (few values).
* `df.describe()` → count, mean, std, min, quartiles, max for numeric columns. Mean ≫ median or a max far above the 75% quartile hints at skew/outliers (e.g. `EstimatedSalary` max ≈ 1.39 M vs 75% ≈ 157 k).
* `df['Exited'].value_counts()` counts each class; dividing by `len(df)` gives percentages.

## Discovered schema
| Column | dtype | distinct values | Notes |
|---|---|---|---|
| id | int64 | 15000 | sequential 0…14999 (row index) |
| CustomerId | float64 | 6349 | repeats across rows |
| Surname | object | 778 | text |
| CreditScore | float64 | 371 | range 431–850 |
| Geography | object | 3 | France 8950, Spain 3296, Germany 2754 |
| Gender | object | 2 | Female, Male |
| Age | float64 | 55 | 18–74 |
| Tenure | float64 | 11 | 0–10 |
| Balance | float64 | 3302 | 64.9% of rows are exactly 0 |
| NumOfProducts | float64 | 4 | 1–4 |
| HasCrCard | float64 | 2 | 0/1 |
| IsActiveMember | float64 | 2 | 0/1 |
| EstimatedSalary | float64 | 6152 | |
| Exited | int64 (after cast) | 2 | target |

Column *meanings* are inferred from their names only; the notebook does not claim documentation that was not inspected. Differences from a "classic" bank-churn layout: none were assumed in advance, but note the CSV has an extra `id` column and stores 0/1 flags as floats.

## Target distribution
`Exited=0`: 11,954 (79.69%); `Exited=1`: 3,046 (20.31%). The positive class is about one-in-five, so the data are imbalanced: a model predicting "stays" for everyone would score ≈ 79.7% accuracy while catching no churners. This is why Recall, F1 and ROC-AUC are used.

## EDA charts
Churn rate for a group = `groupby(col)['Exited'].mean()` (the mean of 0/1 values is the fraction of 1s) × 100. Every bar chart shows group sizes (`n=`) and a dashed red line at the overall rate (20.3%).

| Chart | Question | Calculation / axes | Caveat |
|---|---|---|---|
| Target distribution | How imbalanced is the target? | count per class | — |
| Churn by geography | Do countries differ? | mean `Exited` per country | countries differ in size and possibly in customer mix |
| Churn by age group | Does age matter? | `pd.cut(Age, [17,30,40,50,60,100])` → intervals (17,30], (30,40], … | the (60,100] group has only n=217 |
| Churn by number of products | Does product count matter? | mean `Exited` per value | 3 and 4 products have n=186 and n=32 |
| Churn by `IsActiveMember` | Does activity matter? | mean `Exited` per flag | meaning of "active" not documented |
| Age histogram by outcome | Are churners older? | density histograms, each class normalised (`common_norm=False`) so unequal class sizes do not distort the comparison | |
| Churn by tenure (line) | Does tenure matter? | mean per tenure year | ends (0, 10) have fewer rows |
| Balance box plot | Do balances differ? | box plot by outcome | 64.9% zeros compress the boxes |

## Evidence-based observations (from the computed tables/plots)
1. **Geography:** Germany has the highest churn (39.6%, n=2754) versus France 15.9% and Spain 16.3%; overall is 20.3%.
2. **Age:** churn rises sharply with age: 4.3% for ages 18–30, 9.6% for 31–40, 40.9% for 41–50 and 73.1% for 51–60 (n=947); it falls to 37.3% above 60 (only n=217).
3. **Products / activity:** customers with 2 products churn at 4.4% (n=8405) versus 38.8% with 1 product; 3 and 4 products show 94.6% and 96.9% but from tiny groups (186 and 32), so the extremes are unreliable. Inactive members churn at 28.0% versus 12.4% for active members.
4. **Other:** gender differs (female 28.6%, male 13.9%); tenure shows a modest range (15.2%–29.1%) with higher values for tenure 0–1; churners have a higher mean balance (≈58.6 k vs ≈38.6 k) but both medians are 0.

These are descriptive associations in this sample. They do not show that geography, age or products *cause* churn.

## Unexpected findings
* `CustomerId` has only 6,349 distinct values for 15,000 rows (80.6% of rows share an ID with another row). The same customer may therefore appear in several rows, possibly in different splits (see doc 03).
* `Surname` contains some odd entries (e.g. "L?" in row 3), possibly an encoding artefact of the source; the cause was not investigated, and the column is excluded from modelling anyway.
