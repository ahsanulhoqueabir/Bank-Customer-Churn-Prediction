# 04 — Models and Hyperparameter Tuning

## The three models
Each model is always wrapped as `Pipeline([("preprocess", ColumnTransformer), ("model", estimator)])`. Reasons: (1) the exact same transformation is applied at training and prediction time; (2) cross-validation refits preprocessing inside each fold (no leakage); (3) one object (`fit`, `predict_proba`) is all that is needed to deploy/predict, including for the competition file.

### Logistic Regression
*Learns:* one coefficient per input column plus an intercept. *Predicts:* computes `z = intercept + Σ wᵢxᵢ`, then the sigmoid `p = 1/(1+e^−z)` gives P(`Exited=1`). The decision boundary is linear in the (scaled, one-hot) features. Fitted by minimising log-loss plus an L2 penalty (`penalty='l2'`). *Strengths:* fast, stable, interpretable weights. *Limits:* cannot capture non-linear effects or interactions unless features are engineered. Key settings: `C` (inverse regularisation strength; smaller = stronger shrinkage), `class_weight` (`'balanced'` up-weights the minority class during fitting), `max_iter=1000` (set so the optimiser converges). Uses the scaled preprocessor.

### Random Forest
*Learns:* many decision trees, each grown on a bootstrap resample of the training rows, considering a random subset of features at each split (`max_features='sqrt'`). *Predicts:* each tree votes/outputs a class probability; the forest averages them. Averaging many different deep trees reduces variance (overfitting of a single tree). *Strengths:* captures non-linearities and interactions, needs no scaling, robust. *Limits:* larger/slower, less interpretable, probabilities can be less well calibrated. Settings tuned: `n_estimators`, `max_depth`, `min_samples_leaf`.

### Gradient Boosting (`GradientBoostingClassifier`)
*Learns:* a sequence of shallow regression trees. Each new tree fits the gradient of the log-loss (roughly, the residual errors) of the current ensemble; its contribution is scaled by `learning_rate` and added. *Predicts:* sums the scaled tree outputs into a log-odds score, then applies a sigmoid. *Strengths:* often the most accurate on tabular data; *limits:* more hyperparameter-sensitive, sequential training (slower), can overfit with too many/too deep trees and a high learning rate. Settings tuned: `n_estimators`, `learning_rate`, `max_depth`.

### Linear vs tree ensembles
Linear: global, additive effect per feature. Trees: piecewise-constant regions that can represent "age 45–60 *and* one product" style interactions automatically. Random Forest reduces variance by averaging independent trees; Gradient Boosting reduces bias by sequentially correcting errors.

## Parameters vs hyperparameters
*Parameters* are learned from data by `fit` (logistic coefficients, tree split thresholds, imputer medians). *Hyperparameters* are chosen by the practitioner before fitting (`C`, `max_depth`, `learning_rate`, number of folds) and are selected by experiment.

## Task 6 — baselines (validation split, threshold 0.5, no imbalance remedy)
Default-ish settings (LR `C=1`, `max_iter=1000`; RF 200 trees; GB defaults: 100 trees, learning rate 0.1, depth 3); all with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.8978 | 0.7963 | 0.6674 | 0.7262 | 0.9288 |
| Random Forest | 0.8831 | 0.7487 | 0.6389 | 0.6895 | 0.9137 |
| Logistic Regression | 0.8729 | 0.7568 | 0.5514 | 0.6380 | 0.8760 |

On this validation split Gradient Boosting ranked first on every metric shown and Logistic Regression last. Because churners are 20.3% of rows, all three models miss a substantial fraction of churners at threshold 0.5 (recall 0.55–0.67).

## Task 7 — tuning strategy
* **Selection metric (decided before comparing): ROC-AUC.** It measures ranking quality independent of any threshold, so the threshold can be chosen separately afterwards.
* **Method:** `GridSearchCV(pipeline, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=42), refit=True)` fitted on `X_train, y_train` only. For each grid point and each of 5 folds, scikit-learn clones the pipeline, fits it (preprocessing included) on 4/5 of the training rows, scores on the remaining 1/5, and averages. Stratified folds keep the 20% churn share in every fold. After the search, the best setting is refit on the whole training split (`refit=True`; train+validation is **not** merged, to keep the validation role clean).
* **Search spaces:**
  * Logistic Regression: `C ∈ {0.01, 0.1, 1, 10}` × `class_weight ∈ {None, 'balanced'}` = 8 settings
  * Random Forest: `n_estimators ∈ {100, 300}` × `max_depth ∈ {None, 8, 16}` × `min_samples_leaf ∈ {1, 5}` = 12 settings
  * Gradient Boosting: `n_estimators ∈ {100, 200}` × `learning_rate ∈ {0.05, 0.1}` × `max_depth ∈ {2, 3, 4}` = 12 settings
  Grids are intentionally small so the notebook runs in reasonable time on Colab (32 settings × 5 folds = 160 fits).
* **`class_weight`** appears only as a Logistic Regression hyperparameter, judged by CV evidence; no resampling or other imbalance remedy is used, and the tree models use none.

### Results (best per model; the complete table of all 32 settings with mean ± std CV ROC-AUC is printed in notebook §9)
| Model | Best CV ROC-AUC (train folds) | Best setting |
|---|---|---|
| Logistic Regression | 0.8783 | `C=1`, `class_weight='balanced'` |
| Random Forest | 0.9292 | `max_depth=8`, `min_samples_leaf=1`, `n_estimators=300` |
| Gradient Boosting | 0.9325 | `learning_rate=0.05`, `max_depth=3`, `n_estimators=200` |

Several Logistic Regression settings differ only in the 4th decimal (e.g. 0.8774 vs 0.8783) with a fold standard deviation ≈ 0.01, so those differences are within noise.

### Validation comparison of the tuned models (threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | baseline ROC-AUC |
|---|---|---|---|---|---|---|
| Gradient Boosting | 0.8960 | 0.7911 | 0.6630 | 0.7214 | 0.9283 | 0.9288 |
| Random Forest | 0.8938 | 0.8187 | 0.6127 | 0.7009 | 0.9256 | 0.9137 |
| Logistic Regression | 0.8133 | 0.5268 | 0.7965 | 0.6341 | 0.8783 | 0.8760 |

Observations: tuning improved Random Forest validation ROC-AUC noticeably (0.9137 → 0.9256), Logistic Regression marginally, and Gradient Boosting not at all (0.9288 → 0.9283; essentially unchanged, i.e. its defaults were already near the grid optimum). `class_weight='balanced'` in Logistic Regression raised recall (0.55 → 0.80) at the cost of precision (0.76 → 0.53) — it moves the operating point rather than improving ranking (ROC-AUC barely changed).

**Selected model: Gradient Boosting** (`learning_rate=0.05, max_depth=3, n_estimators=200`) — highest validation ROC-AUC (0.9283) and highest CV ROC-AUC (0.9325). The margin over the tuned Random Forest on validation is small (0.0027), so the choice is supported but not decisive.

### Decision threshold
Chosen on the **validation** split only, from a grid 0.10–0.90 (step 0.01), maximising validation F1: **0.39** (validation F1 0.7410 versus 0.7214 at 0.5). Choosing it on the same data it is later reported on makes the validation numbers slightly optimistic; that is why the final claim rests on the untouched internal test set (doc 05).

## Common problems
* Invalid grid keys: parameters of a step inside a pipeline use the `stepname__param` form (`model__C`).
* Mixing `class_weight='balanced'` with other imbalance fixes double-counts the correction.
* Slow search → reduce grid size, use `n_jobs=-1`, or switch to `RandomizedSearchCV`.
