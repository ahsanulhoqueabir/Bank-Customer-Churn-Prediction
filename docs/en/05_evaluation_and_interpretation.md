# 05 — Evaluation and Interpretation

Positive class = `Exited = 1` (customer churned). Threshold-dependent metrics use probability ≥ threshold ⇒ predicted churn; the final threshold is **0.39** (chosen on validation F1).

## Metrics in plain language
Let TP = churners caught, FN = churners missed, FP = stayers wrongly flagged, TN = stayers correctly left alone.
* **Accuracy** = (TP+TN)/all. Fraction of correct predictions. With 20.3% churners, predicting "nobody leaves" gives ≈79.7% accuracy and zero usefulness, so accuracy alone misleads.
* **Precision** = TP/(TP+FP). Of customers flagged as churn, how many really churned.
* **Recall** = TP/(TP+FN). Of customers who really churned, how many were caught.
* **F1** = 2·P·R/(P+R). Harmonic mean; low if either precision or recall is low.
* **ROC-AUC.** Plot true-positive rate against false-positive rate for every possible threshold; AUC is the area under that curve = the probability that a random churner gets a higher score than a random stayer. 0.5 = chance, 1.0 = perfect. It is computed from probabilities (`predict_proba`), not hard labels, and does not depend on a threshold. The notebook's `safe_roc_auc` returns NaN instead of crashing if a split contains only one class (AUC is undefined then).
* **Confusion matrix.** 2×2 table of counts, rows = true class, columns = predicted class.

## Task 8 — training vs validation (Gradient Boosting, threshold 0.39)
| Metric | Training | Validation | Gap (train − val) |
|---|---|---|---|
| Recall | 0.7402 | 0.7418 | −0.0016 |
| F1 | 0.7523 | 0.7410 | 0.0113 |
| ROC-AUC | 0.9425 | 0.9283 | 0.0142 |

*Reading it:* training and validation scores are close; the largest gap is 0.014 in ROC-AUC. This gives no strong sign of overfitting (a model that memorised training rows would typically show a much higher training score) and the high absolute scores argue against underfitting. This is evidence, not proof: the validation set has only 457 churners, so each number has sampling uncertainty of roughly a few percentage points, and the threshold was chosen on this same validation split. Recall on validation is slightly *above* training, which can only be noise.

## Task 9 — final one-time internal-test evaluation (2,250 rows, 457 churners)
| Metric | Validation | Internal test |
|---|---|---|
| Accuracy | 0.8947 | 0.8920 |
| Precision | 0.7402 | 0.7211 |
| Recall | 0.7418 | 0.7637 |
| F1 | 0.7410 | 0.7418 |
| ROC-AUC | 0.9283 | 0.9329 |

Confusion matrix (rows true, columns predicted): TN = 1658, FP = 135, FN = 108, TP = 349.
* TN 1658 — stayed, predicted to stay.
* FP 135 — stayed but flagged: a retention offer or call wasted (and a possible annoyance).
* FN 108 — churned but not flagged: a lost customer and no chance to intervene.
* TP 349 — churned and flagged: retention action possible.
Recall = 349/457 = 0.764; precision = 349/484 = 0.721. Internal-test results are consistent with validation (ROC-AUC +0.0045, F1 +0.0008); differences of this size are within what sampling variation on 2,250 rows would produce. The notebook read `X_test`/`y_test` for scoring only in §11, after model, hyperparameters and threshold were fixed; nothing was changed afterwards.

## Cost trade-off for a bank
A false negative loses the customer's lifetime value; a false positive costs an offer and goodwill. If lost customers are more expensive than wasted offers (typical in retention), a lower threshold (higher recall, lower precision) would be preferable; F1 treats both errors as equally important, which is an assumption, not a business fact. Real costs were not available and were not modelled.

## Association vs prediction vs causation vs calibration
* *Association:* "Germany has higher churn" describes data. *Prediction:* the model uses such patterns to score new customers. *Causation:* nothing here shows that changing a feature (e.g. selling a second product) would change churn. Interventions need experiments.
* *Calibration:* a predicted 0.39 does not necessarily mean 39% of such customers churn. Boosted-tree probabilities are not guaranteed calibrated, and no calibration check (reliability curve) was done, so probabilities should be treated as risk **scores**.

## Limitations and possible improvements
Observed in this project: duplicate `CustomerId`s may leak customers across splits; feature timing unverified; one split and one seed (no repeated-split variance); a single threshold criterion. Possible improvements (not implemented): grouped splitting by customer, repeated CV for confidence intervals, probability calibration, cost-sensitive threshold selection, fairness analysis by geography/gender/age, learning curves and feature importance.
