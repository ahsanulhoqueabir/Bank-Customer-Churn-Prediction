# CLAUDE.md

Project: CSE 4101 Lab 10 — Bank Customer Churn Prediction (Colab notebook + docs). The full specification is `index.md`; read it and `docs/` before major changes. Do not duplicate it here.

Rules
- Preserve all 10 assignment tasks and the notebook order in `index.md` §5.
- Inspect the real dataset schema; never assume column names or invent results/observations. Never claim a cell ran unless it did.
- Keep the notebook Colab-compatible and Drive-first (`/content/drive/MyDrive/Customer Churn Prediction/train.csv`); the `data/` fallback is local-development only.
- Prevent leakage: split before fitting; preprocessing only inside Pipelines (so it is refit per CV fold); target never in `X`; identifiers (`id`, `CustomerId`, `Surname`) excluded.
- Never use the internal test split (`X_test`/`y_test`) for tuning, model selection or thresholds; it is scored once (Task 9). Validation = `X_val`/`y_val`.
- Positive class is `Exited = 1`; ROC-AUC from probabilities; handle single-class ROC-AUC.
- Keep docs, README and notebook synchronized (quoted metrics/settings come from actual executed runs). If the notebook changes, re-execute it and update the numbers in docs 02–05.
- Run available checks (execute the notebook with `jupyter nbconvert --execute`, `nbformat.validate`) and report what was and was not executed.
- Do not commit data files (`.gitignore` covers `data/`).
