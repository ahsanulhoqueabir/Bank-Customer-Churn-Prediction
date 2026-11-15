# 07 — Google Colab and Google Drive Guide

## Prepare Drive
1. In Google Drive create (exactly) `My Drive/Customer Churn Prediction/`.
2. Upload `train.csv` there. For Task 10 also upload the competition's `test.csv` (optional).
3. Resulting required path: `/content/drive/MyDrive/Customer Churn Prediction/train.csv` (optional: `.../test.csv`).

## Open the notebook in Colab
* Upload `notebooks/bank_customer_churn_prediction.ipynb` via *colab.research.google.com → File → Upload notebook*, or put it in Drive and open it with *Google Colaboratory*, or open it from GitHub if you push the repo.
* Use a standard CPU runtime; no GPU is needed. pandas, NumPy, Matplotlib, Seaborn and scikit-learn are preinstalled, so nothing is `pip install`ed.

## Run
*Runtime → Run all*. In §2 a pop-up asks you to authorise Drive access; approve it with your Google account. The notebook then prints the loaded path, shape and columns. Confirm the printed path is exactly the Drive path above (not `data/train.csv`, which is only the local fallback) and the shape matches your file (the local copy: 15000 × 14).

Total runtime: the hyper-parameter search is the slow part (160 pipeline fits); runtime was not formally timed locally or on Colab, so expect it to take noticeable time and be patient at §9.

## Expected outputs
Printed tables for every task; figures: target distribution + 5 churn/age panels, tenure line + balance box plot, split churn-rate bar chart, CV-score bar plots (one per model), validation F1-vs-threshold curve, train-vs-validation grouped bar chart, internal-test confusion matrix; five-row prediction table (if `test.csv` exists). With the same data, seed and library versions you should get the numbers quoted in docs 02–05; small differences are possible on other versions.

## Missing-file handling
* `train.csv` missing → `FileNotFoundError` listing the expected path; fix by uploading to the exact folder/filename (case and the space in the folder name matter), then re-run §2.
* `test.csv` missing → Task 10 prints "BLOCKED: competition test file not found…" and the notebook still finishes; upload `test.csv` next to `train.csv` and re-run §12 (the model objects from earlier cells must still exist, otherwise Run all).

## Save / export
*File → Save a copy in Drive* or *Download .ipynb*; *File → Print* → save as PDF for a report. Outputs are stored inside the saved `.ipynb`.

## Reproducibility and troubleshooting
* Seed 42 everywhere; Colab's library versions can differ from those printed in the docs.
* "Drive not mounted"/permission error → re-run the mount cell and re-authorise; check you are signed in with the account that owns the file.
* Session restarted → Run all again (cells depend on earlier variables).
* Folder shared with you but not in *My Drive* → add a shortcut to My Drive or copy the folder.
* Notebook was verified locally (see README for what was and was not executed); the Drive mount path was **not** executed in this project's development environment.
