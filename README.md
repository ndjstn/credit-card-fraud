# The Default Threshold Is the Bug: Credit Card Fraud on the ULB Benchmark

A fraud classifier trained on the famous ULB Kaggle dataset reaches 0.98 ROC-AUC and still ships a policy most fraud teams would reject. At the default 0.5 decision threshold, the confusion matrix underneath that impressive headline is the part nobody wants to look at. This repo is the pipeline, the figures, and the cost-sensitive threshold analysis that explain why the threshold is a policy decision, not a hyperparameter.

The companion walkthrough video and the full write-up with figures and narrative are linked at the bottom.

## Key results

| Model | ROC-AUC | Avg. precision | Min-cost threshold | Recall | Precision | Missed fraud | False reviews | Total cost (EUR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9788 | 0.8630 | 0.01 | 0.862 | 0.573 | 17 | 79 | 2,095 |
| Logistic regression | 0.9727 | 0.7062 | 0.97 | 0.854 | 0.507 | 18 | 102 | 2,310 |
| Isolation Forest | 0.9463 | 0.1133 | 0.55 | 0.585 | 0.092 | 51 | 710 | 8,650 |

Cost assumptions: a missed fraud costs 100 EUR, a false positive review costs 5 EUR. The test set has 123 positives out of 71,202 transactions, a 0.17 percent prevalence that matches the full dataset. XGBoost and logistic regression arrive at almost identical operating points from different directions; the Isolation Forest is not a serious contender here, and running it on the same split makes the gap between supervised and unsupervised fraud detection on this benchmark legible.

## What is in this repo

`src/run_analysis.py` is a single end-to-end script that loads the Kaggle ULB dataset, profiles the class balance and the amount and time distributions, fits the three classifiers with the appropriate weighting, runs the cost sweep, and writes every figure and table under `figures/` and `outputs/`. `figures/` holds the cost curves, precision-recall curve, confusion matrices, and exploratory plots used in the write-up and the video. `outputs/` holds `analysis_summary.json`, `analysis_summary.md`, the per-model cost-sweep tables, and the head-to-head model comparison.

`REPORT.md` is a long-form written analysis covering the dataset, the three classifiers, the cost-sensitive threshold sweep, and a discussion of what the findings do and do not support.

## How to reproduce

The dataset is the Kaggle `creditcard.csv` from the Machine Learning Group ULB credit card fraud detection dataset. Download it from <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and place it at `data/creditcard.csv` relative to the repo root.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --data data/creditcard.csv --figures figures --outputs outputs
```

The script writes its figures to `figures/` and its numerical outputs to `outputs/`. Total runtime is roughly thirty seconds on a modern CPU; no GPU is required.

## Further reading

The full write-up with narrative, figures, and the cost-sensitive threshold argument is on my site: <https://ndjstn.github.io/posts/credit-card-fraud-default-threshold/>.

## License

MIT. See [LICENSE](LICENSE).
