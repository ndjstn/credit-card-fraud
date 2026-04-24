# Credit card fraud analysis summary

Dataset rows: 284,807. Positive class: 492 (0.1727%).

## Headline metrics (min-cost operating point)

| Model | ROC-AUC | AP | Threshold | Recall | Precision | Missed fraud | False reviews | Total cost (EUR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.9727 | 0.7062 | 0.97 | 0.854 | 0.507 | 18 | 102 | 2,310.0 |
| xgboost | 0.9788 | 0.863 | 0.01 | 0.862 | 0.573 | 17 | 79 | 2,095.0 |
| isolation_forest | 0.9463 | 0.1133 | 0.55 | 0.585 | 0.092 | 51 | 710 | 8,650.0 |

Cost assumptions: missed fraud = EUR 100 per case, false positive review = EUR 5 per case.