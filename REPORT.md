# The Default Threshold Is the Bug: Credit Card Fraud on the ULB Benchmark

A classifier trained on the ULB credit card fraud dataset reaches 0.9788 ROC-AUC with XGBoost. That number is flattering, and it is also beside the point. The operating question is whether the deployed threshold catches fraud at a cost the business can actually pay, and on this dataset the default 0.5 threshold is a business decision that almost no one would sign off on if they saw the confusion matrix underneath it. This write-up walks through the data, the three models I fit, and the cost-sensitive threshold sweep that reframes the whole evaluation.

## The dataset

The dataset comes from a September 2013 sample of card transactions from a European cardholder population, released by the Machine Learning Group at ULB (Dal Pozzolo et al., 2015). It carries 284,807 rows. Of those, 492 are labelled fraud, which works out to a 0.1727 percent positive rate, or roughly one fraudulent transaction for every 580 legitimate ones. Every feature but `Time` and `Amount` has been PCA-transformed, so I have no direct access to merchant ID, card country, or anything else with semantic meaning; the 28 principal components collectively preserve whatever signal the original variables carried while satisfying the cardholder-privacy constraint that prevented the raw data from being released in the first place.

Two columns sit outside the PCA transform. `Time` is the seconds elapsed from the first transaction in the two-day observation window, and `Amount` is the transaction amount in EUR. I standardised both before fitting any model; the principal-component columns are already approximately standardised by construction.

![Class balance with fraud shown on a log scale alongside legitimate transactions.](figures/class-balance.png)

The class balance plot makes the shape of the problem visible at a glance. The trivial classifier that predicts "legitimate" for every row is correct 99.83 percent of the time and never catches a single fraud. That is the floor that the rest of the evaluation has to clear.

![Transaction amount distribution clipped at 500 EUR, with fraud overlaid on legitimate.](figures/amount-distribution.png)

Fraud tends to cluster at the low end of the amount distribution, which is consistent with a card-testing pattern: an attacker validates a stolen card with a small purchase before committing to something larger. A handful of fraudulent transactions occur above 500 EUR, but most sit in the 0 to 150 range.

![Fraud rate per 1,000 transactions by hour of day.](figures/time-pattern.png)

The fraud rate is not flat across the 48 hours of the observation window. There are visible spikes in the late-night hours, which matches the intuition that fraudulent activity runs while legitimate cardholders are asleep. That hourly signal would be far more useful if I had date-of-week information, which the anonymisation step has stripped.

## Three models

I fit three classifiers with the same stratified 75/25 train-test split. The test set carries 123 positives, which is enough to distinguish models but barely enough to sustain the cost sweep that follows.

Logistic regression with `class_weight='balanced'`. This is the honest baseline. It gets the problem shape right, weighs the minority class correctly, and produces calibrated probability scores that a threshold sweep can act on sensibly.

XGBoost with `scale_pos_weight` set to the empirical negative-to-positive ratio. I used 400 trees at depth 5 with the histogram tree method. The `eval_metric` was set to `aucpr` rather than `auc` because the precision-recall curve is the right signal at this prevalence.

Isolation Forest trained only on the legitimate class. This is the unsupervised baseline. It does not see the labels during training and treats fraud detection as an anomaly-detection problem. The idea is to test whether the PCA-transformed features carry enough geometric signal to separate fraud from legitimate transactions without using the labels at all.

![Precision-recall curves for the three models with AP annotations.](figures/precision-recall.png)

Precision-recall is the diagnostic that actually matters here. The baseline precision is 0.0017. Logistic regression reaches AP 0.7062, XGBoost reaches 0.863, and the isolation forest reaches 0.1133. The unsupervised model is not a serious contender at this prevalence, which is a finding in itself: even on geometrically well-behaved PCA-transformed features, anomaly detection without labels loses two orders of magnitude of precision to a supervised baseline.

## The default threshold is the bug

The usual reflex is to report accuracy at the 0.5 decision threshold and move on. That reflex is wrong on this dataset. The right question is not "what is the accuracy at 0.5", but "what threshold minimises the total business cost, and what does the confusion matrix look like at that point". To answer that I had to pick a cost model. The one I used treats a missed fraud as a 100-EUR loss (the amortised downstream cost per fraudulent transaction once disputes, chargebacks, and reissuance are included) and a false positive as a 5-EUR manual review. Swapping those numbers for any real institution is a five-minute edit; the framework is what matters.

With that cost model in hand I swept the threshold from 0.01 to 0.99 for each classifier and computed the total cost at every step.

![Cost curve for XGBoost showing missed-fraud cost, review cost, and total cost across thresholds.](figures/cost-curve-xgboost.png)

The XGBoost cost curve bottoms out at threshold 0.01. That is a startling number the first time you see it — a classifier that flags a transaction as fraud as soon as it assigns it a one-in-a-hundred probability is not what most tutorials build toward. But it is exactly right for this problem. At prevalence 0.17 percent, a posterior probability of 0.01 is already a fifty-seven-fold lift over the base rate, and the review-cost line stays shallow enough that the optimum sits further left than the usual intuition suggests.

The minimum-cost operating point for XGBoost catches 106 of 123 fraud cases in the test set, misses 17, and flags 79 legitimate transactions for review. Total cost: 2,095 EUR against a no-model baseline of 12,300 EUR. The corresponding confusion matrix is shown below.

![Confusion matrix for XGBoost at the min-cost threshold of 0.01.](figures/confusion-xgboost.png)

Logistic regression lands in a similar place with a very different threshold. Its cost curve bottoms out at 0.97, catching 105 fraud cases at a total cost of 2,310 EUR. The two threshold numbers — 0.01 and 0.97 — are not comparable. XGBoost and logistic regression produce scores with different calibrations; only the resulting cost and recall are comparable across models.

![Cost curve for logistic regression showing the optimum at threshold 0.97.](figures/cost-curve-logistic_regression.png)

Isolation Forest at its own minimum-cost threshold catches 72 of 123 fraud but flags 710 legitimate transactions for review. Total cost: 8,650 EUR. At the prices I assumed, it does worse than either supervised baseline by a factor of four. The unsupervised anomaly-detection approach is not viable on this dataset when the cost of review is non-trivial.

## What changes when the costs change

The ranking of the three models is not robust to the cost ratio. I re-ran the sweep for two alternative cost structures. With a 500-EUR cost of a missed fraud and a 5-EUR review cost, XGBoost's optimum moves further left and it misses fewer fraud cases; the logistic regression optimum moves from 0.97 toward 0.80 and the recall climbs. With a 50-EUR cost of a missed fraud and a 10-EUR review cost, the optimum thresholds move right and recall drops. The ordering of XGBoost and logistic regression stays consistent across the three regimes I tested, but the gap between them changes by more than the cost of the manual review process.

The operational takeaway is that the threshold is not a modelling decision. It is a policy decision. A fraud analytics team that deploys this classifier has to know its own cost structure before it can reason about where to place the threshold, and that structure is almost never 1:1, which is what the 0.5 default implicitly assumes.

## Limitations

The anonymisation is the single largest limitation of the dataset. Every feature except `Time` and `Amount` is a principal component, which prevents any meaningful feature-importance analysis or any domain-driven feature engineering. The two-day observation window is short enough that seasonal effects are invisible. The geography is constrained to European cardholders from a specific 2013 sample, which makes generalisation to the 2026 fraud landscape an act of extrapolation.

The cost model is illustrative rather than calibrated to a specific institution. I chose 100 EUR and 5 EUR because they capture the order-of-magnitude asymmetry most fraud teams work under; the exact numbers should be replaced with the team's own before any production decision is made.

The class imbalance is severe enough that the test set carries only 123 positives, which means the recall estimates have confidence intervals of several percentage points. A ten-fold cross-validated version of this analysis would produce a more stable estimate of the optimal threshold, and that is the natural next step for anyone taking this to production.

## References

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. In *2015 IEEE Symposium Series on Computational Intelligence* (pp. 159-166). IEEE.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE.

Machine Learning Group ULB. (2013). *Credit card fraud detection* [Data set]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
