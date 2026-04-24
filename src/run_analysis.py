"""End-to-end credit card fraud analysis.

Loads the Kaggle ULB creditcardfraud dataset, profiles the 0.17% positive rate,
fits three classifiers (logistic regression, XGBoost, isolation forest), and
runs a cost-sensitive threshold sweep that reframes the operating point as a
business question rather than a hyperparameter. All figures and metrics are
written under figures/ and outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RNG = 42
COST_FN = 100.0
COST_FP = 5.0

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11})
sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--figures", required=True)
    p.add_argument("--outputs", required=True)
    return p.parse_args()


def class_balance_figure(y: pd.Series, path: Path) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4c72b0", "#c44e52"])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}", ha="center", va="bottom")
    ax.set_ylabel("Transactions")
    ax.set_title(f"Class balance: {counts.iloc[1]:,} fraud of {counts.sum():,} total ({counts.iloc[1] / counts.sum() * 100:.3f}%)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def amount_distribution_figure(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df.loc[df["Class"] == 0, "Amount"], bins=80, range=(0, 500), alpha=0.55, color="#4c72b0", label="Legitimate")
    ax.hist(df.loc[df["Class"] == 1, "Amount"], bins=80, range=(0, 500), alpha=0.75, color="#c44e52", label="Fraud")
    ax.set_xlabel("Transaction amount (EUR)")
    ax.set_ylabel("Count")
    ax.set_title("Transaction amount distribution, clipped at 500 EUR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def time_pattern_figure(df: pd.DataFrame, path: Path) -> None:
    hours = (df["Time"] / 3600).astype(int) % 24
    fraud_rate = df.assign(Hour=hours).groupby("Hour")["Class"].mean() * 1000
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(fraud_rate.index, fraud_rate.values, color="#c44e52")
    ax.set_xlabel("Hour of day (UTC, derived from Time seconds)")
    ax.set_ylabel("Fraud per 1,000 transactions")
    ax.set_title("Fraud rate by hour: the raw rate is not flat across the day")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cost_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    total_cost = COST_FN * fn + COST_FP * fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall),
        "fraud_missed_cost": float(COST_FN * fn),
        "review_cost": float(COST_FP * fp),
        "total_cost": float(total_cost),
    }


def sweep_costs(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = [cost_at_threshold(y_true, scores, t) for t in thresholds]
    return pd.DataFrame(rows)


def cost_curve_figure(sweep: pd.DataFrame, best_threshold: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["threshold"], sweep["total_cost"], color="#2d3047", label="Total cost")
    ax.plot(sweep["threshold"], sweep["fraud_missed_cost"], color="#c44e52", ls="--", label="Missed fraud cost")
    ax.plot(sweep["threshold"], sweep["review_cost"], color="#4c72b0", ls="--", label="Review cost")
    ax.axvline(best_threshold, color="#e1a94b", ls=":", lw=2, label=f"Min-cost threshold = {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Expected cost (EUR, with $FN={int(COST_FN)}, $FP={int(COST_FP)})")
    ax.set_title(title)
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pr_curve_figure(y_true: np.ndarray, model_scores: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (scores, color) in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP = {ap:.3f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", ls=":", label=f"Baseline = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves, test set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_figure(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
        xticklabels=["Pred legit", "Pred fraud"],
        yticklabels=["Actual legit", "Actual fraud"], ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.figures)
    out_dir = Path(args.outputs)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    df = pd.read_csv(args.data)
    print(f"  rows={len(df):,}  positives={int(df['Class'].sum()):,}  rate={df['Class'].mean() * 100:.4f}%")

    class_balance_figure(df["Class"], fig_dir / "class-balance.png")
    amount_distribution_figure(df, fig_dir / "amount-distribution.png")
    time_pattern_figure(df, fig_dir / "time-pattern.png")

    y = df["Class"].values
    X = df.drop(columns=["Class"])
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RNG,
    )
    print(f"  train positives={int(y_train.sum())}  test positives={int(y_test.sum())}")

    print("Fitting logistic regression")
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG)
    logreg.fit(X_train, y_train)
    logreg_scores = logreg.predict_proba(X_test)[:, 1]

    print("Fitting XGBoost")
    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        tree_method="hist", random_state=RNG, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_scores = xgb.predict_proba(X_test)[:, 1]

    print("Fitting Isolation Forest (unsupervised)")
    iforest = IsolationForest(
        n_estimators=200, contamination=float(y_train.mean()),
        random_state=RNG, n_jobs=-1,
    )
    iforest.fit(X_train[y_train == 0])
    iforest_raw = -iforest.score_samples(X_test)
    iforest_scores = (iforest_raw - iforest_raw.min()) / (iforest_raw.max() - iforest_raw.min())

    pr_curve_figure(
        y_test,
        {
            "Logistic regression": (logreg_scores, "#4c72b0"),
            "XGBoost": (xgb_scores, "#c44e52"),
            "Isolation Forest": (iforest_scores, "#55a868"),
        },
        fig_dir / "precision-recall.png",
    )

    headline = {}
    model_scores = {
        "logistic_regression": logreg_scores,
        "xgboost": xgb_scores,
        "isolation_forest": iforest_scores,
    }
    for name, scores in model_scores.items():
        sweep = sweep_costs(y_test, scores)
        sweep.to_csv(out_dir / f"cost-sweep-{name}.csv", index=False)
        best = sweep.loc[sweep["total_cost"].idxmin()]
        headline[name] = {
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "average_precision": float(average_precision_score(y_test, scores)),
            "min_cost_threshold": float(best["threshold"]),
            "min_cost_recall": float(best["recall"]),
            "min_cost_precision": float(best["precision"]),
            "min_cost_total": float(best["total_cost"]),
            "min_cost_missed_cost": float(best["fraud_missed_cost"]),
            "min_cost_review_cost": float(best["review_cost"]),
            "min_cost_tp": int(best["tp"]),
            "min_cost_fp": int(best["fp"]),
            "min_cost_fn": int(best["fn"]),
        }
        cost_curve_figure(sweep, best["threshold"], fig_dir / f"cost-curve-{name}.png", f"{name.replace('_', ' ').title()}: expected cost vs. threshold")
        pred = (scores >= best["threshold"]).astype(int)
        confusion_figure(y_test, pred, f"{name.replace('_', ' ').title()} at min-cost threshold", fig_dir / f"confusion-{name}.png")

    # Cost-ratio animation for XGBoost: sweep the ratio of FN:FP cost and see how
    # the minimum-cost threshold migrates. This is the figure that makes the
    # policy-vs-model argument visible on one axis.
    xgb_scores = model_scores["xgboost"]
    fn_over_fp_ratios = np.linspace(2, 60, 30)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (EUR) for flagged half of test set")
    ax.set_title("XGBoost cost curve as the missed-fraud/review-cost ratio grows")
    thresholds = np.linspace(0.001, 0.99, 200)
    frames_data = []
    for ratio in fn_over_fp_ratios:
        fp_cost = 5.0
        fn_cost = 5.0 * ratio
        costs = []
        best_t = None
        best_c = np.inf
        for t in thresholds:
            pred = (xgb_scores >= t).astype(int)
            tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            c = fn_cost * fn_ + fp_cost * fp_
            costs.append(c)
            if c < best_c:
                best_c = c
                best_t = t
        frames_data.append((ratio, np.array(costs), best_t, best_c))

    line, = ax.plot([], [], lw=2, color="#2d3047")
    vline = ax.axvline(0, color="#e1a94b", ls=":", lw=2)
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlim(0, 1)
    all_costs = np.concatenate([fc[1] for fc in frames_data])
    ax.set_ylim(0, all_costs.max() * 1.05)

    def animate(i):
        ratio, costs, best_t, best_c = frames_data[i]
        line.set_data(thresholds, costs)
        vline.set_xdata([best_t, best_t])
        txt.set_text(f"FN : FP cost ratio = {ratio:.0f} : 1\nmin-cost threshold = {best_t:.3f}\nmin cost = {best_c:,.0f} EUR")
        return line, vline, txt

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data), interval=350, blit=True)
    anim.save(str(fig_dir / "cost-ratio-animation.gif"), writer="pillow", fps=3)
    plt.close(fig)

    # Class distribution across V1..V28 for a random pair of features
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax_, feat in zip(axes.flat, ["V1", "V3", "V4", "V10", "V14", "V17"]):
        sns.kdeplot(data=df, x=feat, hue="Class", ax=ax_, common_norm=False, palette=["#4c72b0", "#c44e52"], fill=True, alpha=0.35)
        ax_.set_title(f"{feat} density by class")
    plt.suptitle("Where the PCA features actually separate the classes", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature-kde.png")
    plt.close(fig)

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "positives": int(df["Class"].sum()),
            "positive_rate_pct": float(df["Class"].mean() * 100),
            "features": list(df.columns[:-1]),
        },
        "costs_used": {"cost_false_negative_eur": COST_FN, "cost_false_positive_eur": COST_FP},
        "test_positives": int(y_test.sum()),
        "test_size": int(len(y_test)),
        "models": headline,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    rows = []
    for name, vals in headline.items():
        rows.append({
            "model": name,
            "roc_auc": round(vals["roc_auc"], 4),
            "average_precision": round(vals["average_precision"], 4),
            "threshold": round(vals["min_cost_threshold"], 2),
            "recall": round(vals["min_cost_recall"], 3),
            "precision": round(vals["min_cost_precision"], 3),
            "missed_fraud": vals["min_cost_fn"],
            "false_review": vals["min_cost_fp"],
            "total_cost_eur": round(vals["min_cost_total"], 2),
        })
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)

    md = ["# Credit card fraud analysis summary", ""]
    md.append(f"Dataset rows: {len(df):,}. Positive class: {int(df['Class'].sum()):,} ({df['Class'].mean() * 100:.4f}%).")
    md.append("")
    md.append("## Headline metrics (min-cost operating point)")
    md.append("")
    md.append("| Model | ROC-AUC | AP | Threshold | Recall | Precision | Missed fraud | False reviews | Total cost (EUR) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        md.append(
            f"| {row['model']} | {row['roc_auc']} | {row['average_precision']} | {row['threshold']} | "
            f"{row['recall']} | {row['precision']} | {row['missed_fraud']} | {row['false_review']} | {row['total_cost_eur']:,} |"
        )
    md.append("")
    md.append(f"Cost assumptions: missed fraud = EUR {int(COST_FN)} per case, false positive review = EUR {int(COST_FP)} per case.")
    (out_dir / "analysis_summary.md").write_text("\n".join(md))

    print("Done")


if __name__ == "__main__":
    main()
