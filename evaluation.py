import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from pathlib import Path

from config import OUTPUT_DIR, FIGURES_DIR, RESULTS_FILE


def compute_metrics(true_labels, scores, predictions, model_name="model"):
    """Compute and return a dict of evaluation metrics."""
    auroc = roc_auc_score(true_labels, scores)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()

    metrics = {
        "model": model_name,
        "AUROC": round(auroc, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    return metrics


def print_metrics(metrics):
    print(f"\n{'='*50}")
    print(f"  Results: {metrics['model']}")
    print(f"{'='*50}")
    print(f"  AUROC     : {metrics['AUROC']:.4f}")
    print(f"  F1-Score  : {metrics['F1']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}")
    print(f"{'='*50}\n")


def save_results(results_list, results_file=None):
    results_file = Path(results_file) if results_file else RESULTS_FILE
    df = pd.DataFrame(results_list)
    df.to_csv(results_file, index=False)
    print(f"  Results saved → {results_file}")
    return df


def plot_roc_curves(results_data, save=True, save_path=None):
    plt.figure(figsize=(8, 6))
    for rd in results_data:
        fpr, tpr, _ = roc_curve(rd["true_labels"], rd["scores"])
        auc = roc_auc_score(rd["true_labels"], rd["scores"])
        plt.plot(fpr, tpr, label=f"{rd['name']} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Industrial Defect Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save:
        path = Path(save_path) if save_path else FIGURES_DIR / "roc_curves.png"
        plt.savefig(path, dpi=150)
        print(f"  ROC curve saved → {path}")
    plt.close()


def plot_score_distributions(results_data, save=True, save_path=None):
    n = len(results_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, rd in zip(axes, results_data):
        true_labels = np.array(rd["true_labels"])
        scores = np.array(rd["scores"])
        normal_scores = scores[true_labels == 0]
        defect_scores = scores[true_labels == 1]

        ax.hist(normal_scores, bins=20, alpha=0.6, color="green", label="Normal")
        ax.hist(defect_scores, bins=20, alpha=0.6, color="red", label="Defect")
        ax.set_title(f"{rd['name']} — Score Distribution")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    if save:
        path = Path(save_path) if save_path else FIGURES_DIR / "score_distributions.png"
        plt.savefig(path, dpi=150)
        print(f"  Score distribution saved → {path}")
    plt.close()


def plot_comparison_bar(results_list, save=True, save_path=None):
    models = [r["model"] for r in results_list]
    aurocs = [r["AUROC"] for r in results_list]
    f1s = [r["F1"] for r in results_list]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color="steelblue")
    ax.bar(x + width / 2, f1s, width, label="F1-Score", color="darkorange")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    if save:
        path = Path(save_path) if save_path else FIGURES_DIR / "model_comparison.png"
        plt.savefig(path, dpi=150)
        print(f"  Comparison chart saved → {path}")
    plt.close()
