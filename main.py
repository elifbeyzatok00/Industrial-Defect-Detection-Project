"""
Industrial Defect Detection — Main Entry Point
Transfer Learning (ResNet50) + PatchCore vs Gaussian Baseline
Dataset: MVTec AD
"""

import argparse
import sys
from pathlib import Path

from data_loader import load_train_data, load_test_data, generate_dummy_data
from baseline_method import GaussianBaselineDetector
from patchcore_model import PatchCore
from evaluation import (
    compute_metrics, print_metrics, save_results,
    plot_roc_curves, plot_score_distributions, plot_comparison_bar,
)
from config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR, DATASET_DIR

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial Defect Detection")
    parser.add_argument("--dummy", action="store_true",
                        help="Generate and use dummy data instead of MVTec AD")
    parser.add_argument("--few-shot", action="store_true",
                        help="Also run few-shot scenario (30%% of training data)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip Gaussian baseline evaluation")
    parser.add_argument("--coreset", type=float, default=0.1,
                        help="PatchCore coreset ratio (default 0.1)")
    parser.add_argument("--all-categories", action="store_true",
                        help="Run on all 15 MVTec AD categories and save a combined summary")
    return parser.parse_args()


def check_data_exists():
    train_images = list(TRAIN_DIR.glob("*.png")) + list(TRAIN_DIR.glob("*.jpg"))
    return len(train_images) > 0


def run_baseline_experiment(test_paths, true_labels):
    print("\n--- Gaussian Baseline ---")
    train_paths = list(TRAIN_DIR.glob("*.png")) + list(TRAIN_DIR.glob("*.jpg"))
    detector = GaussianBaselineDetector(blur_sigma=3)
    detector.fit(train_paths)

    import numpy as np
    from tqdm import tqdm
    scores = np.array([detector.score(p) for p in tqdm(test_paths, desc="Baseline")])
    threshold = detector._auto_threshold(scores)
    predictions = (scores >= threshold).astype(int)

    metrics = compute_metrics(true_labels, scores, predictions, model_name="Baseline")
    print_metrics(metrics)
    return metrics, scores


def run_patchcore_experiment(train_loader, test_loader, true_labels,
                              coreset_ratio=0.1, label="PatchCore"):
    print(f"\n--- {label} ---")
    model = PatchCore(coreset_ratio=coreset_ratio)
    model.fit(train_loader)
    model.save()

    scores, predictions, threshold = model.predict(test_loader)

    import numpy as np
    metrics = compute_metrics(true_labels, scores, predictions, model_name=label)
    print_metrics(metrics)
    return metrics, scores


def collect_test_paths(test_dir=None):
    import numpy as np
    test_dir = Path(test_dir) if test_dir else TEST_DIR
    test_paths, true_labels = [], []
    good_dir = test_dir / "good"
    if good_dir.exists():
        for p in sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg")):
            test_paths.append(p)
            true_labels.append(0)
    for d in sorted(test_dir.iterdir()):
        if d.name == "good" or not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            test_paths.append(p)
            true_labels.append(1)
    return test_paths, np.array(true_labels)


def run_for_category(category, args):
    import numpy as np
    train_dir = DATASET_DIR / category / "train" / "good"
    test_dir  = DATASET_DIR / category / "test"

    if not train_dir.exists():
        print(f"  [SKIP] {category} — klasör bulunamadı: {train_dir}")
        return []

    cat_output = OUTPUT_DIR / category
    cat_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Kategori: {category.upper()}")
    print(f"{'='*60}")

    train_loader, n_train = load_train_data(few_shot_ratio=1.0, train_dir=train_dir)
    test_loader, true_labels = load_test_data(test_dir=test_dir)
    test_paths, _ = collect_test_paths(test_dir=test_dir)
    true_labels = np.array(true_labels)

    print(f"  Train: {n_train}  |  Test: {len(true_labels)}"
          f" ({(true_labels==0).sum()} normal, {(true_labels==1).sum()} defect)")

    all_metrics, roc_data = [], []

    if not args.skip_baseline:
        b_metrics, b_scores = run_baseline_experiment(test_paths, true_labels)
        all_metrics.append(b_metrics)
        roc_data.append({"name": "Baseline", "true_labels": true_labels, "scores": b_scores})

    pc_metrics, pc_scores = run_patchcore_experiment(
        train_loader, test_loader, true_labels,
        coreset_ratio=args.coreset, label="PatchCore"
    )
    all_metrics.append(pc_metrics)
    roc_data.append({"name": "PatchCore", "true_labels": true_labels, "scores": pc_scores})

    save_results(all_metrics, results_file=cat_output / "results.csv")
    plot_roc_curves(roc_data,          save_path=cat_output / "roc_curves.png")
    plot_score_distributions(roc_data, save_path=cat_output / "score_distributions.png")
    plot_comparison_bar(all_metrics,   save_path=cat_output / "model_comparison.png")

    for m in all_metrics:
        m["category"] = category
    return all_metrics


def main():
    args = parse_args()

    print("=" * 60)
    print("  Industrial Defect Detection")
    print("  PatchCore + Transfer Learning vs Gaussian Baseline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # All-categories mode
    # ------------------------------------------------------------------
    if args.all_categories:
        import pandas as pd
        import numpy as np
        combined = []
        for cat in ALL_CATEGORIES:
            metrics = run_for_category(cat, args)
            combined.extend(metrics)

        if combined:
            df = pd.DataFrame(combined)
            summary_path = OUTPUT_DIR / "all_categories_summary.csv"
            df.to_csv(summary_path, index=False)
            print("\n" + "=" * 60)
            print("  All-Categories Summary")
            print("=" * 60)
            cols = ["category", "model", "AUROC", "F1"]
            print(df[[c for c in cols if c in df.columns]].to_string(index=False))
            print(f"\nFull summary saved → {summary_path}")
        print("Done.")
        return

    # ------------------------------------------------------------------
    # Single-category mode (default)
    # ------------------------------------------------------------------
    if args.dummy or not check_data_exists():
        if not args.dummy:
            print("\nNo dataset found. Generating dummy data automatically.")
            print("Tip: Use --dummy flag explicitly to suppress this message.")
        generate_dummy_data()

    print("\nLoading data...")
    train_loader, n_train = load_train_data(few_shot_ratio=1.0)
    test_loader, true_labels = load_test_data()
    test_paths, _ = collect_test_paths()

    import numpy as np
    true_labels = np.array(true_labels)
    print(f"  Train images : {n_train}")
    print(f"  Test images  : {len(true_labels)}"
          f" ({(true_labels == 0).sum()} normal, {(true_labels == 1).sum()} defect)")

    all_metrics = []
    roc_data = []

    # ------------------------------------------------------------------
    # Gaussian Baseline
    # ------------------------------------------------------------------
    if not args.skip_baseline:
        b_metrics, b_scores = run_baseline_experiment(test_paths, true_labels)
        all_metrics.append(b_metrics)
        roc_data.append({"name": "Baseline", "true_labels": true_labels,
                         "scores": b_scores})

    # ------------------------------------------------------------------
    # PatchCore (full training set)
    # ------------------------------------------------------------------
    pc_metrics, pc_scores = run_patchcore_experiment(
        train_loader, test_loader, true_labels,
        coreset_ratio=args.coreset, label="PatchCore"
    )
    all_metrics.append(pc_metrics)
    roc_data.append({"name": "PatchCore", "true_labels": true_labels,
                     "scores": pc_scores})

    # ------------------------------------------------------------------
    # Few-shot scenario (30% training data)
    # ------------------------------------------------------------------
    if args.few_shot:
        fs_loader, n_fs = load_train_data(few_shot_ratio=0.3)
        print(f"\nFew-shot: using {n_fs}/{n_train} training images (30%)")
        fs_metrics, fs_scores = run_patchcore_experiment(
            fs_loader, test_loader, true_labels,
            coreset_ratio=args.coreset, label="PatchCore (30% few-shot)"
        )
        all_metrics.append(fs_metrics)
        roc_data.append({"name": "PatchCore (few-shot)", "true_labels": true_labels,
                         "scores": fs_scores})

    # ------------------------------------------------------------------
    # Save results and plots
    # ------------------------------------------------------------------
    print("\nGenerating plots and saving results...")
    df = save_results(all_metrics)
    plot_roc_curves(roc_data)
    plot_score_distributions(roc_data)
    plot_comparison_bar(all_metrics)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\nAll outputs saved to:", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
