#!/usr/bin/env python3
"""
Main pipeline for CP1: Gold Futures Price Direction Prediction.

Runs the full pipeline:
  1. Load and clean data
  2. Engineer features
  3. Create binary target
  4. Time-based train/test split
  5. Train baseline + ML models
  6. Evaluate and compare

Usage:
  python run_pipeline.py
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_dataset, HORIZON
from src.modeling import train_and_evaluate, print_summary, get_feature_importance


def main():
    np.random.seed(42)

    print("=" * 60)
    print("GOLD FUTURES PRICE DIRECTION PREDICTION — CP1 PIPELINE")
    print("=" * 60)

    print(f"\n[1] Building dataset (horizon={HORIZON}) ...")
    X_train, y_train, X_test, y_test, feature_cols, df = build_dataset(
        horizon=HORIZON, test_frac=0.2
    )

    print(f"    Total samples after cleaning:  {len(df)}")
    print(f"    Train size: {len(X_train)}")
    print(f"    Test size:  {len(X_test)}")
    print(f"    Features:   {len(feature_cols)}")
    print(f"    Train target balance: {y_train.mean():.3f} (class 1 fraction)")
    print(f"    Test  target balance: {y_test.mean():.3f} (class 1 fraction)")
    print(f"    Train period: {df.iloc[0]['DateTime']} — {df.iloc[int(len(df)*0.8)-1]['DateTime']}")
    print(f"    Test  period: {df.iloc[int(len(df)*0.8)]['DateTime']} — {df.iloc[-1]['DateTime']}")

    print(f"\n    Feature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i:2d}. {col}")

    print(f"\n[2] Training models ...")
    results = train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)

    print_summary(results)

    print(f"\n[3] Feature importance (tree-based models):")
    importances = get_feature_importance(results, feature_cols)

    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    print(f"{'='*60}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = results[best_name]

    print(f"\nBest model by ROC-AUC: {best_name}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")
    print(f"  F1-score:  {best['f1']:.4f}")
    print(f"  ROC-AUC:   {best['roc_auc']:.4f}")

    baseline_roc = results.get("LogisticRegression", {}).get("roc_auc", 0.5)
    print(f"\nBaseline (LogisticRegression) ROC-AUC: {baseline_roc:.4f}")
    print(f"Random baseline ROC-AUC: 0.5000")

    if best["roc_auc"] < 0.52:
        print("\n⚠ The signal is very weak. Models are barely better than random.")
        print("  This is expected for short-term financial prediction.")
        print("  Consider: different horizons, more features, or ensemble methods.")
    elif best["roc_auc"] < 0.55:
        print("\n◉ There is a weak but potentially useful signal.")
        print("  Further feature engineering and hyperparameter tuning may help.")
    else:
        print("\n✓ The model shows a meaningful signal above random baseline.")

    print(f"\nPipeline completed successfully.")
    return results


if __name__ == "__main__":
    main()
