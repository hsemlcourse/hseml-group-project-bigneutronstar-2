#!/usr/bin/env python3
"""
Main pipeline for Gold Futures Price Direction Prediction.

Runs experiments across multiple horizons, performs walk-forward
cross-validation, tunes hyperparameters, and compares models.

Usage:
  python run_pipeline.py
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    build_dataset, build_full_df, walk_forward_split, HORIZON
)
from src.modeling import (
    train_and_evaluate, print_summary, get_feature_importance,
    walk_forward_evaluate, tune_hyperparameters, get_models
)


def run_holdout_experiment(horizon, test_frac=0.2):
    """Run standard holdout evaluation for a given horizon."""
    print(f"\n{'='*60}")
    print(f"HOLDOUT EXPERIMENT - horizon={horizon}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test, feature_cols, df = build_dataset(
        horizon=horizon, test_frac=test_frac
    )

    print(f"  Samples: {len(df)} total, {len(X_train)} train, {len(X_test)} test")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target balance - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    results = train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)
    print_summary(results)

    return results, feature_cols


def run_walkforward_experiment(horizon, n_splits=5):
    """Run walk-forward CV evaluation for a given horizon."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD CV - horizon={horizon}, {n_splits} folds")
    print(f"{'='*60}")

    df, feature_cols = build_full_df(horizon=horizon)
    splits = walk_forward_split(df, n_splits=n_splits)

    print(f"  Samples: {len(df)}, Features: {len(feature_cols)}")
    print(f"  Splits generated: {len(splits)}")

    wf_results = {}
    for model_name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
        print(f"\n  --- {model_name} ---")
        fold_metrics, mean_metrics = walk_forward_evaluate(
            df, feature_cols, splits, model_name=model_name, verbose=True
        )
        wf_results[model_name] = {"folds": fold_metrics, "mean": mean_metrics}

    print(f"\n  Walk-Forward CV Summary (horizon={horizon}):")
    print(f"  {'Model':<25} {'Mean ROC-AUC':>12} {'Mean Acc':>10} {'Mean F1':>10}")
    print(f"  {'-'*57}")
    for name, r in wf_results.items():
        m = r["mean"]
        print(f"  {name:<25} {m['roc_auc']:>12.4f} {m['accuracy']:>10.4f} {m['f1']:>10.4f}")

    return wf_results, df, feature_cols, splits


def run_tuning(df, feature_cols, splits):
    """Run hyperparameter tuning using walk-forward CV."""
    best_params = tune_hyperparameters(df, feature_cols, splits, verbose=True)
    return best_params


def main():
    np.random.seed(42)

    print("=" * 60)
    print("GOLD FUTURES PRICE DIRECTION PREDICTION")
    print("=" * 60)

    horizons = [1, 4, 24]
    all_holdout_results = {}
    all_wf_results = {}

    for h in horizons:
        holdout_results, feature_cols = run_holdout_experiment(h)
        all_holdout_results[h] = holdout_results

        wf_results, df, feature_cols, splits = run_walkforward_experiment(h)
        all_wf_results[h] = wf_results

    best_horizon = max(
        all_wf_results.keys(),
        key=lambda h: max(r["mean"]["roc_auc"] for r in all_wf_results[h].values())
    )
    print(f"\n{'='*60}")
    print(f"BEST HORIZON: {best_horizon}")
    print(f"{'='*60}")

    print(f"\nRunning hyperparameter tuning on horizon={best_horizon} ...")
    df_best, feature_cols_best = build_full_df(horizon=best_horizon)
    splits_best = walk_forward_split(df_best, n_splits=5)
    best_params = run_tuning(df_best, feature_cols_best, splits_best)

    print(f"\n{'='*60}")
    print(f"TUNED MODELS - holdout evaluation (horizon={best_horizon})")
    print(f"{'='*60}")
    X_train, y_train, X_test, y_test, fc, _ = build_dataset(
        horizon=best_horizon, test_frac=0.2
    )

    from sklearn.preprocessing import StandardScaler
    from src.modeling import _evaluate

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    tuned_results = {}
    for model_key, info in best_params.items():
        params = info["best_params"]
        print(f"\n  {model_key}: {params}")

        if model_key == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**params)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)[:, 1]
        elif model_key == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = _evaluate(y_test, y_pred, y_proba)
        tuned_results[f"{model_key}_tuned"] = {
            "model": model, **metrics, "y_pred": y_pred, "y_proba": y_proba
        }
        print(f"    Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"ROC-AUC={metrics['roc_auc']:.4f}")

    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (best tuned tree model)")
    print(f"{'='*60}")
    get_feature_importance(tuned_results, fc)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY - ALL HORIZONS (holdout)")
    print(f"{'='*60}")
    for h in horizons:
        print(f"\n  Horizon={h}:")
        print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
        print(f"  {'-'*55}")
        for name, r in all_holdout_results[h].items():
            print(f"  {name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")

    print(f"\n  Tuned models (horizon={best_horizon}):")
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*60}")
    for name, r in tuned_results.items():
        print(f"  {name:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")

    save_results(all_holdout_results, all_wf_results, best_params, tuned_results,
                 fc, best_horizon)

    print(f"\nPipeline completed successfully.")


def save_results(holdout_results, wf_results, best_params, tuned_results,
                 feature_cols, best_horizon):
    """Save metrics to JSON and best model to pickle."""
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "best_horizon": best_horizon,
        "feature_cols": feature_cols,
        "holdout": {},
        "walk_forward": {},
        "tuning": {},
        "tuned_holdout": {},
    }

    for h, results in holdout_results.items():
        summary["holdout"][str(h)] = {
            name: {"accuracy": r["accuracy"], "f1": r["f1"], "roc_auc": r["roc_auc"]}
            for name, r in results.items()
        }

    for h, results in wf_results.items():
        summary["walk_forward"][str(h)] = {
            name: r["mean"]
            for name, r in results.items()
        }

    for name, info in best_params.items():
        summary["tuning"][name] = {
            "best_score": info["best_score"],
            "best_params": {k: v for k, v in info["best_params"].items()
                          if k != "random_state"},
        }

    for name, r in tuned_results.items():
        summary["tuned_holdout"][name] = {
            "accuracy": r["accuracy"], "f1": r["f1"], "roc_auc": r["roc_auc"]
        }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")

    best_model_name = max(tuned_results, key=lambda k: tuned_results[k]["roc_auc"])
    best_model = tuned_results[best_model_name]["model"]
    model_path = output_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model ({best_model_name}) saved to {model_path}")


if __name__ == "__main__":
    main()
