#!/usr/bin/env python3
"""
Main pipeline for Gold Futures Price Direction Prediction.
Runs experiments across multiple horizons and thresholds,
performs walk-forward cross-validation, tunes hyperparameters,
runs backtests and compares models.
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
    build_dataset, build_full_df, walk_forward_split
)
from src.modeling import (
    train_and_evaluate, print_summary, get_feature_importance,
    walk_forward_evaluate, tune_hyperparameters, run_simple_backtest
)
def run_holdout_experiment(horizon, threshold, test_frac=0.2):
    """Run standard holdout evaluation for a given horizon and threshold."""
    print(f"\n{'='*60}")
    print(f"HOLDOUT EXPERIMENT - horizon={horizon}, threshold={threshold}")
    print(f"{'='*60}")
    X_train, y_train, X_test, y_test, feature_cols, df, future_rets = build_dataset(
        horizon=horizon, threshold=threshold, test_frac=test_frac
    )
    print(f"  Samples: {len(df)} total, {len(X_train)} train, {len(X_test)} test")
    print(f"  Features: {len(feature_cols)}")
    classes, counts = np.unique(y_train, return_counts=True)
    dist = ", ".join([f"class {c}: {cnt/len(y_train):.2f}" for c, cnt in zip(classes, counts)])
    print(f"  Target balance - train: {dist}")
    results = train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)
    print_summary(results)
    print("\n  --- Backtest Results ---")
    for name, res in results.items():
        bt = run_simple_backtest(res["y_proba"], future_rets)
        res["backtest"] = bt
        print(f"  {name:<25}: HitRate={bt['hit_rate']:.2f}, Trades={bt['n_trades']}, AvgRet={bt['avg_return_trade']:.5f}")
    return results, feature_cols
def run_walkforward_experiment(horizon, threshold, n_splits=3):
    """Run walk-forward CV evaluation for a given horizon."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD CV - horizon={horizon}, threshold={threshold}, {n_splits} folds")
    print(f"{'='*60}")
    df, feature_cols = build_full_df(horizon=horizon, threshold=threshold)
    splits = walk_forward_split(df, n_splits=n_splits)
    print(f"  Samples: {len(df)}, Features: {len(feature_cols)}")
    print(f"  Splits generated: {len(splits)}")
    wf_results = {}
    models_to_test = ["LogisticRegression", "RandomForest", "GradientBoosting", "CatBoost"]
    for model_name in models_to_test:
        print(f"\n  --- {model_name} ---")
        fold_metrics, mean_metrics = walk_forward_evaluate(
            df, feature_cols, splits, model_name=model_name, verbose=True
        )
        wf_results[model_name] = {"folds": fold_metrics, "mean": mean_metrics}
    print(f"\n  Walk-Forward CV Summary (horizon={horizon}, thresh={threshold}):")
    print(f"  {'Model':<25} {'Mean ROC-AUC':>12} {'Mean Acc':>10} {'Mean F1(m)':>10}")
    print(f"  {'-'*57}")
    for name, r in wf_results.items():
        m = r["mean"]
        print(f"  {name:<25} {m['roc_auc']:>12.4f} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f}")
    return wf_results, df, feature_cols, splits
def run_tuning(df, feature_cols, splits):
    """Run hyperparameter tuning using walk-forward CV."""
    best_params = tune_hyperparameters(df, feature_cols, splits, verbose=True)
    return best_params
def main():
    np.random.seed(42)
    print("=" * 60)
    print("GOLD FUTURES PRICE DIRECTION PREDICTION - ITERATION 2")
    print("=" * 60)
    horizons = [24]
    thresholds = [0.001, 0.002]
    all_holdout_results = {}
    all_wf_results = {}
    for h in horizons:
        for t in thresholds:
            key = f"h{h}_t{t}"
            holdout_results, feature_cols = run_holdout_experiment(h, t)
            all_holdout_results[key] = holdout_results
            wf_results, df, feature_cols, splits = run_walkforward_experiment(h, t)
            all_wf_results[key] = wf_results
    best_key = max(
        all_wf_results.keys(),
        key=lambda k: max(r["mean"]["roc_auc"] for r in all_wf_results[k].values())
    )
    b_h, b_t = best_key.split("_")
    best_horizon = int(b_h[1:])
    best_thresh = float(b_t[1:])
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: horizon={best_horizon}, threshold={best_thresh}")
    print(f"{'='*60}")
    print("\nRunning hyperparameter tuning on best config ...")
    df_best, feature_cols_best = build_full_df(horizon=best_horizon, threshold=best_thresh)
    splits_best = walk_forward_split(df_best, n_splits=3)
    best_params = run_tuning(df_best, feature_cols_best, splits_best)
    print(f"\n{'='*60}")
    print("TUNED MODELS - holdout evaluation")
    print(f"{'='*60}")
    X_train, y_train, X_test, y_test, fc, _, future_rets = build_dataset(
        horizon=best_horizon, threshold=best_thresh, test_frac=0.2
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
        if "Logistic" in model_key:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**params)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)
        elif "RandomForest" in model_key:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        elif "GradientBoosting" in model_key:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        elif "CatBoost" in model_key:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        else:
            continue
        metrics = _evaluate(y_test, y_pred, y_proba)
        bt = run_simple_backtest(y_proba, future_rets)
        tuned_results[f"{model_key}_tuned"] = {
            "model": model, **metrics, "y_pred": y_pred, "y_proba": y_proba, "backtest": bt
        }
        print(f"    Accuracy={metrics['accuracy']:.4f}, F1(mac)={metrics['f1_macro']:.4f}, "
              f"ROC-AUC={metrics['roc_auc']:.4f}, Backtest_Trades={bt['n_trades']}, Backtest_AvgRet={bt['avg_return_trade']:.5f}")
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (best tuned tree model)")
    print(f"{'='*60}")
    get_feature_importance(tuned_results, fc)
    save_results(all_holdout_results, all_wf_results, best_params, tuned_results,
                 fc, best_horizon, best_thresh)
    print("\nPipeline completed successfully.")
def save_results(holdout_results, wf_results, best_params, tuned_results,
                 feature_cols, best_horizon, best_thresh):
    """Save metrics to JSON and best model to pickle."""
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    summary = {
        "best_horizon": best_horizon,
        "best_threshold": best_thresh,
        "feature_cols": feature_cols,
        "holdout": {},
        "walk_forward": {},
        "tuning": {},
        "tuned_holdout": {},
    }
    for key, results in holdout_results.items():
        summary["holdout"][key] = {
            name: {"accuracy": r["accuracy"], "f1_macro": r["f1_macro"], "roc_auc": r["roc_auc"], 
                   "backtest": r.get("backtest", {})}
            for name, r in results.items()
        }
    for key, results in wf_results.items():
        summary["walk_forward"][key] = {
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
            "accuracy": r["accuracy"], "f1_macro": r["f1_macro"], "roc_auc": r["roc_auc"],
            "backtest": r.get("backtest", {})
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
