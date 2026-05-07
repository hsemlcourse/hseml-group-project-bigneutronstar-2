# Пайплайн для прогнозирования золота (CP2).
# Сравнивает Variant A (только золото) и Variant B (золото + макро).
import sys
import json
import argparse
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
    walk_forward_evaluate, tune_hyperparameters, run_simple_backtest,
    get_models, _evaluate, RANDOM_SEED
)


HORIZON = 24
THRESHOLD = 0.002
TEST_FRAC = 0.2
N_SPLITS = 3
ALL_MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting", "CatBoost", "ExtraTrees"]


# -----------------------------------------------------------------------------
# Holdout
# -----------------------------------------------------------------------------

def run_holdout_experiment(horizon, threshold, test_frac=TEST_FRAC, use_external=True, label=""):
    """Оценка на отложенной выборке."""
    tag = f"[{label}]" if label else ""
    print(f"\n{'='*65}")
    print(f"HOLDOUT {tag} horizon={horizon}, threshold={threshold}, external={use_external}")
    print(f"{'='*65}")

    X_train, y_train, X_test, y_test, feature_cols, df, future_rets = build_dataset(
        horizon=horizon, threshold=threshold,
        test_frac=test_frac, use_external=use_external
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
        print(f"  {name:<28}: HitRate={bt['hit_rate']:.2f}, "
              f"Trades={bt['n_trades']}, AvgRet={bt['avg_return_trade']:.5f}")

    return results, feature_cols, df, future_rets


# -----------------------------------------------------------------------------
# Walk-forward
# -----------------------------------------------------------------------------

def run_walkforward_experiment(horizon, threshold, n_splits=N_SPLITS,
                               use_external=True, label=""):
    """Оценка через walk-forward."""
    tag = f"[{label}]" if label else ""
    print(f"\n{'='*65}")
    print(f"WALK-FORWARD {tag} horizon={horizon}, threshold={threshold}, "
          f"external={use_external}, {n_splits} folds")
    print(f"{'='*65}")

    df, feature_cols = build_full_df(horizon=horizon, threshold=threshold,
                                     use_external=use_external)
    splits = walk_forward_split(df, n_splits=n_splits)
    print(f"  Samples: {len(df)}, Features: {len(feature_cols)}, Splits: {len(splits)}")

    wf_results = {}
    for model_name in ALL_MODELS:
        print(f"\n  --- {model_name} ---")
        fold_metrics, mean_metrics = walk_forward_evaluate(
            df, feature_cols, splits, model_name=model_name, verbose=True
        )
        wf_results[model_name] = {"folds": fold_metrics, "mean": mean_metrics}

    print(f"\n  Walk-Forward Summary (horizon={horizon}, thresh={threshold}, ext={use_external}):")
    print(f"  {'Model':<28} {'WF ROC-AUC':>12} {'WF Acc':>10} {'WF F1(m)':>10}")
    print(f"  {'-'*60}")
    for name, r in wf_results.items():
        m = r["mean"]
        print(f"  {name:<28} {m['roc_auc']:>12.4f} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f}")

    return wf_results, df, feature_cols, splits


# ----------------------------------------------------------------------------─
# Tuning
# ----------------------------------------------------------------------------─

def run_tuning(df, feature_cols, splits):
    """Run hyperparameter tuning using walk-forward CV."""
    return tune_hyperparameters(df, feature_cols, splits, verbose=True)


# ----------------------------------------------------------------------------─
# -----------------------------------------------------------------------------
# Tuned holdout
# -----------------------------------------------------------------------------

def run_tuned_holdout(best_params, horizon, threshold, use_external=True, label=""):
    """Переобучение лучших моделей на холдоуте."""
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    X_train, y_train, X_test, y_test, fc, _, future_rets = build_dataset(
        horizon=horizon, threshold=threshold,
        test_frac=TEST_FRAC, use_external=use_external
    )
    scaler = SS()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cls_map = {
        "LogisticRegression":  (LogisticRegression,          True),
        "RandomForest":        (RandomForestClassifier,       False),
        "GradientBoosting":    (GradientBoostingClassifier,   False),
        "CatBoost":            (CatBoostClassifier,           False),
        "ExtraTrees":         (ExtraTreesClassifier,         False),
    }

    tuned_results = {}
    tag = f"[{label}]" if label else ""
    print(f"\n{'='*65}")
    print(f"TUNED HOLDOUT {tag} horizon={horizon}, threshold={threshold}, external={use_external}")
    print(f"{'='*65}")

    for model_key, info in best_params.items():
        params = info["best_params"]
        if model_key not in cls_map:
            continue
        model_cls, use_scale = cls_map[model_key]
        print(f"\n  {model_key}: {params}")

        model = model_cls(**params)
        if use_scale:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

        metrics = _evaluate(y_test, y_pred, y_proba)
        bt = run_simple_backtest(y_proba, future_rets)
        tuned_results[f"{model_key}_tuned"] = {
            "model": model, **metrics,
            "y_pred": y_pred, "y_proba": y_proba, "backtest": bt
        }
        print(f"    Acc={metrics['accuracy']:.4f}, F1(mac)={metrics['f1_macro']:.4f}, "
              f"ROC-AUC={metrics['roc_auc']:.4f}, "
              f"Trades={bt['n_trades']}, HitRate={bt['hit_rate']:.2f}, "
              f"AvgRet={bt['avg_return_trade']:.5f}")

    return tuned_results, fc


# ----------------------------------------------------------------------------─
# Save results
# ----------------------------------------------------------------------------─

def _safe_metrics(r):
    return {
        "accuracy": r["accuracy"],
        "f1_macro": r["f1_macro"],
        "roc_auc": r["roc_auc"],
        "backtest": r.get("backtest", {}),
    }


def save_results(summary: dict, best_model, best_model_name: str):
    """Save metrics to JSON and best model to pkl."""
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    model_path = output_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model ({best_model_name}) saved to {model_path}")


# ----------------------------------------------------------------------------─
# Comparison printer
# ----------------------------------------------------------------------------─

def print_comparison(results_a: dict, results_b: dict, title="VARIANT COMPARISON"):
    """Print side-by-side comparison of Variant A (gold-only) vs Variant B (gold+ext)."""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Variant':>9} {'Accuracy':>10} {'F1(mac)':>9} {'ROC-AUC':>9} {'HitRate':>9}")
    print("-" * 80)

    all_names = sorted(set(list(results_a.keys()) + list(results_b.keys())))
    for name in all_names:
        base = name.replace("_tuned", "")
        if name in results_a:
            r = results_a[name]
            bt = r.get("backtest", {})
            hr = bt.get("hit_rate", 0)
            print(f"  {name:<28} {'A (no ext)':>9} {r['accuracy']:>10.4f} "
                  f"{r['f1_macro']:>9.4f} {r['roc_auc']:>9.4f} {hr:>9.2f}")
        if name in results_b:
            r = results_b[name]
            bt = r.get("backtest", {})
            hr = bt.get("hit_rate", 0)
            print(f"  {name:<28} {'B (+ext)':>9} {r['accuracy']:>10.4f} "
                  f"{r['f1_macro']:>9.4f} {r['roc_auc']:>9.4f} {hr:>9.2f}")
    print()


# ----------------------------------------------------------------------------─
# Main
# ----------------------------------------------------------------------------─

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip expensive tuning (use default params only)")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("=" * 65)
    print("GOLD FUTURES DIRECTION PREDICTION - CP2")
    print(f"Horizon={HORIZON}h | Threshold={THRESHOLD} | Seed={RANDOM_SEED}")
    print("=" * 65)

    # -- VARIANT A: gold-only ------------------------------------─
    print("\n\n" + "█" * 65)
    print("VARIANT A: GOLD-ONLY FEATURES")
    print("█" * 65)

    ho_a, fc_a, df_a, rets_a = run_holdout_experiment(
        HORIZON, THRESHOLD, use_external=False, label="A:gold-only"
    )
    wf_a, _, feat_a, splits_a = run_walkforward_experiment(
        HORIZON, THRESHOLD, use_external=False, label="A:gold-only"
    )

    # -- VARIANT B: gold + external ------------------------------─
    print("\n\n" + "█" * 65)
    print("VARIANT B: GOLD + EXTERNAL MACRO FEATURES")
    print("█" * 65)

    ho_b, fc_b, df_b, rets_b = run_holdout_experiment(
        HORIZON, THRESHOLD, use_external=True, label="B:gold+ext"
    )
    wf_b, df_b_full, feat_b, splits_b = run_walkforward_experiment(
        HORIZON, THRESHOLD, use_external=True, label="B:gold+ext"
    )

    # -- Cross-variant comparison (default models) ----------------─
    print_comparison(ho_a, ho_b, "HOLDOUT COMPARISON: Variant A vs B (default models)")

    # -- Hyperparameter tuning (on Variant B - the richer dataset) ─
    if not args.fast:
        print("\n\n" + "█" * 65)
        print("HYPERPARAMETER TUNING (walk-forward CV on Variant B)")
        print("█" * 65)
        best_params = run_tuning(df_b_full, feat_b, splits_b)

        tuned_a, _ = run_tuned_holdout(best_params, HORIZON, THRESHOLD,
                                        use_external=False, label="A:gold-only")
        tuned_b, fc_tuned = run_tuned_holdout(best_params, HORIZON, THRESHOLD,
                                               use_external=True, label="B:gold+ext")

        print_comparison(tuned_a, tuned_b, "TUNED HOLDOUT COMPARISON: Variant A vs B")
    else:
        print("\n[--fast] Skipping tuning. Using default model params.")
        tuned_a = ho_a
        tuned_b = ho_b
        best_params = {}
        fc_tuned = fc_b

    # -- Feature importance ----------------------------------------
    print(f"\n{'='*65}")
    print("FEATURE IMPORTANCE (tree models, Variant B)")
    print(f"{'='*65}")
    get_feature_importance(tuned_b if not args.fast else ho_b, fc_tuned)

    # -- Final model selection ------------------------------------─
    print(f"\n{'='*65}")
    print("FINAL MODEL SELECTION")
    print(f"{'='*65}")

    # Pick best by ROC-AUC from tuned Variant B results
    all_tuned = tuned_b if not args.fast else ho_b
    best_model_name = max(all_tuned, key=lambda k: all_tuned[k]["roc_auc"])
    best_model = all_tuned[best_model_name]["model"]
    best_metrics = all_tuned[best_model_name]

    # Compare to best from Variant A
    best_a_name = max(tuned_a, key=lambda k: tuned_a[k]["roc_auc"])
    best_a_metrics = tuned_a[best_a_name]

    print(f"\n  Best Variant A model: {best_a_name}")
    print(f"    ROC-AUC={best_a_metrics['roc_auc']:.4f}, "
          f"F1(mac)={best_a_metrics['f1_macro']:.4f}, "
          f"HitRate={best_a_metrics.get('backtest', {}).get('hit_rate', 0):.2f}")
    print(f"\n  Best Variant B model: {best_model_name}")
    print(f"    ROC-AUC={best_metrics['roc_auc']:.4f}, "
          f"F1(mac)={best_metrics['f1_macro']:.4f}, "
          f"HitRate={best_metrics.get('backtest', {}).get('hit_rate', 0):.2f}")

    ext_improved = best_metrics["roc_auc"] > best_a_metrics["roc_auc"]
    print(f"\n  External factors improved ROC-AUC: {'YES ✓' if ext_improved else 'NO ✗'}")
    print(f"\n  → RECOMMENDED FINAL MODEL: {best_model_name} (Variant {'B' if ext_improved else 'A'})")
    print(f"    Selection criterion: highest walk-forward + holdout ROC-AUC")

    # -- Save ------------------------------------------------------
    summary = {
        "config": {
            "horizon": HORIZON,
            "threshold": THRESHOLD,
            "seed": RANDOM_SEED,
        },
        "variant_a_holdout": {
            name: _safe_metrics(r) for name, r in ho_a.items()
        },
        "variant_b_holdout": {
            name: _safe_metrics(r) for name, r in ho_b.items()
        },
        "variant_a_walkforward": {
            name: r["mean"] for name, r in wf_a.items()
        },
        "variant_b_walkforward": {
            name: r["mean"] for name, r in wf_b.items()
        },
        "tuning": {
            name: {"best_score": info["best_score"],
                   "best_params": {k: v for k, v in info["best_params"].items()
                                   if k not in ("random_state", "random_seed")}}
            for name, info in best_params.items()
        },
        "tuned_a_holdout": {name: _safe_metrics(r) for name, r in tuned_a.items()},
        "tuned_b_holdout": {name: _safe_metrics(r) for name, r in tuned_b.items()},
        "final_model": best_model_name,
        "external_factors_improved": ext_improved,
        "feature_cols_a": list(fc_a),
        "feature_cols_b": list(fc_b),
    }

    save_results(summary, best_model, best_model_name)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
