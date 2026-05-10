"""
Iteration 6: ROC-AUC improvement via:
  1. Feature selection (top-50 by aggregated tree importance)
  2. Class balancing (class_weight='balanced' + CatBoost class_weights)
  3. LightGBM added as 5th base estimator in Stacking
Compares v1 (baseline Stacking) vs v2 (improved) on all key metrics:
  ROC-AUC, Accuracy, F1-macro, Hit Rate, Trades, Avg Return, Cum Return.
"""
import sys
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import build_dataset_3way
from src.modeling import (
    get_stacking_model, get_stacking_model_v2, get_top_feature_indices,
    run_simple_backtest, optimize_threshold, _evaluate, RANDOM_SEED,
)
HORIZON    = 24
THRESHOLD  = 0.002
TOP_K      = 50       
MIN_TRADES = 100      
def evaluate_full(model, X_test, y_test, fut_rets_test,
                  y_proba_val, fut_rets_val, label=""):
    """Train/predict уже выполнены. Считаем все метрики + оптимизируем threshold."""
    y_proba_test = model.predict_proba(X_test)
    y_pred_test  = model.predict(X_test)
    m = _evaluate(y_test, y_pred_test, y_proba_test)
    best_t, best_val_hr, _ = optimize_threshold(
        y_proba_val, fut_rets_val, min_trades=MIN_TRADES
    )
    bt_def = run_simple_backtest(y_proba_test, fut_rets_test, p_thresh=0.45)
    bt_opt = run_simple_backtest(y_proba_test, fut_rets_test, p_thresh=best_t)
    print(f"\n  ── {label} ──")
    print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1 macro : {m['f1_macro']:.4f}")
    print(f"  F1 weighted: {m['f1_weighted']:.4f}")
    print(f"\n  Backtest default (thresh=0.45):")
    print(f"    HitRate={bt_def['hit_rate']:.2f}  "
          f"Trades={bt_def['n_trades']}  "
          f"AvgRet={bt_def['avg_return_trade']:.5f}  "
          f"CumRet={bt_def['cum_return']:.4f}")
    print(f"\n  Backtest optimised (thresh={best_t:.2f}, val-tuned, ≥{MIN_TRADES} trades):")
    print(f"    HitRate={bt_opt['hit_rate']:.2f}  "
          f"Trades={bt_opt['n_trades']}  "
          f"AvgRet={bt_opt['avg_return_trade']:.5f}  "
          f"CumRet={bt_opt['cum_return']:.4f}")
    print(f"  Val HR @ {best_t:.2f} = {best_val_hr:.2f}")
    return {
        "roc_auc": m["roc_auc"],
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
        "thresh": best_t,
        "val_hr": best_val_hr,
        "test_hr": bt_opt["hit_rate"],
        "trades": bt_opt["n_trades"],
        "avg_ret": bt_opt["avg_return_trade"],
        "cum_ret": bt_opt["cum_return"],
        "y_proba_test": y_proba_test,
    }
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print("=" * 65)
    print("ITERATION 6 — ROC-AUC IMPROVEMENT")
    print("Feature Selection + Class Balancing + LightGBM")
    print(f"Horizon={HORIZON}h | Threshold={THRESHOLD} | Seed={RANDOM_SEED}")
    print("=" * 65)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_cols, _df,
     fut_rets_val, fut_rets_test) = build_dataset_3way(
        horizon=HORIZON, threshold=THRESHOLD,
        val_frac=0.15, test_frac=0.20, use_external=True
    )
    feature_names = list(feature_cols)
    print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Features (total): {len(feature_names)}")
    classes, counts = np.unique(y_train, return_counts=True)
    dist = ", ".join([f"class {c}: {cnt/len(y_train):.1%}" for c, cnt in zip(classes, counts)])
    print(f"  Target balance (train): {dist}")
    print("\n" + "─" * 65)
    print("  STEP 1/3: Baseline Stacking v1 (all 96 features, no balancing)")
    print("─" * 65)
    v1 = get_stacking_model(seed=RANDOM_SEED)
    v1.fit(X_train, y_train)
    y_proba_val_v1 = v1.predict_proba(X_val)
    res_v1 = evaluate_full(v1, X_test, y_test, fut_rets_test,
                           y_proba_val_v1, fut_rets_val, label="Stacking v1 (baseline)")
    print("\n" + "─" * 65)
    print("  STEP 2/3: Feature selection (top-50 by aggregated tree importance)")
    print("─" * 65)
    top_idx, top_names = get_top_feature_indices(v1, feature_names, top_k=TOP_K)
    print(f"  Selected {len(top_idx)} features. Top-10:")
    for i, fn in enumerate(top_names[:10]):
        print(f"    {i+1:>2}. {fn}")
    print(f"    ... ({len(top_names) - 10} more)")
    X_train_sel = X_train[:, top_idx]
    X_val_sel   = X_val[:, top_idx]
    X_test_sel  = X_test[:, top_idx]
    print("\n" + "─" * 65)
    print("  STEP 3/3: Stacking v2 (class balancing + LightGBM + top-50 features)")
    print("─" * 65)
    v2 = get_stacking_model_v2(seed=RANDOM_SEED, use_lgbm=True)
    v2.fit(X_train_sel, y_train)
    y_proba_val_v2 = v2.predict_proba(X_val_sel)
    res_v2 = evaluate_full(v2, X_test_sel, y_test, fut_rets_test,
                           y_proba_val_v2, fut_rets_val, label="Stacking v2 (improved)")
    print("\n" + "=" * 65)
    print("  ИТОГОВОЕ СРАВНЕНИЕ v1 vs v2")
    print("=" * 65)
    hdr = f"  {'Метрика':<25} {'v1 baseline':>14} {'v2 improved':>14} {'Delta':>10}"
    print(hdr)
    print("  " + "-" * 63)
    def row(name, k, fmt=".4f"):
        v = res_v1[k];  w = res_v2[k]
        delta = w - v
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<25} {v:>14{fmt}} {w:>14{fmt}} {sign}{delta:>9{fmt}}")
    row("ROC-AUC",    "roc_auc")
    row("Accuracy",   "accuracy")
    row("F1 macro",   "f1_macro")
    row("F1 weighted","f1_weighted")
    print("  " + "-" * 63)
    print(f"  {'Threshold (opt)':}  {res_v1['thresh']:.2f}  →  {res_v2['thresh']:.2f}")
    print(f"  {'Val HitRate':}      {res_v1['val_hr']:.2f}  →  {res_v2['val_hr']:.2f}")
    print(f"  {'Test HitRate':}     {res_v1['test_hr']:.2f}  →  {res_v2['test_hr']:.2f}")
    print(f"  {'Trades':}           {res_v1['trades']}  →  {res_v2['trades']}")
    print(f"  {'Avg Return/trade':} {res_v1['avg_ret']:.5f}  →  {res_v2['avg_ret']:.5f}")
    print(f"  {'Cum Return':}       {res_v1['cum_ret']:.4f}  →  {res_v2['cum_ret']:.4f}")
    print("=" * 65)
    improved = res_v2["roc_auc"] > res_v1["roc_auc"]
    print(f"\n  ROC-AUC improved: {'YES ✓' if improved else 'NO ✗'}")
    print(f"  Delta ROC-AUC: {res_v2['roc_auc'] - res_v1['roc_auc']:+.4f}")
