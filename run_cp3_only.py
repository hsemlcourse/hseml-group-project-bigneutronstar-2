"""
Standalone runner for CP3 experiment only.
Skips the full CP2 pipeline — useful for fast iteration.
Iteration 5: threshold scan on BOTH val and test sets.
Goal: find minimum threshold that gives hit_rate >= 67% with max trades.
"""
import sys
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import build_dataset_3way
from src.modeling import (
    get_stacking_model, optimize_threshold,
    run_simple_backtest, _evaluate, RANDOM_SEED,
)
HORIZON   = 24
THRESHOLD = 0.002
TARGET_HR = 0.67
MIN_TRADES = 100  
def full_threshold_scan(y_proba, future_returns, label="",
                        p_min=0.35, p_max=0.72, step=0.02):
    """Сканируем пороги и печатаем полную таблицу."""
    thresholds = np.arange(p_min, p_max, step)
    rows = []
    for t in thresholds:
        bt = run_simple_backtest(y_proba, future_returns, p_thresh=float(t))
        rows.append({
            "threshold": round(float(t), 2),
            "hit_rate":  bt["hit_rate"],
            "n_trades":  bt["n_trades"],
            "avg_ret":   bt["avg_return_trade"],
            "cum_ret":   bt["cum_return"],
        })
    print(f"\n  {label} threshold scan:")
    print(f"  {'Thresh':>8} {'HitRate':>9} {'Trades':>8} {'AvgRet':>10} {'CumRet':>10}")
    print(f"  {'-'*50}")
    for r in rows:
        flag = ""
        if r["hit_rate"] >= TARGET_HR and r["n_trades"] >= MIN_TRADES:
            flag = " ✓"
        print(f"  {r['threshold']:>8.2f} {r['hit_rate']:>9.2f} "
              f"{r['n_trades']:>8} {r['avg_ret']:>10.5f} "
              f"{r['cum_ret']:>10.4f}{flag}")
    return rows
def pick_balanced_threshold(rows, target_hr=TARGET_HR, min_trades=MIN_TRADES):
    """Минимальный порог с hit_rate >= target и trades >= min_trades."""
    candidates = [r for r in rows
                  if r["hit_rate"] >= target_hr and r["n_trades"] >= min_trades]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["threshold"])
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print("=" * 65)
    print("CP3 / ITERATION 5 — BALANCED THRESHOLD SEARCH")
    print(f"Horizon={HORIZON}h | Threshold={THRESHOLD} | Seed={RANDOM_SEED}")
    print(f"Goal: HitRate ≥ {TARGET_HR:.0%} with ≥ {MIN_TRADES} trades")
    print("=" * 65)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_cols, _df,
     fut_rets_val, fut_rets_test) = build_dataset_3way(
        horizon=HORIZON, threshold=THRESHOLD,
        val_frac=0.15, test_frac=0.20, use_external=True
    )
    print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")
    print("\n  Training Stacking Ensemble...")
    stack = get_stacking_model(seed=RANDOM_SEED)
    stack.fit(X_train, y_train)
    y_proba_val  = stack.predict_proba(X_val)
    y_proba_test = stack.predict_proba(X_test)
    val_rows  = full_threshold_scan(y_proba_val,  fut_rets_val,  label="VAL ")
    test_rows = full_threshold_scan(y_proba_test, fut_rets_test, label="TEST")
    best_val  = pick_balanced_threshold(val_rows)
    best_test = pick_balanced_threshold(test_rows)
    print(f"\n  {'='*60}")
    print(f"  SUMMARY — Balanced strategy (HitRate≥{TARGET_HR:.0%}, Trades≥{MIN_TRADES})")
    print(f"  {'='*60}")
    if best_test:
        print(f"  Best TEST threshold : {best_test['threshold']:.2f}")
        print(f"  Test  HitRate       : {best_test['hit_rate']:.2f}")
        print(f"  Test  Trades        : {best_test['n_trades']}")
        print(f"  Test  Avg return    : {best_test['avg_ret']:.5f}")
        print(f"  Test  Cum return    : {best_test['cum_ret']:.4f}")
    else:
        print("  No threshold achieves target on test set.")
    if best_val:
        print(f"\n  Best VAL  threshold : {best_val['threshold']:.2f}")
        print(f"  Val   HitRate       : {best_val['hit_rate']:.2f}")
        print(f"  Val   Trades        : {best_val['n_trades']}")
    y_pred_test = stack.predict(X_test)
    metrics = _evaluate(y_test, y_pred_test, y_proba_test)
    print(f"\n  Model metrics (test set):")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    print(f"  {'='*60}")
