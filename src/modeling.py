# Обучение и оценка моделей для золота.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from catboost import CatBoostClassifier

RANDOM_SEED = 42


def get_models(tuned: bool = False):
    """Список моделей для оценки."""
    if tuned:
        return get_tuned_models()
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs",
            multi_class="multinomial"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def get_tuned_models():
    """Тюненые модели."""
    return {
        "LogisticRegression_tuned": LogisticRegression(
            max_iter=1000, C=0.1, random_state=RANDOM_SEED, solver="lbfgs",
            multi_class="multinomial"
        ),
        "RandomForest_tuned": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=30,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting_tuned": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
        "CatBoost_tuned": CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=5,
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1,
        ),
        "ExtraTrees_tuned": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=30,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def get_ensemble(best_results: dict):
    """Создает Soft Voting Ensemble из лучших моделей."""
    estimators = []
    if "LogisticRegression" in best_results:
        p = best_results["LogisticRegression"]["best_params"]
        estimators.append(("lr", LogisticRegression(**p)))
    if "ExtraTrees" in best_results:
        p = best_results["ExtraTrees"]["best_params"]
        estimators.append(("et", ExtraTreesClassifier(**p)))
    if "CatBoost" in best_results:
        p = best_results["CatBoost"]["best_params"]
        estimators.append(("cb", CatBoostClassifier(**p)))
    if len(estimators) < 2: return None
    return VotingClassifier(estimators=estimators, voting="soft")


def _evaluate(y_true, y_pred, y_proba):
    """Compute standard classification metrics for multiclass."""
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def run_simple_backtest(y_proba, future_returns, p_thresh=0.45):
    """
    Very simple vectorized backtest logic.
    Goes LONG if probability of UP (class 2) > p_thresh AND > probability of DOWN.
    Goes SHORT if probability of DOWN (class 0) > p_thresh AND > probability of UP.
    """
    p_down = y_proba[:, 0]
    p_up = y_proba[:, 2]
    positions = np.zeros(len(y_proba))
    long_mask = (p_up > p_thresh) & (p_up > p_down)
    short_mask = (p_down > p_thresh) & (p_down > p_up)
    positions[long_mask] = 1
    positions[short_mask] = -1
    returns = positions * future_returns.values
    n_trades = np.sum(positions != 0)
    hit_rate = np.mean(returns[positions != 0] > 0) if n_trades > 0 else 0.0
    avg_return_trade = np.mean(returns[positions != 0]) if n_trades > 0 else 0.0
    cum_return = np.sum(returns)
    return {
        "n_trades": int(n_trades),
        "hit_rate": hit_rate,
        "avg_return_trade": avg_return_trade,
        "cum_return": cum_return,
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names=None,
                       tuned: bool = False, verbose: bool = True):
    """
    Train all models, evaluate on test set.
    Returns dict of {name: {model, metrics, predictions}}.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models(tuned=tuned)
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")

        is_linear = "Logistic" in name
        X_tr = X_train_scaled if is_linear else X_train
        X_te = X_test_scaled if is_linear else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        metrics = _evaluate(y_test, y_pred, y_proba)

        if verbose:
            print(f"Accuracy:   {metrics['accuracy']:.4f}")
            print(f"F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
            print(f"\nConfusion matrix:\n{metrics['confusion_matrix']}")
            print(f"\nClassification report:\n{metrics['report']}")

        results[name] = {
            "model": model,
            "scaler": scaler if is_linear else None,
            **metrics,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    return results


def walk_forward_evaluate(df, feature_cols, splits, model_name="GradientBoosting",
                          tuned: bool = False, verbose: bool = True):
    """
    Evaluate a single model using walk-forward cross-validation splits.
    Returns list of per-fold metrics dicts and the mean metrics.
    """
    models_dict = get_models(tuned=tuned)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr = df.iloc[train_idx][feature_cols].values
        y_tr = df.iloc[train_idx]["target"].values
        X_te = df.iloc[test_idx][feature_cols].values
        y_te = df.iloc[test_idx]["target"].values

        is_linear = "Logistic" in model_name
        if is_linear:
            scaler_fold = StandardScaler()
            X_tr = scaler_fold.fit_transform(X_tr)
            X_te = scaler_fold.transform(X_te)

        model = _clone_model(models_dict[model_name])
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        metrics = _evaluate(y_te, y_pred, y_proba)
        metrics["fold"] = fold_i
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        fold_metrics.append(metrics)

        if verbose:
            print(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}, "
                  f"ROC-AUC={metrics['roc_auc']:.4f}, Acc={metrics['accuracy']:.4f}")

    mean_metrics = {
        "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
        "f1_macro": np.mean([m["f1_macro"] for m in fold_metrics]),
        "roc_auc": np.mean([m["roc_auc"] for m in fold_metrics]),
    }
    if verbose:
        print(f"  Mean:  ROC-AUC={mean_metrics['roc_auc']:.4f}, "
              f"Acc={mean_metrics['accuracy']:.4f}, F1(macro)={mean_metrics['f1_macro']:.4f}")
    return fold_metrics, mean_metrics


def _clone_model(model):
    """Create a fresh copy of a model with the same hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def tune_hyperparameters(df, feature_cols, splits, verbose=True):
    """
    Grid search over key hyperparameters using walk-forward CV.
    Returns best params dict per model.
    """
    best_results = {}
    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (walk-forward CV)")
        print("=" * 60)

    # -- Logistic Regression --------------------------------------
    lr_configs = [
        {"C": c, "solver": "lbfgs", "max_iter": 1000,
         "random_state": RANDOM_SEED, "multi_class": "multinomial"}
        for c in [0.01, 0.1, 0.5, 1.0, 5.0]
    ]
    best_results["LogisticRegression"] = _grid_search_model(
        LogisticRegression, lr_configs, df, feature_cols,
        splits, scale=True, verbose=verbose, name="LogisticRegression"
    )

    # -- Random Forest --------------------------------------------
    rf_configs = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m,
         "max_features": f, "random_state": RANDOM_SEED, "n_jobs": -1}
        for n in [200, 400]
        for d in [4, 6, 8]
        for m in [20, 40]
        for f in ["sqrt"]
    ]
    best_results["RandomForest"] = _grid_search_model(
        RandomForestClassifier, rf_configs, df, feature_cols,
        splits, scale=False, verbose=verbose, name="RandomForest"
    )

    # -- Gradient Boosting ----------------------------------------
    gb_configs = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr,
         "min_samples_leaf": m, "subsample": ss, "random_state": RANDOM_SEED}
        for n in [100, 200]
        for d in [3, 4]
        for lr in [0.05, 0.1]
        for m in [20, 30]
        for ss in [0.8]
    ]
    best_results["GradientBoosting"] = _grid_search_model(
        GradientBoostingClassifier, gb_configs, df, feature_cols,
        splits, scale=False, verbose=verbose, name="GradientBoosting"
    )

    # -- CatBoost (optimized grid) --------------------------------
    cb_configs = [
        {"iterations": itr, "depth": dep, "learning_rate": lr,
         "l2_leaf_reg": 3, "random_seed": RANDOM_SEED, "verbose": 0,
         "thread_count": -1}
        for itr in [300, 500]
        for dep in [4, 6]
        for lr in [0.03, 0.05, 0.1]
    ]
    best_results["CatBoost"] = _grid_search_model(
        CatBoostClassifier, cb_configs, df, feature_cols,
        splits, scale=False, verbose=verbose, name="CatBoost"
    )

    # -- ExtraTrees ----------------------------------------------
    et_configs = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m,
         "max_features": f, "random_state": RANDOM_SEED, "n_jobs": -1}
        for n in [200, 400]
        for d in [4, 6, 8]
        for m in [20, 40]
        for f in ["sqrt"]
    ]
    best_results["ExtraTrees"] = _grid_search_model(
        ExtraTreesClassifier, et_configs, df, feature_cols,
        splits, scale=False, verbose=verbose, name="ExtraTrees"
    )

    return best_results


def _grid_search_model(model_class, configs, df, feature_cols, splits,
                        scale=False, verbose=True, name=""):
    """Run walk-forward CV for each config, return best config and score."""
    best_score = -1
    best_config = None

    for config in configs:
        scores = []
        for train_idx, test_idx in splits:
            X_tr = df.iloc[train_idx][feature_cols].values
            y_tr = df.iloc[train_idx]["target"].values
            X_te = df.iloc[test_idx][feature_cols].values
            y_te = df.iloc[test_idx]["target"].values

            if scale:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            model = model_class(**config)
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_te)
            try:
                scores.append(roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro"))
            except ValueError:
                scores.append(0.5)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_config = config

    if verbose:
        print(f"\n  {name}: best ROC-AUC={best_score:.4f}, params={best_config}")
    return {"best_score": best_score, "best_params": best_config}


def print_summary(results: dict):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1 (mac)':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<30} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['roc_auc']:>10.4f}")
    print()


def get_feature_importance(results: dict, feature_names: list) -> dict:
    """
    Extract feature importances from tree-based models.
    Returns dict of {model_name: sorted list of (feature, importance)}.
    """
    importances = {}
    for name, r in results.items():
        model = r["model"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
            importances[name] = pairs
            print(f"\nTop-10 features for {name}:")
            for feat, val in pairs[:10]:
                print(f"  {feat:<30} {val:.4f}")
    return importances


# ---------------------------------------------------------------------------
# CP3: Stacking Ensemble
# ---------------------------------------------------------------------------

def get_stacking_model(seed: int = RANDOM_SEED) -> StackingClassifier:
    """
    Stacking ensemble:
      Base:  RF, GradientBoosting, CatBoost, ExtraTrees
      Meta:  LogisticRegression (trained on out-of-fold predictions)
    n_jobs=1 on base models to avoid MacOS multiprocessing fork issues
    inside the stacking CV loop.
    """
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=30,
                max_features="sqrt", random_state=seed, n_jobs=1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=30, subsample=0.8, random_state=seed,
            ),
        ),
        (
            "cb",
            CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.03,
                l2_leaf_reg=5, random_seed=seed, verbose=0, thread_count=1,
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=30,
                max_features="sqrt", random_state=seed, n_jobs=1,
            ),
        ),
    ]
    meta = LogisticRegression(
        C=0.5, max_iter=1000, random_state=seed,
        solver="lbfgs", multi_class="multinomial",
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=3,
        passthrough=False,
        n_jobs=1,  # avoid fork issues on MacOS
    )


# ---------------------------------------------------------------------------
# CP3: Threshold optimisation (on val set, NOT test)
# ---------------------------------------------------------------------------

def optimize_threshold(
    y_proba: np.ndarray,
    future_returns,
    min_trades: int = 25,
    p_min: float = 0.35,
    p_max: float = 0.72,
    step: float = 0.02,
):
    """
    Grid-search the confidence threshold that maximises hit_rate on a
    held-out validation set.  Only thresholds that produce at least
    `min_trades` trades are considered.

    Returns (best_threshold, best_hit_rate, all_results_list).
    """
    thresholds = np.arange(p_min, p_max, step)
    best_thresh = 0.45
    best_hr = 0.0
    results = []
    for t in thresholds:
        t_f = float(t)
        bt = run_simple_backtest(y_proba, future_returns, p_thresh=t_f)
        results.append({
            "threshold": round(t_f, 3),
            "hit_rate": bt["hit_rate"],
            "n_trades": bt["n_trades"],
            "avg_return": bt["avg_return_trade"],
        })
        if bt["n_trades"] >= min_trades and bt["hit_rate"] > best_hr:
            best_hr = bt["hit_rate"]
            best_thresh = t_f
    return best_thresh, best_hr, results
