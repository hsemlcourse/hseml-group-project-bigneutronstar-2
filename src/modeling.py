"""
Modeling module for gold futures price direction prediction.

Trains baseline (Logistic Regression), Random Forest, and Gradient Boosting
classifiers. Evaluates with accuracy, F1, ROC-AUC. Supports walk-forward
cross-validation and hyperparameter tuning.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_SEED = 42


def get_models(tuned: bool = False):
    """Return dict of {name: model} for all models to evaluate."""
    if tuned:
        return get_tuned_models()

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        ),
    }


def get_tuned_models():
    """Return models with tuned hyperparameters (found via walk-forward CV)."""
    return {
        "LogisticRegression_tuned": LogisticRegression(
            max_iter=1000, C=0.1, random_state=RANDOM_SEED, solver="lbfgs"
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
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
    }


def _evaluate(y_true, y_pred, y_proba):
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4),
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
        y_proba = model.predict_proba(X_te)[:, 1]

        metrics = _evaluate(y_test, y_pred, y_proba)

        if verbose:
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"F1-score:  {metrics['f1']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
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

    scaler = StandardScaler()

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
        y_proba = model.predict_proba(X_te)[:, 1]

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
        "f1": np.mean([m["f1"] for m in fold_metrics]),
        "roc_auc": np.mean([m["roc_auc"] for m in fold_metrics]),
    }

    if verbose:
        print(f"  Mean:  ROC-AUC={mean_metrics['roc_auc']:.4f}, "
              f"Acc={mean_metrics['accuracy']:.4f}, F1={mean_metrics['f1']:.4f}")

    return fold_metrics, mean_metrics


def _clone_model(model):
    """Create a fresh copy of a model with the same hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def tune_hyperparameters(df, feature_cols, splits, verbose=True):
    """
    Simple grid search over key hyperparameters using walk-forward CV.
    Returns best params dict per model.
    """
    best_results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (walk-forward CV)")
        print("=" * 60)

    lr_configs = [
        {"C": c, "solver": "lbfgs", "max_iter": 1000, "random_state": RANDOM_SEED}
        for c in [0.01, 0.1, 1.0, 10.0]
    ]
    best_lr = _grid_search_model(LogisticRegression, lr_configs, df, feature_cols,
                                  splits, scale=True, verbose=verbose, name="LogisticRegression")
    best_results["LogisticRegression"] = best_lr

    rf_configs = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m,
         "random_state": RANDOM_SEED, "n_jobs": -1}
        for n in [200, 500]
        for d in [4, 6, 8]
        for m in [20, 30, 50]
    ]
    best_rf = _grid_search_model(RandomForestClassifier, rf_configs, df, feature_cols,
                                  splits, scale=False, verbose=verbose, name="RandomForest")
    best_results["RandomForest"] = best_rf

    gb_configs = [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
         "min_samples_leaf": 30, "subsample": 0.8, "random_state": RANDOM_SEED},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
         "min_samples_leaf": 20, "subsample": 0.8, "random_state": RANDOM_SEED},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03,
         "min_samples_leaf": 20, "subsample": 0.8, "random_state": RANDOM_SEED},
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
         "min_samples_leaf": 30, "subsample": 0.8, "random_state": RANDOM_SEED},
    ]
    best_gb = _grid_search_model(GradientBoostingClassifier, gb_configs, df, feature_cols,
                                  splits, scale=False, verbose=verbose, name="GradientBoosting")
    best_results["GradientBoosting"] = best_gb

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
            y_proba = model.predict_proba(X_te)[:, 1]
            scores.append(roc_auc_score(y_te, y_proba))

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
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
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
