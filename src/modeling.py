"""
Modeling module for gold futures price direction prediction.

Trains baseline (Logistic Regression), Random Forest, and Gradient Boosting
classifiers. Evaluates all models with accuracy, F1, ROC-AUC, confusion matrix,
and classification report.
"""
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


def get_models():
    """Return dict of {name: model} for all models to evaluate."""
    models = {
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
    }

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    )

    return models


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train all models, evaluate on test set.
    Returns dict of {name: {model, metrics, predictions}}.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        if name == "LogisticRegression":
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)

        print(f"Accuracy:  {acc:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc:.4f}")
        print(f"\nConfusion matrix:\n{cm}")
        print(f"\nClassification report:\n{report}")

        results[name] = {
            "model": model,
            "scaler": scaler if name == "LogisticRegression" else None,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "confusion_matrix": cm,
            "report": report,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return results


def print_summary(results: dict):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
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
