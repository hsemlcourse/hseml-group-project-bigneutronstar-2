
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_dataset_3way
from src.modeling import get_stacking_model, RANDOM_SEED

HORIZON = 24
THRESHOLD = 0.002
MODEL_PATH = PROJECT_ROOT / "models" / "cp3_stacking_model.pkl"

def save_model():
    print("Loading data and building dataset...")
    # Get 3-way split just to use the same feature generation
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_cols, _df,
     _, _) = build_dataset_3way(
        horizon=HORIZON, threshold=THRESHOLD,
        val_frac=0.15, test_frac=0.20, use_external=True
    )
    
    import numpy as np
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    
    print(f"Training Stacking Ensemble on {len(X_all)} samples...")
    model = get_stacking_model(seed=RANDOM_SEED)
    model.fit(X_all, y_all)
    
    artifact = {
        "model": model,
        "feature_cols": list(feature_cols),
        "horizon": HORIZON,
        "threshold": THRESHOLD,
        "optimal_p_thresh": 0.59 # Found in Iteration 5
    }
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    save_model()
