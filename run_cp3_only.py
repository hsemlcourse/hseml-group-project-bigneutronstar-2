"""
Standalone runner for CP3 experiment only.
Skips the full CP2 pipeline — useful for fast iteration.
"""
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import run_cp3_experiment, HORIZON, THRESHOLD, RANDOM_SEED

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print("=" * 65)
    print("RUNNING CP3 EXPERIMENT (standalone)")
    print(f"Horizon={HORIZON}h | Threshold={THRESHOLD} | Seed={RANDOM_SEED}")
    print("=" * 65)
    run_cp3_experiment(horizon=HORIZON, threshold=THRESHOLD)
