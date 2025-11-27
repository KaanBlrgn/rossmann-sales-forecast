import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from joblib import load

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
DATA_DIR = os.path.join(ROOT, "dataset")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUT_PATH = os.path.join(ROOT, "submission.csv")


def main():
    # Load artifacts
    model = load(os.path.join(MODELS_DIR, "model.pkl"))
    with open(os.path.join(MODELS_DIR, "features.json"), "r", encoding="utf-8") as f:
        feat_cols = json.load(f)

    # Load data
    train, test, store = load_datasets(DATA_DIR)
    train, test = merge_store(train, test, store)

    # Build features in a consistent way
    train_fe, test_fe = build_features(train, test)

    # Prepare X in the same column order, filling missing columns with 0 if needed
    X_test = test_fe.reindex(columns=feat_cols, fill_value=0)

    # Predict in log space if model trained that way (as in our train script) and back-transform
    y_hat_log = model.predict(X_test)
    y_hat = np.expm1(y_hat_log)

    # Apply Open==0 -> Sales=0 rule
    if "Open" in test_fe.columns:
        open_zero = (test_fe["Open"] == 0).values
        y_hat[open_zero] = 0.0

    # Clip negatives
    y_hat = np.clip(y_hat, 0, None)

    # Save submission in original order of test.csv
    submission = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), usecols=["Id"])  # keep order
    submission["Sales"] = y_hat
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved submission to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
