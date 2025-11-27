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

# Try preferred models, fall back to sklearn
ModelClass = None
model_name = ""
use_early_stopping = False
try:
    import lightgbm as lgb
    ModelClass = lgb.LGBMRegressor
    model_name = "LightGBM"
    use_early_stopping = True
except Exception:
    try:
        import xgboost as xgb
        ModelClass = xgb.XGBRegressor
        model_name = "XGBoost"
        use_early_stopping = True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
        ModelClass = HGBR
        model_name = "HistGradientBoosting"
        use_early_stopping = False

from joblib import dump

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
from src.metrics import rmspe
from src.validation import time_series_cv_indices
DATA_DIR = os.path.join(ROOT, "dataset")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_params():
    if model_name == "LightGBM":
        return dict(
            n_estimators=2000,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        return dict(
            n_estimators=2000,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        return dict(
            max_depth=None,
            learning_rate=0.05,
            max_bins=255,
            l2_regularization=0.0,
            random_state=42,
        )


def main():
    print(f"Training with: {model_name}")
    train, test, store = load_datasets(DATA_DIR)
    train, test = merge_store(train, test, store)

    # Build features jointly (for lags)
    train_fe, test_fe = build_features(train, test)

    # Train filter: Open==1 and Sales present
    train_fit = train_fe.copy()
    if "Open" in train_fit.columns:
        train_fit = train_fit[train_fit["Open"] == 1]
    train_fit = train_fit.dropna(subset=["Sales"]) 

    y = np.log1p(train_fit["Sales"].values)
    feat_cols = select_feature_columns(train_fit)
    X = train_fit[feat_cols].copy()

    params = get_model_params()

    # CV
    folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)
    scores = []
    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
        X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]

        if model_name == "LightGBM":
            model = ModelClass(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            pred = model.predict(X_va)
        elif model_name == "XGBoost":
            model = ModelClass(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                verbose=False,
                early_stopping_rounds=100,
            )
            pred = model.predict(X_va)
        else:
            model = ModelClass(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)

        score = rmspe(np.expm1(y_va), np.expm1(pred))
        scores.append(score)
        print(f"Fold {i} RMSPE: {score:.4f}")

    if scores:
        print(f"CV RMSPE mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

    # Fit final model with early stopping on last 6 weeks
    final_model = None
    if model_name in ("LightGBM", "XGBoost"):
        folds_once = time_series_cv_indices(train_fit, n_splits=1, val_weeks=6)
        if folds_once:
            tr_idx, va_idx = folds_once[0]
            X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
            X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]
            if model_name == "LightGBM":
                final_model = ModelClass(**params)
                final_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="rmse",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
            else:
                final_model = ModelClass(**params)
                final_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="rmse",
                    verbose=False,
                    early_stopping_rounds=100,
                )
        else:
            final_model = ModelClass(**params)
            final_model.fit(X, y)
    else:
        final_model = ModelClass(**params)
        final_model.fit(X, y)

    # Save artifacts
    dump(final_model, os.path.join(MODELS_DIR, "model.pkl"))
    with open(os.path.join(MODELS_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(feat_cols, f)
    print("Saved model to models/model.pkl and features to models/features.json")


if __name__ == "__main__":
    main()
