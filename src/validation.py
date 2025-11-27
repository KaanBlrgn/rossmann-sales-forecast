import pandas as pd
from typing import List, Tuple


def time_series_cv_indices(df: pd.DataFrame, n_splits: int = 3, val_weeks: int = 6) -> List[Tuple[list, list]]:
    df_sorted = df.sort_values("Date").copy()
    max_date = df_sorted["Date"].max()
    fold_indices = []
    end = max_date
    for _ in range(n_splits):
        val_end = end
        val_start = val_end - pd.Timedelta(weeks=val_weeks)
        train_mask = df["Date"] < val_start
        val_mask = (df["Date"] >= val_start) & (df["Date"] <= val_end)
        tr_idx = df.index[train_mask].tolist()
        va_idx = df.index[val_mask].tolist()
        if len(tr_idx) == 0 or len(va_idx) == 0:
            break
        fold_indices.append((tr_idx, va_idx))
        end = val_start - pd.Timedelta(days=1)
    return fold_indices[::-1]
