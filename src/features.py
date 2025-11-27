import numpy as np
import pandas as pd
from datetime import datetime

MONTH_ABBR = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sept",10:"Oct",11:"Nov",12:"Dec"}

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]) 
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    return df


def _normalize_promo_interval(x):
    if pd.isna(x) or x == "":
        return []
    return [m.strip() for m in str(x).split(',')]


def add_store_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PromoInterval active flag
    month_abbr = df["month"].map(MONTH_ABBR)
    promo_lists = df["PromoInterval"].apply(_normalize_promo_interval)
    df["promo_interval_active"] = [int(ma in pls) for ma, pls in zip(month_abbr, promo_lists)]

    # Promo2 active based on since week/year
    since_year = df["Promo2SinceYear"].fillna(0).astype(int)
    since_week = df["Promo2SinceWeek"].fillna(1).astype(int).clip(1, 53)

    def iso_to_date(y, w):
        try:
            return datetime.fromisocalendar(int(y), int(w), 1)
        except Exception:
            return pd.Timestamp("2100-01-01")

    since_dates = [iso_to_date(y, w) if y > 0 else pd.Timestamp("2100-01-01") for y, w in zip(since_year, since_week)]
    df["promo2_active_on_date"] = (df["Date"].values >= np.array(since_dates, dtype="datetime64[ns]")) & (df["Promo2"].fillna(0).astype(int) == 1)
    df["promo2_active_on_date"] = df["promo2_active_on_date"].astype(int)

    # Competition open months
    comp_month = df["CompetitionOpenSinceMonth"].fillna(1).astype(int).clip(1, 12)
    comp_year = df["CompetitionOpenSinceYear"].fillna(df["year"]).astype(int)
    comp_since = pd.to_datetime(comp_year.astype(str) + "-" + comp_month.astype(str) + "-01", errors="coerce")
    months = (df["Date"].dt.to_period("M").astype(int) - comp_since.dt.to_period("M").astype(int))
    df["competition_open_months"] = months.fillna(0).clip(lower=0).astype(int)

    # Distance log transform
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["log_competition_distance"] = np.log1p(df["CompetitionDistance"].clip(lower=0))

    # Drop raw string column after deriving flag
    if "PromoInterval" in df.columns:
        df = df.drop(columns=["PromoInterval"])  

    return df


def add_event_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for days before/after events (promo, holidays)"""
    df = df.copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    # Promo event proximity (3 days before/after promo starts/ends)
    for shift_days in [1, 2, 3]:
        df[f"promo_starts_in_{shift_days}d"] = (df.groupby("Store")["Promo"].shift(-shift_days).fillna(0) > df["Promo"]).astype(int)
        df[f"promo_ended_{shift_days}d_ago"] = (df.groupby("Store")["Promo"].shift(shift_days).fillna(0) > df["Promo"]).astype(int)
    
    # StateHoliday proximity
    df["holiday_tomorrow"] = (df.groupby("Store")["StateHoliday"].shift(-1).fillna("0") != "0").astype(int)
    df["holiday_yesterday"] = (df.groupby("Store")["StateHoliday"].shift(1).fillna("0") != "0").astype(int)
    
    return df


def add_store_status_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add store status flags (closed tomorrow, was closed yesterday, etc.)"""
    df = df.copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    if "Open" in df.columns:
        df["closes_tomorrow"] = (df.groupby("Store")["Open"].shift(-1).fillna(1) == 0).astype(int)
        df["was_closed_yesterday"] = (df.groupby("Store")["Open"].shift(1).fillna(1) == 0).astype(int)
        # Was closed on last Sunday (DayOfWeek=7)
        last_sunday_open = df[df["DayOfWeek"] == 7].groupby("Store")["Open"].shift(1)
        df["was_closed_last_sunday"] = 0
        df.loc[last_sunday_open.index, "was_closed_last_sunday"] = (last_sunday_open == 0).astype(int)
    
    return df


def add_store_dow_historical_avg(df: pd.DataFrame, target: str = "Sales") -> pd.DataFrame:
    """Add per-store, per-DOW historical average (leakage-safe expanding mean)"""
    df = df.copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    # Expanding mean per Store+DOW (only past data)
    df["store_dow_mean"] = df.groupby(["Store", "DayOfWeek"])[target].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["store_dow_std"] = df.groupby(["Store", "DayOfWeek"])[target].transform(
        lambda x: x.shift(1).expanding().std()
    )
    
    return df


def add_categorical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dh = pd.get_dummies(df["StateHoliday"].astype(str), prefix="StateHoliday")
    st = pd.get_dummies(df["StoreType"].astype(str), prefix="StoreType")
    ass = pd.get_dummies(df["Assortment"].astype(str), prefix="Assortment")
    df = pd.concat([df.drop(columns=["StateHoliday", "StoreType", "Assortment"]), dh, st, ass], axis=1)
    return df


def add_lag_rolling_features(df: pd.DataFrame, group_key: str = "Store", target: str = "Sales") -> pd.DataFrame:
    df = df.sort_values([group_key, "Date"]).copy()
    for lag in [7, 14, 28]:
        df[f"{target}_lag_{lag}"] = df.groupby(group_key)[target].shift(lag)
    for w in [7, 14, 28]:
        df[f"{target}_rollmean_{w}"] = df.groupby(group_key)[target].shift(1).rolling(window=w).mean().reset_index(level=0, drop=True)
        df[f"{target}_rollstd_{w}"] = df.groupby(group_key)[target].shift(1).rolling(window=w).std().reset_index(level=0, drop=True)
    df["sales_momentum_7"] = df[f"{target}_rollmean_{7}"] / (df[f"{target}_rollmean_{14}"] + 1e-6)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday proximity features"""
    df = df.copy()
    
    # Easter dates (Germany) 2013-2015
    easter_dates = {
        2013: pd.Timestamp('2013-03-31'),
        2014: pd.Timestamp('2014-04-20'),
        2015: pd.Timestamp('2015-04-05')
    }
    
    # Christmas and New Year
    df['is_christmas_week'] = ((df['month'] == 12) & (df['day'] >= 20)).astype(int)
    df['is_newyear_week'] = (
        ((df['month'] == 12) & (df['day'] >= 27)) |
        ((df['month'] == 1) & (df['day'] <= 3))
    ).astype(int)
    
    # Easter proximity (±7 days)
    df['days_to_easter'] = 999  # Default: far from Easter
    df['is_easter_week'] = 0
    
    for year, easter_date in easter_dates.items():
        mask = df['year'] == year
        if mask.any():
            days_diff = (df.loc[mask, 'Date'] - easter_date).dt.days
            df.loc[mask, 'days_to_easter'] = days_diff.abs()
            df.loc[mask, 'is_easter_week'] = (days_diff.abs() <= 7).astype(int)
    
    # School holiday proximity (already exists as SchoolHoliday)
    # Add before/after flags
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    df['school_holiday_tomorrow'] = df.groupby('Store')['SchoolHoliday'].shift(-1).fillna(0).astype(int)
    df['school_holiday_yesterday'] = df.groupby('Store')['SchoolHoliday'].shift(1).fillna(0).astype(int)
    
    return df


def add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add store cluster labels from clustering analysis as features"""
    import os
    df = df.copy()
    
    # Try to load clustering results
    try:
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cluster_path = os.path.join(ROOT, 'outputs', 'reports', 'clustering_labels.csv')
        
        if os.path.exists(cluster_path):
            clusters = pd.read_csv(cluster_path)[['Store', 'cluster']]
            df = df.merge(clusters, on='Store', how='left')
            
            # Fill missing clusters with -1 (unknown)
            df['cluster'] = df['cluster'].fillna(-1).astype(int)
            
            # One-hot encode clusters
            for i in range(5):  # 5 clusters (0-4)
                df[f'cluster_{i}'] = (df['cluster'] == i).astype(int)
            
            # Drop original cluster column after encoding
            df = df.drop(columns=['cluster'])
            
            print(f"✓ Cluster features eklendi: cluster_0 to cluster_4")
        else:
            print(f"⚠️  Clustering sonuçları bulunamadı: {cluster_path}")
    except Exception as e:
        print(f"⚠️  Cluster feature eklenemedi: {e}")
    
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy(); te = test.copy()
    tr["is_train"] = 1; te["is_train"] = 0
    union = pd.concat([tr, te], axis=0, ignore_index=True, sort=False)

    union = add_calendar_features(union)
    union = add_store_features(union)
    union = add_categorical_encodings(union)
    union = add_holiday_features(union)  # ← YENİ: Holiday features ekle
    union = add_cluster_features(union)  # ← Cluster features ekle

    if "Sales" not in union.columns:
        union["Sales"] = np.nan
    union = union.sort_values(["Store", "Date"]).reset_index(drop=True)
    union = add_lag_rolling_features(union, group_key="Store", target="Sales")

    train_fe = union[union["is_train"] == 1].copy()
    test_fe = union[union["is_train"] == 0].copy()
    train_fe.drop(columns=["is_train"], inplace=True)
    test_fe.drop(columns=["is_train"], inplace=True)
    return train_fe, test_fe


def select_feature_columns(df: pd.DataFrame) -> list:
    drop_cols = ["Sales", "Customers", "Date"]
    feats = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feats = feats.select_dtypes(include=[np.number])
    return list(feats.columns)
