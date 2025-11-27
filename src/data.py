import os
import pandas as pd

def load_datasets(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    store = pd.read_csv(os.path.join(data_dir, "store.csv"))
    for df in (train, test):
        df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
        df["StateHoliday"] = df["StateHoliday"].astype(str)
    return train, test, store


def merge_store(train: pd.DataFrame, test: pd.DataFrame, store: pd.DataFrame):
    train = train.merge(store, on="Store", how="left")
    test = test.merge(store, on="Store", how="left")
    return train, test
