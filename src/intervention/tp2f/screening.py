from __future__ import annotations

import numpy as np
import pandas as pd


def receiver_stratum(sender_seed: int, receiver_seed: int) -> str:
    return "SUPPORTING_SAME_SEED_ID" if int(sender_seed) == int(receiver_seed) else "PRIMARY_CROSS_SEED"


def add_receiver_stratum(df: pd.DataFrame, sender_col="sender_seed", receiver_col="receiver_seed") -> pd.DataFrame:
    out = df.copy()
    out["receiver_stratum"] = [
        receiver_stratum(s, r) for s, r in zip(out[sender_col].astype(int), out[receiver_col].astype(int))
    ]
    return out


def equal_country_median_directed(
    df: pd.DataFrame,
    metric: str,
    *,
    country_col: str = "country",
    stratum_col: str | None = None,
    stratum: str | None = None,
):
    d = df.copy()
    if stratum_col is not None and stratum is not None:
        d = d[d[stratum_col] == stratum]
    country_values = d.groupby(country_col)[metric].median()
    return float(country_values.median()), country_values


def forecasting_country_medians(df: pd.DataFrame, metric: str, country_col: str = "country"):
    return df.groupby(country_col)[metric].median()

