"""Per-stock features, each known at the close of day t, and the day t+1 target.

Every feature uses data up to and including day t only. The target is the
next day's close-to-close return, so a forecast of r for day t+1 maps to a
price of ``close_t * (1 + r)``. ``tests/test_features.py`` checks that
changing future prices never changes past features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "target"


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss  # inf when there were no losses, which gives an RSI of 100
    return (100 - 100 / (1 + rs)).where(~((gain == 0) & (loss == 0)), 50.0)


def rating_change(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """1 on days an analyst changed their rating for the stock."""
    t = ticker.lower()
    to_col, from_col, action_col = f"ToGrade_{t}", f"FromGrade_{t}", f"Action_{t}"
    if action_col not in prices.columns:
        return pd.Series(0, index=prices.index)
    changed = prices[action_col].notna() & (prices[to_col].fillna("") != prices[from_col].fillna(""))
    return changed.astype(int)


def build_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Feature frame for one stock. The last row has features but no target yet."""
    close, open_ = prices[f"Close_{ticker}"], prices[f"Open_{ticker}"]
    high, low, volume = prices[f"High_{ticker}"], prices[f"Low_{ticker}"], prices[f"Volume_{ticker}"]
    ret = close.pct_change()

    f = pd.DataFrame(index=prices.index)
    f["ret_1d"] = ret
    for lag in range(1, 5):
        f[f"ret_lag{lag}"] = ret.shift(lag)
    f["intraday_ret"] = close / open_ - 1
    f["high_low_range"] = (high - low) / open_
    f["mom_5d"] = close.pct_change(5)
    f["mom_10d"] = close.pct_change(10)
    f["vol_5d"] = ret.rolling(5).std()
    f["vol_20d"] = ret.rolling(20).std()
    f["rsi_14"] = rsi(close) / 100
    log_volume = np.log(volume.replace(0, np.nan))
    f["volume_z20"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    for name in ("SP500", "NASDAQ"):
        col = f"Close_{name}"
        if col in prices.columns:
            f[f"{name.lower()}_ret"] = prices[col].ffill(limit_area="inside").pct_change(fill_method=None)
    if "sp500_ret" in f:
        f["excess_ret_sp500"] = ret - f["sp500_ret"]
    f["rating_change"] = rating_change(prices, ticker)
    f["day_of_week"] = prices.index.day_of_week / 4

    f["close"] = close
    f[TARGET] = ret.shift(-1)
    # Drop the warm-up rows where rolling features are undefined; keep the final row.
    feature_cols = feature_columns(f)
    return f.dropna(subset=feature_cols)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in (TARGET, "close")]
