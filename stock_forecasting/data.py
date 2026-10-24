"""Download and load daily prices.

Prices are stored wide, one row per trading day, with columns such as
``Close_AAPL`` and ``Close_SP500``. The committed dataset in ``data/`` makes
every result in the README reproducible without network access.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

INDICES = {"SP500": "^GSPC", "NASDAQ": "^IXIC"}
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.sort_index()


def download_prices(tickers: list[str], history_days: int = 730, end: dt.date | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV for the tickers, plus S&P 500 and NASDAQ closes, from Yahoo Finance."""
    import yfinance as yf

    end = end or dt.date.today()
    start = end - dt.timedelta(days=history_days)
    frames = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
        frames.append(raw[FIELDS].add_suffix(f"_{ticker}"))
    for name, symbol in INDICES.items():
        raw = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
        frames.append(raw[["Close"]].rename(columns={"Close": f"Close_{name}"}))
    prices = pd.concat(frames, axis=1).sort_index()
    prices.index.name = "Date"
    return prices.dropna(subset=[f"Close_{t}" for t in tickers])


def available_tickers(prices: pd.DataFrame) -> list[str]:
    return sorted(c.split("_", 1)[1] for c in prices.columns if c.startswith("Close_") and c[6:] not in INDICES)
