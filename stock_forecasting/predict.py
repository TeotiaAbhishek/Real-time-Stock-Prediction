"""Train on all available history and forecast the next trading day."""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data import load_prices
from .features import TARGET, build_features
from .models import make_model


def forecast_next_day(
    config: Config, models: list[str] | None = None, tickers: list[str] | None = None
) -> pd.DataFrame:
    prices = load_prices(config.data_path)
    rows = []
    for ticker in tickers or config.tickers:
        frame = build_features(prices, ticker)
        train = frame[frame[TARGET].notna()]
        latest = frame.index[-1:]
        for name in models or config.models:
            pred = float(make_model(name, config).fit(train).predict(frame, latest)[0])
            close = float(frame.loc[latest[0], "close"])
            rows.append(
                {
                    "ticker": ticker,
                    "as_of": latest[0].date(),
                    "model": name,
                    "predicted_return_pct": round(pred * 100, 3),
                    "last_close": round(close, 2),
                    "predicted_close": round(close * (1 + pred), 2),
                }
            )
    return pd.DataFrame(rows)
