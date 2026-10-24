"""Walk-forward backtest.

For each stock the first ``initial_train_days`` rows train the models. They then
forecast the next ``refit_every_days`` days one day at a time, are refitted on
everything seen so far, and so on to the end of the data. Every forecast is
made with information available at the close of the day before.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import Config
from .data import load_prices
from .features import TARGET, build_features
from .models import make_model


def folds(n_rows: int, initial_train: int, step: int):
    """Yield (train_end, test_end) positions: train on [0, train_end), test on [train_end, test_end)."""
    if n_rows <= initial_train:
        raise ValueError(f"need more than {initial_train} rows, got {n_rows}")
    for start in range(initial_train, n_rows, step):
        yield start, min(start + step, n_rows)


def backtest_stock(frame: pd.DataFrame, ticker: str, config: Config, models: list[str]) -> pd.DataFrame:
    labelled = frame[frame[TARGET].notna()]
    out = []
    for train_end, test_end in folds(len(labelled), config.initial_train_days, config.refit_every_days):
        train = labelled.iloc[:train_end]
        history = labelled.iloc[:test_end]
        rows = labelled.index[train_end:test_end]
        for name in models:
            model = make_model(name, config).fit(train)
            preds = model.predict(history, rows)
            out.append(
                pd.DataFrame(
                    {
                        "date": rows,
                        "ticker": ticker,
                        "model": name,
                        "fold_start": rows[0],
                        "close": labelled.loc[rows, "close"].to_numpy(),
                        "predicted_return": preds,
                        "actual_return": labelled.loc[rows, TARGET].to_numpy(),
                    }
                )
            )
    return pd.concat(out, ignore_index=True)


def run_backtest(
    config: Config, models: list[str] | None = None, tickers: list[str] | None = None, verbose: bool = True
) -> pd.DataFrame:
    prices = load_prices(config.data_path)
    models = models or config.models
    results = []
    for ticker in tickers or config.tickers:
        t0 = time.perf_counter()
        results.append(backtest_stock(build_features(prices, ticker), ticker, config, models))
        if verbose:
            print(f"{ticker}: done in {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.concat(results, ignore_index=True)


def score(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics per stock and model. Return errors equal percentage errors on the next close."""

    def metrics(g: pd.DataFrame) -> pd.Series:
        err = g.predicted_return - g.actual_return
        moved = (g.actual_return != 0) & (g.predicted_return != 0)
        hits = np.sign(g.predicted_return[moved]) == np.sign(g.actual_return[moved])
        return pd.Series(
            {
                "days": len(g),
                "mae_pct": err.abs().mean() * 100,
                "rmse_pct": np.sqrt((err**2).mean()) * 100,
                "direction_hits": int(hits.sum()),
                "direction_calls": int(moved.sum()),
                "directional_accuracy": hits.mean() if moved.any() else np.nan,
                "up_day_share": (g.actual_return > 0).mean(),
            }
        )

    per_stock = preds.groupby(["ticker", "model"]).apply(metrics, include_groups=False).reset_index()
    naive = per_stock[per_stock.model == "naive"].set_index("ticker")["mae_pct"]
    per_stock["skill_vs_naive"] = 1 - per_stock.mae_pct / per_stock.ticker.map(naive)
    return per_stock


def summarise(per_stock: pd.DataFrame) -> pd.DataFrame:
    """One row per model, averaged across stocks, with a binomial test on direction calls."""
    rows = []
    for model, g in per_stock.groupby("model", sort=False):
        hits, calls = int(g.direction_hits.sum()), int(g.direction_calls.sum())
        rows.append(
            {
                "model": model,
                "mae_pct": g.mae_pct.mean(),
                "rmse_pct": g.rmse_pct.mean(),
                "skill_vs_naive": g.skill_vs_naive.mean(),
                "stocks_beating_naive": int((g.skill_vs_naive > 0).sum()),
                "directional_accuracy": hits / calls if calls else np.nan,
                "direction_p_value": binomtest(hits, calls, 0.5, alternative="greater").pvalue if calls else np.nan,
                "up_day_share": g.up_day_share.mean(),
                "forecasts": int(g.days.sum()),
            }
        )
    order = {m: i for i, m in enumerate(["naive", "ridge", "arima", "lstm", "transformer"])}
    return pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
