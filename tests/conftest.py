import numpy as np
import pandas as pd
import pytest

from stock_forecasting.config import Config


@pytest.fixture
def prices():
    """Synthetic random-walk prices for two stocks and two indices, 320 trading days."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=320, name="Date")
    frame = {}
    for ticker in ("AAA", "BBB"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
        open_ = close * (1 + rng.normal(0, 0.005, len(dates)))
        frame |= {
            f"Close_{ticker}": close,
            f"Open_{ticker}": open_,
            f"High_{ticker}": np.maximum(close, open_) * 1.01,
            f"Low_{ticker}": np.minimum(close, open_) * 0.99,
            f"Volume_{ticker}": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        }
    for name in ("SP500", "NASDAQ"):
        frame[f"Close_{name}"] = 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame(frame, index=dates)


@pytest.fixture
def config(tmp_path, prices):
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    return Config(
        data_path=path,
        tickers=["AAA", "BBB"],
        initial_train_days=200,
        refit_every_days=40,
        models=["naive", "ridge", "arima"],
        window=5,
        model_params={
            "ridge": {"alpha": 1.0},
            "arima": {"order": [1, 0, 0]},
            "lstm": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
            "transformer": {"model_dim": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.0},
        },
        training={"max_epochs": 5, "patience": 2, "learning_rate": 0.01, "val_fraction": 0.2},
    )
