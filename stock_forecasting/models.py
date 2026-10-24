"""Forecasting models with one interface.

``fit(train)`` sees only the training rows of a feature frame. ``predict(history, rows)``
returns next-day return forecasts for ``rows``, using ``history`` (all rows up to
and including the last one in ``rows``) for lagged inputs such as the sequence
window. Scalers are fitted inside ``fit`` on training rows only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import TARGET, feature_columns


class Forecaster:
    name = "base"

    def fit(self, train: pd.DataFrame) -> Forecaster:
        return self

    def predict(self, history: pd.DataFrame, rows: pd.Index) -> np.ndarray:
        raise NotImplementedError


class Naive(Forecaster):
    """Tomorrow's close equals today's close (a zero return forecast)."""

    name = "naive"

    def predict(self, history, rows):
        return np.zeros(len(rows))


class RidgeForecaster(Forecaster):
    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, train):
        cols = feature_columns(train)
        self.cols = cols
        self.scaler = StandardScaler().fit(train[cols])
        self.model = Ridge(alpha=self.alpha).fit(self.scaler.transform(train[cols]), train[TARGET])
        return self

    def predict(self, history, rows):
        return self.model.predict(self.scaler.transform(history.loc[rows, self.cols]))


class ArimaForecaster(Forecaster):
    """ARIMA on daily returns. Parameters are estimated on training rows only,
    then applied to the full history to get one-step-ahead forecasts."""

    name = "arima"

    def __init__(self, order=(5, 0, 2)):
        self.order = tuple(order)

    def fit(self, train):
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = ARIMA(train["ret_1d"].to_numpy(), order=self.order).fit()
        return self

    def predict(self, history, rows):
        series = history["ret_1d"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.result.apply(series)
            # forecast for day t+1 made at day t is the one-step prediction for position t+1
            nxt = applied.forecast(1)[0]
            onestep = np.append(applied.predict(start=1, end=len(series) - 1), nxt)
        pos = history.index.get_indexer(rows)
        return onestep[pos]


class _SequenceForecaster(Forecaster):
    """Shared training loop for the LSTM and Transformer."""

    def __init__(
        self, window=7, seed=42, max_epochs=300, patience=20, learning_rate=1e-3, val_fraction=0.15, **net_params
    ):
        self.window = window
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.val_fraction = val_fraction
        self.net_params = net_params

    def build_net(self, input_dim):
        raise NotImplementedError

    def _windows(self, scaled: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return np.stack([scaled[p - self.window + 1 : p + 1] for p in positions])

    def fit(self, train):
        import torch

        torch.manual_seed(self.seed)
        self.cols = feature_columns(train)
        self.x_scaler = StandardScaler().fit(train[self.cols])
        self.y_mean, self.y_std = train[TARGET].mean(), train[TARGET].std()
        scaled = self.x_scaler.transform(train[self.cols])
        y = ((train[TARGET] - self.y_mean) / self.y_std).to_numpy()
        positions = np.arange(self.window - 1, len(train))
        X = self._windows(scaled, positions)
        y = y[positions]

        n_val = max(1, int(len(X) * self.val_fraction))
        X_tr, y_tr = torch.tensor(X[:-n_val], dtype=torch.float32), torch.tensor(y[:-n_val], dtype=torch.float32)
        X_va, y_va = torch.tensor(X[-n_val:], dtype=torch.float32), torch.tensor(y[-n_val:], dtype=torch.float32)

        self.net = self.build_net(len(self.cols))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.SmoothL1Loss()
        best, best_state, waited = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            optimizer.zero_grad()
            loss = loss_fn(self.net(X_tr), y_tr)
            loss.backward()
            optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(X_va), y_va).item()
            if val < best - 1e-6:
                best, waited = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, history, rows):
        import torch

        scaled = self.x_scaler.transform(history[self.cols])
        positions = history.index.get_indexer(rows)
        if positions.min() < self.window - 1:
            raise ValueError("not enough history before the first forecast row")
        with torch.no_grad():
            out = self.net(torch.tensor(self._windows(scaled, positions), dtype=torch.float32)).numpy()
        return out * self.y_std + self.y_mean


def _torch_nets():
    import torch
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(1)

    class TransformerNet(nn.Module):
        def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, dropout=0.2, max_len=64):
            super().__init__()
            self.proj = nn.Linear(input_dim, model_dim)
            self.pos = nn.Parameter(torch.zeros(1, max_len, model_dim))
            layer = nn.TransformerEncoderLayer(
                model_dim, num_heads, dim_feedforward=model_dim * 2, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
            self.head = nn.Linear(model_dim, 1)

        def forward(self, x):
            h = self.proj(x) + self.pos[:, : x.shape[1]]
            return self.head(self.encoder(h)[:, -1]).squeeze(1)

    return LSTMNet, TransformerNet


class LSTMForecaster(_SequenceForecaster):
    name = "lstm"

    def build_net(self, input_dim):
        return _torch_nets()[0](input_dim, **self.net_params)


class TransformerForecaster(_SequenceForecaster):
    name = "transformer"

    def build_net(self, input_dim):
        return _torch_nets()[1](input_dim, **self.net_params)


def make_model(name: str, config) -> Forecaster:
    params = config.model_params.get(name, {})
    if name == "naive":
        return Naive()
    if name == "ridge":
        return RidgeForecaster(**params)
    if name == "arima":
        return ArimaForecaster(**params)
    if name in ("lstm", "transformer"):
        cls = LSTMForecaster if name == "lstm" else TransformerForecaster
        return cls(window=config.window, seed=config.seed, **config.training, **params)
    raise ValueError(f"unknown model: {name}")
