import numpy as np
import pandas as pd
import pytest

from stock_forecasting.backtest import backtest_stock, folds, run_backtest, score, summarise
from stock_forecasting.features import TARGET, build_features
from stock_forecasting.models import Forecaster


def test_folds_cover_test_period_once_without_overlap():
    spans = list(folds(100, 60, 15))
    assert spans == [(60, 75), (75, 90), (90, 100)]
    with pytest.raises(ValueError):
        list(folds(50, 60, 15))


class Spy(Forecaster):
    """Records the last training date it was given and forecasts zero."""

    name = "spy"
    seen = []

    def fit(self, train):
        Spy.seen.append(train.index.max())
        return self

    def predict(self, history, rows):
        return np.zeros(len(rows))


def test_models_are_trained_only_on_rows_before_each_test_window(prices, config, monkeypatch):
    monkeypatch.setattr("stock_forecasting.backtest.make_model", lambda name, cfg: Spy())
    Spy.seen = []
    frame = build_features(prices, "AAA")
    preds = backtest_stock(frame, "AAA", config, ["spy"])
    for train_end, (start, _rows) in zip(Spy.seen, preds.groupby("fold_start"), strict=True):
        assert train_end < start


def test_scores_naive_against_hand_calculation():
    preds = pd.DataFrame(
        {
            "ticker": "AAA",
            "model": ["naive"] * 4 + ["m"] * 4,
            "predicted_return": [0, 0, 0, 0, 0.01, -0.01, 0.02, 0.01],
            "actual_return": [0.01, -0.02, 0.03, -0.01] * 2,
        }
    )
    s = score(preds).set_index("model")
    assert np.isclose(s.loc["naive", "mae_pct"], 1.75)
    assert np.isclose(s.loc["m", "mae_pct"], (0 + 1 + 1 + 2) / 4)
    assert s.loc["m", "directional_accuracy"] == 0.75
    assert np.isclose(s.loc["m", "skill_vs_naive"], 1 - 1.0 / 1.75)
    summary = summarise(s.reset_index()).set_index("model")
    assert summary.loc["m", "stocks_beating_naive"] == 1


def test_end_to_end_backtest_on_synthetic_prices(config):
    preds = run_backtest(config, verbose=False)
    assert set(preds.model) == {"naive", "ridge", "arima"}
    assert preds.groupby(["ticker", "model"]).size().nunique() == 1
    assert not preds.predicted_return.isna().any()
    summary = summarise(score(preds))
    # Random-walk prices are unpredictable, so nothing should beat naive by much.
    assert (summary.skill_vs_naive < 0.05).all()


def test_target_column_never_reaches_models(prices, config):
    from stock_forecasting.features import feature_columns

    frame = build_features(prices, "AAA")
    assert TARGET not in feature_columns(frame)
