import numpy as np
import pytest

from stock_forecasting.features import TARGET, build_features
from stock_forecasting.models import make_model
from stock_forecasting.predict import forecast_next_day
from stock_forecasting.report import plot_summary


@pytest.mark.parametrize("name", ["naive", "ridge", "arima", "lstm", "transformer"])
def test_each_model_fits_and_forecasts(prices, config, name):
    frame = build_features(prices, "AAA")
    labelled = frame[frame[TARGET].notna()]
    train, rows = labelled.iloc[:200], labelled.index[200:210]
    model = make_model(name, config).fit(train)
    preds = model.predict(labelled.iloc[:210], rows)
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()
    assert np.abs(preds).max() < 0.2


def test_scaler_is_fitted_on_training_rows_only(prices, config):
    frame = build_features(prices, "AAA")
    train = frame.iloc[:150]
    model = make_model("ridge", config).fit(train)
    assert np.allclose(model.scaler.mean_, train[model.cols].mean().to_numpy())


def test_arima_forecasts_only_use_past_returns(prices, config):
    frame = build_features(prices, "AAA")
    labelled = frame[frame[TARGET].notna()]
    model = make_model("arima", config).fit(labelled.iloc[:200])
    rows = labelled.index[200:205]
    base = model.predict(labelled.iloc[:205], rows)
    changed = labelled.iloc[:205].copy()
    changed.loc[rows[-1], "ret_1d"] += 0.5  # only the last day's own return changes
    after = model.predict(changed, rows)
    assert np.allclose(base[:-1], after[:-1])


def test_sequence_model_is_reproducible(prices, config):
    frame = build_features(prices, "AAA")
    labelled = frame[frame[TARGET].notna()]
    rows = labelled.index[200:205]
    a = make_model("lstm", config).fit(labelled.iloc[:200]).predict(labelled.iloc[:205], rows)
    b = make_model("lstm", config).fit(labelled.iloc[:200]).predict(labelled.iloc[:205], rows)
    assert np.allclose(a, b)


def test_predict_next_day(config):
    out = forecast_next_day(config, models=["naive", "ridge"])
    assert len(out) == 4
    naive = out[out.model == "naive"]
    assert (naive.predicted_close == naive.last_close).all()


def test_summary_chart_is_written(tmp_path, config):
    from stock_forecasting.backtest import run_backtest, score, summarise

    summary = summarise(score(run_backtest(config, verbose=False)))
    plot_summary(summary, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 10_000
