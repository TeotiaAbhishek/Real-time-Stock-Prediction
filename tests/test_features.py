import numpy as np
import pandas as pd

from stock_forecasting.features import TARGET, build_features, feature_columns, rsi


def test_target_is_next_day_close_to_close_return(prices):
    f = build_features(prices, "AAA")
    close = prices["Close_AAA"]
    day = f.index[10]
    nxt = close.index[close.index.get_loc(day) + 1]
    assert np.isclose(f.loc[day, TARGET], close[nxt] / close[day] - 1)
    assert np.isnan(f[TARGET].iloc[-1])


def test_features_never_use_future_prices(prices):
    """Shocking every price after a cut-off must not change any feature up to that day."""
    cut = prices.index[200]
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= 1.5
    a, b = build_features(prices, "AAA"), build_features(shocked, "AAA")
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a.loc[:cut, cols], b.loc[:cut, cols])


def test_no_missing_features_and_no_price_levels(prices):
    f = build_features(prices, "AAA")
    cols = feature_columns(f)
    assert not f[cols].isna().any().any()
    assert not any(c.startswith(("Close_", "Open_", "High_", "Low_")) for c in cols)


def test_index_gap_at_end_is_not_filled_with_zero_returns(prices):
    prices = prices.copy()
    prices.loc[prices.index[-3:], "Close_SP500"] = np.nan
    f = build_features(prices, "AAA")
    assert f.index[-1] == prices.index[-4]


def test_rsi_bounds():
    close = pd.Series(np.r_[np.arange(1, 30), np.arange(30, 1, -1)], dtype=float)
    values = rsi(close).dropna()
    assert values.between(0, 100).all()
    assert values.iloc[0] == 100


def test_rating_change_flag(prices):
    prices = prices.copy()
    for col in ("Action_aaa", "ToGrade_aaa", "FromGrade_aaa"):
        prices[col] = pd.Series(None, index=prices.index, dtype=object)
    day = prices.index[100]
    prices.loc[day, ["Action_aaa", "ToGrade_aaa", "FromGrade_aaa"]] = ["up", "Buy", "Hold"]
    f = build_features(prices, "AAA")
    assert f.loc[day, "rating_change"] == 1
    assert f["rating_change"].sum() == 1
