import pandas as pd
import numpy as np
import pytest
from ra_testbed.strategies.base import Strategy
from ra_testbed.strategies.fixed_weight import EqualWeightStrategy, FixedWeightStrategy
from ra_testbed.strategies.moving_average import MovingAverageCrossStrategy


def make_prices(n_days: int = 300, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = ["SPY", "TLT", "GLD"]
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = {t: 100 + np.cumsum(np.random.randn(n_days)) for t in tickers}
    return pd.DataFrame(data, index=dates)


class TestEqualWeightStrategy:
    def test_returns_equal_weights(self):
        prices = make_prices()
        weights = EqualWeightStrategy().generate_weights(prices)
        assert set(weights.keys()) == {"SPY", "TLT", "GLD"}
        assert all(abs(w - 1 / 3) < 1e-6 for w in weights.values())

    def test_sums_to_one(self):
        weights = EqualWeightStrategy().generate_weights(make_prices())
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_two_assets(self):
        prices = make_prices(tickers=["A", "B"])
        weights = EqualWeightStrategy().generate_weights(prices)
        assert abs(weights["A"] - 0.5) < 1e-6

    def test_ignores_price_values(self):
        """비중이 가격 수준에 독립적이어야 한다."""
        p1 = make_prices(n_days=10)
        p2 = make_prices(n_days=100)
        w1 = EqualWeightStrategy().generate_weights(p1)
        w2 = EqualWeightStrategy().generate_weights(p2)
        assert w1 == w2


class TestFixedWeightStrategy:
    def test_returns_given_weights(self):
        target = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}
        weights = FixedWeightStrategy(target).generate_weights(make_prices())
        assert weights == target

    def test_rejects_weights_not_summing_to_one(self):
        with pytest.raises(ValueError):
            FixedWeightStrategy({"SPY": 0.6, "TLT": 0.5})

    def test_weights_unchanged_regardless_of_prices(self):
        target = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}
        strategy = FixedWeightStrategy(target)
        assert strategy.generate_weights(make_prices(10)) == strategy.generate_weights(make_prices(200))


class TestMovingAverageCrossStrategy:
    def test_rejects_invalid_windows(self):
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(short_window=200, long_window=50)

    def test_fallback_when_insufficient_data(self):
        prices = make_prices(n_days=50)
        strategy = MovingAverageCrossStrategy(short_window=50, long_window=200)
        weights = strategy.generate_weights(prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_always_sum_to_one(self):
        prices = make_prices(n_days=300)
        strategy = MovingAverageCrossStrategy(short_window=50, long_window=200)
        weights = strategy.generate_weights(prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_bullish_asset_gets_positive_weight(self):
        """상승추세 자산은 양의 비중을 가진다."""
        # 한 자산만 단조 상승 → 반드시 bullish
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        prices = pd.DataFrame({
            "UP": np.linspace(50, 150, 300),
            "DOWN": np.linspace(150, 50, 300),
        }, index=dates)
        strategy = MovingAverageCrossStrategy(short_window=50, long_window=200)
        weights = strategy.generate_weights(prices)
        assert weights["UP"] > 0
