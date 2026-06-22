import pandas as pd
import pytest
from ra_testbed.strategies.factory import strategy_from_config
from ra_testbed.strategies.fixed_weight import EqualWeightStrategy, FixedWeightStrategy
from ra_testbed.strategies.moving_average import MovingAverageCrossStrategy


def make_prices(tickers, n=10):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({t: range(100, 100 + n) for t in tickers}, index=dates)


class TestStrategyFromConfig:
    def test_equal_weight(self):
        s = strategy_from_config({"type": "equal_weight"})
        assert isinstance(s, EqualWeightStrategy)

    def test_fixed_weight(self):
        s = strategy_from_config({"type": "fixed_weight", "weights": {"SPY": 0.6, "TLT": 0.4}})
        assert isinstance(s, FixedWeightStrategy)
        w = s.generate_weights(make_prices(["SPY", "TLT"]))
        assert w == {"SPY": 0.6, "TLT": 0.4}

    def test_ma_cross_with_params(self):
        s = strategy_from_config({"type": "ma_cross", "short_window": 20, "long_window": 100})
        assert isinstance(s, MovingAverageCrossStrategy)
        assert s.short_window == 20 and s.long_window == 100

    def test_ma_cross_defaults(self):
        s = strategy_from_config({"type": "ma_cross"})
        assert s.short_window == 50 and s.long_window == 200

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="알 수 없는 전략 type"):
            strategy_from_config({"type": "deep_learning_magic"})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="'type'"):
            strategy_from_config({"weights": {"SPY": 1.0}})

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            strategy_from_config("equal_weight")

    def test_fixed_weight_missing_weights_raises(self):
        with pytest.raises(ValueError, match="weights"):
            strategy_from_config({"type": "fixed_weight"})

    def test_fixed_weight_non_numeric_raises(self):
        with pytest.raises(ValueError, match="숫자"):
            strategy_from_config({"type": "fixed_weight", "weights": {"SPY": "많이"}})

    def test_fixed_weight_not_summing_to_one_raises(self):
        # FixedWeightStrategy 생성자의 합계 검증으로 전파
        with pytest.raises(ValueError):
            strategy_from_config({"type": "fixed_weight", "weights": {"SPY": 0.6, "TLT": 0.6}})

    def test_ma_cross_invalid_windows_raises(self):
        # short >= long → MovingAverageCrossStrategy 생성자가 거부
        with pytest.raises(ValueError):
            strategy_from_config({"type": "ma_cross", "short_window": 200, "long_window": 50})

    def test_ma_cross_non_numeric_raises(self):
        with pytest.raises(ValueError, match="숫자"):
            strategy_from_config({"type": "ma_cross", "short_window": "fast"})
