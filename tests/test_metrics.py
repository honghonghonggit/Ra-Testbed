import pandas as pd
import numpy as np
import pytest
from ra_testbed.metrics.risk import cagr, max_drawdown, sharpe_ratio, annual_volatility


def make_portfolio(start: str, periods: int, daily_return: float) -> pd.Series:
    dates = pd.date_range(start, periods=periods, freq="B")
    values = 1_000_000 * (1 + daily_return) ** np.arange(periods)
    return pd.Series(values, index=dates)


class TestCagr:
    def test_flat_portfolio_is_zero(self):
        pv = make_portfolio("2020-01-01", 252, 0.0)
        assert abs(cagr(pv)) < 1e-4

    def test_positive_for_rising_portfolio(self):
        pv = make_portfolio("2020-01-01", 252, 0.001)
        assert cagr(pv) > 0

    def test_known_doubling_over_two_years(self):
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        values = pd.Series(np.linspace(1_000_000, 2_000_000, len(dates)), index=dates)
        expected = 2 ** (1 / 2) - 1  # ≈ 41.4%
        assert abs(cagr(values) - expected) < 0.01


class TestMaxDrawdown:
    def test_no_drawdown_for_rising(self):
        pv = make_portfolio("2020-01-01", 252, 0.001)
        assert abs(max_drawdown(pv)) < 1e-4

    def test_50_percent_drawdown(self):
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        pv = pd.Series([100, 200, 100, 150], index=dates, dtype=float)
        assert abs(max_drawdown(pv) - (-0.5)) < 1e-6

    def test_returns_negative_value(self):
        pv = make_portfolio("2020-01-01", 252, -0.001)
        assert max_drawdown(pv) < 0


class TestSharpeRatio:
    def test_positive_for_good_strategy(self):
        pv = make_portfolio("2020-01-01", 252 * 5, 0.0004)
        assert sharpe_ratio(pv) > 0

    def test_negative_for_losing_strategy(self):
        pv = make_portfolio("2020-01-01", 252 * 5, -0.0003)
        assert sharpe_ratio(pv) < 0

    def test_zero_for_constant_portfolio(self):
        pv = make_portfolio("2020-01-01", 252, 0.0)
        assert sharpe_ratio(pv) == 0.0


class TestAnnualVolatility:
    def test_zero_for_constant_daily_return(self):
        """일별 수익률이 일정하면 변동성은 0."""
        pv = make_portfolio("2020-01-01", 252, 0.001)
        assert abs(annual_volatility(pv)) < 1e-6

    def test_positive_for_volatile_portfolio(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        pv = pd.Series(
            1_000_000 * np.cumprod(1 + np.random.randn(252) * 0.01), index=dates
        )
        assert annual_volatility(pv) > 0
