import pandas as pd
import numpy as np
import pytest
from ra_testbed.backtest.compare import compare_strategies
from ra_testbed.strategies.fixed_weight import EqualWeightStrategy, FixedWeightStrategy


def create_fake_cache(tmp_path, tickers, n_days=60, price=100.0):
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    for ticker in tickers:
        df = pd.DataFrame(
            {
                "Open": np.full(n_days, price),
                "High": np.full(n_days, price * 1.01),
                "Low": np.full(n_days, price * 0.99),
                "Close": np.full(n_days, price),
                "Volume": np.full(n_days, 1_000_000.0),
            },
            index=dates,
        )
        df.to_parquet(tmp_path / f"{ticker}.parquet")


class TestCompareStrategies:
    def test_returns_one_result_per_strategy(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers)
        results = compare_strategies(
            {
                "균등": EqualWeightStrategy(),
                "고정": FixedWeightStrategy({"SPY": 0.6, "TLT": 0.4}),
            },
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        )
        assert set(results.keys()) == {"균등", "고정"}

    def test_all_results_share_same_index(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers)
        results = compare_strategies(
            {
                "A": EqualWeightStrategy(),
                "B": FixedWeightStrategy({"SPY": 0.5, "TLT": 0.5}),
            },
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        )
        idx_a = results["A"].portfolio_values.index
        idx_b = results["B"].portfolio_values.index
        assert idx_a.equals(idx_b)

    def test_deterministic_same_strategy_same_result(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers)
        results = compare_strategies(
            {"first": EqualWeightStrategy(), "second": EqualWeightStrategy()},
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        )
        pd.testing.assert_series_equal(
            results["first"].portfolio_values,
            results["second"].portfolio_values,
        )

    def test_each_result_has_metrics(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers)
        results = compare_strategies(
            {"균등": EqualWeightStrategy()},
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        )
        assert all(
            k in results["균등"].metrics for k in ["CAGR", "MDD", "Sharpe", "Volatility"]
        )

    def test_empty_strategies_raises(self, tmp_path):
        with pytest.raises(ValueError):
            compare_strategies(
                {}, tickers=["SPY"], start="2020-01-01", end="2020-04-30"
            )
