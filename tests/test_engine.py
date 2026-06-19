import pandas as pd
import numpy as np
import pytest
from ra_testbed.backtest.engine import BacktestEngine
from ra_testbed.strategies.fixed_weight import EqualWeightStrategy, FixedWeightStrategy


def create_fake_cache(tmp_path, tickers: list, n_days: int = 60, price: float = 100.0):
    """테스트용 parquet 캐시 파일 생성. 가격은 모두 동일한 고정값."""
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


class TestLookaheadBias:
    def test_strategy_only_receives_past_signal_dates(self, tmp_path):
        """
        신호일(t)에 전략이 받는 prices의 마지막 날짜가 단조 증가해야 한다.
        미래 데이터를 포함하면 역전이 발생한다.
        """
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers, n_days=60)

        last_dates_seen: list[pd.Timestamp] = []

        class RecordingStrategy(EqualWeightStrategy):
            def generate_weights(self_inner, prices):
                last_dates_seen.append(prices.index[-1])
                return super().generate_weights(prices)

        BacktestEngine(
            strategy=RecordingStrategy(),
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        ).run()

        assert len(last_dates_seen) > 0
        # 날짜가 단조 증가 = 미래 데이터 누출 없음
        assert last_dates_seen == sorted(last_dates_seen), (
            "전략이 미래 데이터를 수신했습니다 (lookahead bias)."
        )

    def test_strategy_never_sees_final_date(self, tmp_path):
        """마지막 신호일은 마지막 거래일보다 반드시 하루 전이어야 한다."""
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers, n_days=60)

        last_dates_seen: list[pd.Timestamp] = []

        class RecordingStrategy(EqualWeightStrategy):
            def generate_weights(self_inner, prices):
                last_dates_seen.append(prices.index[-1])
                return super().generate_weights(prices)

        engine = BacktestEngine(
            strategy=RecordingStrategy(),
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        )
        result = engine.run()
        last_execution_date = result.portfolio_values.index[-1]
        # 신호일이 마지막 실행일과 같거나 이후일 수 없다
        assert all(d < last_execution_date for d in last_dates_seen)


class TestTransactionCosts:
    def test_costs_reduce_final_value(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers, n_days=60)
        strategy = FixedWeightStrategy({"SPY": 0.5, "TLT": 0.5})

        def run(bps):
            return BacktestEngine(
                strategy=strategy,
                tickers=tickers,
                start="2020-01-01",
                end="2020-04-30",
                transaction_cost_bps=bps,
                rebalance_freq="M",
                cache_dir=str(tmp_path),
            ).run()

        result_0 = run(0)
        result_10 = run(10)
        assert result_10.portfolio_values.iloc[-1] < result_0.portfolio_values.iloc[-1]

    def test_flat_prices_zero_cost_preserves_capital(self, tmp_path):
        """가격이 일정하고 거래비용이 없으면 최종 자산은 초기 자본과 동일해야 한다."""
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers, n_days=30, price=100.0)

        result = BacktestEngine(
            strategy=FixedWeightStrategy({"SPY": 0.5, "TLT": 0.5}),
            tickers=tickers,
            start="2020-01-01",
            end="2020-02-28",
            initial_capital=1_000_000,
            transaction_cost_bps=0,
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        ).run()

        assert abs(result.portfolio_values.iloc[-1] - 1_000_000) < 1.0


class TestRebalanceFreq:
    def test_invalid_freq_raises_value_error(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers)
        with pytest.raises(ValueError):
            BacktestEngine(
                strategy=EqualWeightStrategy(),
                tickers=tickers,
                start="2020-01-01",
                end="2020-04-30",
                rebalance_freq="W",
                cache_dir=str(tmp_path),
            ).run()

    def test_result_contains_all_metrics(self, tmp_path):
        tickers = ["SPY", "TLT"]
        create_fake_cache(tmp_path, tickers, n_days=60)
        result = BacktestEngine(
            strategy=EqualWeightStrategy(),
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        ).run()
        assert all(k in result.metrics for k in ["CAGR", "MDD", "Sharpe", "Volatility"])

    def test_portfolio_series_length_matches_trading_days(self, tmp_path):
        tickers = ["SPY", "TLT"]
        n_days = 60
        create_fake_cache(tmp_path, tickers, n_days=n_days)
        result = BacktestEngine(
            strategy=EqualWeightStrategy(),
            tickers=tickers,
            start="2020-01-01",
            end="2020-04-30",
            rebalance_freq="M",
            cache_dir=str(tmp_path),
        ).run()
        # 결과 길이는 캐시된 날짜 수 이하 (날짜 범위로 슬라이싱되므로)
        assert len(result.portfolio_values) <= n_days
        assert len(result.portfolio_values) > 0
