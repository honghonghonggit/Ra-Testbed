import pandas as pd
import numpy as np
import pytest
from ra_testbed.backtest.lookahead import (
    detect_lookahead,
    wrap_strategy_as_audit,
    lookahead_cheating_audit,
    CLEAN,
    DETECTED,
)
from ra_testbed.strategies.fixed_weight import EqualWeightStrategy
from ra_testbed.strategies.moving_average import MovingAverageCrossStrategy


def make_prices(n=300, tickers=None, seed=42):
    if tickers is None:
        tickers = ["SPY", "TLT", "GLD"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # 양수 가격 (난수 교란이 곱셈이므로 양수 유지)
    data = {t: 100 + np.cumsum(rng.normal(0, 1, n)) + 50 for t in tickers}
    return pd.DataFrame(data, index=dates)


class TestDetectLookahead:
    def test_cheating_strategy_is_detected(self):
        prices = make_prices()
        as_of = prices.index[200]
        report = detect_lookahead(lookahead_cheating_audit, prices, as_of)
        assert report.status == DETECTED
        assert report.max_weight_diff > 0

    def test_wrapped_equal_weight_is_clean(self):
        prices = make_prices()
        as_of = prices.index[200]
        audit = wrap_strategy_as_audit(EqualWeightStrategy())
        report = detect_lookahead(audit, prices, as_of)
        assert report.status == CLEAN
        assert report.max_weight_diff <= 1e-9

    def test_wrapped_ma_cross_is_clean(self):
        """추세 전략도 슬라이싱으로 감싸면 미래 참조가 없어 CLEAN이어야 한다."""
        prices = make_prices()
        as_of = prices.index[250]
        audit = wrap_strategy_as_audit(
            MovingAverageCrossStrategy(short_window=20, long_window=50)
        )
        report = detect_lookahead(audit, prices, as_of)
        assert report.status == CLEAN

    def test_report_carries_as_of_and_detail(self):
        prices = make_prices()
        as_of = prices.index[200]
        report = detect_lookahead(lookahead_cheating_audit, prices, as_of)
        assert report.as_of == as_of
        assert report.detail

    def test_no_future_data_raises(self):
        prices = make_prices(n=100)
        as_of = prices.index[-1]  # 미래 구간 없음
        with pytest.raises(ValueError):
            detect_lookahead(lookahead_cheating_audit, prices, as_of)

    def test_as_of_before_start_raises(self):
        prices = make_prices(n=100)
        early = prices.index[0] - pd.Timedelta(days=10)
        with pytest.raises(ValueError):
            detect_lookahead(lookahead_cheating_audit, prices, early)

    def test_deterministic_with_same_seed(self):
        prices = make_prices()
        as_of = prices.index[200]
        r1 = detect_lookahead(lookahead_cheating_audit, prices, as_of, seed=7)
        r2 = detect_lookahead(lookahead_cheating_audit, prices, as_of, seed=7)
        assert r1.max_weight_diff == r2.max_weight_diff


class TestCheatingAudit:
    def test_allocates_fully_to_one_asset(self):
        prices = make_prices()
        weights = lookahead_cheating_audit(prices, prices.index[200])
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert sorted(weights.values())[-1] == 1.0  # 한 자산에 100%
