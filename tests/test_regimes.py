import pandas as pd
import numpy as np
import pytest
from ra_testbed.backtest.regimes import (
    classify_regimes,
    decompose_by_regime,
    PREDEFINED_SCENARIOS,
    BULL,
    BEAR,
    RECOVERY,
)


def make_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    dates = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=dates, dtype=float)


class TestClassifyRegimes:
    def test_monotonic_rise_is_all_bull(self):
        regimes = classify_regimes(make_series(list(np.linspace(100, 200, 50))))
        assert (regimes == BULL).all()

    def test_drop_over_20pct_is_bear(self):
        # 100 정점에서 75로 하락 (-25%) → 하락장
        regimes = classify_regimes(make_series([100, 100, 90, 80, 75]))
        assert regimes.iloc[-1] == BEAR
        assert regimes.iloc[-2] == BEAR  # 80은 -20% 이므로 임계 도달

    def test_shallow_pullback_stays_bull(self):
        # 정점 대비 -10%까지만 하락 → 하락장 아님
        regimes = classify_regimes(make_series([100, 95, 92, 90]))
        assert (regimes == BULL).all()

    def test_recovery_after_bear(self):
        # 100 → 70(-30%, 하락장) → 95(-5%, 회복) → 105(신고가, 상승장)
        regimes = classify_regimes(make_series([100, 70, 95, 105]))
        assert regimes.iloc[1] == BEAR
        assert regimes.iloc[2] == RECOVERY
        assert regimes.iloc[3] == BULL

    def test_index_preserved(self):
        s = make_series([100, 90, 80, 75])
        regimes = classify_regimes(s)
        assert regimes.index.equals(s.index)
        assert len(regimes) == len(s)

    def test_custom_threshold(self):
        # -10% 임계값이면 90에서 이미 하락장
        regimes = classify_regimes(make_series([100, 90, 85]), drawdown_threshold=-0.10)
        assert regimes.iloc[1] == BEAR


class TestDecomposeByRegime:
    def test_only_present_regimes_returned(self):
        pv = make_series(list(np.linspace(100, 150, 30)))
        regimes = classify_regimes(pv)
        breakdown = decompose_by_regime(pv, regimes)
        # 단조 상승이므로 상승장만 존재
        assert set(breakdown.keys()) == {BULL}

    def test_each_regime_has_all_metrics(self):
        pv = make_series([100, 70, 95, 105, 110, 80, 90])
        regimes = classify_regimes(pv)
        breakdown = decompose_by_regime(pv, regimes)
        for stats in breakdown.values():
            assert set(stats.keys()) == {"수익률", "MDD", "변동성", "거래일수"}

    def test_day_counts_sum_to_total_minus_one(self):
        # pct_change로 첫날이 빠지므로 합은 전체-1
        pv = make_series([100, 70, 95, 105, 110, 80, 90])
        regimes = classify_regimes(pv)
        breakdown = decompose_by_regime(pv, regimes)
        total_days = sum(s["거래일수"] for s in breakdown.values())
        assert total_days == len(pv) - 1

    def test_bear_regime_has_negative_return(self):
        # 정점 후 지속 하락 구간 → 하락장 수익률 음수
        pv = make_series([100, 100, 60, 55, 50])
        regimes = classify_regimes(pv)
        breakdown = decompose_by_regime(pv, regimes)
        assert BEAR in breakdown
        assert breakdown[BEAR]["수익률"] < 0


class TestPredefinedScenarios:
    def test_contains_three_crisis_windows(self):
        assert set(PREDEFINED_SCENARIOS.keys()) == {
            "2008 금융위기",
            "2020 코로나 급락",
            "2022 금리인상기",
        }

    def test_each_scenario_is_valid_date_range(self):
        for start, end in PREDEFINED_SCENARIOS.values():
            assert pd.Timestamp(start) < pd.Timestamp(end)
