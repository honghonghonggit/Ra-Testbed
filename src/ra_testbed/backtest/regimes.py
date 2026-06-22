"""
시장 국면 분류 + 국면별 성과 분해.

국면은 매매 신호가 아니라 '분석용 라벨'이다. 전체 기간 cummax(정점)를 기준으로
사후적으로 구간을 나누므로 hindsight를 쓰지만, 전략의 의사결정에 쓰이지 않으므로
lookahead bias 대상이 아니다 (README 설계 결정 참조).
"""
import pandas as pd
from ..metrics import risk as risk_metrics

BULL = "상승장"
BEAR = "하락장"
RECOVERY = "회복"

# 대표 위기/국면 구간 (벤치마크 데이터 가용 범위 내에서 사전 정의).
# 클릭 한 번으로 해당 구간만 즉시 백테스트하기 위한 프리셋.
PREDEFINED_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008 금융위기": ("2007-10-01", "2009-06-30"),
    "2020 코로나 급락": ("2020-02-01", "2020-06-30"),
    "2022 금리인상기": ("2022-01-01", "2022-12-31"),
}


def classify_regimes(
    benchmark: pd.Series, drawdown_threshold: float = -0.20
) -> pd.Series:
    """
    벤치마크 지수(기본 SPY)를 거래일별 국면 라벨로 분류한다.

    규칙:
      - 정점(누적 최고가) 대비 낙폭이 drawdown_threshold(-20%) 이하 → 하락장(BEAR)
      - 하락장 진입 후 정점을 회복하기 전까지(낙폭 -20%~0%) → 회복(RECOVERY)
      - 그 외(정점 부근/신고가) → 상승장(BULL)

    국면을 전략과 독립적인 벤치마크 기준으로 매김으로써, 같은 시장 국면에서
    서로 다른 전략을 비교할 수 있다.
    """
    peak = benchmark.cummax()
    drawdown = benchmark / peak - 1.0

    # 부동소수점 경계 보정: 정확히 -20%(예: 80/100-1 = -0.199...996)도
    # "임계 도달"로 포함하기 위해 미세 허용오차를 둔다.
    threshold = drawdown_threshold + 1e-9

    labels: list[str] = []
    in_episode = False
    for dd in drawdown:
        if dd <= threshold:
            in_episode = True
            labels.append(BEAR)
        elif in_episode and dd < 0:
            labels.append(RECOVERY)
        else:
            in_episode = False
            labels.append(BULL)

    return pd.Series(labels, index=benchmark.index, name="regime")


def decompose_by_regime(
    portfolio_values: pd.Series, regimes: pd.Series
) -> dict[str, dict[str, float]]:
    """
    포트폴리오 가치를 국면 라벨로 분해해 국면별 성과 지표를 계산한다.

    국면 구간은 비연속적일 수 있으므로, 해당 국면 거래일의 일별 수익률만 모아
    합성 수익곡선을 만든 뒤 누적수익률·MDD·변동성을 계산한다(근사).
    """
    returns = portfolio_values.pct_change().dropna()
    aligned = regimes.reindex(returns.index)

    breakdown: dict[str, dict[str, float]] = {}
    for label in (BULL, BEAR, RECOVERY):
        mask = aligned == label
        n_days = int(mask.sum())
        if n_days == 0:
            continue

        regime_returns = returns[mask]
        equity = (1.0 + regime_returns).cumprod()

        breakdown[label] = {
            "수익률": float(equity.iloc[-1] - 1.0),
            "MDD": risk_metrics.max_drawdown(equity),
            "변동성": risk_metrics.annual_volatility(equity),
            "거래일수": n_days,
        }

    return breakdown
