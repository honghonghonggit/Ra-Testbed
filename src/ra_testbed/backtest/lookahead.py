"""
Lookahead bias 가드레일 — 전략이 미래 데이터를 참조하는지 코드로 검증한다.

Phase 1의 백테스트 엔진은 `close.loc[:signal_date]`로 데이터를 슬라이싱해 전략에
넘김으로써 lookahead를 '구조적으로 방지'한다. 이 가드레일은 그 방어가 없는 상황,
즉 전략에게 미래까지 포함된 패널을 통째로 넘겼을 때 전략이 스스로 as_of 시점까지만
사용하는지를 '검증'한다 (방지와 검증의 분리).

원리 — 차등 교란(differential poisoning):
  동일한 패널의 as_of 이후 행만 강하게 교란한 두 입력(real / poisoned)을 만든다.
  as_of 이하 행은 완전히 동일하다. 전략이 as_of까지만 본다면 두 입력에 대한 출력이
  같아야 한다(CLEAN). 출력이 달라졌다면 전략이 as_of 이후 행을 참조한 것이다(DETECTED).
  입력 shape을 고정한 채 미래 구간만 바꾸므로 '마지막 행=현재' 모호성이 없고
  이동평균 같은 추세 전략에도 공정하다.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

CLEAN = "CLEAN"
DETECTED = "DETECTED"

# 감사 대상 계약: (prices, as_of) -> {ticker: weight}. 전략은 as_of 이후 행을 쓰면 안 된다.
AuditFn = Callable[[pd.DataFrame, pd.Timestamp], dict[str, float]]


@dataclass
class LookaheadReport:
    status: str            # CLEAN | DETECTED
    as_of: pd.Timestamp
    max_weight_diff: float
    detail: str


def detect_lookahead(
    audit_fn: AuditFn,
    prices: pd.DataFrame,
    as_of,
    future_horizon: int = 20,
    tol: float = 1e-9,
    seed: int = 0,
) -> LookaheadReport:
    """전략이 as_of 이후 데이터를 참조하는지 차등 교란으로 검증한다."""
    as_of = pd.Timestamp(as_of)
    pos = prices.index.searchsorted(as_of, side="right")  # as_of 이하 행 개수
    if pos == 0:
        raise ValueError("as_of가 데이터 시작보다 빠릅니다.")
    if pos >= len(prices):
        raise ValueError("미래 구간이 없습니다. as_of 이후 데이터가 필요합니다.")

    end = min(pos + future_horizon, len(prices))
    if end <= pos:
        raise ValueError("future_horizon으로 확보되는 미래 구간이 없습니다.")

    real = prices.iloc[:end].copy()
    poisoned = real.copy()
    # as_of 이후 행만 셀별 독립 난수로 교란 (as_of 이하 행은 두 입력이 완전히 동일).
    # 셀마다 다른 배수를 곱해 레벨·추세·자산간 순위를 모두 흔들므로, 미래에 대한
    # 어떤 함수적 의존성이든 출력 변화로 드러난다. 균일 배수와 달리 순위가 보존되지 않는다.
    rng = np.random.default_rng(seed)
    future = poisoned.iloc[pos:]
    factors = rng.uniform(2.0, 10.0, size=future.shape)
    poisoned.iloc[pos:] = future.to_numpy() * factors

    w_real = audit_fn(real, as_of)
    w_poison = audit_fn(poisoned, as_of)

    tickers = set(w_real) | set(w_poison)
    max_diff = max(
        (abs(w_real.get(t, 0.0) - w_poison.get(t, 0.0)) for t in tickers),
        default=0.0,
    )

    if max_diff > tol:
        return LookaheadReport(
            status=DETECTED,
            as_of=as_of,
            max_weight_diff=float(max_diff),
            detail=(
                f"as_of({as_of.date()}) 이후 데이터만 바꿨는데 비중이 최대 "
                f"{max_diff:.4f} 변했습니다 — 전략이 미래를 참조합니다."
            ),
        )
    return LookaheadReport(
        status=CLEAN,
        as_of=as_of,
        max_weight_diff=float(max_diff),
        detail=f"as_of({as_of.date()}) 이후 데이터를 바꿔도 비중이 동일합니다 — 미래 참조 없음.",
    )


# ── 데모용 감사 전략 ───────────────────────────────────────────────
def wrap_strategy_as_audit(strategy) -> AuditFn:
    """기존 Strategy를 as_of로 슬라이싱해 감사 계약에 맞춘다.

    엔진과 동일하게 prices.loc[:as_of]만 넘기므로 항상 CLEAN —
    Phase 1의 슬라이싱 방어가 옳음을 보여준다.
    """
    def audit_fn(prices: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
        return strategy.generate_weights(prices.loc[:as_of])

    return audit_fn


def lookahead_cheating_audit(prices: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """의도적으로 미래를 훔쳐보는 컨닝 전략 (가드레일 시연용).

    as_of를 무시하고 패널의 '미래' 마지막 구간 모멘텀에 비례해 비중을 배분한다.
    출력이 미래 봉 값에 연속적으로 의존하므로, 가드레일이 미래 구간을 교란하면
    반드시 비중이 달라져 DETECTED로 잡힌다.
    """
    tickers = list(prices.columns)
    window = min(5, len(prices) - 1)
    # ← 미래 마지막 봉(iloc[-1])과 그 직전 봉을 참조 (lookahead!)
    recent = (prices.iloc[-1] / prices.iloc[-1 - window] - 1.0).clip(lower=0.0)
    total = float(recent.sum())
    if total <= 0:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
    return {t: float(recent[t] / total) for t in tickers}
