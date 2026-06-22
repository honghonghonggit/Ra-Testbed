"""
표준화된 신뢰성 리포트.

백테스트 리스크 지표를 사전 정의된 임계값과 대조해 지표별 통과(PASS)/경고(WARN)를
판정한다. 임계값과 그 근거는 README 설계 결정에 명시한다(코드에서 조정 가능).
"""
from dataclasses import dataclass

PASS = "PASS"
WARN = "WARN"

# 비교 방향:
#   "min" → 값이 threshold보다 작으면 경고 (낮을수록 위험한 지표)
#   "max" → 값이 threshold보다 크면 경고 (높을수록 위험한 지표)
DEFAULT_THRESHOLDS: dict[str, dict] = {
    "MDD": {
        "threshold": -0.40,
        "direction": "min",
        "rationale": "최대 낙폭 -40%는 회복에 +67%가 필요한 수준 — 투자자 이탈 임계로 간주.",
    },
    "Sharpe": {
        "threshold": 0.5,
        "direction": "min",
        "rationale": "샤프비율 0.5 미만은 위험 대비 보상이 불충분 (통상 1.0 이상을 양호로 봄).",
    },
    "Volatility": {
        "threshold": 0.25,
        "direction": "max",
        "rationale": "연간 변동성 25% 초과는 주식 100% 포트폴리오 수준 이상의 위험.",
    },
    "CAGR": {
        "threshold": 0.0,
        "direction": "min",
        "rationale": "연평균 수익률이 음수면 장기 원금 손실.",
    },
}


@dataclass
class ReliabilityCheck:
    name: str
    value: float
    threshold: float
    status: str       # PASS | WARN
    rationale: str


@dataclass
class ReliabilityReport:
    checks: list[ReliabilityCheck]
    overall: str      # PASS | WARN

    @property
    def warnings(self) -> list[ReliabilityCheck]:
        return [c for c in self.checks if c.status == WARN]


def evaluate(
    metrics: dict[str, float], thresholds: dict[str, dict] = DEFAULT_THRESHOLDS
) -> ReliabilityReport:
    """리스크 지표 dict를 임계값과 대조해 신뢰성 리포트를 생성한다."""
    checks: list[ReliabilityCheck] = []

    for name, spec in thresholds.items():
        if name not in metrics:
            continue
        value = metrics[name]
        threshold = spec["threshold"]

        if spec["direction"] == "min":
            status = WARN if value < threshold else PASS
        else:  # "max"
            status = WARN if value > threshold else PASS

        checks.append(
            ReliabilityCheck(
                name=name,
                value=value,
                threshold=threshold,
                status=status,
                rationale=spec["rationale"],
            )
        )

    overall = WARN if any(c.status == WARN for c in checks) else PASS
    return ReliabilityReport(checks=checks, overall=overall)
