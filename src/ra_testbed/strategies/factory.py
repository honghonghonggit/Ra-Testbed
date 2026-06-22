"""
선언적 설정(dict/JSON)으로부터 전략 인스턴스를 생성한다.

사용자가 임의의 Python 코드를 올리는 대신, 안전한 선언적 포맷으로 전략을 끼워 넣는다.
코드 실행이 없으므로 Streamlit Cloud 배포 환경에서도 안전하다.

지원 포맷:
    {"type": "equal_weight"}
    {"type": "fixed_weight", "weights": {"SPY": 0.6, "TLT": 0.4}}
    {"type": "ma_cross", "short_window": 50, "long_window": 200}
"""
from .base import Strategy
from .fixed_weight import EqualWeightStrategy, FixedWeightStrategy
from .moving_average import MovingAverageCrossStrategy

SUPPORTED_TYPES = ("equal_weight", "fixed_weight", "ma_cross")


def strategy_from_config(config: dict) -> Strategy:
    """선언적 설정 dict를 Strategy 인스턴스로 변환한다.

    알 수 없는 type이나 잘못된 파라미터는 명확한 ValueError를 발생시킨다.
    """
    if not isinstance(config, dict):
        raise ValueError("전략 설정은 dict(JSON 객체)여야 합니다.")
    if "type" not in config:
        raise ValueError("전략 설정에 'type' 키가 필요합니다.")

    strategy_type = config["type"]

    if strategy_type == "equal_weight":
        return EqualWeightStrategy()

    if strategy_type == "fixed_weight":
        weights = config.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("fixed_weight 전략에는 비어있지 않은 'weights' 객체가 필요합니다.")
        if not all(isinstance(w, (int, float)) for w in weights.values()):
            raise ValueError("'weights'의 모든 값은 숫자여야 합니다.")
        # 합계 1.0 검증은 FixedWeightStrategy 생성자가 수행
        return FixedWeightStrategy({t: float(w) for t, w in weights.items()})

    if strategy_type == "ma_cross":
        short = config.get("short_window", 50)
        long = config.get("long_window", 200)
        if not isinstance(short, (int, float)) or not isinstance(long, (int, float)):
            raise ValueError("'short_window'와 'long_window'는 숫자여야 합니다.")
        # short < long 검증은 MovingAverageCrossStrategy 생성자가 수행
        return MovingAverageCrossStrategy(short_window=int(short), long_window=int(long))

    raise ValueError(
        f"알 수 없는 전략 type: '{strategy_type}'. "
        f"지원: {', '.join(SUPPORTED_TYPES)}"
    )
