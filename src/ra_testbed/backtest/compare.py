"""
여러 전략을 동일한 조건(자산·기간·엔진 설정)으로 백테스트해 한 번에 비교한다.

엔진(BacktestEngine)을 전략 수만큼 그대로 호출할 뿐, 엔진 자체는 수정하지 않는다.
동일한 데이터/거래비용/리밸런싱 규칙 아래에서 평가하므로 결과를 공정하게 비교할 수 있다.
"""
from .engine import BacktestEngine, BacktestResult
from ..strategies.base import Strategy


def compare_strategies(
    strategies: dict[str, Strategy],
    *,
    tickers: list[str],
    start: str,
    end: str,
    **engine_kwargs,
) -> dict[str, BacktestResult]:
    """이름→전략 매핑을 받아 각각 백테스트하고 이름→결과 매핑을 반환한다.

    모든 전략은 동일한 tickers/start/end/engine_kwargs로 실행되어
    같은 잣대 위에서 비교된다.
    """
    if not strategies:
        raise ValueError("비교할 전략이 최소 1개 필요합니다.")

    results: dict[str, BacktestResult] = {}
    for name, strategy in strategies.items():
        results[name] = BacktestEngine(
            strategy=strategy,
            tickers=tickers,
            start=start,
            end=end,
            **engine_kwargs,
        ).run()
    return results
