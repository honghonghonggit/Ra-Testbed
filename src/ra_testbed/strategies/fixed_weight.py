import pandas as pd
from .base import Strategy


class EqualWeightStrategy(Strategy):
    """prices의 모든 티커에 동일 비중 배분."""

    def generate_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        tickers = list(prices.columns)
        weight = 1.0 / len(tickers)
        return {t: weight for t in tickers}


class FixedWeightStrategy(Strategy):
    """사용자가 지정한 고정 비중을 유지. 엔진의 rebalance_freq에 따라 주기적으로 리밸런싱."""

    def __init__(self, weights: dict[str, float]):
        Strategy.validate_weights(weights)
        self._weights = weights

    def generate_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        return self._weights
