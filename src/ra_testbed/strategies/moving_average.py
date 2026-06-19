import pandas as pd
from .base import Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    단기 MA > 장기 MA인 자산(상승추세)에 동일 비중 배분.
    데이터 부족 또는 전 자산 하락추세 시 균등 배분으로 폴백.
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        tickers = list(prices.columns)
        n = len(tickers)

        if len(prices) < self.long_window:
            return {t: 1.0 / n for t in tickers}

        bullish = [
            t for t in tickers
            if prices[t].rolling(self.short_window).mean().iloc[-1]
            > prices[t].rolling(self.long_window).mean().iloc[-1]
        ]

        if not bullish:
            # 전 자산 하락추세 — 균등 배분 (현금 포지션 없음, Phase1 단순화)
            return {t: 1.0 / n for t in tickers}

        weight = 1.0 / len(bullish)
        return {t: weight if t in bullish else 0.0 for t in tickers}
