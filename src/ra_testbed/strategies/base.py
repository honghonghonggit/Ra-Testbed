from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        prices: signal_date까지의 종가만 포함된 DataFrame (엔진이 슬라이싱해서 전달)
        returns: {ticker: weight}, 합계 반드시 1.0
        """

    @staticmethod
    def validate_weights(weights: dict[str, float]) -> None:
        total = sum(weights.values())
        if not (0.9999 < total < 1.0001):
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
