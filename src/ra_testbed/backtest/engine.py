from dataclasses import dataclass
import pandas as pd
from ..strategies.base import Strategy
from ..data.loader import DataLoader
from ..metrics import risk as risk_metrics


@dataclass
class BacktestResult:
    portfolio_values: pd.Series   # DatetimeIndex, 일별 포트폴리오 가치
    weights_history: pd.DataFrame  # 리밸런싱 실행일 × 티커, 적용 비중
    metrics: dict[str, float]      # CAGR, MDD, Sharpe, Volatility


class BacktestEngine:
    """
    Lookahead Bias 방지 설계:
      signal_date (t):   전략이 prices.loc[:t] 만 받아 목표 비중 계산
      execution_date (t+1): 다음 거래일 시가(Open)로 실제 체결

    이 분리는 코드 레벨에서 구조적으로 강제된다.
    전략 함수는 절대 t+1 이후 데이터에 접근할 수 없다.
    """

    def __init__(
        self,
        strategy: Strategy,
        tickers: list[str],
        start: str,
        end: str,
        initial_capital: float = 1_000_000,
        transaction_cost_bps: float = 10.0,
        rebalance_freq: str = "M",
        risk_free_rate: float = 0.02,
        cache_dir: str = "data/",
    ):
        self.strategy = strategy
        self.tickers = tickers
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.rebalance_freq = rebalance_freq
        self.risk_free_rate = risk_free_rate
        self.cache_dir = cache_dir

    def run(self) -> BacktestResult:
        loader = DataLoader(self.tickers, cache_dir=self.cache_dir)
        close, open_ = loader.load(self.start, self.end)

        dates = close.index
        if len(dates) < 2:
            raise ValueError("Not enough trading days in the specified date range.")

        rebal_signals = self._get_rebal_signal_dates(dates)
        rebal_signals.add(dates[0])  # 첫 거래일은 항상 초기 투자 신호

        cash = float(self.initial_capital)
        holdings: dict[str, float] = {t: 0.0 for t in self.tickers}
        portfolio_vals: list[float] = []
        weights_records: list[dict] = []

        for i, date in enumerate(dates):
            # 전날이 신호일이면 오늘 시가(Open)로 체결
            if i > 0 and dates[i - 1] in rebal_signals:
                signal_date = dates[i - 1]
                exec_prices = open_.loc[date]

                current_value = cash + sum(
                    holdings[t] * exec_prices[t] for t in self.tickers
                )

                # 핵심: signal_date까지의 종가만 전략에 전달 (미래 데이터 차단)
                past_prices = close.loc[:signal_date]
                target_weights = self.strategy.generate_weights(past_prices)
                Strategy.validate_weights(target_weights)

                # 거래비용: 비중 변화에 비례 (매수/매도 절댓값 기준)
                total_cost = sum(
                    abs(current_value * target_weights.get(t, 0.0) - holdings[t] * exec_prices[t])
                    * self.transaction_cost_bps / 10_000
                    for t in self.tickers
                )
                investable = current_value - total_cost

                for t in self.tickers:
                    holdings[t] = (
                        investable * target_weights.get(t, 0.0) / exec_prices[t]
                    )
                cash = 0.0

                weights_records.append({"date": date, **target_weights})

            daily_value = cash + sum(holdings[t] * close.loc[date, t] for t in self.tickers)
            portfolio_vals.append(daily_value)

        portfolio_series = pd.Series(portfolio_vals, index=dates, name="portfolio_value")

        weights_df = (
            pd.DataFrame(weights_records).set_index("date")
            if weights_records
            else pd.DataFrame()
        )

        metrics = {
            "CAGR": risk_metrics.cagr(portfolio_series),
            "MDD": risk_metrics.max_drawdown(portfolio_series),
            "Sharpe": risk_metrics.sharpe_ratio(portfolio_series, self.risk_free_rate),
            "Volatility": risk_metrics.annual_volatility(portfolio_series),
        }

        return BacktestResult(
            portfolio_values=portfolio_series,
            weights_history=weights_df,
            metrics=metrics,
        )

    def _get_rebal_signal_dates(self, dates: pd.DatetimeIndex) -> set:
        if self.rebalance_freq == "D":
            return set(dates)

        freq_map = {"M": "ME", "Q": "QE"}
        if self.rebalance_freq not in freq_map:
            raise ValueError(
                f"Invalid rebalance_freq: '{self.rebalance_freq}'. Choose 'D', 'M', or 'Q'."
            )

        dummy = pd.Series(1, index=dates)
        signal_dates: set = set()
        for _, group in dummy.resample(freq_map[self.rebalance_freq]):
            if not group.empty:
                signal_dates.add(group.index[-1])
        return signal_dates
