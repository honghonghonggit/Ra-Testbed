import pandas as pd
import numpy as np


def cagr(portfolio_values: pd.Series) -> float:
    """연평균 복합 수익률 (Compound Annual Growth Rate)."""
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1.0 / years) - 1.0


def max_drawdown(portfolio_values: pd.Series) -> float:
    """최대 낙폭 (Maximum Drawdown). 음수 반환 (예: -0.35 = -35%)."""
    rolling_max = portfolio_values.cummax()
    drawdown = portfolio_values / rolling_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(portfolio_values: pd.Series, risk_free_rate: float = 0.02) -> float:
    """연간화 샤프 비율. 252 거래일 기준."""
    daily_returns = portfolio_values.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / 252
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def annual_volatility(portfolio_values: pd.Series) -> float:
    """연간화 변동성 (일별 수익률 표준편차 × √252)."""
    daily_returns = portfolio_values.pct_change().dropna()
    return float(daily_returns.std() * np.sqrt(252))
