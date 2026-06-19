import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ra_testbed.strategies import (
    EqualWeightStrategy,
    FixedWeightStrategy,
    MovingAverageCrossStrategy,
)
from ra_testbed.backtest.engine import BacktestEngine

st.set_page_config(page_title="RA Testbed", layout="wide")
st.title("로보어드바이저 신뢰성 검증 테스트베드")
st.caption("투자 전략의 과거 성과를 백테스트하여 신뢰성 지표를 자동 계산합니다.")

# ── 사이드바 설정 ──────────────────────────────────────────────
with st.sidebar:
    st.header("전략 설정")

    strategy_name = st.selectbox(
        "전략",
        ["균등 비중 (Equal Weight)", "고정 비중 (Custom)", "이동평균 교차 (MA Cross)"],
    )

    tickers_input = st.text_input("티커 (쉼표 구분)", value="SPY,TLT,GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.subheader("백테스트 기간")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("시작일", value=pd.Timestamp("2010-01-01"))
    with col2:
        end = st.date_input("종료일", value=pd.Timestamp("2023-12-31"))

    st.subheader("엔진 설정")
    rebalance_freq = st.selectbox(
        "리밸런싱 주기",
        ["M", "Q", "D"],
        format_func=lambda x: {"M": "월별", "Q": "분기별", "D": "매일"}[x],
    )
    initial_capital = st.number_input("초기 자본", value=1_000_000, step=100_000)
    transaction_cost_bps = st.slider("거래비용 (bps)", 0, 50, 10)

    custom_weights: dict[str, float] = {}
    if strategy_name == "고정 비중 (Custom)" and tickers:
        st.subheader("비중 설정")
        for ticker in tickers:
            custom_weights[ticker] = st.slider(ticker, 0.0, 1.0, round(1.0 / len(tickers), 2), 0.05)

    run_btn = st.button("백테스트 실행", type="primary", use_container_width=True)

# ── 메인 ───────────────────────────────────────────────────────
if run_btn:
    if not tickers:
        st.error("티커를 입력해주세요.")
        st.stop()

    if strategy_name == "균등 비중 (Equal Weight)":
        strategy = EqualWeightStrategy()
    elif strategy_name == "고정 비중 (Custom)":
        total_w = sum(custom_weights.values())
        if total_w < 0.01:
            st.error("비중 합계가 너무 작습니다.")
            st.stop()
        normalized = {t: w / total_w for t, w in custom_weights.items()}
        try:
            strategy = FixedWeightStrategy(normalized)
        except ValueError as e:
            st.error(str(e))
            st.stop()
    else:
        strategy = MovingAverageCrossStrategy(short_window=50, long_window=200)

    with st.spinner("데이터 로딩 및 백테스트 실행 중..."):
        try:
            engine = BacktestEngine(
                strategy=strategy,
                tickers=tickers,
                start=str(start),
                end=str(end),
                initial_capital=float(initial_capital),
                transaction_cost_bps=float(transaction_cost_bps),
                rebalance_freq=rebalance_freq,
            )
            result = engine.run()
        except Exception as e:
            st.error(f"백테스트 오류: {e}")
            st.stop()

    # ── 리스크 지표 카드 ──
    m = result.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{m['CAGR']:.2%}")
    c2.metric("MDD", f"{m['MDD']:.2%}")
    c3.metric("Sharpe", f"{m['Sharpe']:.2f}")
    c4.metric("연간 변동성", f"{m['Volatility']:.2%}")

    # ── 포트폴리오 가치 곡선 ──
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.portfolio_values.index,
            y=result.portfolio_values.values,
            mode="lines",
            name="포트폴리오 가치",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        title="포트폴리오 가치 추이",
        xaxis_title="날짜",
        yaxis_title="가치 (원)",
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 비중 변화 (리밸런싱 이력) ──
    if not result.weights_history.empty:
        st.subheader("리밸런싱 비중 추이")
        fig2 = go.Figure()
        for ticker in tickers:
            if ticker in result.weights_history.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=result.weights_history.index,
                        y=result.weights_history[ticker],
                        mode="lines",
                        name=ticker,
                        stackgroup="one",
                        groupnorm="percent",
                    )
                )
        fig2.update_layout(
            xaxis_title="날짜",
            yaxis_title="비중 (%)",
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)
