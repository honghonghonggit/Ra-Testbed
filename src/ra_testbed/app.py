import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ra_testbed.strategies import (
    EqualWeightStrategy,
    FixedWeightStrategy,
    MovingAverageCrossStrategy,
    strategy_from_config,
)
from ra_testbed.backtest.engine import BacktestEngine
from ra_testbed.backtest.regimes import (
    classify_regimes,
    decompose_by_regime,
    PREDEFINED_SCENARIOS,
    BEAR,
    RECOVERY,
)
from ra_testbed.data.loader import DataLoader
from ra_testbed.report.reliability import evaluate as evaluate_reliability, PASS, WARN

st.set_page_config(page_title="RA Testbed", layout="wide")
st.title("로보어드바이저 신뢰성 검증 테스트베드")
st.caption("투자 전략의 과거 성과를 백테스트하여 신뢰성 지표를 자동 계산합니다.")

CACHE_DIR = "data/"

ASSET_PRESETS = {
    "미국 ETF (SPY/TLT/GLD)": "SPY,TLT,GLD",
    "국내 ETF (KODEX 주식/채권/금)": "069500.KS,114260.KS,132030.KS",
    "직접 입력": "SPY,TLT,GLD",
}

JSON_PLACEHOLDER = (
    '{\n'
    '  "type": "fixed_weight",\n'
    '  "weights": {"SPY": 0.6, "TLT": 0.4}\n'
    '}'
)


# ── 사이드바 설정 ──────────────────────────────────────────────
with st.sidebar:
    st.header("자산 유니버스")
    preset = st.selectbox("프리셋", list(ASSET_PRESETS))
    if preset == "직접 입력":
        tickers_input = st.text_input("티커 (쉼표 구분)", value="SPY,TLT,GLD")
    else:
        tickers_input = ASSET_PRESETS[preset]
        st.caption(f"티커: {tickers_input}")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.header("전략")
    strategy_mode = st.radio("입력 방식", ["프리셋 전략", "JSON 직접 입력"])

    strategy_name = None
    json_text = ""
    custom_weights: dict[str, float] = {}
    if strategy_mode == "프리셋 전략":
        strategy_name = st.selectbox(
            "전략",
            ["균등 비중 (Equal Weight)", "고정 비중 (Custom)", "이동평균 교차 (MA Cross)"],
        )
        if strategy_name == "고정 비중 (Custom)" and tickers:
            st.caption("비중 설정")
            for ticker in tickers:
                custom_weights[ticker] = st.slider(
                    ticker, 0.0, 1.0, round(1.0 / len(tickers), 2), 0.05
                )
    else:
        json_text = st.text_area("전략 JSON", value=JSON_PLACEHOLDER, height=160)
        st.caption("지원 type: equal_weight · fixed_weight · ma_cross")

    st.header("백테스트 기간")
    col1, col2 = st.columns(2)
    with col1:
        start_input = st.date_input("시작일", value=pd.Timestamp("2010-01-01"))
    with col2:
        end_input = st.date_input("종료일", value=pd.Timestamp("2023-12-31"))

    st.header("엔진 설정")
    rebalance_freq = st.selectbox(
        "리밸런싱 주기",
        ["M", "Q", "D"],
        format_func=lambda x: {"M": "월별", "Q": "분기별", "D": "매일"}[x],
    )
    initial_capital = st.number_input("초기 자본", value=1_000_000, step=100_000)
    transaction_cost_bps = st.slider("거래비용 (bps)", 0, 50, 10)

    run_btn = st.button("백테스트 실행", type="primary", use_container_width=True)

    st.header("시나리오 (대표 위기 구간)")
    st.caption("클릭하면 해당 구간만 즉시 평가합니다.")
    scenario_clicked = None
    for name in PREDEFINED_SCENARIOS:
        if st.button(name, use_container_width=True):
            scenario_clicked = name


# ── 전략 생성 ──────────────────────────────────────────────────
def build_strategy():
    """현재 UI 상태로부터 (strategy, error_message) 튜플 반환."""
    if strategy_mode == "JSON 직접 입력":
        try:
            cfg = json.loads(json_text)
        except json.JSONDecodeError as e:
            return None, f"JSON 파싱 오류: {e}"
        try:
            return strategy_from_config(cfg), None
        except ValueError as e:
            return None, str(e)

    if strategy_name == "균등 비중 (Equal Weight)":
        return EqualWeightStrategy(), None
    if strategy_name == "고정 비중 (Custom)":
        total_w = sum(custom_weights.values())
        if total_w < 0.01:
            return None, "비중 합계가 너무 작습니다."
        normalized = {t: w / total_w for t, w in custom_weights.items()}
        try:
            return FixedWeightStrategy(normalized), None
        except ValueError as e:
            return None, str(e)
    return MovingAverageCrossStrategy(short_window=50, long_window=200), None


# ── 국면 음영 구간 계산 ────────────────────────────────────────
def regime_segments(regimes: pd.Series):
    """연속된 동일 국면 구간을 (label, start_date, end_date) 리스트로 반환."""
    segments = []
    if regimes.empty:
        return segments
    cur_label = regimes.iloc[0]
    seg_start = prev = regimes.index[0]
    for idx, label in regimes.items():
        if label != cur_label:
            segments.append((cur_label, seg_start, prev))
            cur_label, seg_start = label, idx
        prev = idx
    segments.append((cur_label, seg_start, prev))
    return segments


# ── 실행 트리거 결정 ───────────────────────────────────────────
start, end = str(start_input), str(end_input)
scenario_label = None
if scenario_clicked:
    s, e = PREDEFINED_SCENARIOS[scenario_clicked]
    start, end, scenario_label = s, e, scenario_clicked
    do_run = True
else:
    do_run = run_btn

if not do_run:
    st.info("좌측에서 전략을 설정하고 **백테스트 실행** 또는 시나리오 버튼을 눌러주세요.")
    st.stop()

if not tickers:
    st.error("티커를 입력해주세요.")
    st.stop()

strategy, err = build_strategy()
if err:
    st.error(err)
    st.stop()

if scenario_label:
    st.subheader(f"시나리오: {scenario_label}  ({start} ~ {end})")

with st.spinner("데이터 로딩 및 백테스트 실행 중..."):
    try:
        engine = BacktestEngine(
            strategy=strategy,
            tickers=tickers,
            start=start,
            end=end,
            initial_capital=float(initial_capital),
            transaction_cost_bps=float(transaction_cost_bps),
            rebalance_freq=rebalance_freq,
            cache_dir=CACHE_DIR,
        )
        result = engine.run()
    except Exception as e:
        msg = str(e)
        kr_tickers = [t for t in tickers if t.endswith(".KS")]
        if "Not enough trading days" in msg and kr_tickers and pd.Timestamp(start) < pd.Timestamp("2010-01-01"):
            st.error(
                f"**데이터 없음**: 선택한 국내 ETF({', '.join(kr_tickers)})는 "
                f"2010년 이전 데이터가 제공되지 않습니다. "
                f"**2008 금융위기** 시나리오는 미국 ETF 프리셋(SPY/TLT/GLD)으로 변경 후 실행해주세요."
            )
        else:
            st.error(f"백테스트 오류: {msg}")
        st.stop()

pv = result.portfolio_values

# ── 리스크 지표 카드 ──
m = result.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", f"{m['CAGR']:.2%}")
c2.metric("MDD", f"{m['MDD']:.2%}")
c3.metric("Sharpe", f"{m['Sharpe']:.2f}")
c4.metric("연간 변동성", f"{m['Volatility']:.2%}")

# ── 신뢰성 리포트 ──
st.subheader("신뢰성 리포트")
report = evaluate_reliability(m)
if report.overall == PASS:
    st.success("종합 판정: ✅ 통과 (PASS)")
else:
    st.warning(f"종합 판정: ⚠️ 경고 (WARN) — {len(report.warnings)}개 항목")

report_rows = []
for chk in report.checks:
    badge = "✅ 통과" if chk.status == PASS else "⚠️ 경고"
    report_rows.append(
        {"지표": chk.name, "값": f"{chk.value:.2%}" if chk.name != "Sharpe" else f"{chk.value:.2f}",
         "판정": badge, "근거": chk.rationale}
    )
st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)

# ── 국면 분류 (벤치마크 = 첫 티커) ──
benchmark_ticker = tickers[0]
regimes = None
try:
    bench_close, _ = DataLoader([benchmark_ticker], cache_dir=CACHE_DIR).load(start, end)
    regimes = classify_regimes(bench_close[benchmark_ticker]).reindex(pv.index).ffill()
except Exception:
    regimes = None

# ── 포트폴리오 가치 곡선 (국면 음영) ──
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=pv.index, y=pv.values, mode="lines",
        name="포트폴리오 가치", line=dict(color="#2563eb", width=2),
    )
)
if regimes is not None:
    shade = {BEAR: "rgba(239,68,68,0.13)", RECOVERY: "rgba(245,158,11,0.13)"}
    for label, s_date, e_date in regime_segments(regimes.dropna()):
        if label in shade:
            fig.add_vrect(x0=s_date, x1=e_date, fillcolor=shade[label],
                          line_width=0, layer="below")
fig.update_layout(
    title=f"포트폴리오 가치 추이 (음영: 하락장🔴 / 회복🟠, 기준={benchmark_ticker})",
    xaxis_title="날짜", yaxis_title="가치 (원)",
    hovermode="x unified", height=480,
)
st.plotly_chart(fig, use_container_width=True)

# ── 국면별 성과 분해 ──
if regimes is not None:
    breakdown = decompose_by_regime(pv, regimes)
    if breakdown:
        st.subheader("국면별 성과 분해")
        st.caption(f"국면은 벤치마크({benchmark_ticker}) 정점 대비 -20% 규칙으로 분류 (분석용 사후 라벨).")
        rows = []
        for label, stats in breakdown.items():
            rows.append({
                "국면": label,
                "거래일수": stats["거래일수"],
                "구간 수익률": f"{stats['수익률']:.2%}",
                "구간 MDD": f"{stats['MDD']:.2%}",
                "구간 변동성": f"{stats['변동성']:.2%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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
                    mode="lines", name=ticker,
                    stackgroup="one", groupnorm="percent",
                )
            )
    fig2.update_layout(xaxis_title="날짜", yaxis_title="비중 (%)", height=300)
    st.plotly_chart(fig2, use_container_width=True)
