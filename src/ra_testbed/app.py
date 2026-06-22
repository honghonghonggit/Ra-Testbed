import sys
import pathlib

# 배포 환경(Streamlit Cloud)에서 설치본(site-packages) staleness와 무관하게 항상
# 라이브 소스를 import하도록 강제한다. (this file: src/ra_testbed/app.py → _SRC=src/)
# 1) src/를 경로 맨 앞에 둔다. 2) startup 때 stale 설치본이 먼저 캐시됐을 수 있으므로
#    ra_testbed 모듈 캐시를 비워 src/에서 재해석되게 한다.
_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
for _m in [m for m in sys.modules if m == "ra_testbed" or m.startswith("ra_testbed.")]:
    del sys.modules[_m]

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
from ra_testbed.backtest.compare import compare_strategies
from ra_testbed.backtest.lookahead import (
    detect_lookahead,
    wrap_strategy_as_audit,
    lookahead_cheating_audit,
    CLEAN,
)
from ra_testbed.backtest.regimes import (
    classify_regimes,
    decompose_by_regime,
    PREDEFINED_SCENARIOS,
    BEAR,
    RECOVERY,
)
from ra_testbed.data.loader import DataLoader
from ra_testbed.report.reliability import evaluate as evaluate_reliability, PASS

st.set_page_config(page_title="RA Testbed", layout="wide")
st.title("로보어드바이저 신뢰성 검증 테스트베드")
st.caption("투자 전략의 과거 성과를 백테스트하여 신뢰성 지표를 자동 계산합니다.")

# 번들된 시세 캐시(data/)는 레포 루트 기준 절대 경로로 참조한다 — 배포 환경의
# 작업 디렉터리가 레포 루트가 아니어도 항상 찾도록. (_SRC=src/ → 레포 루트=_SRC.parent)
CACHE_DIR = str(_SRC.parent / "data")

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

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]


# ── 공통 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.header("모드")
    app_mode = st.radio(
        "분석 모드",
        ["단일 전략 백테스트", "전략 비교", "Lookahead 가드레일"],
        label_visibility="collapsed",
    )

    st.header("자산 유니버스")
    preset = st.selectbox("프리셋", list(ASSET_PRESETS))
    if preset == "직접 입력":
        tickers_input = st.text_input("티커 (쉼표 구분)", value="SPY,TLT,GLD")
    else:
        tickers_input = ASSET_PRESETS[preset]
        st.caption(f"티커: {tickers_input}")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

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

start, end = str(start_input), str(end_input)
engine_kwargs = dict(
    initial_capital=float(initial_capital),
    transaction_cost_bps=float(transaction_cost_bps),
    rebalance_freq=rebalance_freq,
    cache_dir=CACHE_DIR,
)


# ── 공용 헬퍼 ──────────────────────────────────────────────────
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


def load_regimes(benchmark_ticker, index, period_start, period_end):
    """벤치마크 종가로 국면을 분류해 주어진 인덱스에 맞춰 반환 (실패 시 None)."""
    try:
        bench_close, _ = DataLoader([benchmark_ticker], cache_dir=CACHE_DIR).load(period_start, period_end)
        return classify_regimes(bench_close[benchmark_ticker]).reindex(index).ffill()
    except Exception:
        return None


def render_reliability(metrics: dict):
    report = evaluate_reliability(metrics)
    if report.overall == PASS:
        st.success("종합 판정: ✅ 통과 (PASS)")
    else:
        st.warning(f"종합 판정: ⚠️ 경고 (WARN) — {len(report.warnings)}개 항목")
    rows = []
    for chk in report.checks:
        badge = "✅ 통과" if chk.status == PASS else "⚠️ 경고"
        val = f"{chk.value:.2%}" if chk.name != "Sharpe" else f"{chk.value:.2f}"
        rows.append({"지표": chk.name, "값": val, "판정": badge, "근거": chk.rationale})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def friendly_engine_error(e: Exception):
    """엔진 예외를 친절한 메시지로 변환해 표시하고 중단."""
    msg = str(e)
    kr_tickers = [t for t in tickers if t.endswith(".KS")]
    if "Not enough trading days" in msg and kr_tickers and pd.Timestamp(start) < pd.Timestamp("2010-01-01"):
        st.error(
            f"**데이터 없음**: 선택한 국내 ETF({', '.join(kr_tickers)})는 "
            f"2010년 이전 데이터가 제공되지 않습니다. "
            f"**2008 금융위기** 시나리오는 미국 ETF 프리셋(SPY/TLT/GLD)으로 변경 후 실행해주세요."
        )
    elif "Not enough trading days" in msg:
        st.error(
            f"**시세 데이터를 불러오지 못했습니다** ({', '.join(tickers)}). "
            f"기본 프리셋(미국·국내 ETF)은 시세가 레포에 포함돼 항상 동작합니다. "
            f"직접 입력한 티커는 외부 데이터 조회가 필요하며, 배포 환경에서 일시적으로 제한될 수 있습니다."
        )
    else:
        st.error(f"백테스트 오류: {msg}")
    st.stop()


def concentrated_weights(ts: list[str]) -> dict[str, float]:
    """첫 자산 60%, 나머지 40% 균등 분배."""
    if len(ts) == 1:
        return {ts[0]: 1.0}
    rest = 0.4 / (len(ts) - 1)
    return {t: (0.6 if i == 0 else rest) for i, t in enumerate(ts)}


# ══════════════════════════════════════════════════════════════
# 모드 1: 단일 전략 백테스트
# ══════════════════════════════════════════════════════════════
def render_single():
    with st.sidebar:
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

        run_btn = st.button("백테스트 실행", type="primary", use_container_width=True)

        st.header("시나리오 (대표 위기 구간)")
        st.caption("클릭하면 해당 구간만 즉시 평가합니다.")
        scenario_clicked = None
        for name in PREDEFINED_SCENARIOS:
            if st.button(name, use_container_width=True):
                scenario_clicked = name

    def build_strategy():
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

    # 실행 트리거
    run_start, run_end, scenario_label = start, end, None
    if scenario_clicked:
        run_start, run_end = PREDEFINED_SCENARIOS[scenario_clicked]
        scenario_label = scenario_clicked
        do_run = True
    else:
        do_run = run_btn

    if not do_run:
        st.info("좌측에서 전략을 설정하고 **백테스트 실행** 또는 시나리오 버튼을 눌러주세요.")
        return
    if not tickers:
        st.error("티커를 입력해주세요.")
        return

    strategy, err = build_strategy()
    if err:
        st.error(err)
        return

    if scenario_label:
        st.subheader(f"시나리오: {scenario_label}  ({run_start} ~ {run_end})")

    with st.spinner("데이터 로딩 및 백테스트 실행 중..."):
        try:
            result = BacktestEngine(
                strategy=strategy, tickers=tickers, start=run_start, end=run_end, **engine_kwargs
            ).run()
        except Exception as e:
            friendly_engine_error(e)

    pv = result.portfolio_values
    m = result.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{m['CAGR']:.2%}")
    c2.metric("MDD", f"{m['MDD']:.2%}")
    c3.metric("Sharpe", f"{m['Sharpe']:.2f}")
    c4.metric("연간 변동성", f"{m['Volatility']:.2%}")

    st.subheader("신뢰성 리포트")
    render_reliability(m)

    # 국면 분류 (벤치마크 = 첫 티커) — 시나리오 구간에선 run_start/run_end 기준
    benchmark_ticker = tickers[0]
    regimes = load_regimes(benchmark_ticker, pv.index, run_start, run_end)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pv.index, y=pv.values, mode="lines",
        name="포트폴리오 가치", line=dict(color="#2563eb", width=2),
    ))
    if regimes is not None:
        shade = {BEAR: "rgba(239,68,68,0.13)", RECOVERY: "rgba(245,158,11,0.13)"}
        for label, s_date, e_date in regime_segments(regimes.dropna()):
            if label in shade:
                fig.add_vrect(x0=s_date, x1=e_date, fillcolor=shade[label], line_width=0, layer="below")
    fig.update_layout(
        title=f"포트폴리오 가치 추이 (음영: 하락장🔴 / 회복🟠, 기준={benchmark_ticker})",
        xaxis_title="날짜", yaxis_title="가치 (원)", hovermode="x unified", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    if regimes is not None:
        breakdown = decompose_by_regime(pv, regimes)
        if breakdown:
            st.subheader("국면별 성과 분해")
            st.caption(f"국면은 벤치마크({benchmark_ticker}) 정점 대비 -20% 규칙으로 분류 (분석용 사후 라벨).")
            rows = [{
                "국면": label, "거래일수": s["거래일수"],
                "구간 수익률": f"{s['수익률']:.2%}", "구간 MDD": f"{s['MDD']:.2%}",
                "구간 변동성": f"{s['변동성']:.2%}",
            } for label, s in breakdown.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if not result.weights_history.empty:
        st.subheader("리밸런싱 비중 추이")
        fig2 = go.Figure()
        for ticker in tickers:
            if ticker in result.weights_history.columns:
                fig2.add_trace(go.Scatter(
                    x=result.weights_history.index, y=result.weights_history[ticker],
                    mode="lines", name=ticker, stackgroup="one", groupnorm="percent",
                ))
        fig2.update_layout(xaxis_title="날짜", yaxis_title="비중 (%)", height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 모드 2: 전략 비교
# ══════════════════════════════════════════════════════════════
def render_compare():
    available = {
        "균등 비중": lambda: EqualWeightStrategy(),
        "첫 자산 집중 (60/40)": lambda: FixedWeightStrategy(concentrated_weights(tickers)),
        "이동평균 교차 (50/200)": lambda: MovingAverageCrossStrategy(50, 200),
    }
    with st.sidebar:
        st.header("비교할 전략")
        chosen = st.multiselect(
            "전략 선택 (2개 이상 권장)",
            list(available),
            default=list(available),
        )
        run_btn = st.button("비교 실행", type="primary", use_container_width=True)

    if not run_btn:
        st.info("좌측에서 비교할 전략을 선택하고 **비교 실행**을 눌러주세요. 동일한 자산·기간·거래비용 위에서 전략을 나란히 평가합니다.")
        return
    if not tickers:
        st.error("티커를 입력해주세요.")
        return
    if not chosen:
        st.error("전략을 1개 이상 선택해주세요.")
        return

    strategies = {name: available[name]() for name in chosen}
    with st.spinner("전략별 백테스트 실행 중..."):
        try:
            results = compare_strategies(strategies, tickers=tickers, start=start, end=end, **engine_kwargs)
        except Exception as e:
            friendly_engine_error(e)

    # 수익곡선 오버레이
    st.subheader("수익곡선 비교")
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        pv = res.portfolio_values
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values, mode="lines", name=name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(
        xaxis_title="날짜", yaxis_title="가치 (원)",
        hovermode="x unified", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 지표 + 신뢰성 비교 표
    st.subheader("지표 · 신뢰성 비교")
    rows = []
    for name, res in results.items():
        m = res.metrics
        report = evaluate_reliability(m)
        rows.append({
            "전략": name,
            "CAGR": f"{m['CAGR']:.2%}",
            "MDD": f"{m['MDD']:.2%}",
            "Sharpe": f"{m['Sharpe']:.2f}",
            "변동성": f"{m['Volatility']:.2%}",
            "신뢰성": "✅ 통과" if report.overall == PASS else f"⚠️ 경고({len(report.warnings)})",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("모든 전략은 동일한 자산·기간·거래비용·리밸런싱 규칙으로 평가되어 공정하게 비교됩니다.")


# ══════════════════════════════════════════════════════════════
# 모드 3: Lookahead 가드레일
# ══════════════════════════════════════════════════════════════
def render_guardrail():
    st.subheader("Lookahead 가드레일 — 미래 참조 자동 탐지")
    st.markdown(
        "백테스트 엔진은 신호일까지의 데이터만 전략에 넘겨 lookahead bias를 **구조적으로 방지**합니다.\n"
        "이 가드레일은 한 걸음 더 나아가, 전략에게 **미래까지 포함된 데이터를 통째로 넘겼을 때**도 "
        "스스로 신호일까지만 사용하는지를 **검증**합니다.\n\n"
        "원리: 신호일(`as_of`) **이후 구간만** 무작위로 교란한 두 데이터를 전략에 넣어, "
        "출력 비중이 달라지면 → 전략이 미래를 참조한 것(**DETECTED**), 같으면 → 미래 참조 없음(**CLEAN**)."
    )

    with st.sidebar:
        st.header("가드레일 검사")
        check_clean = st.button("정상 전략 검사 (균등 비중)", use_container_width=True)
        check_cheat = st.button("미래 참조(컨닝) 전략 검사", type="primary", use_container_width=True)

    if not (check_clean or check_cheat):
        st.info("좌측 버튼으로 정상 전략과 컨닝 전략을 각각 검사해보세요.")
        return
    if not tickers:
        st.error("티커를 입력해주세요.")
        return

    with st.spinner("데이터 로딩 중..."):
        try:
            close, _ = DataLoader(tickers, cache_dir=CACHE_DIR).load(start, end)
        except Exception as e:
            st.error(f"데이터 로딩 오류: {e}")
            return

    if len(close) < 50:
        st.error("검사에 필요한 데이터가 부족합니다. 기간을 늘려주세요.")
        return

    # 신호일 = 기간의 80% 지점 (이후에 미래 구간이 남도록)
    as_of = close.index[int(len(close) * 0.8)]

    if check_clean:
        audit = wrap_strategy_as_audit(EqualWeightStrategy())
        report = detect_lookahead(audit, close, as_of)
        label = "정상 전략 (균등 비중, 엔진과 동일하게 신호일까지만 사용)"
    else:
        report = detect_lookahead(lookahead_cheating_audit, close, as_of)
        label = "컨닝 전략 (신호일 무시, 미래 구간 모멘텀에 비례 배분)"

    st.markdown(f"**검사 대상:** {label}")
    if report.status == CLEAN:
        st.success(f"✅ CLEAN — {report.detail}")
    else:
        st.error(f"⚠️ DETECTED — {report.detail}")

    st.dataframe(pd.DataFrame([{
        "신호일 (as_of)": str(report.as_of.date()),
        "판정": report.status,
        "최대 비중 변화": f"{report.max_weight_diff:.4f}",
    }]), hide_index=True, use_container_width=True)


# ── 라우팅 ─────────────────────────────────────────────────────
if app_mode == "단일 전략 백테스트":
    render_single()
elif app_mode == "전략 비교":
    render_compare()
else:
    render_guardrail()
