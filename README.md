# 로보어드바이저 신뢰성 검증 테스트베드

투자 알고리즘(로보어드바이저)을 검증하는 평가 인프라. 어떤 전략이든 끼워 넣으면 신뢰성 지표를 자동으로 계산한다.

> 코스콤 RA테스트베드 개념을 이해하고 직접 구현해본 포트폴리오 프로젝트입니다. 실제 투자 자문이 아닙니다 (교육·포트폴리오 목적).

---

## 파이프라인

```
전략 (Strategy ABC plug-in)
        │
        ▼
과거 시장 데이터 ─── yfinance → parquet 캐시
        │
        ▼
백테스트 엔진
  ├─ 신호일 t : prices.loc[:t] → generate_weights()
  ├─ 체결일 t+1 : 다음 거래일 시가(Open)로 매매
  ├─ 거래비용 반영 (bps 단위, 비중 변화에 비례)
  └─ 리밸런싱 주기 적용 (M / Q / D)
        │
        ▼
리스크 지표 (CAGR · MDD · Sharpe · 변동성)
        │
        ▼
Streamlit 대시보드
```

## 데모

*GIF 추가 예정*

---

## 설계 결정

설계 결정은 이 프로젝트에서 가장 중요하게 다루는 부분입니다.

### 1. 전략 인터페이스 — Strategy ABC

모든 전략은 `Strategy` 추상 클래스를 상속하고 단 하나의 메서드를 구현한다.

```python
class Strategy(ABC):
    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        """
        prices: 신호 계산 시점까지의 종가만 포함 (엔진이 슬라이싱해서 전달)
        returns: {'SPY': 0.6, 'TLT': 0.3, 'GLD': 0.1}  합계 1.0
        """
```

**선택 근거:** 규칙 기반 전략이든 ML 기반 전략이든 동일한 시그니처를 따르면 백테스트 엔진을 전혀 수정하지 않고 전략만 교체할 수 있다. ABC 방식은 Protocol 방식 대비 `isinstance` 체크가 가능해 Phase 2의 사용자 전략 업로드 기능과 호환성이 좋다.

**현재 구현된 전략:**

| 전략 | 클래스 | 설명 |
|------|--------|------|
| 균등 비중 | `EqualWeightStrategy` | 모든 자산에 동일 비중 배분, 기준선 역할 |
| 고정 비중 | `FixedWeightStrategy` | 사용자 지정 비중 유지, 주기적 리밸런싱 |
| 이동평균 교차 | `MovingAverageCrossStrategy` | 단기 MA > 장기 MA인 자산에만 투자 |

---

### 2. Lookahead Bias 방지 — 가장 중요한 설계 결정

백테스트의 가장 흔한 오류는 전략이 미래 데이터를 보고 의사결정하는 것이다. 이 프로젝트는 이를 코드 레벨에서 구조적으로 차단한다.

```
신호일 (t):    prices.loc[:t] 만 전략에 전달 → 목표 비중 계산
체결일 (t+1):  다음 거래일 시가(Open)로 실제 매매 체결
```

엔진 내부 코드:

```python
# 핵심: signal_date까지의 종가만 전략에 전달 (미래 데이터 차단)
past_prices = close.loc[:signal_date]
target_weights = self.strategy.generate_weights(past_prices)

# 체결은 다음 거래일 시가(Open)로
exec_prices = open_.loc[execution_date]
```

전략 함수는 `signal_date` 이후의 데이터에 구조적으로 접근할 수 없다. 이 분리 방식은 테스트에서도 명시적으로 검증한다 (`tests/test_engine.py::TestLookaheadBias`).

**체결가를 시가(Open)로 쓰는 이유:** 종가 기준으로 신호를 계산하고 같은 날 종가로 체결하면, 실제로는 불가능한 타이밍에 거래한다고 가정하는 셈이다. 다음 거래일 시가가 현실에 가장 근접한 체결가다.

---

### 3. 거래비용 모델링

```python
BacktestEngine(transaction_cost_bps=10)  # 기본 10bps = 0.1%
```

비중 변화가 있는 자산에 대해서만 `|새 포지션 가치 - 기존 포지션 가치| × bps/10000`를 차감한다.

**단순화의 한계 (의도적 선택):**
- 실제 시장에서는 슬리피지, 스프레드, 세금, 환전 비용이 추가로 존재한다.
- 본 프로젝트는 고정 비율 모델로 단순화한다. 이 한계를 인지하고 bps 파라미터를 높여 민감도 분석을 할 수 있다.

---

### 4. 자산 유니버스 — 하드코딩 배제

자산 목록을 코드에 하드코딩하지 않는다. 엔진은 `tickers: list[str]` 파라미터로 어떤 자산이든 받을 수 있다.

```python
# Phase 1 기본값: 미국 ETF 3종
BacktestEngine(tickers=["SPY", "TLT", "GLD"], ...)

# Phase 2: 동일한 엔진으로 국내 ETF 적용 (케이스 스터디)
BacktestEngine(tickers=["069500", "114820"], ...)
```

**기본 유니버스로 SPY/TLT/GLD를 선택한 이유:**
- yfinance 데이터 품질이 우수하고 2004년부터 히스토리가 존재
- 주식(SPY)·채권(TLT)·원자재(GLD)로 상관관계가 낮은 3개 자산 클래스를 포괄
- 면접 설명 시 직관적으로 이해하기 쉬운 조합

---

### 5. 리밸런싱 주기

```python
BacktestEngine(rebalance_freq="M")  # 'M'=월별, 'Q'=분기별, 'D'=매일
```

엔진이 각 기간의 마지막 거래일을 신호일로 결정한다. 전략은 주기를 몰라도 되고, `generate_weights()`를 호출할 때마다 응답하면 된다.

---

### 6. DB 미사용 결정

과거 시세는 parquet 파일로 로컬 캐싱하고, 백테스트 결과는 매번 새로 계산한다.

**이유:**
- 과거 시세는 변하지 않으므로 DB의 Write 기능이 필요 없다.
- 백테스트는 파라미터 조합이 무한하고 결과 자체는 빠르게 계산된다 — 결과를 저장하는 것보다 매번 계산하는 게 단순하고 정확하다.
- Streamlit Community Cloud 배포 환경에서 DB 인프라 없이 동작한다.

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 언어 | Python 3.12 |
| 데이터 수집 | yfinance |
| 데이터 처리 | pandas, numpy |
| 시각화 | Plotly |
| UI | Streamlit |
| 테스트 | pytest |
| 배포 | Streamlit Community Cloud |

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt
pip install -e .   # 패키지 편집 모드 설치 (테스트용)

# 단위 테스트
pytest

# Streamlit 앱 실행
streamlit run src/ra_testbed/app.py
```

첫 실행 시 `data/` 폴더에 각 티커의 parquet 파일이 생성된다. 이후 실행부터는 캐시에서 로드한다.

---

## Phase 2 예고

- **시장 국면 분류**: 2008 금융위기 / 2020 코로나 급락 / 2022 금리인상기 구간 정의 및 국면별 성과 분해
- **국내 ETF 케이스 스터디**: KODEX200(069500) + KODEX국채10년(114820)으로 동일한 엔진 적용
- **신뢰성 리포트**: CAGR·MDD·Sharpe 임계값 기반 통과/경고 판정 자동화

---

## 회고

*프로젝트 완성 후 추가 예정*
