# 프로젝트: 로보어드바이저 신뢰성 검증 테스트베드

투자 전략 자체가 아니라, 어떤 투자 전략이든 끼워 넣으면 자동으로 신뢰성을 검증해주는 평가 인프라를 구현한다.
전체 기획/스코프/스택은 @docs/PROJECT_BRIEF.md 참고. MINI-Exchange 프로젝트와 동일한 운영 원칙을 따른다.

## 기술 스택
- Backend/분석: Python 3.12, pandas, numpy, yfinance(과거 시세 수집), pytest
- UI: Streamlit (Phase1~2), FastAPI는 Phase3 스트레치(REST API로 노출하고 싶을 때만)
- 시각화: Plotly 또는 matplotlib
- 배포: Streamlit Community Cloud
- DB 없음 — 과거 시세는 parquet/csv로 캐싱, 백테스트 결과는 즉시 계산해서 보여줌 (의도적 선택, README에 이유 명시)

## 빌드 / 테스트
- `pip install -r requirements.txt`
- `pytest` (전략/백테스트엔진/리스크지표 단위 테스트)
- `streamlit run src/ra_testbed/app.py` (로컬 실행)

## 개발 순서
1. 전략 인터페이스(표준 입출력 포맷) 정의 + 예시 전략 2~3개 구현, 단위 테스트부터 (TDD)
2. 데이터 로더(yfinance 수집 + 캐싱) 구현
3. 백테스트 엔진 구현 — lookahead bias 방지가 최우선 검증 대상
4. 리스크 지표 계산 모듈 (CAGR, MDD, Sharpe, 변동성)
5. 시장 국면 분류 + 국면별 리포트 (Phase2)
6. Streamlit UI 연결
7. 신뢰성 리포트 생성 + 배포

## 핵심 원칙
- 범위는 Phase1(MVP) → Phase2(차별화 기능) → Phase3(스트레치) 순서로 단계적으로 확장한다. Phase1이 끝나기 전에 다음 단계에 손대지 않는다.
- 백테스트 엔진은 lookahead bias(미래 정보 누설)를 구조적으로 막아야 한다 — 신호 계산일과 체결일을 명확히 분리하고, 이 분리 방식을 설계 결정으로 반드시 문서화한다.
- README는 "기능 나열"보다 "설계 결정"(전략 인터페이스 설계, lookahead bias 방지 방법, 거래비용 모델링, 국면 분류 기준, 통과/경고 임계값 설정 근거)을 가장 비중 있게 다룬다. 맨 앞에는 한 줄 소개 + 아키텍처/파이프라인 다이어그램 + 데모 GIF를 먼저 배치한다.
- 코스콤이 운영하는 RA테스트베드를 "그대로 재현했다"는 단정적 표현은 쓰지 않는다. "그 개념을 이해하고 직접 구현해본 것"이라는 식으로 서술한다.
- 큰 설계 변경 전에는 plan mode로 먼저 합의받는다.
- 커밋 메시지에 "Co-Authored-By: Claude"나 "Generated with Claude Code" 같은 attribution을 넣지 않는다.

## 폴더 구조 (초안)
```
ra-testbed/
├── data/                       # 캐싱된 과거 시세 데이터
├── src/ra_testbed/
│   ├── strategies/             # 전략 인터페이스 + 예시 전략
│   ├── data/                   # 데이터 로더
│   ├── backtest/               # 백테스트 엔진 + 국면 분류
│   ├── metrics/                # 리스크 지표 계산
│   ├── report/                 # 신뢰성 리포트 생성
│   └── app.py                  # Streamlit 앱
├── tests/
├── requirements.txt
└── README.md
```
