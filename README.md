# 비상장기업 유사기업(베타풀) 자동 선정

비상장기업 가치평가에서 가장 주관이 개입하는 **유사기업(peer) 선정**을 DART 공시
데이터로 자동화하고, 그 선택이 WACC에 미치는 영향을 민감도로 정량화하는 대시보드.

배경은 [개요.md](./개요.md), 단계별 구현 기록은 [계획.md](./계획.md) 참고.

## 대시보드 화면

1. **유사기업 후보** — 1차 업종분류(KSIC) + 사업 텍스트 임베딩 → 2차 재무비율 거리로 재정렬
2. **최종 후보 스코어카드** — 항목별 유사도 %, 종합 유사도 랭킹, 레이더 차트, Beta·시가총액
3. **전체 상장기업 지도** — 사업 텍스트 임베딩을 PCA/UMAP으로 2D 축소, 평가대상 업종·Top 20 강조
4. **베타풀 구성** — 후보를 체크박스로 넣고 빼면 중앙값 베타 → WACC 실시간 갱신
5. **민감도 분석** — 후보 n개 중 k개를 고르는 전체 조합의 WACC 분포

## 실행

```bash
git clone https://github.com/sejeans/comparable-company-selection
cd comparable-company-selection

py -3.11 -m venv comp_comp_selec          # Streamlit Cloud 배포 환경과 동일한 3.11
comp_comp_selec\Scripts\python -m pip install -r requirements.txt
comp_comp_selec\Scripts\python -m streamlit run src/app/streamlit_app.py
```

- 앱 구동에 **DART API 키는 필요 없다.** 미리 만들어둔
  `data/processed/universe_2023_app.parquet`(상장사 2,651곳, 임베딩 포함)만 읽는다.
- `sentence-transformers`가 torch를 끌고 와 설치가 수 GB / 수 분 걸린다.
- "직접 입력(비상장기업)" 모드를 처음 쓸 때 한국어 임베딩 모델
  (`jhgan/ko-sroberta-multitask`, 약 500MB)이 자동으로 내려받아진다.

## 데이터 다시 수집하기

`data/` 원천 데이터는 용량 때문에 git에 없다(GitHub 100MB 단일파일 제한).
저장소에 커밋된 건 앱이 읽는 경량 파일 하나뿐이다.

| 파일 | 크기 | git |
|---|---|---|
| `data/processed/universe_2023_app.parquet` | 8MB | ✅ 커밋됨 (앱이 읽는 파일) |
| `data/processed/universe_2023.parquet` | 197MB | ❌ 사업보고서 원문 텍스트 포함 |
| `data/raw/universe_2023_rows.jsonl` | 419MB | ❌ 배치 재시작용 스크래치 |

처음부터 다시 만들려면:

```bash
# 0. DART API 키 (https://opendart.fss.or.kr 에서 무료 발급)
copy .env.example .env                    # DART_API_KEY= 뒤에 키를 채운다
comp_comp_selec\Scripts\python -m pip install -r requirements-pipeline.txt

# 1. 전체 상장사 수집 → data/raw/universe_2023_rows.jsonl + processed parquet
#    상장사 2,651곳을 도는 배치라 몇 시간 걸리고 DART 일일 호출한도에 걸릴 수 있다.
#    한도에 걸리면 그 자리에서 멈추고, 다음 날 같은 명령을 다시 실행하면 이어서 진행된다.
comp_comp_selec\Scripts\python -m src.pipeline.build_universe --year 2023 --finalize

# 2. '사업의 내용' 텍스트 → 768차원 임베딩을 parquet에 추가
comp_comp_selec\Scripts\python -m src.pipeline.embed_text --year 2023

# 3. 원문 텍스트를 뺀 배포용 경량 parquet 생성 (197MB → 8MB)
comp_comp_selec\Scripts\python -m src.pipeline.build_app_dataset --year 2023
```

다른 연도를 쓰려면 `--year`를 바꾸고, `src/app/streamlit_app.py`의 `UNIVERSE_YEAR`도
같이 맞춘다.

## 구조

```
src/
├── collectors/     # DART 기업개황·재무제표·사업보고서 원문, KRX 주가/시가총액
├── pipeline/       # 배치 수집 → 임베딩 → 배포용 경량 데이터셋
├── analysis/       # 유사도 선정 · 스코어카드 · 2D 지도 · 베타 회귀 · WACC 민감도
├── app/            # Streamlit 대시보드
└── paths.py        # 공통 경로 상수 (의존성 없음)
```

`requirements.txt`는 **대시보드 실행에 필요한 것만** 담는다. DART 수집 전용 패키지
(OpenDartReader 등)는 `requirements-pipeline.txt`로 분리돼 있다 — 배포 환경에
불필요한 의존성을 올리지 않기 위해서다.

## 배포 (Streamlit Community Cloud)

- Main file path: `src/app/streamlit_app.py`
- **Advanced settings → Python version: 3.11** (기본값이 3.14면 pandas 2.x·numba 휠이
  없어 설치가 깨진다)
- Secrets 설정 불필요

`FinanceDataReader`는 import 이름이고 PyPI 배포명은 `finance-datareader`다.
requirements에 전자로 적으면 "No matching distribution found"로 배포가 실패한다.
