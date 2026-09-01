"""비상장기업 유사기업(베타풀) 자동 선정 대시보드.

실행: comp_comp_selec/Scripts/python.exe -m streamlit run src/app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.beta import compute_beta_pool, median_beta
from src.analysis.similarity import embed_business_description, select_candidates
from src.analysis.wacc import beta_pool_sensitivity, compute_wacc, debt_weight_from_ratio, sensitivity_summary

UNIVERSE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "universe_2023.parquet"
UNIVERSE_YEAR = 2023

COLOR_PRIMARY = "#2a78d6"  # blue — 단일 시리즈 기본색 (베타 막대, WACC 분포)
COLOR_ACCENT = "#eb6834"  # orange — 강조/현재값 마커

st.set_page_config(page_title="유사기업 자동 선정 & WACC 민감도", layout="wide")


@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_parquet(UNIVERSE_PATH)


@st.cache_data(show_spinner="사업 설명 임베딩 계산 중...")
def embed_text_cached(text: str) -> np.ndarray:
    return embed_business_description(text)


@st.cache_data(show_spinner="주가 데이터로 베타 계산 중... (최대 수십 초 소요될 수 있습니다)")
def compute_beta_pool_cached(stock_codes: tuple, end_date: str, years: int) -> pd.DataFrame:
    return compute_beta_pool(list(stock_codes), end_date=end_date, years=years)


universe = load_universe()

st.title("비상장기업 유사기업(베타풀) 자동 선정")
st.caption(
    "DART 공시 데이터 기반으로 유사기업을 자동 선정하고, 베타풀 구성에 따라 "
    "WACC이 얼마나 달라지는지 정량화합니다."
)

# ---------- 사이드바: 평가대상 기업 입력 ----------
st.sidebar.header("1. 평가대상 기업")
input_mode = st.sidebar.radio("입력 방식", ["상장기업에서 선택 (데모)", "직접 입력 (비상장기업)"])

target = None
target_label = None

if input_mode == "상장기업에서 선택 (데모)":
    names = sorted(universe["corp_name"].unique())
    default_idx = names.index("LG화학") if "LG화학" in names else 0
    picked = st.sidebar.selectbox("상장기업 선택", names, index=default_idx)
    row = universe[universe["corp_name"] == picked].iloc[0]
    target = {
        "corp_code": row["corp_code"],
        "induty_code": row["induty_code"],
        "embedding": row["embedding"],
        "revenue_growth": row["revenue_growth"],
        "operating_margin": row["operating_margin"],
        "debt_ratio": row["debt_ratio"],
        "asset_turnover": row["asset_turnover"],
        "total_assets": row["total_assets"],
    }
    target_label = picked
else:
    target_label = st.sidebar.text_input("회사명", value="평가대상기업")
    ref_names = sorted(universe["corp_name"].unique())
    ref_name = st.sidebar.selectbox("업종이 비슷한 상장기업 선택 (업종코드 차용)", ref_names)
    induty_code = universe.loc[universe["corp_name"] == ref_name, "induty_code"].iloc[0]
    description = st.sidebar.text_area(
        "사업의 내용 (사업보고서 'II. 사업의 내용' 수준의 설명)", height=150
    )

    st.sidebar.caption("재무비율")
    revenue_growth = st.sidebar.number_input("매출성장률(%)", value=0.0) / 100
    operating_margin = st.sidebar.number_input("영업이익률(%)", value=0.0) / 100
    debt_ratio_pct = st.sidebar.number_input("부채비율(%, 부채총계/자본총계)", value=100.0) / 100
    asset_turnover = st.sidebar.number_input("자산회전율(매출/자산)", value=1.0)
    total_assets = st.sidebar.number_input(
        "자산총계(원)", value=1_000_000_000_000.0, step=1e11, format="%.0f"
    )

    target = {
        "induty_code": induty_code,
        "business_description": description,
        "revenue_growth": revenue_growth,
        "operating_margin": operating_margin,
        "debt_ratio": debt_ratio_pct,
        "asset_turnover": asset_turnover,
        "total_assets": total_assets,
    }

st.sidebar.header("2. WACC 파라미터")
risk_free_rate = st.sidebar.slider("무위험이자율 (%)", 0.0, 8.0, 3.5, 0.1) / 100
equity_risk_premium = st.sidebar.slider("시장위험프리미엄 (%)", 0.0, 12.0, 6.0, 0.1) / 100
cost_of_debt = st.sidebar.slider("타인자본비용 (%)", 0.0, 12.0, 5.0, 0.1) / 100
tax_rate = st.sidebar.slider("법인세율 (%)", 0.0, 40.0, 24.2, 0.1) / 100
st.sidebar.caption("자본구조 가중치는 평가대상 기업의 장부가 부채비율에서 자동 계산합니다.")

st.sidebar.header("3. 베타 계산 기간")
end_date = st.sidebar.date_input("평가기준일", value=pd.Timestamp(f"{UNIVERSE_YEAR}-12-31"))
beta_years = st.sidebar.slider("회귀 기간(년)", 2, 5, 3)


if input_mode.startswith("직접") and not target.get("business_description", "").strip():
    st.info("왼쪽 사이드바에서 평가대상 기업의 '사업의 내용'을 입력하면 분석이 시작됩니다.")
    st.stop()

debt_weight = debt_weight_from_ratio(target["debt_ratio"]) if target.get("debt_ratio") else 0.5

# ---------- 화면 1: 유사기업 후보 ----------
st.header(f"① {target_label} — 유사기업 후보")

top_n_final = st.slider("후보 개수", 5, 30, 20)

if target.get("embedding") is None:
    target["embedding"] = embed_text_cached(target["business_description"])

candidates = select_candidates(target, universe, top_n_final=top_n_final)

if candidates.empty:
    st.error("업종이 일치하는 후보를 찾지 못했습니다. 참조 상장기업을 다시 선택해보세요.")
    st.stop()

st.caption(
    f"업종코드 매칭 자릿수: {candidates['industry_match_level'].iloc[0]}자리 "
    "(자릿수가 작을수록 더 넓은 업종 범위까지 완화해서 찾았다는 뜻) · "
    "1차: 업종분류 + 사업 텍스트 유사도 → 2차: 재무비율 거리로 재정렬"
)

display_cols = {
    "corp_name": "기업명",
    "stock_code": "종목코드",
    "induty_code": "업종코드",
    "text_similarity": "텍스트 유사도",
    "financial_distance": "재무비율 거리",
    "revenue_growth": "매출성장률",
    "operating_margin": "영업이익률",
    "debt_ratio": "부채비율",
    "asset_turnover": "자산회전율",
}
show_df = candidates[list(display_cols)].rename(columns=display_cols)
st.dataframe(
    show_df.style.format(
        {
            "텍스트 유사도": "{:.3f}",
            "재무비율 거리": "{:.2f}",
            "매출성장률": "{:.1%}",
            "영업이익률": "{:.1%}",
            "부채비율": "{:.1%}",
            "자산회전율": "{:.2f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

# ---------- 화면 2: 베타풀 구성 ----------
st.header("② 베타풀 구성")

candidate_key = tuple(candidates["stock_code"].tolist())

if st.button("베타 계산 실행", type="primary"):
    st.session_state["beta_df"] = compute_beta_pool_cached(
        candidate_key, end_date.strftime("%Y-%m-%d"), beta_years
    )
    st.session_state["beta_df_key"] = candidate_key

if st.session_state.get("beta_df_key") != candidate_key:
    st.info("위 '베타 계산 실행' 버튼을 눌러 후보 기업들의 베타를 계산하세요 (KRX 주가 조회).")
    st.stop()

beta_df = st.session_state["beta_df"].merge(
    candidates[["stock_code", "corp_name"]], on="stock_code", how="left"
)
beta_df["포함"] = beta_df["beta"].notna()

edited = st.data_editor(
    beta_df[["포함", "corp_name", "stock_code", "beta", "r_squared", "n_obs"]].rename(
        columns={
            "corp_name": "기업명",
            "stock_code": "종목코드",
            "beta": "베타",
            "r_squared": "R²",
            "n_obs": "관측주수",
        }
    ),
    disabled=["기업명", "종목코드", "베타", "R²", "관측주수"],
    hide_index=True,
    width="stretch",
    key="beta_editor",
)

included = beta_df[edited["포함"].values & beta_df["beta"].notna()]

if included.empty:
    st.warning("포함된 베타가 없습니다. 위 표에서 최소 1개 이상 체크하세요.")
    st.stop()

beta_median = median_beta(included["beta"])
current_wacc = compute_wacc(
    beta_median,
    risk_free_rate=risk_free_rate,
    equity_risk_premium=equity_risk_premium,
    cost_of_debt=cost_of_debt,
    tax_rate=tax_rate,
    debt_weight=debt_weight,
)

c1, c2, c3 = st.columns(3)
c1.metric("포함된 기업 수", len(included))
c2.metric("베타풀 중앙값 베타", f"{beta_median:.3f}")
c3.metric("현재 WACC", f"{current_wacc * 100:.2f}%")

bar_df = included.sort_values("beta", ascending=False)
fig_beta = go.Figure(go.Bar(x=bar_df["corp_name"], y=bar_df["beta"], marker_color=COLOR_PRIMARY))
fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#898781", annotation_text="시장베타 1.0")
fig_beta.update_layout(yaxis_title="베타", xaxis_title=None, margin=dict(t=10, b=10), height=350)
st.plotly_chart(fig_beta, width="stretch", theme="streamlit")

# ---------- 화면 3: 민감도 분석 ----------
st.header("③ 민감도 분석 — 베타풀 구성에 따라 WACC이 얼마나 벌어지는가")

max_k = len(included)
if max_k < 2:
    st.warning("민감도 분석에는 베타가 유효한 기업이 최소 2개 필요합니다.")
    st.stop()

k = st.slider("몇 개를 골라 베타풀을 구성할지 (k)", 2, max_k, min(5, max_k))

wacc_kwargs = dict(
    risk_free_rate=risk_free_rate,
    equity_risk_premium=equity_risk_premium,
    cost_of_debt=cost_of_debt,
    tax_rate=tax_rate,
    debt_weight=debt_weight,
)

sens = beta_pool_sensitivity(included["beta"].tolist(), k=k, wacc_kwargs=wacc_kwargs)
summary = sensitivity_summary(sens)

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 조합 수", f"{summary['n_combinations']:,}")
c2.metric("WACC 최소", f"{summary['wacc_min'] * 100:.2f}%")
c3.metric("WACC 최대", f"{summary['wacc_max'] * 100:.2f}%")
c4.metric("변동폭", f"{summary['wacc_range'] * 100:.2f}%p")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=sens["wacc"] * 100, marker_color=COLOR_PRIMARY, nbinsx=40))
fig_hist.add_vline(
    x=current_wacc * 100,
    line_dash="dash",
    line_color=COLOR_ACCENT,
    annotation_text=f"현재 선택 조합 {current_wacc * 100:.2f}%",
    annotation_position="top",
)
fig_hist.update_layout(
    xaxis_title="WACC (%)", yaxis_title="조합 수", margin=dict(t=40, b=10), height=350, showlegend=False
)
st.plotly_chart(fig_hist, width="stretch", theme="streamlit")

st.caption(
    f"베타풀 후보 {max_k}개 중 {k}개를 고르는 {summary['n_combinations']:,}가지 조합을 전부 계산했습니다. "
    f"어떤 {k}개를 고르느냐에 따라 WACC이 {summary['wacc_min'] * 100:.2f}%~{summary['wacc_max'] * 100:.2f}%"
    f"(폭 {summary['wacc_range'] * 100:.2f}%p)로 달라집니다 — "
    "평가자 주관이 개입하는 지점을 수치로 보여주는 것이 이 프로젝트의 핵심입니다."
)

with st.expander("방법론 / 단순화 지점"):
    st.markdown(
        f"""
- 후보 풀은 {UNIVERSE_YEAR}년 사업보고서 기준 상장기업 {len(universe):,}개(KOSPI+KOSDAQ, KONEX 제외)에서 뽑습니다.
- 1차 필터는 DART 업종코드 일치 + 사업 텍스트 임베딩(`jhgan/ko-sroberta-multitask`) 코사인 유사도.
- 2차 필터는 매출성장률·영업이익률·부채비율·자산회전율·규모(log 자산총계)를 표준화한 유클리드 거리.
- 베타는 KOSPI 대비 주간수익률 OLS 회귀(관측치 20주 미만이면 제외).
- WACC 자본구조 가중치는 시가총액이 아니라 **장부가 기준 부채비율**로 단순화했습니다.
        """
    )
