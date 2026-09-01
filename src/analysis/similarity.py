"""1차(산업분류+텍스트 유사도) → 2차(재무비율 거리)로 유사기업 후보를 뽑는다.

개요.md의 프로세스("전체 상장종목 Dashboard 구성 → 키워드 검색 → 1차 산업분류
→ 2차 재무비율 유사성")를 그대로 구현한다.

1차: 업종코드(induty_code)가 일치하는 풀 안에서 사업 텍스트 임베딩 코사인
     유사도로 상위 top_k_text개를 추린다. 풀이 너무 작으면 업종코드 앞자리만
     맞춰 느슨하게 확장한다 (DART induty_code는 회사마다 자릿수가 3~5자리로
     들쭉날쭉해서 고정 자릿수 매칭이 안 통함).
2차: 그 후보들을 재무비율(매출성장률/영업이익률/부채비율/자산회전율/규모)
     표준화 벡터의 유클리드 거리로 재정렬해 최종 top_n_final개를 뽑는다.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.embed_text import DESCRIPTION_TRUNCATE_CHARS, MODEL_NAME

FINANCIAL_FEATURES = ["revenue_growth", "operating_margin", "debt_ratio", "asset_turnover", "log_total_assets"]
MIN_INDUSTRY_POOL = 15

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_business_description(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode(text[:DESCRIPTION_TRUNCATE_CHARS], normalize_embeddings=True)


def _industry_pool(universe: pd.DataFrame, induty_code) -> tuple[pd.DataFrame, int]:
    """업종코드가 일치하는 풀을 찾는다. MIN_INDUSTRY_POOL보다 작으면 앞자리만
    맞춰 넓힌다 (예: 5자리 정확매칭 실패 시 4자리 → 3자리 → 2자리 순으로 완화)."""
    codes = universe["induty_code"].astype(int).astype(str)
    induty_code = str(int(induty_code))
    for prefix_len in range(len(induty_code), 1, -1):
        prefix = induty_code[:prefix_len]
        pool = universe[codes.str.startswith(prefix)]
        if len(pool) >= MIN_INDUSTRY_POOL or prefix_len == 2:
            return pool, prefix_len
    return universe.iloc[0:0], 0


def _rank_by_text_similarity(pool: pd.DataFrame, target_embedding: np.ndarray, top_k: int) -> pd.DataFrame:
    embeddings = np.stack(pool["embedding"].values)
    sims = embeddings @ target_embedding
    pool = pool.copy()
    pool["text_similarity"] = sims
    return pool.sort_values("text_similarity", ascending=False).head(top_k)


def _with_log_scale(df: pd.DataFrame, asset_col: str = "total_assets") -> pd.DataFrame:
    df = df.copy()
    df["log_total_assets"] = np.log(df[asset_col].clip(lower=1))
    return df


def _rank_by_financial_distance(pool: pd.DataFrame, target_ratios: dict, top_n: int) -> pd.DataFrame:
    pool = _with_log_scale(pool)
    X = pool[FINANCIAL_FEATURES].astype(float)
    # 결측치는 후보 풀 중앙값으로 채운다. 한 지표 없다고 후보에서 통째로 빼면
    # 진짜 유사한 기업을 놓칠 수 있다 (예: 롯데케미칼은 revenue_growth 결측이지만
    # LG화학과 업종/텍스트가 거의 동일한 진짜 비교대상).
    medians = X.median()
    X = X.fillna(medians)

    target_raw = pd.Series({f: target_ratios.get(f) for f in FINANCIAL_FEATURES}, dtype=float)
    target_raw = target_raw.fillna(medians)

    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    target_z = scaler.transform(target_raw[FINANCIAL_FEATURES].to_frame().T)

    pool = pool.copy()
    pool["financial_distance"] = np.linalg.norm(Xz - target_z, axis=1)
    return pool.sort_values("financial_distance").head(top_n)


def select_candidates(
    target: dict,
    universe: pd.DataFrame,
    top_k_text: int = 40,
    top_n_final: int = 20,
) -> pd.DataFrame:
    """평가대상 기업의 유사기업 후보를 뽑는다.

    target 딕셔너리 키:
      - induty_code (필수)
      - business_description 또는 embedding 중 하나 (필수)
      - revenue_growth, operating_margin, debt_ratio, asset_turnover, total_assets
        (있는 만큼만 써도 됨 — 없는 값은 후보 풀 중앙값으로 대체)
      - corp_code (선택, universe에 포함된 상장사를 타겟으로 검증할 때 자기 자신 제외용)
    """
    pool = universe
    if "corp_code" in target:
        pool = pool[pool["corp_code"] != target["corp_code"]]

    industry_pool, matched_prefix_len = _industry_pool(pool, target["induty_code"])
    if industry_pool.empty:
        return industry_pool

    target_embedding = target.get("embedding")
    if target_embedding is None:
        target_embedding = embed_business_description(target["business_description"])

    text_ranked = _rank_by_text_similarity(industry_pool, target_embedding, top_k=top_k_text)

    target_ratios = dict(target)
    target_ratios["log_total_assets"] = float(np.log(max(target.get("total_assets") or 1, 1)))
    final = _rank_by_financial_distance(text_ranked, target_ratios, top_n=top_n_final)

    final = final.copy()
    final["industry_match_level"] = matched_prefix_len
    return final[
        [
            "corp_code",
            "corp_name",
            "stock_code",
            "induty_code",
            "industry_match_level",
            "text_similarity",
            "financial_distance",
            *FINANCIAL_FEATURES,
        ]
    ].reset_index(drop=True)
