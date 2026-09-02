"""전체 상장기업 사업 텍스트 임베딩을 2차원으로 축소해 지도처럼 보여주기 위한 로직.

PCA는 빠르고 결정적(항상 같은 결과)이라 기본값으로 쓰고, UMAP은 군집 구조를
더 뚜렷하게 보여주지만 계산이 무겁고 fit마다 조금씩 달라질 수 있어 선택 옵션으로
둔다.
"""
import numpy as np
import pandas as pd

from src.analysis.ksic import is_semiconductor, midclass_code, midclass_name

TOP_N_INDUSTRIES_IN_LEGEND = 12
OTHER_INDUSTRY_LABEL = "기타 업종"


def usable_embedding_mask(df: pd.DataFrame) -> pd.Series:
    """business_description이 없어 임베딩이 0벡터인 행을 걸러낸다."""
    norms = df["embedding"].apply(lambda v: float(np.linalg.norm(v)) if v is not None else 0.0)
    return norms > 1e-6


def fit_reducer(embeddings: np.ndarray, method: str, random_state: int = 42):
    """method: 'PCA' 또는 'UMAP'. 학습된 reducer 객체를 반환한다 (transform용)."""
    if method == "UMAP":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state, metric="cosine")
    else:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=random_state)
    reducer.fit(embeddings)
    return reducer


def reduce_2d(reducer, embeddings: np.ndarray) -> np.ndarray:
    return reducer.transform(embeddings)


def build_plot_frame(df: pd.DataFrame, coords: np.ndarray, top_candidate_codes: set[str] | None = None) -> pd.DataFrame:
    """산점도용 DataFrame 생성: 좌표 + 업종명(범례 상위 N개 외엔 '기타 업종') + 반도체 여부 + 강조 여부."""
    plot_df = df[["corp_name", "stock_code", "induty_code"]].copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["midclass_code"] = plot_df["induty_code"].apply(midclass_code)
    plot_df["is_semiconductor"] = plot_df["induty_code"].apply(is_semiconductor)

    top_codes = plot_df["midclass_code"].value_counts().head(TOP_N_INDUSTRIES_IN_LEGEND).index
    plot_df["industry_group"] = plot_df["midclass_code"].where(
        plot_df["midclass_code"].isin(top_codes), other=None
    )
    plot_df["industry_label"] = plot_df["midclass_code"].apply(midclass_name)
    plot_df.loc[plot_df["industry_group"].isna(), "industry_label"] = OTHER_INDUSTRY_LABEL

    if top_candidate_codes:
        plot_df["is_top_candidate"] = plot_df["stock_code"].isin(top_candidate_codes)
    else:
        plot_df["is_top_candidate"] = False

    return plot_df
