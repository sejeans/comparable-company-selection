"""후보 기업 하나를 최종 선택했을 때 보여줄 유사도 스코어카드 + 레이더 차트 데이터.

select_candidates()가 뽑은 top_similarity/financial_distance는 "정렬용" 지표라
그 자체로는 사람이 읽을 퍼센트 점수가 아니다. 여기서는 재무비율 각 항목을
전체 유니버스 표준편차 기준 z-score 차이로 바꾸고, 가우시안 커널로 0~100%
유사도 점수를 만든다 (차이가 0이면 100%, 표준편차만큼 벌어지면 약 60%,
2표준편차면 약 14%로 매끄럽게 감소).
"""
import numpy as np
import pandas as pd

from src.analysis.ksic import midclass_name

FINANCIAL_FEATURES = ["revenue_growth", "operating_margin", "debt_ratio", "asset_turnover", "log_total_assets"]

METRIC_LABELS = {
    "revenue_growth": "매출성장률",
    "operating_margin": "영업이익률",
    "debt_ratio": "부채비율",
    "asset_turnover": "자산회전율",
    "log_total_assets": "기업규모",
}

# 임베딩이 0벡터(사업의 내용 미확보)인 기업을 가려내는 기준. 정규화된 임베딩끼리의
# 코사인 유사도가 정확히 0에 붙는 경우는 사실상 0벡터일 때뿐이다.
EMBEDDING_EPSILON = 1e-9

# 종합 유사도 계산에 쓰는 가중치 (사업 유사도에 가장 큰 비중을 둔다)
WEIGHTS = {
    "business": 0.35,
    "revenue_growth": 0.13,
    "operating_margin": 0.13,
    "debt_ratio": 0.13,
    "asset_turnover": 0.13,
    "log_total_assets": 0.13,
}


def _gaussian_similarity(z_diff: float) -> float:
    return float(np.exp(-0.5 * z_diff**2)) * 100


def _industry_match(target_code, candidate_code) -> tuple[str, float]:
    """(표시용 라벨, 진행바용 0~1 비율)을 반환한다."""
    target_code = str(int(target_code))
    candidate_code = str(int(candidate_code))
    if target_code == candidate_code:
        return "동일", 1.0
    common = 0
    for a, b in zip(target_code, candidate_code):
        if a != b:
            break
        common += 1
    if common == 0:
        return "불일치", 0.0
    return f"부분일치 ({common}자리)", common / len(target_code)


def reference_stds(universe: pd.DataFrame) -> pd.Series:
    """z-score 계산 기준이 되는 전체 유니버스의 재무비율 표준편차.

    universe를 통째로 copy()하면 768차원 임베딩 컬럼까지 복사돼 비싸므로
    필요한 컬럼만 떼어 쓴다 (후보 전체 순위를 매길 때 반복 호출되는 경로).
    """
    raw_cols = [f for f in FINANCIAL_FEATURES if f != "log_total_assets"]
    ref = universe[raw_cols + ["total_assets"]].copy()
    ref["log_total_assets"] = np.log(ref["total_assets"].clip(lower=1))
    return ref[FINANCIAL_FEATURES].std()


def build_scorecard(
    target: dict,
    candidate: pd.Series,
    universe: pd.DataFrame | None = None,
    stds: pd.Series | None = None,
) -> dict:
    """target(평가대상) vs candidate(후보 기업 1개)의 항목별/종합 유사도를 계산한다.

    universe는 재무비율 표준편차를 구할 참조 모집단(전체 유니버스)이다.
    여러 후보를 연속으로 계산할 때는 reference_stds()로 한 번만 구한 stds를
    직접 넘겨 중복 계산을 피한다.
    """
    if stds is None:
        stds = reference_stds(universe)

    target_vals = dict(target)
    target_vals["log_total_assets"] = float(np.log(max(target.get("total_assets") or 1, 1)))

    candidate_vals = {f: candidate.get(f) for f in FINANCIAL_FEATURES if f != "log_total_assets"}
    # select_candidates()가 반환하는 후보 행에는 total_assets 없이 log_total_assets만
    # 있으므로 그걸 우선 쓰고, 유니버스 원본 행(total_assets 보유)이 넘어온 경우엔
    # 거기서 계산한다.
    if candidate.get("log_total_assets") is not None and not pd.isna(candidate.get("log_total_assets")):
        candidate_vals["log_total_assets"] = float(candidate["log_total_assets"])
    else:
        candidate_vals["log_total_assets"] = float(np.log(max(candidate.get("total_assets") or 1, 1)))

    metric_scores: dict[str, float | None] = {}
    for feat in FINANCIAL_FEATURES:
        t, c, std = target_vals.get(feat), candidate_vals.get(feat), stds.get(feat)
        if t is None or c is None or pd.isna(t) or pd.isna(c) or not std or pd.isna(std):
            metric_scores[feat] = None
            continue
        metric_scores[feat] = _gaussian_similarity((t - c) / std)

    # 사업의 내용을 못 파싱한 기업은 임베딩이 0벡터라 코사인 유사도가 정확히 0으로
    # 나온다. 이걸 "사업 유사도 0%"로 세면 데이터가 없다는 이유만으로 순위가
    # 밀리므로, 재무비율 결측과 똑같이 '데이터 없음'(None)으로 처리해 가중평균에서
    # 빼고 나머지 항목으로만 종합 점수를 낸다.
    text_similarity = candidate.get("text_similarity")
    if text_similarity is None or pd.isna(text_similarity) or abs(float(text_similarity)) < EMBEDDING_EPSILON:
        business_score = None
    else:
        business_score = float(text_similarity) * 100

    parts = {"business": business_score, **metric_scores}
    weighted_sum, weight_total, n_scored = 0.0, 0.0, 0
    for key, score in parts.items():
        if score is None:
            continue
        weighted_sum += WEIGHTS[key] * score
        weight_total += WEIGHTS[key]
        n_scored += 1
    overall = weighted_sum / weight_total if weight_total > 0 else None

    def avg(*keys):
        vals = [metric_scores[k] for k in keys if metric_scores.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    radar = {
        "사업유사도": business_score,
        "성장률": avg("revenue_growth", "operating_margin"),
        "재무구조": avg("debt_ratio", "asset_turnover"),
        "규모": metric_scores.get("log_total_assets"),
    }

    industry_label, industry_ratio = _industry_match(target["induty_code"], candidate["induty_code"])

    metric_rows = [{"label": "사업 유사도", "score": business_score}]
    metric_rows.append(
        {
            "label": "산업분류",
            "score": None,
            "text": industry_label,
            "ratio": industry_ratio,
        }
    )
    for feat in FINANCIAL_FEATURES:
        metric_rows.append({"label": METRIC_LABELS[feat], "score": metric_scores[feat]})

    return {
        "overall": overall,
        "industry_name": midclass_name(candidate["induty_code"]),
        "metric_rows": metric_rows,
        "radar": radar,
        # 종합 점수가 몇 개 항목으로 계산됐는지. 결측이 많은 기업은 점수 자체를
        # 곧이곧대로 믿으면 안 되므로 화면에 같이 표시한다.
        "n_scored": n_scored,
        "n_scorable": len(parts),
    }


def rank_candidates(target: dict, candidates: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """후보 전체의 종합 유사도를 계산해 높은 순으로 순위를 매긴다.

    select_candidates()가 돌려주는 순서는 2차 필터(재무비율 거리)만 반영한
    순서라, 사업 텍스트 유사도까지 가중평균한 종합 유사도 순위와는 다를 수 있다.
    스코어카드에서 "몇 위짜리 회사를 고른 것인지" 보여주려면 이 함수의 순위를 쓴다.
    """
    stds = reference_stds(universe)
    cards = [build_scorecard(target, row, stds=stds) for _, row in candidates.iterrows()]

    ranked = candidates.copy()
    ranked["overall_score"] = [c["overall"] for c in cards]
    ranked["n_scored"] = [c["n_scored"] for c in cards]
    ranked["n_scorable"] = [c["n_scorable"] for c in cards]
    ranked = ranked.sort_values("overall_score", ascending=False, na_position="last").reset_index(drop=True)
    ranked["similarity_rank"] = range(1, len(ranked) + 1)
    return ranked
