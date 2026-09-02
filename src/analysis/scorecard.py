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


def build_scorecard(target: dict, candidate: pd.Series, universe: pd.DataFrame) -> dict:
    """target(평가대상) vs candidate(후보 기업 1개)의 항목별/종합 유사도를 계산한다.

    universe는 재무비율 표준편차를 구할 참조 모집단(전체 유니버스)이다.
    """
    ref = universe.copy()
    ref["log_total_assets"] = np.log(ref["total_assets"].clip(lower=1))
    stds = ref[FINANCIAL_FEATURES].std()

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

    text_similarity = candidate.get("text_similarity")
    business_score = float(text_similarity) * 100 if text_similarity is not None and not pd.isna(text_similarity) else None

    parts = {"business": business_score, **metric_scores}
    weighted_sum, weight_total = 0.0, 0.0
    for key, score in parts.items():
        if score is None:
            continue
        weighted_sum += WEIGHTS[key] * score
        weight_total += WEIGHTS[key]
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
    }
