"""WACC 산출 + 베타풀 조합에 따른 민감도 분석.

개요.md의 핵심 어필 포인트: "베타풀 후보 20개 중 어떤 5개를 고르느냐에 따라
WACC이 얼마나 벌어지는가"를 정량화한다. 조합 수가 감당할 수준(기본 20만개 이하)이면
전수조합(itertools.combinations)을 그대로 다 돌리고, 그 이상이면 랜덤 샘플링으로
근사한다.
"""
import random
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

DEFAULT_MAX_COMBINATIONS = 200_000


def cost_of_equity_capm(beta: float, risk_free_rate: float, equity_risk_premium: float) -> float:
    return risk_free_rate + beta * equity_risk_premium


def compute_wacc(
    beta: float,
    risk_free_rate: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    tax_rate: float,
    debt_weight: float,
) -> float:
    """자기자본비용(CAPM) + 세후 타인자본비용을 자본구조로 가중평균한다.

    debt_weight/equity_weight는 장부가 기준 부채비율에서 유도한 값을 쓴다는
    전제 (시가총액 기준이 이론적으로 더 맞지만, 이 프로젝트 스코프에서는
    재무제표로 바로 구할 수 있는 장부가 기준으로 단순화함).
    """
    equity_weight = 1 - debt_weight
    cost_of_equity = cost_of_equity_capm(beta, risk_free_rate, equity_risk_premium)
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    return equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt


def debt_weight_from_ratio(debt_ratio: float) -> float:
    """debt_ratio = 부채총계/자본총계 (부채비율) 을 D/(D+E)로 변환한다."""
    return debt_ratio / (1 + debt_ratio)


def beta_pool_sensitivity(
    betas: list[float],
    k: int,
    wacc_kwargs: dict,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    random_seed: int = 42,
) -> pd.DataFrame:
    """베타풀에서 k개를 고르는 모든(또는 샘플링한) 조합에 대해 WACC을 계산한다.

    wacc_kwargs: compute_wacc의 beta를 제외한 나머지 인자
      (risk_free_rate, equity_risk_premium, cost_of_debt, tax_rate, debt_weight)
    """
    betas = list(betas)
    n = len(betas)
    if k > n:
        raise ValueError(f"베타풀 크기({n})보다 큰 조합 크기({k})는 뽑을 수 없습니다")

    total = comb(n, k)
    if total <= max_combinations:
        index_combos = list(combinations(range(n), k))
    else:
        rng = random.Random(random_seed)
        seen = set()
        index_combos = []
        while len(index_combos) < max_combinations:
            combo = tuple(sorted(rng.sample(range(n), k)))
            if combo not in seen:
                seen.add(combo)
                index_combos.append(combo)

    rows = []
    for combo in index_combos:
        chosen = [betas[i] for i in combo]
        beta_median = float(np.median(chosen))
        wacc = compute_wacc(beta_median, **wacc_kwargs)
        rows.append({"combo": combo, "beta_median": beta_median, "wacc": wacc})

    return pd.DataFrame(rows)


def sensitivity_summary(sensitivity_df: pd.DataFrame) -> dict:
    wacc = sensitivity_df["wacc"]
    return {
        "n_combinations": len(sensitivity_df),
        "wacc_min": float(wacc.min()),
        "wacc_max": float(wacc.max()),
        "wacc_range": float(wacc.max() - wacc.min()),
        "wacc_mean": float(wacc.mean()),
        "wacc_std": float(wacc.std()),
    }
