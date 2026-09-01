"""Phase 4 검증: Phase 3 후보풀(20개)로 개별 베타를 계산하고, 20개 중 5개를 고르는
조합에 따라 WACC이 얼마나 벌어지는지 확인한다 (이 프로젝트의 핵심 어필 포인트).

실행: comp_comp_selec/Scripts/python.exe notebooks/phase4_validate.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis.beta import compute_beta_pool, median_beta
from src.analysis.similarity import select_candidates
from src.analysis.wacc import beta_pool_sensitivity, debt_weight_from_ratio, sensitivity_summary

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "universe_2023.parquet"
TARGET_NAME = "LG화학"
END_DATE = "2023-12-31"
BETA_YEARS = 3
POOL_SIZE = 20
CHOOSE_K = 5

WACC_PARAMS = {
    "risk_free_rate": 0.035,
    "equity_risk_premium": 0.06,
    "cost_of_debt": 0.05,
    "tax_rate": 0.242,
}


def main():
    universe = pd.read_parquet(UNIVERSE_PATH)
    target_row = universe[universe["corp_name"] == TARGET_NAME].iloc[0]

    target = {
        "corp_code": target_row["corp_code"],
        "induty_code": target_row["induty_code"],
        "embedding": target_row["embedding"],
        "revenue_growth": target_row["revenue_growth"],
        "operating_margin": target_row["operating_margin"],
        "debt_ratio": target_row["debt_ratio"],
        "asset_turnover": target_row["asset_turnover"],
        "total_assets": target_row["total_assets"],
    }

    print(f"[1] {TARGET_NAME} 유사기업 후보 {POOL_SIZE}개 산출...")
    candidates = select_candidates(target, universe, top_n_final=POOL_SIZE)
    print(candidates[["corp_name", "text_similarity", "financial_distance"]].to_string())

    print(f"\n[2] 후보 {len(candidates)}개 베타 계산 중 (최근 {BETA_YEARS}년 주간수익률 회귀)...")
    t0 = time.time()
    beta_df = compute_beta_pool(candidates["stock_code"].tolist(), end_date=END_DATE, years=BETA_YEARS)
    beta_df = beta_df.merge(
        candidates[["stock_code", "corp_name"]], on="stock_code", how="left"
    )
    print(f"    {time.time()-t0:.0f}초 소요")
    print(beta_df[["corp_name", "stock_code", "beta", "r_squared", "n_obs"]].to_string())

    valid_betas = beta_df["beta"].dropna().tolist()
    print(f"\n[3] 유효 베타 {len(valid_betas)}/{len(beta_df)}개, 전체 풀 중앙값 베타: {median_beta(beta_df['beta']):.3f}")

    debt_weight = debt_weight_from_ratio(target["debt_ratio"])
    wacc_kwargs = {**WACC_PARAMS, "debt_weight": debt_weight}
    print(f"    타겟 debt_weight(D/(D+E))={debt_weight:.3f}, WACC 파라미터={WACC_PARAMS}")

    print(f"\n[4] {len(valid_betas)}개 중 {CHOOSE_K}개를 고르는 모든 조합의 WACC 분포...")
    sens = beta_pool_sensitivity(valid_betas, k=CHOOSE_K, wacc_kwargs=wacc_kwargs)
    summary = sensitivity_summary(sens)
    print(f"    조합 수: {summary['n_combinations']}")
    print(f"    WACC 범위: {summary['wacc_min']*100:.2f}% ~ {summary['wacc_max']*100:.2f}% "
          f"(폭 {summary['wacc_range']*100:.2f}%p)")
    print(f"    평균 {summary['wacc_mean']*100:.2f}%, 표준편차 {summary['wacc_std']*100:.2f}%p")


if __name__ == "__main__":
    main()
