"""Phase 3 검증: 알려진 동종업계 기업 2~3쌍으로 실제 유사도 상위(top 20)에 잡히는지 확인.

실행: comp_comp_selec/Scripts/python.exe notebooks/phase3_validate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis.similarity import select_candidates

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "universe_2023.parquet"

# (타겟 회사명, 여기 잡혀야 하는 known peer 회사명)
KNOWN_PAIRS = [
    ("LG화학", "롯데케미칼"),
    ("하나금융지주", "우리금융지주"),
    ("현대제철", "한국철강"),
]


def make_target(row: pd.Series) -> dict:
    return {
        "corp_code": row["corp_code"],
        "induty_code": row["induty_code"],
        "embedding": row["embedding"],
        "revenue_growth": row["revenue_growth"],
        "operating_margin": row["operating_margin"],
        "debt_ratio": row["debt_ratio"],
        "asset_turnover": row["asset_turnover"],
        "total_assets": row["total_assets"],
    }


def main():
    universe = pd.read_parquet(UNIVERSE_PATH)

    for target_name, expected_peer in KNOWN_PAIRS:
        target_rows = universe[universe["corp_name"] == target_name]
        if target_rows.empty:
            print(f"[SKIP] '{target_name}' 유니버스에 없음")
            continue
        target_row = target_rows.iloc[0]
        target = make_target(target_row)

        candidates = select_candidates(target, universe, top_n_final=20)

        names = candidates["corp_name"].tolist()
        if expected_peer in names:
            rank = names.index(expected_peer) + 1
            print(f"[PASS] {target_name} -> {expected_peer}: top20 중 {rank}위")
        else:
            print(f"[FAIL] {target_name} -> {expected_peer}: top20에 없음")

        print(f"       업종매치자릿수={candidates['industry_match_level'].iloc[0]}, "
              f"top5={names[:5]}")
        print()


if __name__ == "__main__":
    main()
