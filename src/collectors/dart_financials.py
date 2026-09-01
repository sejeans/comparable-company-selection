"""DART 재무제표 수집 및 재무비율 계산."""
import re

import pandas as pd

from .dart_company import get_client

REPRT_CODE_ANNUAL = "11011"  # 사업보고서(연간) 기준 재무제표

# 계정명이 회사/업종마다 다르게 표기된다 (예: "자산총계" vs "자산 총계", "매출액" vs "매출").
# 공백 제거 후 비교하고, 그래도 다른 표현은 대체명을 둔다.
ACCOUNT_ALIASES = {
    "revenue": ["매출액", "수익(매출액)", "영업수익", "매출"],
    "operating_income": ["영업이익", "영업이익(손실)"],
    "total_assets": ["자산총계"],
    "total_liabilities": ["부채총계"],
    "total_equity": ["자본총계"],
}


def _normalize(name: str) -> str:
    return re.sub(r"\s+", "", str(name))


def get_financial_statements(corp_code: str, year: int, fs_div: str = "CFS") -> pd.DataFrame:
    """사업보고서 재무제표를 조회한다. 연결(CFS)이 없으면 별도(OFS)로 재시도."""
    dart = get_client()
    fs = dart.finstate_all(corp_code, year, reprt_code=REPRT_CODE_ANNUAL, fs_div=fs_div)
    if fs is None or fs.empty:
        if fs_div == "CFS":
            return get_financial_statements(corp_code, year, fs_div="OFS")
        return pd.DataFrame()
    return fs


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _get_amounts(fs: pd.DataFrame, account_key: str, norm_names: pd.Series) -> tuple[float | None, float | None]:
    """(당기금액, 전기금액)을 반환. 공백 무시 + 별칭 목록을 순서대로 탐색."""
    for name in ACCOUNT_ALIASES[account_key]:
        row = fs[norm_names == _normalize(name)]
        if not row.empty:
            r = row.iloc[0]
            return _to_float(r.get("thstrm_amount")), _to_float(r.get("frmtrm_amount"))
    return None, None


def compute_financial_ratios(fs: pd.DataFrame) -> dict:
    """유사기업 재무비율 비교에 쓸 지표를 계산한다.

    매출성장률 / 영업이익률 / 부채비율 / 자산회전율 / 자산규모
    """
    if fs is None or fs.empty:
        return {}

    norm_names = fs["account_nm"].map(_normalize)

    revenue, revenue_prev = _get_amounts(fs, "revenue", norm_names)
    operating_income, _ = _get_amounts(fs, "operating_income", norm_names)
    total_assets, _ = _get_amounts(fs, "total_assets", norm_names)
    total_liabilities, _ = _get_amounts(fs, "total_liabilities", norm_names)
    total_equity, _ = _get_amounts(fs, "total_equity", norm_names)

    revenue_growth = (
        (revenue - revenue_prev) / revenue_prev
        if revenue is not None and revenue_prev not in (None, 0)
        else None
    )
    operating_margin = operating_income / revenue if operating_income is not None and revenue else None
    debt_ratio = total_liabilities / total_equity if total_liabilities is not None and total_equity else None
    asset_turnover = revenue / total_assets if revenue is not None and total_assets else None

    return {
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "operating_income": operating_income,
        "operating_margin": operating_margin,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_equity": total_equity,
        "debt_ratio": debt_ratio,
        "asset_turnover": asset_turnover,
    }
