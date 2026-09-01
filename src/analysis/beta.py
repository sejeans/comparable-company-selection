"""KRX 주간수익률 vs KOSPI 주간수익률 회귀로 개별 종목 베타를 계산한다."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.collectors.krx_price import get_kospi_index, get_stock_prices, get_weekly_returns

MIN_OBSERVATIONS = 20  # 약 20주 미만이면 회귀 신뢰도가 낮다고 보고 베타를 None 처리


def compute_beta(
    stock_code: str,
    end_date: str,
    years: int = 3,
    kospi_weekly: pd.Series | None = None,
) -> dict:
    """개별 종목의 베타(KOSPI 대비 주간수익률 회귀 기울기)를 계산한다.

    kospi_weekly를 넘기면 재계산하지 않는다 (풀 전체를 돌릴 때 KOSPI는 1번만 조회).
    """
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

    if kospi_weekly is None:
        kospi_weekly = get_weekly_returns(get_kospi_index(start_date, end_date))

    try:
        stock_prices = get_stock_prices(stock_code, start_date, end_date)
    except Exception:
        stock_prices = pd.DataFrame()

    if stock_prices.empty:
        return {"stock_code": stock_code, "beta": None, "r_squared": None, "n_obs": 0}

    stock_weekly = get_weekly_returns(stock_prices)
    aligned = pd.concat([stock_weekly.rename("stock"), kospi_weekly.rename("market")], axis=1).dropna()

    if len(aligned) < MIN_OBSERVATIONS:
        return {"stock_code": stock_code, "beta": None, "r_squared": None, "n_obs": len(aligned)}

    X = sm.add_constant(aligned["market"])
    model = sm.OLS(aligned["stock"], X).fit()
    return {
        "stock_code": stock_code,
        "beta": float(model.params["market"]),
        "r_squared": float(model.rsquared),
        "n_obs": len(aligned),
    }


def compute_beta_pool(stock_codes: list[str], end_date: str, years: int = 3) -> pd.DataFrame:
    """후보 풀 전체의 베타를 계산한다. KOSPI 지수는 한 번만 받아서 재사용한다."""
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    kospi_weekly = get_weekly_returns(get_kospi_index(start_date, end_date))

    rows = [compute_beta(code, end_date, years=years, kospi_weekly=kospi_weekly) for code in stock_codes]
    return pd.DataFrame(rows)


def median_beta(betas: pd.Series) -> float:
    return float(betas.dropna().median())
