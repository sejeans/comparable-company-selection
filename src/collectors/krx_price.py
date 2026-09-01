"""KRX 주가/지수 수집.

pykrx의 티커목록/지수 조회 API는 최근 KRX 정책 변경으로 KRX_ID/KRX_PW
로그인을 요구해서 무인증으로는 실패한다 (개별 종목 시세 get_market_ohlcv는
로그인 없이 동작 확인함). 로그인 없이 전부 되는 FinanceDataReader로 통일한다.
"""
import pandas as pd
import FinanceDataReader as fdr

KOSPI_TICKER = "KS11"


def get_stock_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """개별 종목 일별 시세(OHLCV)를 조회한다. ticker는 6자리 종목코드."""
    return fdr.DataReader(ticker, start, end)


def get_kospi_index(start: str, end: str) -> pd.DataFrame:
    """KOSPI 지수 일별 시세를 조회한다."""
    return fdr.DataReader(KOSPI_TICKER, start, end)


def get_weekly_returns(price_df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """일별 종가를 주간 수익률로 변환한다 (베타 회귀용)."""
    weekly_close = price_df[price_col].resample("W-FRI").last()
    return weekly_close.pct_change().dropna()
