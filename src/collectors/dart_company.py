"""DART 고유번호/기업개황 수집."""
import os

import FinanceDataReader as fdr
import OpenDartReader
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# 실제로 현재 거래되는 시장만 유사기업 후보 풀에 포함한다.
# KONEX는 초기 중소기업 전용 시장이라 거래가 희박해 베타 추정에 부적합하므로 제외.
INCLUDE_MARKETS = {"KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"}

_dart = None


def get_client() -> OpenDartReader:
    """DART API 클라이언트를 생성(1회)해서 재사용한다."""
    global _dart
    if _dart is None:
        api_key = os.getenv("DART_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DART_API_KEY가 .env에 설정되어 있지 않습니다.")
        _dart = OpenDartReader(api_key)
    return _dart


def get_listed_corp_codes() -> pd.DataFrame:
    """DART 전체 법인 중 '현재' 상장된 법인만 반환한다.

    DART corp_codes의 stock_code는 상장폐지된 종목의 옛 코드도 그대로 남아있어서
    (예: 한빛네트 036720 등 실제로는 이미 폐지된 종목), stock_code 존재 여부만으로는
    필터링이 안 된다. FinanceDataReader의 실시간 KRX 상장종목 목록과 교집합을 취한다.
    """
    dart = get_client()
    corp_codes = dart.corp_codes
    has_stock_code = corp_codes["stock_code"].notna() & (corp_codes["stock_code"].str.strip() != "")

    krx_listing = fdr.StockListing("KRX")
    active_codes = set(krx_listing.loc[krx_listing["Market"].isin(INCLUDE_MARKETS), "Code"])

    listed = corp_codes[has_stock_code & corp_codes["stock_code"].isin(active_codes)]
    return listed.reset_index(drop=True)


def find_corp_code(stock_code: str) -> str | None:
    """종목코드(예: 005930)로 DART corp_code를 찾는다."""
    listed = get_listed_corp_codes()
    row = listed[listed["stock_code"] == stock_code]
    if row.empty:
        return None
    return row.iloc[0]["corp_code"]


def get_company_info(corp_code: str) -> dict:
    """기업개황(업종코드 induty_code 포함)을 조회한다."""
    dart = get_client()
    return dart.company(corp_code)
