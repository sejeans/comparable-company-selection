"""Phase 1 검증: 임의 상장사 5곳으로 5가지 데이터가 정상적으로 나오는지 확인.

실행: comp_comp_selec/Scripts/python.exe notebooks/phase1_validate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collectors.dart_company import find_corp_code, get_company_info, get_listed_corp_codes
from src.collectors.dart_financials import compute_financial_ratios, get_financial_statements
from src.collectors.dart_report_text import extract_business_description, find_annual_report_rcept_no
from src.collectors.krx_price import get_kospi_index, get_stock_prices

SAMPLE_TICKERS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "051910": "LG화학",
}

YEAR = 2023


def main():
    print(f"[0] 상장기업 전체 목록: {len(get_listed_corp_codes())}개\n")

    kospi = get_kospi_index(f"{YEAR}-01-01", f"{YEAR}-12-31")
    print(f"[KOSPI] {len(kospi)}일치 지수 데이터 확보\n")

    for ticker, name in SAMPLE_TICKERS.items():
        print(f"===== {name} ({ticker}) =====")

        corp_code = find_corp_code(ticker)
        print(f"  corp_code: {corp_code}")

        info = get_company_info(corp_code)
        print(f"  업종코드(induty_code): {info.get('induty_code')}")

        fs = get_financial_statements(corp_code, YEAR)
        ratios = compute_financial_ratios(fs)
        print(f"  재무비율: {ratios}")

        rcept_no = find_annual_report_rcept_no(corp_code, YEAR)
        if rcept_no:
            desc = extract_business_description(rcept_no)
            desc_len = len(desc) if desc else 0
            preview = (desc[:80] + "...") if desc else None
            print(f"  사업의 내용: {desc_len}자 / 미리보기: {preview}")
        else:
            print("  사업의 내용: 사업보고서 접수번호를 찾지 못함")

        prices = get_stock_prices(ticker, f"{YEAR}-01-01", f"{YEAR}-12-31")
        print(f"  주가: {len(prices)}일치 시세 확보 (마지막 종가 {prices['Close'].iloc[-1]})")

        print()


if __name__ == "__main__":
    main()
