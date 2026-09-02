"""전체 상장사 대상으로 Phase 1 collector들을 일괄 실행해 유사기업 후보 풀 원본 데이터를 만든다.

재시작 가능한 구조:
- 결과를 `data/raw/universe_{year}_rows.jsonl`(성공)과
  `_failures.jsonl`(실패)에 한 줄씩 append하면서 진행한다.
- 재실행하면 이미 두 파일에 기록된 corp_code는 건너뛰고 이어서 처리한다.
- DART 일일 호출한도(status='020')에 걸리면 그 자리에서 멈춘다.
  (재무제표 API(finstate_all)만 예외적으로 에러를 raise하지 않고 빈 결과로
  삼켜버려서, 정확히 그 호출에서 한도를 넘기면 해당 회사 하나는 재무비율이
  비어있는 채로 성공 처리될 수 있음 — 다음 회사에서 바로 감지되어 멈춘다.)

실행 예:
    python -m src.pipeline.build_universe --year 2023 --limit 30
    python -m src.pipeline.build_universe --year 2023          # 전체
"""
import argparse
import json
import time
from pathlib import Path

import requests

from src.collectors.dart_company import get_company_info, get_listed_corp_codes
from src.collectors.dart_financials import compute_financial_ratios, get_financial_statements
from src.collectors.dart_report_text import extract_business_description, find_annual_report_rcept_no
from src.paths import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT  # noqa: F401 (기존 import 경로 호환)

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2


def _paths(year: int) -> tuple[Path, Path]:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    return DATA_RAW / f"universe_{year}_rows.jsonl", DATA_RAW / f"universe_{year}_failures.jsonl"


def _load_done_corp_codes(*paths: Path) -> set[str]:
    done = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done.add(json.loads(line)["corp_code"])
    return done


def _is_quota_exceeded(exc: Exception) -> bool:
    arg = exc.args[0] if exc.args else None
    return isinstance(arg, dict) and arg.get("status") == "020"


def _collect_one_attempt(corp_code: str, stock_code: str, corp_name: str, year: int) -> dict:
    info = get_company_info(corp_code)
    fs = get_financial_statements(corp_code, year)
    ratios = compute_financial_ratios(fs)
    rcept_no = find_annual_report_rcept_no(corp_code, year)
    description = extract_business_description(rcept_no) if rcept_no else None

    return {
        "corp_code": corp_code,
        "stock_code": stock_code,
        "corp_name": corp_name,
        "year": year,
        "induty_code": info.get("induty_code"),
        "rcept_no": rcept_no,
        "business_description": description,
        **ratios,
    }


def collect_one(corp_code: str, stock_code: str, corp_name: str, year: int) -> dict:
    """일시적 네트워크 오류(SSL/커넥션 등)는 재시도하고, DART API 에러(사용한도초과/
    데이터없음 등 ValueError)는 즉시 위로 올려서 호출부가 처리하게 한다."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _collect_one_attempt(corp_code, stock_code, corp_name, year)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    raise last_exc


def build_universe(year: int, limit: int | None = None, sleep_sec: float = 0.0) -> None:
    rows_path, fail_path = _paths(year)
    done = _load_done_corp_codes(rows_path, fail_path)

    listed = get_listed_corp_codes()
    if limit:
        listed = listed.head(limit)
    targets = listed[~listed["corp_code"].isin(done)]

    print(
        f"[build_universe] year={year} 대상 {len(listed)}개 중 "
        f"이미 처리 {len(done & set(listed['corp_code']))}개, 남은 {len(targets)}개"
    )

    t0 = time.time()
    processed = 0
    with open(rows_path, "a", encoding="utf-8") as rf, open(fail_path, "a", encoding="utf-8") as ff:
        for row in targets.itertuples(index=False):
            corp_code, corp_name, _corp_eng_name, stock_code, _modify_date = row
            try:
                result = collect_one(corp_code, stock_code, corp_name, year)
                rf.write(json.dumps(result, ensure_ascii=False) + "\n")
                rf.flush()
            except Exception as e:
                if _is_quota_exceeded(e):
                    print(
                        f"[build_universe] DART 일일 호출한도 초과. "
                        f"{processed}/{len(targets)} 처리 후 중단 — 재실행하면 이어서 진행됩니다."
                    )
                    return
                ff.write(
                    json.dumps(
                        {"corp_code": corp_code, "corp_name": corp_name, "error": str(e)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ff.flush()

            processed += 1
            if processed % 20 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta_min = (len(targets) - processed) / rate / 60 if rate > 0 else float("inf")
                print(f"[build_universe] {processed}/{len(targets)} 처리 ({elapsed:.0f}s 경과, 남은 예상 {eta_min:.0f}분)")

            if sleep_sec:
                time.sleep(sleep_sec)

    print(f"[build_universe] 이번 실행에서 {processed}개 처리 완료")


def load_universe_df(year: int):
    """jsonl 원본을 DataFrame으로 읽는다 (성공 건만)."""
    import pandas as pd

    rows_path, _ = _paths(year)
    if not rows_path.exists():
        return pd.DataFrame()
    return pd.read_json(rows_path, lines=True)


def finalize_to_parquet(year: int) -> Path:
    """jsonl 누적 결과를 data/processed/universe_{year}.parquet으로 저장한다."""
    df = load_universe_df(year)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / f"universe_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[build_universe] {len(df)}행 저장: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--limit", type=int, default=None, help="상위 N개만 처리 (테스트용)")
    parser.add_argument("--sleep", type=float, default=0.0, help="회사 처리 사이 대기(초)")
    parser.add_argument("--finalize", action="store_true", help="처리 후 parquet으로 저장")
    args = parser.parse_args()

    build_universe(args.year, limit=args.limit, sleep_sec=args.sleep)
    if args.finalize:
        finalize_to_parquet(args.year)
