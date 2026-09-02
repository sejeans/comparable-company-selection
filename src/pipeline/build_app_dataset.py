"""배포용 경량 데이터셋 생성.

data/processed/universe_{year}.parquet은 "사업의 내용" 원문 텍스트를 통째로
갖고 있어서 200MB에 육박한다 (GitHub 100MB 단일 파일 제한 초과, Streamlit
Community Cloud 배포 불가). 앱 실행에는 원문 텍스트가 필요 없고 이미 계산된
임베딩 벡터만 있으면 되므로, 원문 텍스트 컬럼을 뺀 경량 버전을 따로 만들어
git에 커밋하고 배포용으로 쓴다.

실행 예:
    python -m src.pipeline.build_app_dataset --year 2023
"""
import argparse
from pathlib import Path

import pandas as pd

from src.paths import DATA_PROCESSED

DROP_COLUMNS = ["business_description", "rcept_no"]


def build_app_dataset(year: int) -> Path:
    in_path = DATA_PROCESSED / f"universe_{year}.parquet"
    df = pd.read_parquet(in_path)
    slim = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    out_path = DATA_PROCESSED / f"universe_{year}_app.parquet"
    slim.to_parquet(out_path, index=False)
    print(f"[build_app_dataset] {in_path.stat().st_size / 1e6:.1f}MB → {out_path.stat().st_size / 1e6:.1f}MB: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args()

    build_app_dataset(args.year)
