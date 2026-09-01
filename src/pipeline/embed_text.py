"""'사업의 내용' 텍스트를 문장임베딩 벡터로 변환해 universe parquet에 붙인다.

모델은 한국어 특화 sentence-transformers 모델을 쓴다. 모델의 max_seq_length를
넘는 부분은 어차피 자동 잘리기 때문에, 토큰화 비용을 줄이려고 미리
DESCRIPTION_TRUNCATE_CHARS 만큼만 잘라서 넣는다 (보통 "사업의 개요" 도입부에
가장 응축된 설명이 나와서 앞부분만 써도 됨).

실행 예:
    python -m src.pipeline.embed_text --year 2023
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.build_universe import DATA_PROCESSED

MODEL_NAME = "jhgan/ko-sroberta-multitask"
DESCRIPTION_TRUNCATE_CHARS = 3000


def embed_universe(year: int) -> Path:
    from sentence_transformers import SentenceTransformer

    in_path = DATA_PROCESSED / f"universe_{year}.parquet"
    df = pd.read_parquet(in_path)

    texts = df["business_description"].fillna("").str.slice(0, DESCRIPTION_TRUNCATE_CHARS)
    has_text = texts.str.len() > 0

    print(f"[embed_text] {len(df)}개 중 텍스트 있는 {has_text.sum()}개 임베딩 생성 (model={MODEL_NAME})")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = np.zeros((len(df), model.get_embedding_dimension()), dtype=np.float32)
    embeddings[has_text.values] = model.encode(
        texts[has_text].tolist(), show_progress_bar=True, normalize_embeddings=True
    )

    df["embedding"] = list(embeddings)
    out_path = DATA_PROCESSED / f"universe_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[embed_text] 저장 완료: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args()

    embed_universe(args.year)
