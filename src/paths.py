"""프로젝트 공통 경로 상수.

이 모듈은 의존성이 없어야 한다. 예전에는 이 상수들이 build_universe.py에 있었는데,
그러면 `similarity.py → embed_text.py → build_universe.py → dart_company.py` 순으로
딸려 들어가 대시보드가 쓰지도 않는 OpenDartReader/bs4/lxml까지 import 타임에
로드됐다 (Streamlit Cloud 배포 시 불필요한 의존성이 됨).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
