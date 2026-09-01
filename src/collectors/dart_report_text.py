"""사업보고서 원문에서 'N. 사업의 내용' 섹션 텍스트를 추출한다.

DART 원문은 dart4.xsd 기반 커스텀 XML이며, 최상위 장(章)은
<SECTION-1><TITLE>II. 사업의 내용</TITLE>...</SECTION-1> 구조를 가진다.

주의: sub_docs()가 주는 목차의 eleId는 원문 TITLE의 ATOCID와 항상
일치하지는 않는다 (정정공시처럼 "정정 신고(보고)" 커버 섹션이 앞에
붙는 필정정 사업보고서에서 실제로 어긋나는 사례를 확인함 — eleId=11이
가리키는 것은 "II. 사업의 내용"이 아니라 커버 섹션의 "1. 요약재무정보"였음).
그래서 eleId/ATOCID 매칭 대신, sub_docs로 얻은 "정확한 제목 텍스트"를
원문 TITLE 태그들과 직접 문자열 비교해서 찾는다.

원문 XML은 군데군데 escape 안 된 '&' 등으로 인해 strict XML 파서로는
파싱이 중간에 끊긴다. lxml의 recover 모드로 우회한다.
"""
from lxml import etree

from .dart_company import get_client


def find_annual_report_rcept_no(corp_code: str, year: int) -> str | None:
    """특정 사업연도(year)의 사업보고서 접수번호를 찾는다.

    사업보고서는 보통 다음 해 3월에 제출되므로 조회 기간을 다음 해 전체로 잡는다.
    """
    dart = get_client()
    start = f"{year + 1}-01-01"
    end = f"{year + 1}-12-31"
    reports = dart.list(corp_code, kind="A", start=start, end=end)
    if reports is None or reports.empty:
        return None
    # 정정 공시는 "[기재정정]사업보고서 (2023.12)"처럼 접두사가 붙어 startswith로는 못 잡는다.
    target = f"사업보고서 ({year}.12)"
    match = reports[reports["report_nm"].str.contains(target, regex=False, na=False)]
    if match.empty:
        return None
    # DART list()는 최신순으로 오므로 첫 행이 가장 최근(=최신 정정) 공시.
    return match.iloc[0]["rcept_no"]


def _normalize_title(text: str) -> str:
    return " ".join(text.split())


def _find_business_section_title(rcept_no: str) -> str | None:
    """목차에서 최상위 '사업의 내용' 장(章)의 정확한 제목 텍스트를 찾는다."""
    dart = get_client()
    subs = dart.sub_docs(rcept_no)
    if subs is None or subs.empty:
        return None
    # 최상위 장만: "II. 사업의 내용" (하위 목차 "1. 사업의 개요" 등은 제외)
    top_level = subs[
        subs["title"].str.contains("사업의 내용", na=False)
        & subs["title"].str.match(r"^[IVXLC]+\.")
    ]
    if top_level.empty:
        return None
    return _normalize_title(top_level.iloc[0]["title"])


def extract_business_description(rcept_no: str) -> str | None:
    """사업보고서 rcept_no로부터 '사업의 내용' 섹션 전체 텍스트를 반환한다."""
    section_title = _find_business_section_title(rcept_no)
    if section_title is None:
        return None

    dart = get_client()
    raw = dart.document(rcept_no)

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    root = etree.fromstring(raw.encode("utf-8"), parser=parser)

    target = None
    for title_el in root.findall(".//TITLE"):
        if title_el.text and _normalize_title(title_el.text) == section_title:
            target = title_el
            break
    if target is None:
        return None

    section = target.getparent()
    text = "".join(section.itertext())
    return " ".join(text.split())
