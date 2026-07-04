"""
trend_data.py — 구글 트렌드(pytrends) 연결 + pkl 저장/로드.

이 환경은 네트워크가 막혀 실제 호출은 사용자 로컬에서 돈다.
pytrends는 API 키가 필요 없다(비공식 라이브러리).

pkl 저장: 수집한 트렌드 시계열을 축적 → 나중에 불러와 국면 분석.
"""
from __future__ import annotations
import os, pickle, time


def fetch_trend(keyword, timeframe="today 12-m", geo=""):
    """
    구글 트렌드에서 키워드 시계열 + 관련 검색어를 받아온다.
    timeframe 예: 'today 3-m'(최근3개월), 'today 12-m', 'now 7-d'
    geo 예: ''(전세계), 'KR'(한국), 'US'
    반환: dict(series, dates, rising_count, top_count)
    """
    from pytrends.request import TrendReq
    py = TrendReq(hl="ko", tz=540)
    py.build_payload([keyword], timeframe=timeframe, geo=geo)

    # 시계열 관심도
    df = py.interest_over_time()
    if df.empty:
        raise RuntimeError(f"'{keyword}' 트렌드 데이터 없음")
    series = df[keyword].tolist()
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    # 관련 검색어(급상승/고정) 개수 → 양적/포화 신호
    rising_count = top_count = 0
    has_related = False
    try:
        related = py.related_queries()
        rq = related.get(keyword, {})
        got = False
        if rq.get("rising") is not None:
            rising_count = len(rq["rising"]); got = True
        if rq.get("top") is not None:
            top_count = len(rq["top"]); got = True
        has_related = got
    except Exception:
        pass

    return {
        "keyword": keyword, "timeframe": timeframe, "geo": geo,
        "series": series, "dates": dates,
        "rising_count": rising_count, "top_count": top_count,
        "has_related": has_related,
        "fetched_at": time.time(),
    }


# ── pkl 저장/로드 (임용 앱과 동일 방식) ─────────────────────
def save_snapshots(snapshots, path):
    """수집한 트렌드 스냅샷들을 pkl로 저장."""
    with open(path, "wb") as f:
        pickle.dump(snapshots, f)


def load_snapshots(path):
    """pkl에서 스냅샷 리스트 복원. 없으면 빈 리스트."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def is_valid_pkl(raw_bytes):
    """
    업로드된 파일이 유효한 pkl인지 코드에서 판별.
    (모바일에서 모든 파일형식 업로드 허용 → 여기서 pkl만 통과)
    """
    try:
        pickle.loads(raw_bytes)
        return True
    except Exception:
        return False
