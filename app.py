"""
app.py — 트렌드 평형 분석기 (Streamlit)

여론/관심 트렌드를 정·반·합 평형 엔진으로 국면 진단.
  - 구글 트렌드(pytrends) 수집 또는 가상 데이터
  - 평형 엔진: 양(상승)·음(하락) 신호가 평형 이루면 국면 확정,
    팽팽하면 '전환점·불확실'로 정직하게
  - LLM 속기사: 판단(엔진)을 사람이 읽을 문장으로 옮김(선택)
  - pkl 저장/불러오기 (모바일: 전체 파일 허용 + 코드에서 pkl만 판별)

실행: streamlit run app.py
"""
import sys, os, io, pickle, time
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
from core_base import Core, Judgment, Observation
from equilibrium import EquilibriumEngine
from trend_cores import (ThesisCore, AntithesisCore, SynthesisCore,
                         extract_signals, diagnose_phase)
from trend_data import (save_snapshots, load_snapshots, is_valid_pkl)

st.set_page_config(page_title="트렌드 평형 분석기", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
PKL_PATH = os.path.join(DATA_DIR, "trend_snapshots.pkl")


def analyze(signals, steps=30):
    """평형 엔진으로 국면 진단. 반환: (phase, 마지막 결과, 엔진)"""
    T, A, S = ThesisCore("정"), AntithesisCore("반"), SynthesisCore("합")
    eng = EquilibriumEngine(T, A, S)
    r = None
    for _ in range(steps):
        r = eng.step({"signals": signals})
    eq = eng.is_equilibrium()
    phase = diagnose_phase(r["synthesis"].stance, r["tension"], eq, signals)
    return phase, r, eng


# ══════════════════════════════════════════════════
st.title("📊 트렌드 평형 분석기")
st.caption("여론·관심 트렌드를 정·반·합 평형으로 진단. "
           "양(상승)·음(하락)이 평형이면 국면 확정, 팽팽하면 '전환점·불확실'로 정직하게.")

# 사이드바: pkl 저장/불러오기
st.sidebar.title("💾 데이터")
snapshots = load_snapshots(PKL_PATH)
st.sidebar.write(f"저장된 스냅샷: **{len(snapshots)}개**")

if snapshots:
    with open(PKL_PATH, "rb") as f:
        st.sidebar.download_button("⬇️ pkl 다운로드", f.read(),
                                   file_name="trend_snapshots.pkl",
                                   mime="application/octet-stream",
                                   use_container_width=True)

# 복원: 모든 파일 허용 + 코드에서 pkl만 판별 (임용 앱 방식)
with st.sidebar.expander("📥 pkl 불러오기(업로드)"):
    st.caption("모든 파일 선택 가능 — pkl만 자동 판별해 복원")
    ups = st.file_uploader("파일 (pkl)", type=None,
                           accept_multiple_files=True, key="restore")
    if ups and st.button("복원", key="restore_btn"):
        merged = list(snapshots)
        ok, skip = 0, 0
        for up in ups:
            raw = up.read()
            if is_valid_pkl(raw):
                try:
                    data = pickle.loads(raw)
                    if isinstance(data, list):
                        merged.extend(data)
                    else:
                        merged.append(data)
                    ok += 1
                except Exception:
                    skip += 1
            else:
                skip += 1
        save_snapshots(merged, PKL_PATH)
        st.success(f"{ok}개 pkl 복원, {skip}개 건너뜀")
        st.rerun()

tab1, tab_field, tab2 = st.tabs(["🔍 단일 키워드", "🕸️ 분야 종합 분석", "📈 저장된 스냅샷"])

# ── 탭 분야 종합 분석 ──────────────────────────────
with tab_field:
    st.subheader("분야 종합 분석")
    st.caption("키워드 하나 → LLM이 연관 키워드 확장 → 각각 트렌드 수집 → "
               "평형 엔진이 종합 판단. '이 분야가 어디로 가는가'를 입체적으로.")
    seed = st.text_input("분야 키워드", value="생성형 AI", key="field_seed")
    fc1, fc2 = st.columns(2)
    fgeo = fc1.selectbox("지역", ["전세계", "KR", "US"], key="field_geo")
    fgeo_code = {"전세계": "", "KR": "KR", "US": "US"}[fgeo]
    n_kw = fc2.slider("연관 키워드 수", 4, 10, 6, key="field_n")
    fkey = st.text_input("OpenAI Key (키워드 확장 + 속기)", type="password", key="field_key")

    if st.button("분야 분석 시작", type="primary", key="field_run"):
        if not fkey:
            st.warning("OpenAI 키가 필요합니다(연관 키워드 확장용).")
        else:
            try:
                from field_analysis import (expand_keywords, judge_one,
                                            synthesize_field, scribe_field)
                from trend_data import fetch_trend
                # 1) 키워드 확장
                with st.spinner("연관 키워드 확장 중..."):
                    keywords, kw_err = expand_keywords(seed, fkey, n=n_kw)
                if kw_err:
                    st.error(f"키워드 확장 실패 원인: {kw_err}")
                if len(keywords) <= 1:
                    st.warning("연관 키워드가 확장되지 않아 seed만 진행합니다. "
                               "위 오류를 확인하세요.")
                st.write("**탐색 키워드:**", ", ".join(keywords))

                # 2~3) 각 키워드 트렌드 수집 + 국면 판정
                per_kw = []
                prog = st.progress(0)
                for i, kw in enumerate(keywords):
                    try:
                        snap = fetch_trend(kw, timeframe="today 12-m", geo=fgeo_code)
                        res = judge_one(snap["series"], snap.get("rising_count", 0),
                                        snap.get("top_count", 0),
                                        snap.get("has_related", False))
                        if res:
                            per_kw.append({"keyword": kw, "stance": res["stance"],
                                           "phase": res["phase"], "series": snap["series"]})
                    except Exception as e:
                        st.caption(f"⚠️ '{kw}' 수집 실패: {e}")
                    prog.progress((i + 1) / len(keywords))

                if not per_kw:
                    st.error("트렌드를 하나도 못 받았습니다(네트워크/pytrends 확인).")
                else:
                    # 4) 종합
                    field = synthesize_field(per_kw)
                    st.session_state["field_result"] = {
                        "seed": seed, "field": field, "per_kw": per_kw, "key": fkey,
                    }
            except Exception as e:
                st.error(f"분석 실패: {e}")

    # 결과 표시(세션 유지)
    fr = st.session_state.get("field_result")
    if fr:
        field = fr["field"]
        st.markdown("---")
        if "재편" in field["phase"]:
            st.warning(f"### 종합 국면: {field['phase']}")
        elif "성장" in field["phase"]:
            st.success(f"### 종합 국면: {field['phase']}")
        else:
            st.info(f"### 종합 국면: {field['phase']}")

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("📈 상승 축", len(field["up_keywords"]))
        cc2.metric("📉 하락 축", len(field["down_keywords"]))
        cc3.metric("➡️ 횡보 축", len(field["flat_keywords"]))

        if field["up_keywords"]:
            st.write("**📈 오르는 축:**", ", ".join(field["up_keywords"]))
        if field["down_keywords"]:
            st.write("**📉 식는 축:**", ", ".join(field["down_keywords"]))
        if field["flat_keywords"]:
            st.write("**➡️ 횡보:**", ", ".join(field["flat_keywords"]))

        with st.expander("키워드별 상세 국면 + 그래프"):
            for k in fr["per_kw"]:
                st.write(f"**{k['keyword']}** — {k['phase']} ({k['stance']:+.2f})")
                st.line_chart(k["series"])

        # 속기사
        if st.button("🖋️ 종합 해설 생성", key="field_scribe"):
            try:
                from field_analysis import scribe_field
                with st.spinner("종합 판단을 문장으로..."):
                    text = scribe_field(fr["seed"], field, fr["per_kw"], fr["key"])
                st.session_state["field_scribe_text"] = text
            except Exception as e:
                st.error(f"속기 실패: {e}")
        if st.session_state.get("field_scribe_text"):
            st.info(st.session_state["field_scribe_text"])

# ── 탭1: 단일 키워드 분석 ──────────────────────────
with tab1:
    kw = st.text_input("키워드", value="생성형 AI")
    col1, col2, col3 = st.columns(3)
    source = col1.radio("데이터 소스",
                        ["구글 트렌드(pytrends)", "가상 데이터(로직 테스트)"])
    geo = col2.selectbox("지역", ["전세계", "KR", "US"])
    geo_code = {"전세계": "", "KR": "KR", "US": "US"}[geo]
    tf_label = col3.selectbox("기간", ["1년(권장)", "3개월", "5년"])
    timeframe = {"1년(권장)": "today 12-m", "3개월": "today 3-m",
                 "5년": "today 5-y"}[tf_label]
    st.caption("💡 트렌드 국면은 1년 이상으로 봐야 이벤트성 스파이크에 안 흔들려요. "
               "3개월은 하루 급등이 고점이 돼 대부분 '하락'으로 보일 수 있어요.")

    series = None
    rising = top = 0

    if source == "가상 데이터(로직 테스트)":
        shape = st.selectbox("가상 패턴",
                             ["상승형", "정점형", "하락형", "횡보형"])
        np.random.seed(0)
        if shape == "상승형":
            series = list(np.linspace(10, 90, 24) + np.random.normal(0, 3, 24))
            rising, top = 8, 2
        elif shape == "정점형":
            series = list(np.concatenate([np.linspace(20, 95, 14),
                                          np.linspace(95, 78, 10)]))
            rising, top = 3, 7
        elif shape == "하락형":
            series = list(np.linspace(90, 25, 24) + np.random.normal(0, 3, 24))
            rising, top = 1, 8
        else:
            series = list(50 + np.random.normal(0, 4, 24))
            rising, top = 4, 4

    if st.button("분석", type="primary"):
        if source == "구글 트렌드(pytrends)":
            try:
                from trend_data import fetch_trend
                with st.spinner("구글 트렌드 수집 중..."):
                    snap = fetch_trend(kw, timeframe=timeframe, geo=geo_code)
                series = snap["series"]
                rising, top = snap["rising_count"], snap["top_count"]
                snapshots.append(snap)
                save_snapshots(snapshots, PKL_PATH)
                st.success(f"수집 완료 ({len(series)}포인트) — 스냅샷 저장됨")
            except Exception as e:
                st.error(f"수집 실패: {e}\n(로컬에서 pytrends 설치 필요, "
                         "이 환경은 네트워크 제한)")
                series = None

        if series is not None:
            has_rel = True
            if source == "구글 트렌드(pytrends)":
                has_rel = snap.get("has_related", (rising + top) > 0)
            sig = extract_signals(series, rising, top, has_related=has_rel)
            if sig is None:
                st.warning("데이터가 너무 짧습니다(최소 4포인트).")
            else:
                phase, r, eng = analyze(sig)
                # ★ 결과를 세션에 저장 → 이후 키 입력·재실행에도 안 날아감
                st.session_state["result"] = {
                    "kw": kw, "series": series, "phase": phase,
                    "thesis": r["thesis"].stance, "antithesis": r["antithesis"].stance,
                    "synthesis": r["synthesis"].stance,
                    "grounds_t": r["thesis"].grounds, "grounds_a": r["antithesis"].grounds,
                    "grounds_s": r["synthesis"].grounds,
                    "survival": r["survival"], "eq": eng.is_equilibrium(),
                }

    # ── 결과 표시 (세션에서 읽음 → 재실행돼도 유지) ──────────
    res = st.session_state.get("result")
    if res:
        st.line_chart(res["series"])
        if "불확실" in res["phase"] or "전환점" in res["phase"]:
            st.warning(f"**국면: {res['phase']}**")
        else:
            st.success(f"**국면: {res['phase']}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("정(양적)", f"{res['thesis']:+.2f}")
        c2.metric("반(음적)", f"{res['antithesis']:+.2f}")
        c3.metric("합(통합)", f"{res['synthesis']:+.2f}")

        with st.expander("판단 근거(정·반·합)"):
            st.write("**정(양적 신호):**", res["grounds_t"])
            st.write("**반(음적 신호):**", res["grounds_a"])
            st.write("**합(통합):**", res["grounds_s"])
            st.write(f"**생존값(평형):** {res['survival']}")
            st.write(f"**평형 도달:** {res['eq']}")

        # LLM 속기(선택) — 결과가 세션에 있으므로 키 넣어도 안 날아감
        st.markdown("---")
        st.write("**🖋️ 자연어 해설(속기, 선택)**")
        okey = st.text_input("OpenAI Key", type="password", key="okey")
        if st.button("자연어 해설 생성", key="scribe_btn"):
            if not okey:
                st.warning("OpenAI 키를 입력하세요.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=okey)
                    facts = {
                        "키워드": res["kw"], "국면": res["phase"],
                        "양적_stance": round(res["thesis"], 2),
                        "음적_stance": round(res["antithesis"], 2),
                        "통합_stance": round(res["synthesis"], 2),
                        "평형": res["eq"],
                    }
                    prompt = (
                        "너는 트렌드 분석 속기사다. 분석·추측하지 말고 "
                        "아래 시스템이 계산한 결과를 자연스러운 한국어로 옮겨라. "
                        "수치에 없는 예측을 지어내지 마라.\n\n"
                        f"{facts}\n\n출력: 2~3문장 해설."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1)
                    st.session_state["scribe"] = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"속기 실패: {e}")
        if st.session_state.get("scribe"):
            st.info(st.session_state["scribe"])

# ── 탭2: 저장된 스냅샷 ──────────────────────────────
with tab2:
    if not snapshots:
        st.info("아직 저장된 스냅샷이 없습니다. 구글 트렌드로 분석하면 자동 저장됩니다.")
    else:
        for i, snap in enumerate(reversed(snapshots[-20:])):
            with st.expander(f"[{snap.get('keyword','?')}] "
                             f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(snap.get('fetched_at',0)))}"):
                st.line_chart(snap["series"])
                st.write(f"급상승어 {snap.get('rising_count',0)} / "
                         f"고정어 {snap.get('top_count',0)} / "
                         f"지역 {snap.get('geo','') or '전세계'}")
