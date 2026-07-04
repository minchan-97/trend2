"""
field_analysis.py — 분야 종합 분석.

한 키워드가 아니라 '분야'를 본다:
  1) LLM이 연관 키워드 후보 제안 (탐색만 — 판단 아님)
  2) 각 키워드의 실제 트렌드를 구글에서 수집 (실데이터)
  3) 각 키워드를 평형 엔진으로 국면 판정
  4) 개별 국면들을 다시 종합(메타 평형) → 분야 전체 국면
  5) 속기사 LLM이 번역

LLM은 두 곳에만: 키워드 확장(탐색)과 최종 속기(번역). 판단엔 개입 안 함.
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.append(os.path.dirname(__file__))
from trend_cores import (ThesisCore, AntithesisCore, SynthesisCore,
                         extract_signals, diagnose_phase)
from equilibrium import EquilibriumEngine


# ── 1) 연관 키워드 확장 (LLM 탐색) ──────────────────────────
def expand_keywords(seed, api_key, n=8, model="gpt-4o-mini"):
    """
    LLM이 seed와 연관된 트렌드 키워드 후보를 제안(탐색).
    실패 시 에러를 raise하지 않고 (seed, 에러메시지)로 알린다.
    반환: (keywords_list, error_or_None)
    """
    import json, re
    try:
        from openai import OpenAI
    except Exception as e:
        return [seed], f"openai 패키지 없음: {e} (pip install openai)"
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            f"'{seed}' 분야의 현재 트렌드를 다각도로 보려 한다.\n"
            f"이 분야를 구성하는 세부·연관 검색 키워드를 {n}개 제안하라.\n"
            "실제로 사람들이 검색할 법한 구체적 용어로. seed 자체도 포함.\n"
            "반드시 JSON만 출력: {\"keywords\": [\"키워드1\", \"키워드2\", ...]}"
        )
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.4)
        txt = resp.choices[0].message.content.strip()
    except Exception as e:
        return [seed], f"OpenAI 호출 실패: {type(e).__name__}: {e}"

    kws = []
    txt_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(),
                       flags=re.MULTILINE).strip()
    try:
        kws = json.loads(txt_clean).get("keywords", [])
    except Exception:
        quoted = re.findall(r'"([^"]+)"', txt)
        if quoted:
            kws = [q for q in quoted if q.lower() != "keywords"]
        else:
            for line in txt.splitlines():
                line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('",[]')
                if line and len(line) < 40 and "keyword" not in line.lower():
                    kws.append(line)

    err = None
    if not kws:
        err = f"응답 파싱 실패. 원문: {txt[:150]}"
        kws = [seed]
    out = [seed] + [k for k in kws if k and k != seed]
    seen, uniq = set(), []
    for k in out:
        k = k.strip()
        if k and k not in seen:
            seen.add(k); uniq.append(k)
    return uniq[:n], err


# ── 3) 개별 키워드 국면 판정 ────────────────────────────────
def judge_one(series, rising=0, top=0, has_related=True):
    sig = extract_signals(series, rising, top, has_related=has_related)
    if sig is None:
        return None
    T, A, S = ThesisCore("정"), AntithesisCore("반"), SynthesisCore("합")
    eng = EquilibriumEngine(T, A, S)
    r = None
    for _ in range(30):
        r = eng.step({"signals": sig})
    phase = diagnose_phase(r["synthesis"].stance, r["tension"],
                           eng.is_equilibrium(), sig)
    return {"phase": phase, "stance": float(r["synthesis"].stance),
            "signals": sig}


# ── 4) 메타 종합: 개별 국면들 → 분야 전체 국면 ──────────────
def synthesize_field(per_keyword):
    """
    per_keyword: [{keyword, stance, phase}, ...]
    개별 stance들을 다시 정·반·합 평형으로 종합해 분야 국면을 낸다.
    - 정(양): 상승 키워드들의 힘
    - 반(음): 하락 키워드들의 힘
    - 합: 종합 국면
    """
    stances = [k["stance"] for k in per_keyword if k.get("stance") is not None]
    if not stances:
        return None
    up = [s for s in stances if s > 0.15]
    down = [s for s in stances if s < -0.15]
    flat = [s for s in stances if -0.15 <= s <= 0.15]

    # 메타 신호를 만들어 평형 엔진에 다시 통과
    # (개별 stance 분포를 하나의 '분야 시계열'처럼 요약)
    up_force = float(np.mean(up)) if up else 0.0
    down_force = float(np.mean(down)) if down else 0.0
    field_stance = float(np.tanh(sum(stances) / max(len(stances), 1) * 2))

    # 분야 국면 라벨
    n = len(stances)
    up_n, down_n, flat_n = len(up), len(down), len(flat)
    if up_n > 0 and down_n > 0 and abs(up_n - down_n) <= max(1, n // 4):
        phase = "재편 중 (오르는 축과 식는 축이 공존 — 무게 이동)"
    elif field_stance > 0.3:
        phase = "성장 (분야 전반 상승)"
    elif field_stance > 0.1:
        phase = "완만한 성장"
    elif field_stance < -0.3:
        phase = "쇠퇴 (분야 전반 하락)"
    elif field_stance < -0.1:
        phase = "정체·포화"
    else:
        phase = "횡보 (뚜렷한 방향 없음)"

    return {
        "field_stance": field_stance,
        "phase": phase,
        "up_keywords": [k["keyword"] for k in per_keyword if k.get("stance", 0) > 0.15],
        "down_keywords": [k["keyword"] for k in per_keyword if k.get("stance", 0) < -0.15],
        "flat_keywords": [k["keyword"] for k in per_keyword
                          if -0.15 <= k.get("stance", 0) <= 0.15],
        "up_force": up_force, "down_force": down_force,
    }


# ── 5) 속기사: 분야 종합 판단을 문장으로 ────────────────────
def scribe_field(seed, field_result, per_keyword, api_key, model="gpt-4o-mini"):
    """엔진의 종합 판단을 사람이 읽을 문장으로 옮김(판단 아님, 번역)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    facts = {
        "분야": seed,
        "종합국면": field_result["phase"],
        "상승_키워드": field_result["up_keywords"],
        "하락_키워드": field_result["down_keywords"],
        "횡보_키워드": field_result["flat_keywords"],
    }
    prompt = (
        "너는 트렌드 종합 분석 속기사다. 분석·추측하지 말고, 아래 시스템이 "
        "여러 연관 키워드의 트렌드를 종합해 계산한 결과를 자연스러운 한국어로 "
        "옮겨라. 어떤 축이 오르고 어떤 축이 식는지를 짚어 분야의 무게 이동을 "
        "설명하되, 수치에 없는 예측은 지어내지 마라.\n\n"
        f"{facts}\n\n출력: 3~4문장."
    )
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.2)
    return resp.choices[0].message.content
