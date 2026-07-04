"""
trend_cores.py — 트렌드 국면 진단용 정·반·합 코어.

목적: 구글 트렌드 시계열(관심도 0~100)을 받아 '지금 어느 국면인가'를 진단.
  국면: 상승 초입 / 정점 / 하락 / 횡보 / 전환점(불확실)

양/음 신호는 순수 계산(비-LLM)으로 뽑는다:
  - 정(正, 양적): 트렌드가 '살아있다/오른다'는 증거
  - 반(反, 음적): 트렌드가 '식는다/포화'라는 증거
  - 합(合): 정·반의 긴장을 종합해 국면 점수(-1 하락 ~ +1 상승)를 냄
  정·반이 합을 다시 관측·조정 → 평형 도달 시 그 국면을 확정.

핵심: 양·음이 팽팽하면(평형 안 됨) '전환점/불확실'이라 정직하게 말한다.
"""
from __future__ import annotations
import numpy as np
from core_base import Core, Judgment, Observation


def _slope(y):
    """시계열 기울기(선형회귀). 정규화된 최근 추세."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    x = (x - x.mean())
    denom = (x ** 2).sum()
    if denom < 1e-9:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def _trim_incomplete_tail(y):
    """
    구글 트렌드 today N-m의 마지막 며칠은 집계가 덜 돼 낮게 나온다.
    → 끝에서 급락하는 미집계 구간을 잘라낸다.
    판단: 마지막 값이 직전 구간 중앙값의 40% 미만으로 뚝 떨어지면 미집계로 보고 제거.
    """
    y = list(y)
    if len(y) < 8:
        return y, 0
    trimmed = 0
    # 최대 마지막 3점까지 검사
    for _ in range(3):
        if len(y) < 8:
            break
        body_median = float(np.median(y[-8:-1]))
        if body_median > 0 and y[-1] < 0.4 * body_median:
            y = y[:-1]
            trimmed += 1
        else:
            break
    return y, trimmed


def extract_signals(series, rising_count=0, top_count=0, has_related=True,
                    trim_tail=True):
    """
    시계열 → 양적/음적 원신호 딕셔너리.
    trim_tail: 구글 트렌드 마지막 미집계 구간을 잘라낼지(기본 True).
    """
    y = np.asarray(series, dtype=float)
    trimmed = 0
    if trim_tail:
        y_list, trimmed = _trim_incomplete_tail(y)
        y = np.asarray(y_list, dtype=float)
    n = len(y)
    if n < 4:
        return None
    recent = y[-max(4, n // 4):]              # 최근 구간
    full_slope = _slope(y)
    recent_slope = _slope(recent)
    peak = y.max()
    cur = y[-1]
    scale = max(peak, 1.0)

    # 고점 대비를 '단일 최고점'이 아니라 '구간 평균 비교'로 (스파이크 내성)
    # 최근 1/3 구간 평균 vs 이전 2/3 구간 평균
    third = max(2, n // 3)
    recent_mean = float(np.mean(y[-third:]))
    earlier_mean = float(np.mean(y[:-third])) if n > third else recent_mean
    trend_ratio = (recent_mean - earlier_mean) / scale   # 양=최근이 더 높음(상승추세)

    # 스파이크 감지
    if n >= 6:
        tail = y[-5:-1]
        spike = float((cur - np.mean(tail)) / scale) if len(tail) else 0.0
        sustained = _slope(y[-8:]) / scale if n >= 8 else recent_slope / scale
    else:
        spike, sustained = 0.0, recent_slope / scale

    return {
        "recent_slope": recent_slope / scale,
        "full_slope": full_slope / scale,
        "from_peak": (cur - peak) / scale,
        "trend_ratio": trend_ratio,            # 구간평균 비교(스파이크 내성 추세)
        "recent_mean": recent_mean / scale,
        "accel": (recent_slope - full_slope) / scale,
        "cur_level": cur / scale,
        "volatility": float(np.std(np.diff(y)) / scale),
        "rising_count": rising_count,
        "top_count": top_count,
        "has_related": has_related,
        "spike": spike,
        "sustained": sustained,
        "trimmed_tail": trimmed,
    }


class ThesisCore(Core):
    """정(正) — 양적: 트렌드가 상승/생동한다는 증거를 모은다."""
    def judge(self, context):
        s = context["signals"]
        fb = context.get("feedback", 0.0)
        # 급상승어 신호는 데이터가 실제 있을 때만 반영(없으면 0 넣지 않고 제외)
        rising_term = 0.3 * np.tanh(s["rising_count"] / 5.0) if s.get("has_related") else 0.0
        score = (1.5 * max(0, s["recent_slope"])
                 + 1.5 * max(0, s.get("trend_ratio", 0))
                 + 1.0 * max(0, s["accel"])
                 + rising_term
                 + 0.3 * s["cur_level"])
        score = score - 0.2 * max(0, fb)
        conf = float(np.clip(abs(s["recent_slope"]) * 3, 0.1, 0.95))
        rel = f", 급상승어 {s['rising_count']}" if s.get("has_related") else ", 급상승어 N/A"
        return Judgment(stance=float(np.tanh(score)), confidence=conf,
                        grounds=f"양적: 최근기울기 {s['recent_slope']:+.3f}, "
                                f"가속 {s['accel']:+.3f}{rel}")

    def observe(self, other, context):
        # 정이 합(또는 반)을 관측: 내 양적 확신 대비 상대가 얼마나 낮은가
        mine = self.memory[-1]["stance"] if self.memory else 0.0
        diff = mine - other.stance
        agree = float(np.tanh(-abs(diff) + 0.3))
        return Observation(agree=agree, info_gain=float(min(1.0, abs(diff))),
                           reason=f"정 관측(양적-상대 차 {diff:+.2f})")


class AntithesisCore(Core):
    """반(反) — 음적: 트렌드가 식음/포화한다는 증거를 모은다."""
    def judge(self, context):
        s = context["signals"]
        # 음적 점수: 고점대비 하락 + 감속 + 포화 + 고변동
        saturation = np.tanh(max(0, s["top_count"] - s["rising_count"]) / 5.0)
        # 정점 신호: 아직 높은 수준(cur_level 큼)인데 가속이 음(꺾임) → 강한 음적
        peaking = max(0, -s["accel"]) * s["cur_level"] * 2.0
        score = (1.5 * max(0, -s["recent_slope"])
                 + 1.5 * max(0, -s.get("trend_ratio", 0))
                 + 1.0 * max(0, -s["accel"])
                 + 1.5 * peaking
                 + 0.5 * saturation
                 + 0.3 * s["volatility"])
        conf = float(np.clip((abs(min(0, s["from_peak"])) + abs(min(0, s["recent_slope"]))
                              + max(0, -s["accel"])) * 3, 0.1, 0.95))
        return Judgment(stance=float(-np.tanh(score)), confidence=conf,
                        grounds=f"음적: 고점대비 {s['from_peak']:+.3f}, "
                                f"꺾임 {peaking:.2f}, 포화 {saturation:.2f}")

    def observe(self, other, context):
        mine = self.memory[-1]["stance"] if self.memory else 0.0
        diff = mine - other.stance
        agree = float(np.tanh(-abs(diff) + 0.3))
        return Observation(agree=agree, info_gain=float(min(1.0, abs(diff))),
                           reason=f"반 관측(음적-상대 차 {diff:+.2f})")


class SynthesisCore(Core):
    """합(合) — 정·반의 긴장을 종합해 국면 점수를 낸다."""
    def judge(self, context):
        t = context.get("thesis", 0.0)      # 정의 양적 stance
        a = context.get("antithesis", 0.0)  # 반의 음적 stance(음수)
        # 종합: 양적 + 음적을 합침(반은 이미 음수라 더하면 상쇄)
        combined = t + a
        return Judgment(stance=float(np.tanh(combined)), confidence=0.6,
                        grounds=f"통합: 양{t:+.2f} + 음{a:+.2f} = {combined:+.2f}")

    def observe(self, other, context):
        mine = self.memory[-1]["stance"] if self.memory else 0.0
        diff = mine - other.stance
        agree = float(np.tanh(-abs(diff) + 0.3))
        return Observation(agree=agree, info_gain=float(min(1.0, abs(diff))),
                           reason=f"합 관측(통합-상대 차 {diff:+.2f})")


def diagnose_phase(synthesis_stance, tension, equilibrium, signals=None):
    """
    합의 최종 stance + 긴장 + 평형여부 → 사람이 읽는 국면 라벨.
    평형 안 됐으면(양·음 팽팽) 전환점/불확실로 정직하게.
    signals 주어지면 정점(높은데 꺾임) 특별 감지.
    """
    # 스파이크 감지: 마지막 점만 튀고 지속 추세는 약함 → 일시적
    if signals is not None:
        if signals.get("spike", 0) > 0.4 and signals.get("sustained", 0) < 0.1:
            return "일시적 급등·스파이크 (지속 상승 아님 — 뉴스성 반짝일 가능성)"
        # 정점 특별 감지: 아직 높은 수준인데 가속이 음(꺾이기 시작)
        if signals["cur_level"] > 0.7 and signals["accel"] < -0.01:
            return "정점·전환점 (높은 관심이 꺾이기 시작 — 진입 주의)"
    if not equilibrium:
        return "전환점·불확실 (양·음 신호가 팽팽 — 판단 유보)"
    s = synthesis_stance
    # 신호 대비 변동성이 크면(노이즈에 묻힌 방향) 횡보로
    if signals is not None and signals.get("volatility", 0) > abs(s) * 0.6:
        return "횡보 (변동성이 커 방향 불명확)"
    if abs(s) < 0.15:
        return "횡보 (뚜렷한 방향 없음)"
    if s > 0.4:
        return "상승 (관심 오르는 국면)"
    if s > 0.15:
        return "완만한 상승·초입"
    if s < -0.4:
        return "하락 (관심 식는 국면)"
    return "완만한 하락·포화"
