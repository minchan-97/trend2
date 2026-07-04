"""
core_base.py — 코어(정·반·합)의 추상 뼈대.

핵심 원칙:
  - 코어의 '성향(인격)'은 지금 정하지 않는다. 나중에 이 인터페이스를
    구현한 클래스를 꽂으면 된다(플러그인).
  - 지금 만드는 건 '어떤 성향이든 담을 수 있는 그릇'과 그들이 서로
    관측하며 평형을 이루는 순환 엔진이다.

관측의 정의(합의됨, 3번 옵션):
  코어 A가 코어 B를 관측한다 = B의 판단에 대해 '동의/이견' 신호를 되돌린다.
  이 신호가 변증법의 반(反)이자, 서로의 생존 근거가 된다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import abc
import numpy as np


@dataclass
class Judgment:
    """한 코어가 내놓는 판단."""
    stance: Any                    # 판단 내용(무엇이든 — 성향이 정의)
    confidence: float              # 0~1, 이 판단을 얼마나 확신하는가
    grounds: str = ""              # 근거(사람이 읽을 속기용)
    meta: dict = field(default_factory=dict)


@dataclass
class Observation:
    """한 코어가 다른 코어를 관측한 결과(동의/이견 신호)."""
    agree: float                   # -1(완전 이견) ~ +1(완전 동의)
    reason: str = ""               # 왜 동의/이견인지(속기용)
    info_gain: float = 0.0         # 이 관측이 '새로운 정보'를 줬는가(담합 방지 핵심)


class Core(abc.ABC):
    """
    정·반·합이 공통으로 상속하는 추상 코어.
    성향(인격)은 judge()/observe()를 구현하는 하위 클래스가 정한다.
    """
    def __init__(self, name: str):
        self.name = name
        self.survival = 1.0        # 생존값(관측받은 정도로 오르내림). 평형 대상.
        self.memory = []           # 경험 축적(성장의 근거)

    @abc.abstractmethod
    def judge(self, context: dict) -> Judgment:
        """입력+현재 경험을 바탕으로 판단을 내린다. (성향이 여기서 발현)"""
        ...

    @abc.abstractmethod
    def observe(self, other_judgment: Judgment, context: dict) -> Observation:
        """다른 코어의 판단을 관측하고 동의/이견 신호를 만든다. (성향이 여기서 발현)"""
        ...

    def remember(self, experience: dict):
        """경험을 축적한다(되먹임으로 성장). 최근 N개만 유지."""
        self.memory.append(experience)
        if len(self.memory) > 200:
            self.memory = self.memory[-200:]


class NullCore(Core):
    """
    성향 미정 상태의 자리표시 코어. 뼈대 검증용.
    - judge: 입력을 그대로 중립 판단으로
    - observe: 확신도 차이만 보고 기계적으로 동의/이견
    실제 성향(정의 독단성, 반의 회의성, 합의 통합성)은 나중에 이걸 대체.
    """
    def judge(self, context: dict) -> Judgment:
        prev = self.memory[-1]["stance"] if self.memory else 0.0
        signal = context.get("signal", 0.0)
        # 되먹임: 이전 경험 반영(성장의 씨앗)
        fb = context.get("feedback", prev)
        # 반이면 정에 반대 방향, 합이면 정·반 중재 — 자리표시 수준의 흉내
        if "against" in context:                    # 반의 판단
            stance = -0.7 * context["against"] + 0.3 * signal
        elif "thesis" in context:                   # 합의 판단
            stance = 0.5 * context["thesis"] + 0.5 * context["antithesis"]
        else:                                        # 정의 판단
            stance = 0.6 * signal + 0.4 * fb
        return Judgment(stance=float(stance), confidence=0.5,
                        grounds=f"{self.name}: 자리표시 판단")

    def observe(self, other: Judgment, context: dict) -> Observation:
        # 내 최근 입장과 상대 입장의 차이 = 이견 정도 = 정보량
        mine = self.memory[-1]["stance"] if self.memory else 0.0
        diff = other.stance - mine
        agree = float(np.tanh(-abs(diff) + 0.3))    # 차이 크면 이견(-)
        info = float(min(1.0, abs(diff)))            # 차이 클수록 정보 있음
        return Observation(agree=agree, reason=f"{self.name} 관측(차이 {diff:.2f})",
                           info_gain=info)
