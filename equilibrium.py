"""
equilibrium.py — 평형 엔진.

세 코어(정·반·합)가 서로를 관측하며 동적 평형을 이룬다.
- 입력마다 정→반→합 한 바퀴 + 잠정 결론(합의됨: 경험 성장형)
- 합의 결론이 다음 입력의 정에 되먹임 → 경험 축적 → 성장
- 관측 = 동의/이견 신호. 서로가 서로를 관측해야 생존(판다 원리)
- 평형 메커니즘:
    * 강해지면 눌리고(음성 피드백), 약해지면 희소성으로 되살아남
    * 담합 방지: '정보를 준 관측'만 생존값으로 인정(info_gain 가중)

성향(인격)은 코어에 나중에 꽂는다. 엔진 자체는 성향과 무관하게 돈다.
"""
from __future__ import annotations
import numpy as np
from core_base import Core, Judgment, Observation


class EquilibriumEngine:
    def __init__(self, thesis: Core, antithesis: Core, synthesis: Core,
                 decay=0.05, gain=0.15, scarcity=0.1, seed=42):
        # 정·반·합
        self.T = thesis
        self.A = antithesis
        self.S = synthesis
        self.cores = [self.T, self.A, self.S]
        # 평형 파라미터
        self.decay = decay        # 매 스텝 생존값 자연 감소(관측 없으면 시듦)
        self.gain = gain          # 의미있는 관측 1회가 주는 생존값
        self.scarcity = scarcity  # 희소한 코어가 받는 회복 보너스
        self.history = []         # 매 스텝 기록(속기·분석용)
        self.prev_synthesis = None  # 되먹임용

    def step(self, context: dict) -> dict:
        """
        입력 한 개에 대해 정→반→합 한 바퀴 + 평형 갱신 + 되먹임.
        반환: 이번 스텝의 잠정 결론 + 관측 로그.
        """
        ctx = dict(context)
        # 되먹임: 이전 합의 결론을 이번 정의 맥락에 주입 (경험 성장)
        if self.prev_synthesis is not None:
            ctx["feedback"] = self.prev_synthesis.stance

        # ── 정(正): 입장을 세운다 ──────────────────────────
        jT = self.T.judge(ctx)

        # ── 반(反): 정을 관측하고 이견을 낸다 ───────────────
        oA = self.A.observe(jT, ctx)          # 반이 정을 관측(동의/이견)
        ctx_anti = dict(ctx); ctx_anti["against"] = jT.stance
        jA = self.A.judge(ctx_anti)           # 반의 대립 판단

        # ── 합(合): 정·반의 긴장을 관측해 통합 ──────────────
        oS_t = self.S.observe(jT, ctx)        # 합이 정을 관측
        oS_a = self.S.observe(jA, ctx)        # 합이 반을 관측
        ctx_syn = dict(ctx)
        ctx_syn["thesis"] = jT.stance
        ctx_syn["antithesis"] = jA.stance
        ctx_syn["tension"] = abs(jT.stance - jA.stance)  # 정·반 실제 입장차 = 긴장
        jS = self.S.judge(ctx_syn)

        # ── 순환을 닫는다: 정·반이 합을 관측 (되먹임 관측) ──
        # 합의 결론이 다음 정에 되먹임되기 전에, 정·반이 합을 평가한다.
        # → 합도 관측받아야 생존(누구도 무조건 살지 않음 = 평형)
        oT_s = self.T.observe(jS, ctx_syn)    # 정이 합을 관측
        oA_s = self.A.observe(jS, ctx_syn)    # 반이 합을 관측

        # ── 관측 기반 생존값 갱신 (평형) ────────────────────
        observations = {
            "A_observes_T": oA,     # 반이 정을 관측
            "S_observes_T": oS_t,   # 합이 정을 관측
            "S_observes_A": oS_a,   # 합이 반을 관측
            "T_observes_S": oT_s,   # 정이 합을 관측 (순환 닫음)
            "A_observes_S": oA_s,   # 반이 합을 관측 (순환 닫음)
        }
        self._update_survival(observations)

        # ── 되먹임 + 경험 축적 (성장) ───────────────────────
        self.prev_synthesis = jS
        exp = {"stance": jS.stance, "tension": ctx_syn["tension"]}
        for c in self.cores:
            c.remember(exp)

        result = {
            "thesis": jT, "antithesis": jA, "synthesis": jS,
            "observations": observations,
            "survival": {c.name: round(c.survival, 3) for c in self.cores},
            "tension": round(ctx_syn["tension"], 3),
        }
        self.history.append(result)
        return result

    def _update_survival(self, observations):
        """
        평형의 심장.
        1) 관측받은 코어는 생존값 상승 (단, info_gain 가중 → 담합 방지)
        2) 모든 코어 자연 감소 (관측 안 받으면 시듦)
        3) 강해지면 눌림(음성 피드백), 약해지면 희소성 보너스
        → 자동으로 평형점에 수렴
        """
        # 누가 얼마나 '의미있게' 관측됐나 집계
        observed_gain = {c.name: 0.0 for c in self.cores}
        # 반이 정을 관측
        observed_gain[self.T.name] += self.gain * observations["A_observes_T"].info_gain
        # 합이 정·반을 관측
        observed_gain[self.T.name] += self.gain * observations["S_observes_T"].info_gain
        observed_gain[self.A.name] += self.gain * observations["S_observes_A"].info_gain
        # 정·반이 합을 관측 (순환을 닫음 → 합도 관측받아야 생존)
        observed_gain[self.S.name] += self.gain * observations["T_observes_S"].info_gain
        observed_gain[self.S.name] += self.gain * observations["A_observes_S"].info_gain

        mean_surv = np.mean([c.survival for c in self.cores])
        # 관측 이득의 총량을 정규화(한 코어가 이득을 독식하지 못하게)
        total_gain = sum(observed_gain.values()) + 1e-9
        for c in self.cores:
            # 1) 관측 이득 (상대적 몫으로 — 독식 방지)
            share = observed_gain[c.name] / total_gain
            c.survival += self.gain * share * 3.0   # 몫 기반 분배
            # 2) 자연 감소
            c.survival -= self.decay
            # 3) 강한 음성 피드백: 평균보다 강하면 그 초과분에 비례해 크게 눌림
            gap = c.survival - mean_surv
            if gap > 0:
                c.survival -= 0.8 * gap             # 강하게 끌어내림
            else:
                c.survival += self.scarcity * (-gap)  # 약하면 희소성 회복
            c.survival = float(np.clip(c.survival, 0.05, 2.0))

    def is_equilibrium(self, window=10, tol=0.15):
        """
        최근 스텝들의 생존값이 서로 가깝고(분산 작음) 안정적이면 평형.
        평형은 '정지'가 아니라 '세 코어가 좁은 띠 안에서 함께 움직이는' 상태.
        """
        if len(self.history) < window:
            return False
        recent = self.history[-window:]
        # 각 스텝에서 세 코어 생존값의 분산이 계속 작으면 평형
        for h in recent:
            vals = list(h["survival"].values())
            if float(np.var(vals)) > tol:
                return False
        return True
