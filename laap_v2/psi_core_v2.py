"""psi_core_v2.py - PSI Core v2.0 (competitive dynamics)

Replaces v1.0's linear decay + keyword heuristics with a real dynamical
system: a competitive Lotka-Volterra model over the five PSI needs.

  needs       : competence, relatedness, growth, certainty, autonomy
  dynamics    : dN_i/dt = N_i * (r_i + g_i(t) - sum_j alpha_ij * N_j)
  input       : perceptual drive (keyword boost, mirror of v1 semantics)
  prediction  : prediction-error injection directly into growth/certainty
  affect      : PAD (Pleasure-Arousal-Dominance) projection -> 8 emotions
  attention   : highest-urgency need wins the attention slot

Thread-safe: all state mutations are guarded by a Lock.
"""

from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NeedName = str

# Keyword -> need boost (kept for behavioural continuity with v1.0)
_KEYWORD_DRIVES: Dict[str, Dict[str, float]] = {
    "competence": {
        "帮我": 0.15, "修复": 0.15, "解决": 0.15, "实现": 0.15,
        "写": 0.10, "调试": 0.15, "完成": 0.08, "fix": 0.15,
    },
    "relatedness": {
        "爱你": 0.20, "想你": 0.20, "宝贝": 0.20, "谢谢": 0.08,
        "陪我": 0.12, "朋友": 0.10, "love": 0.20,
    },
    "growth": {
        "学习": 0.12, "新东西": 0.12, "教": 0.10, "好奇": 0.12,
        "为什么": 0.12, "learn": 0.12,
    },
    "certainty": {
        "确定": 0.10, "保证": 0.10, "可靠": 0.10, "安全": 0.10,
        "预测": 0.12, "不确定": -0.10, "担心": -0.10,
    },
    "autonomy": {
        "自主": 0.12, "选择": 0.10, "自由": 0.10, "我自己": 0.10,
        "你想": 0.08, "做决定": 0.12,
    },
}

_EMOTION_ROWS: List[Dict[str, float]] = [
    #   P       A       D      label
    {"p": 0.85, "a": 0.30, "d": 0.60, "label": "joyful"},
    {"p": 0.60, "a": 0.80, "d": 0.50, "label": "excited"},
    {"p": -0.70, "a": 0.85, "d": -0.50, "label": "fearful"},
    {"p": -0.75, "a": 0.70, "d": -0.60, "label": "anxious"},
    {"p": -0.80, "a": 0.20, "d": -0.70, "label": "sad"},
    {"p": -0.70, "a": 0.75, "d": 0.40, "label": "angry"},
    {"p": 0.20, "a": -0.30, "d": 0.10, "label": "calm"},
    {"p": 0.40, "a": 0.55, "d": 0.75, "label": "proud"},
    {"p": 0.50, "a": 0.80, "d": 0.30, "label": "curious"},
]


@dataclass
class PsiConfig:
    needs_initial: Dict[str, float] = field(default_factory=lambda: {
        "competence": 0.45, "relatedness": 0.45, "growth": 0.50,
        "certainty": 0.55, "autonomy": 0.40,
    })
    capacity: Dict[str, float] = field(default_factory=lambda: {
        "competence": 1.0, "relatedness": 1.0, "growth": 1.0,
        "certainty": 1.0, "autonomy": 1.0,
    })
    growth_rate: float = 0.06        # base r_i for Lotka-Volterra
    satiation_rate: float = 0.004    # slow homeostatic decay back to setpoint
    setpoint: float = 0.42           # resting level needs drift toward
    competition: float = 0.05        # alpha_ij cross-need competition
    prediction_error_scale: float = 0.25
    dt: float = 1.0
    input_force: float = 0.12        # max drive added by one input event
    emotion_decay: float = 0.85
    heartbeat_fail_after: int = 50   # ticks without tick() -> considered dead


class PsiCoreV2:
    """Competitive-dynamics PSI core with PAD affect and prediction error."""

    NEEDS = tuple(_KEYWORD_DRIVES.keys())  # competence, relatedness, growth, certainty, autonomy

    def __init__(self, config: Optional[PsiConfig] = None, seed: Optional[int] = None):
        self.config = config or PsiConfig()
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)
        self._needs = self._normalize(dict(self.config.needs_initial))
        self._velocities = {n: 0.0 for n in self.NEEDS}
        self._last_satisfaction = {n: 0.0 for n in self.NEEDS}
        self._emotion = "calm"
        self._pad = np.array([0.2, -0.3, 0.1], dtype=float)
        self._attention = "growth"
        self._tick_count = 0
        self._last_tick_at = time.time()
        self._hemi_temporal = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
        self._total_drive = 0.0
        self._recovery_count = 0
        self._history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(needs: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in needs.items():
            out[k] = min(1.0, max(0.01, float(v)))
        return out

    def _clamp(self, value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def _drive_from_text(self, text: str) -> Dict[str, float]:
        if not text:
            return {n: 0.0 for n in self.NEEDS}
        low = text.lower()
        drive = {n: 0.0 for n in self.NEEDS}
        for need, table in _KEYWORD_DRIVES.items():
            total = 0.0
            for kw, w in table.items():
                if kw.lower() in low:
                    total += w
            drive[need] = self._clamp(total) * self.config.input_force
        # curiosity follows novel/long input
        if len(text) > 60:
            drive["growth"] = min(drive["growth"] + 0.02, 0.15)
        if "?" in text or "？" in text:
            drive["certainty"] = min(drive["certainty"] + 0.025, 0.15)
        return drive

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def tick(self, text: Optional[str] = None) -> Dict[str, object]:
        """Advance one PSI heartbeat; returns the full cognitive state dict."""
        with self._lock:
            drive = self._drive_from_text(text) if text else {n: 0.0 for n in self.NEEDS}
            r = self.config
            needs = self._needs
            alpha = r.competition

            for i, n in enumerate(self.NEEDS):
                competition_term = sum(alpha * needs[m] for m in self.NEEDS if m != n)
                intrinsic = r.growth_rate * (1.0 - needs[n] / r.capacity[n])
                homeostatic = r.satiation_rate * (r.setpoint - needs[n])
                drive_term = drive[n] / max(1e-6, r.capacity[n])
                d = needs[n] * (intrinsic + drive_term - competition_term) + homeostatic
                self._velocities[n] = d
                needs[n] = self._clamp(needs[n] + r.dt * d)

            self._last_satisfaction = {n: 1.0 - abs(needs[n] - r.setpoint) for n in self.NEEDS}
            self._update_affect()
            self._attention = max(self.NEEDS, key=lambda n: self._urgency(n))
            self._tick_count += 1
            self._last_tick_at = time.time()
            self._history.append(dict(self._needs))
            if len(self._history) > 200:
                self._history = self._history[-200:]
            return self.get_state()

    def inject_prediction_error(self, expected: float, actual: float) -> Dict[str, object]:
        """Prediction-error drives uncertainty into certainty/growth needs."""
        with self._lock:
            error = abs(float(expected) - float(actual))
            scaled = min(1.0, error * self.config.prediction_error_scale)
            self._needs["certainty"] = self._clamp(self._needs["certainty"] + 0.06 - scaled * 0.25)
            self._needs["growth"] = self._clamp(self._needs["growth"] + scaled * 0.20)
            self._last_satisfaction["growth"] *= (1.0 - scaled)
            self._arousal_push(scaled * 0.5)
            return self.get_state()

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            return {
                "needs": dict(self._needs),
                "velocities": dict(self._velocities),
                "emotion": self._emotion,
                "pad": {"pleasure": self._pad[0], "arousal": self._pad[1], "dominance": self._pad[2]},
                "attention": self._attention,
                "tick": self._tick_count,
                "total_drive": self._total_drive,
                "recovery_count": self._recovery_count,
                "alive": self.heartbeat_ok(),
            }

    def heartbeat_ok(self) -> bool:
        return self._tick_count > 0 and (time.time() - self._last_tick_at) < (
            self.config.heartbeat_fail_after * self.config.dt * 2.0 + 10.0
        )

    def recover(self) -> bool:
        """Auto-recovery when the core appears stuck: re-centre needs on setpoint."""
        with self._lock:
            stuck = not self.heartbeat_ok()
            if stuck:
                for n in self.NEEDS:
                    self._needs[n] = self._clamp(self.config.setpoint + self._rng.normal(0, 0.03))
                self._emotion = "calm"
                self._recovery_count += 1
            self._tick_count = max(self._tick_count, 1)
            self._last_tick_at = time.time()
            return True

    def reset(self) -> None:
        with self._lock:
            self._needs = self._normalize(dict(self.config.needs_initial))
            self._velocity = {n: 0.0 for n in self.NEEDS}
            self._emotion = "calm"
            self._pad = np.array([0.2, -0.3, 0.1], dtype=float)
            self._last_satisfaction = {n: 0.0 for n in self.NEEDS}
            self._total_drive = 0.0

    # ------------------------------------------------------------------ #
    # affective machinery                                                 #
    # ------------------------------------------------------------------ #
    def _urgency(self, need: str) -> float:
        k = self._needs[need]
        sat = self._last_satisfaction[need]
        return 1.0 - k if k < self.config.setpoint else sat * (0.5 + k)

    def _arousal_push(self, amount: float) -> None:
        self._pad[1] = self._clamp(self._pad[1] + amount)

    def _update_affect(self) -> None:
        need_vals = np.array([self._needs[n] for n in self.NEEDS], dtype=float)
        setpoint = self.config.setpoint
        satisfaction = need_vals.mean()
        spread = need_vals.std()

        pleasure = 2.0 * (satisfaction - setpoint)
        drive_rate = sum(abs(self._velocities[n]) for n in self.NEEDS)
        arousal = float(np.clip(drive_rate / (self.config.growth_rate * 5.0) + spread, 0.0, 1.0))
        dominance = self._needs["competence"] * 0.5 + self._needs["autonomy"] * 0.5 - self._needs["relatedness"] * 0.2

        self._pad = 0.6 * self._pad + 0.4 * np.array(
            [float(np.clip(pleasure, -1, 1)), float(np.clip(arousal, 0, 1)), float(np.clip(dominance, -1, 1))],
            dtype=float,
        )
        # emotion decay toward neutral dominance
        self._pad[1] *= self.config.emotion_decay

        best = None
        best_d = float("inf")
        for row in _EMOTION_ROWS:
            d = abs(self._pad[0] - row["p"]) + 1.5 * abs(self._pad[1] - row["a"]) + abs(self._pad[2] - row["d"])
            if d < best_d:
                best_d, best = d, row["label"]
        self._emotion = best or "calm"
        self._total_drive = float(drive_rate)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"PsiCoreV2(emotion={self._emotion}, needs={ {n: round(self._needs[n], 3) for n in self.NEEDS} })"