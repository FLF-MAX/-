"""meta_learning_engine.py - Meta-Learning via LinUCB + REINFORCE.

Replaces v1.0's plain EMA success tables with two real learners:

  LinUCB    - contextual bandit;  upper-confidence-bound arm selection.
  REINFORCE - simple-policy-gradient learner refining a softmax policy.

Both share one context encoder (fixed random projection) and one reward
stream, giving the system both statistically sound exploration/exploitation
(LinUCB) and a differentiable policy (REINFORCE).  A proficiency ladder and
binned confidence calibration sit on top, mirroring v1's self-model role.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------- #
# context encoder                                                              #
# --------------------------------------------------------------------------- #
class ContextEncoder:
    """Deterministic fixed random projection to a compact context vector."""

    def __init__(self, d_in: int = 64, d_out: int = 16, seed: int = 7):
        self._rng = np.random.default_rng(seed)
        self._proj = self._rng.normal(0.0, 1.0 / math.sqrt(d_out), size=(d_out, d_in)).astype(np.float64)
        self.d_out = d_out

    def encode(self, raw: Sequence[float]) -> np.ndarray:
        if len(raw) == self.d_out:
            return np.asarray(raw, dtype=float)
        x = np.zeros(self._proj.shape[1], dtype=float)
        n = min(len(raw), x.shape[0])
        x[:n] = raw[:n]
        v = self._proj @ x
        norm = float(np.linalg.norm(v))
        return v / norm if norm > EPS else v


# --------------------------------------------------------------------------- #
# LinUCB arm                                                                  #
# --------------------------------------------------------------------------- #
class LinUcbArm:
    """One arm: ridge-regression estimate of the linear value function."""

    def __init__(self, name: str, d: int, alpha: float = 1.2, lmbda: float = 1.0):
        self.name = name
        self.d = d
        self.A = lmbda * np.eye(d, dtype=float)
        self.b = np.zeros(d, dtype=float)
        self.A_inv = np.linalg.inv(self.A)
        self.pulls = 0
        self.reward_sum = 0.0
        self.alpha = alpha

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        mu = float(x @ self.theta)
        uncertainty = float(math.sqrt(max(0.0, x @ self.A_inv @ x)))
        return mu, uncertainty

    def upper_bound(self, x: np.ndarray) -> float:
        mu, u = self.predict(x)
        return mu + self.alpha * u

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.A_inv = np.linalg.inv(self.A + EPS * np.eye(self.d))
        self.pulls += 1
        self.reward_sum += reward

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0


# --------------------------------------------------------------------------- #
# proficiency / calibration                                                   #
# --------------------------------------------------------------------------- #
def proficiency_level(attempts: int, success_rate: float) -> str:
    if attempts < 10:
        return "BEGINNER"
    if attempts < 50:
        return "APPRENTICE" if success_rate >= 0.5 else "BEGINNER"
    if attempts < 200:
        return "COMPETENT" if success_rate >= 0.65 else "APPRENTICE"
    return "MASTER" if success_rate >= 0.8 else "VETERAN"


# --------------------------------------------------------------------------- #
# meta-learning engine                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class MetaLearningConfig:
    d_context: int = 16
    alpha: float = 1.2                 # LinUCB exploration scale
    lmbda: float = 1.0
    reinforce_lr: float = 0.05
    reinforce_beta: float = 0.01       # entropy-ish exploration term
    policy_share: float = 0.3          # fraction of pulls from softmax policy
    exploit_threshold: int = 40        # pulls before aggressive exploitation
    ema_alpha: float = 0.1
    cliff: float = 0.75                # success counts as hit above this


class MetaLearningEngine:
    """LinUCB + REINFORCE hybrid meta-learner with proficiency & calibration."""

    DEFAULT_STRATEGIES = (
        "decompose", "direct_attempt", "ask_clarify", "retry", "defer", "search_memory",
    )

    def __init__(
        self,
        strategies: Optional[Sequence[str]] = None,
        config: Optional[MetaLearningConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or MetaLearningConfig()
        self.strategies = list(strategies) if strategies else list(self.DEFAULT_STRATEGIES)
        self._lock = threading.RLock()
        d = self.config.d_context
        self.encoder = ContextEncoder(d_in=32, d_out=d, seed=seed if seed is not None else 7)
        self._arms = {s: LinUcbArm(s, d, self.config.alpha, self.config.lmbda) for s in self.strategies}
        self._rng = np.random.default_rng(seed)
        self._policy_cache: Optional[np.ndarray] = None
        self._w = self._rng.normal(0, 0.01, size=(len(self.strategies), d)).astype(np.float64)
        self._baseline = 0.0
        self._ema = {s: 0.5 for s in self.strategies}
        self._attempts = {s: 0 for s in self.strategies}
        self._successes = {s: 0 for s in self.strategies}
        self._calib_bins: Dict[int, List[Tuple[float, float]]] = {}
        self._total_pulls = 0
        self._reward_history: List[float] = []

    # ------------------------------------------------------------------ #
    def _context(self, features: Sequence[float]) -> np.ndarray:
        return self.encoder.encode(np.asarray(list(features), dtype=float))

    def _softmax_probs(self, x: np.ndarray) -> np.ndarray:
        logits = self._w @ x
        logits = np.clip(logits, -30, 30)
        e = np.exp(logits - logits.max())
        probs = e / (e.sum() + EPS)
        return probs

    def select(self, features: Sequence[float], exploit: Optional[bool] = None) -> Tuple[str, float]:
        """Return (strategy, confidence)."""
        with self._lock:
            x = self._context(features)
            use_policy = (self._rng.random() < self.config.policy_share)
            use_policy = use_policy and (self._total_pulls < self.config.exploit_threshold)
            if use_policy:
                probs = self._softmax_probs(x)
                idx = int(self._rng.choice(len(self.strategies), p=probs))
                conf = float(probs[idx])
            else:
                idx = int(np.argmax([self._arms[s].upper_bound(x) for s in self.strategies]))
                conf = float(self._arms[self.strategies[idx]].predict(x)[0])
            return self.strategies[idx], max(0.0, min(1.0, conf))

    def update(self, features: Sequence[float], strategy: str, reward: float) -> None:
        """Online update after observing the reward."""
        with self._lock:
            if strategy not in self.strategies:
                return
            x = self._context(features)
            self._arms[strategy].update(x, reward)
            # REINFORCE gradient
            probs = self._softmax_probs(x)
            idx = self.strategies.index(strategy)
            self._baseline += self.config.reinforce_beta * (reward - self._baseline)
            grad_log_pi = x - (probs[idx] * x).sum()
            self._w[idx] += self.config.reinforce_lr * (reward - self._baseline) * grad_log_pi
            # EMA success tracker
            self._attempts[strategy] += 1
            self._successes[strategy] += float(reward)
            self._ema[strategy] = (1 - self.config.ema_alpha) * self._ema[strategy] + \
                self.config.ema_alpha * (1.0 if reward >= self.config.cliff else 0.0)
            self._total_pulls += 1
            self._reward_history.append(reward)
            if len(self._reward_history) > 500:
                self._reward_history = self._reward_history[-500:]

    # ------------------------------------------------------------------ #
    def best_strategy(self) -> str:
        with self._lock:
            return max(self.strategies, key=lambda s: self._ema[s])

    def get_proficiency(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            out = {}
            for s in self.strategies:
                a = self._attempts[s]
                rate = self._successes[s] / a if a else 0.0
                out[s] = {
                    "attempts": a,
                    "success_rate": rate,
                    "self_efficacy": self._ema[s],
                    "level": proficiency_level(a, rate),
                    "mean_reward": self._arms[s].mean_reward,
                }
            return out

    def calibrate(self, predicted: float, actual: float) -> None:
        """Binned calibration of confidence vs realised correctness."""
        with self._lock:
            bin_id = int(np.clip(round(predicted * 10), 0, 10))
            self._calib_bins.setdefault(bin_id, []).append((predicted, actual))
            if len(self._calib_bins[bin_id]) > 200:
                self._calib_bins[bin_id] = self._calib_bins[bin_id][-200:]

    def calibration_summary(self) -> Dict[str, float]:
        with self._lock:
            out = {}
            for b in sorted(self._calib_bins):
                pts = self._calib_bins[b]
                predicted = float(np.mean([p for p, _ in pts]))
                actual = float(np.mean([a for _, a in pts]))
                out[str(b)] = {"n": len(pts), "predicted": predicted, "actual": actual}
            return out

    @property
    def total_pulls(self) -> int:
        return self._total_pulls

    def summary(self) -> Dict[str, object]:
        return {
            "pulls": self._total_pulls,
            "best": self.best_strategy(),
            "proficiency": self.get_proficiency(),
            "avg_reward": float(np.mean(self._reward_history)) if self._reward_history else 0.0,
        }