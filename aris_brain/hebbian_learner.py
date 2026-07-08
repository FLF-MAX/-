"""Hebbian Learner + Emotional RL - runtime weight evolution"""
import numpy as np
from typing import Dict, List, Tuple


class HebbianLearner:
    """Runtime learning via Hebbian plasticity + emotional reinforcement."""

    def __init__(self, dim=1024, n_patterns=64):
        self.dim = dim
        self.n_patterns = n_patterns
        rng = np.random.RandomState(2)
        self.W = np.eye(dim, dtype=np.float32) * 0.1
        self.W += rng.randn(dim, dim).astype(np.float32) * 0.01
        self.patterns = []
        self._max = n_patterns
        self.base_lr = 0.01
        self._n_updates = 0
        self._n_matches = 0

    def update(self, pre_state, post_state, valence, reward=0.0):
        lr = self.base_lr * (1.0 + valence * 0.5) * (1.0 + reward * 0.5)
        delta = lr * (np.outer(pre_state, post_state) -
                      np.outer(post_state, post_state) @ self.W)
        self.W += delta
        trace_norm = np.trace(self.W) / self.dim
        if trace_norm > 1.0:
            self.W = self.W / trace_norm
        self._consolidate(pre_state, post_state, valence)
        self._n_updates += 1

    def _consolidate(self, inp, out, valence):
        for p in self.patterns:
            sim = float(np.dot(p["input"], inp))
            if sim > 0.85:
                p["freq"] += 1
                p["valence_sum"] += max(0, valence)
                p["output"] = p["output"] * 0.8 + out * 0.2
                nrm = np.linalg.norm(p["output"])
                if nrm > 0:
                    p["output"] = p["output"] / nrm
                self._n_matches += 1
                return
        self.patterns.append({
            "input": inp.copy(), "output": out.copy(),
            "freq": 1, "valence_sum": max(0, valence),
        })
        if len(self.patterns) > self._max:
            self.patterns.sort(key=lambda p: p["freq"] * (p["valence_sum"] + 1))
            self.patterns.pop(0)

    def predict(self, state):
        return self.W @ state

    def apply_patterns(self, state):
        for p in self.patterns:
            sim = float(np.dot(p["input"], state))
            if sim > 0.75 and p["freq"] > 2:
                alpha = min(0.5, p["freq"] * 0.01)
                mod = state * (1 - alpha) + p["output"] * alpha
                nrm = np.linalg.norm(mod)
                if nrm > 0:
                    mod = mod / nrm
                return True, mod
        return False, state

    def stats(self):
        return {
            "n_updates": self._n_updates,
            "n_patterns": len(self.patterns),
            "match_rate": self._n_matches / max(1, self._n_updates),
            "W_trace": float(np.trace(self.W) / self.dim),
        }
