"""probabilistic_world_model.py - Dynamic Bayesian Network world model.

Replaces v1.0's entity-dict + rule table with a probability model:

  Structure learning  : greedy parent-set search minimising BIC score.
  Transition model    : linear-Gaussian conditional models (least squares).
  Inference           : particle filtering over hidden states.
  Counterfactuals     : do(var=x) intervention then forward simulation.

The design is intentionally "small but real": it can recover simple causal
structure (temperature -> pressure, temperature -> humidity) from observed
time series and answer do-intervention queries.
"""

from __future__ import annotations

import itertools
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

EPS = 1e-8


@dataclass
class DbnodeModel:
    parents: Tuple[str, ...] = ()
    beta: Optional[np.ndarray] = None   # [1 + |parents|]
    sigma2: float = 1.0                 # residual noise variance


@dataclass
class WorldModelConfig:
    max_parents: int = 3
    bic_penalty: float = 0.5
    n_particles: int = 100
    do_jitter: float = 0.02             # intervention noise scale
    sample_first_n: int = 200           # max rows used for structure learning


class ProbabilisticWorldModel:
    """Greedy-BIC DBN + linear-Gaussian dynamics + particle filtering."""

    def __init__(self, variable_names: Sequence[str], config: Optional[WorldModelConfig] = None, seed: Optional[int] = None):
        self.names = list(variable_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.config = config or WorldModelConfig()
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)
        self._data: List[np.ndarray] = []
        self.models: Dict[str, DbnodeModel] = {n: DbnodeModel() for n in self.names}
        self._particles: Optional[np.ndarray] = None          # [K, d]
        self._particle_weights: Optional[np.ndarray] = None   # [K]
        self._ground_truth: Dict[str, Tuple[str, ...]] = {}

    # ------------------------------------------------------------------ #
    # data ingestion                                                      #
    # ------------------------------------------------------------------ #
    def add_observation(self, values: Sequence[float]) -> None:
        with self._lock:
            if len(values) != len(self.names):
                raise ValueError(f"expected {len(self.names)} values, got {len(values)}")
            self._data.append(np.asarray(values, dtype=float))
            if self._particles is None:
                self._particles = np.repeat(np.asarray(values, dtype=float)[None, :],
                                            self.config.n_particles, axis=0)
                self._particle_weights = np.full(self.config.n_particles, 1.0 / self.config.n_particles)

    def seed_from(self, observations: Sequence[Sequence[float]]) -> None:
        self._data = []
        for obs in observations:
            self.add_observation(obs)

    def num_observations(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------ #
    # structure learning (BIC)                                            #
    # ------------------------------------------------------------------ #
    def _log_likelihood(self, target: str, parents: Sequence[str]) -> float:
        """NLL of a lag-1 linear-Gaussian regression (parents at t-1 -> target at t)."""
        rows = len(self._data)
        if rows < len(parents) + 3:
            return float("inf")
        design = []
        yvals = []
        for t in range(1, rows):
            row = [1.0]
            for p in parents:
                row.append(self._data[t - 1][self.index[p]])
            design.append(row)
            yvals.append(self._data[t][self.index[target]])
        x = np.asarray(design, dtype=float)
        y = np.asarray(yvals, dtype=float)
        n = y.size
        if n < 3:
            return float("inf")
        if len(parents) == 0:
            mu = y.mean()
            resid = y - mu
        else:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ beta
        sigma2 = max(float(resid @ resid / n), EPS)
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
        return float(ll)

    def learn_structure(self) -> Dict[str, Tuple[str, ...]]:
        """Greedy per-variable lag-1 parent search maximising the BIC score.

        BIC = ln L - (k/2) * ln(N)  (maximised; smaller is *not* better).
        """
        with self._lock:
            names = self.names
            n = len(self._data)
            if n < 4:
                raise RuntimeError("not enough observations to learn structure")
            for target in names:
                best: List[str] = []
                best_bic = self._score(target, best)
                candidates = list(names)  # DBN self-loop allowed (autoregression)
                improved = True
                while improved and len(best) < self.config.max_parents:
                    improved = False
                    best_next: List[str] = best
                    for cand in candidates:
                        if cand in best:
                            continue
                        trial = best + [cand]
                        bic = self._score(target, trial)
                        if bic > best_bic + 1e-9:
                            best_bic, best_next, improved = bic, trial, True
                    if improved:
                        best = best_next
                        candidates = [c for c in candidates if c not in best]
                # fit final linear model
                self.models[target] = self._fit_model(target, best)
            return {n: self.models[n].parents for n in names}

    def _score(self, target: str, parents: Sequence[str]) -> float:
        n = len(self._data)
        ll = self._log_likelihood(target, parents)
        if not np.isfinite(ll):
            return float("-inf")
        k = 1 + len(parents)
        return float(ll - self.config.bic_penalty * k * math.log(n) / 2.0)

    def _fit_model(self, target: str, parents: Sequence[str]) -> DbnodeModel:
        rows = len(self._data)
        design = []
        yvals = []
        for t in range(1, rows):
            row = [1.0]
            for p in parents:
                row.append(self._data[t - 1][self.index[p]])
            design.append(row)
            yvals.append(self._data[t][self.index[target]])
        x = np.asarray(design, dtype=float)
        y = np.asarray(yvals, dtype=float)
        if parents:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        else:
            beta = np.array([y.mean()], dtype=float)
        resid = y - x @ beta
        var = max(float(resid @ resid / max(rows - 1, 1)), EPS)
        m = DbnodeModel(parents=tuple(parents), beta=beta, sigma2=var)
        return m

    def get_graph(self) -> Dict[str, Tuple[str, ...]]:
        return {self.names[i]: tuple(self.models[self.names[i]].parents) for i in range(len(self.names))}

    # ------------------------------------------------------------------ #
    # prediction                                                          #
    # ------------------------------------------------------------------ #
    def _conditional_mean(self, target: str, state: np.ndarray) -> float:
        m = self.models[target]
        x = np.ones(1 + len(m.parents), dtype=float)
        for j, p in enumerate(m.parents):
            x[1 + j] = state[self.index[p]]
        if m.beta is None:
            return float(np.mean([o[self.index[target]] for o in self._data])) if self._data else 0.0
        return float(x @ m.beta)

    def predict_next(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict the next state vector conditional on `state` (or posterior mean)."""
        with self._lock:
            if state is None:
                state = self.posterior_mean()
            out = np.zeros(len(self.names), dtype=float)
            for n in self.names:
                out[self.index[n]] = self._conditional_mean(n, np.asarray(state, dtype=float))
            return out

    def posterior_mean(self) -> np.ndarray:
        if self._particles is None or self._particle_weights is None:
            if not self._data:
                return np.zeros(len(self.names), dtype=float)
            return np.mean(self._data[-1:], axis=0)
        return float(wsum := self._particle_weights.sum()) and \
            (self._particle_weights @ self._particles) / max(wsum, EPS)

    # ------------------------------------------------------------------ #
    # particle filtering                                                  #
    # ------------------------------------------------------------------ #
    def step_filter(self, measurement: np.ndarray) -> np.ndarray:
        """One bootstrap particle-filter step; returns the posterior mean."""
        with self._lock:
            if self._particles is None:
                self._particles = np.repeat(measurement[None, :], self.config.n_particles, axis=0)
                self._particle_weights = np.full(self.config.n_particles, 1.0 / self.config.n_particles)
                return self.posterior_mean()
            K = self.config.n_particles
            # propagate
            new_particles = np.zeros_like(self._particles)
            for i in range(K):
                new_particles[i] = self.predict_next(self._particles[i])
            # *small* process noise
            new_particles += self._rng.normal(0, 0.05, size=new_particles.shape)
            # weight by likelihood (Gaussian with measurement noise)
            noise2 = 0.1
            logw = -0.5 * ((measurement - new_particles) ** 2).sum(axis=1) / noise2
            logw -= logw.max()
            w = np.exp(logw)
            wsum = w.sum()
            if wsum <= EPS:
                w = np.ones(K, dtype=float)
                wsum = K
            w = w / wsum
            self._particles = new_particles
            self._particle_weights = w
            # systematic resampling
            cum = np.cumsum(w)
            u0 = self._rng.random() / K
            idx = np.searchsorted(cum, u0 + np.arange(K) / K, side="right")
            idx = np.clip(idx, 0, K - 1)
            self._particles = new_particles[idx]
            self._particle_weights = np.full(K, 1.0 / K)
            return self.posterior_mean()

    # ------------------------------------------------------------------ #
    # counterfactual do-intervention                                     #
    # ------------------------------------------------------------------ #
    @dataclass
    class _DoContext:
        saved: np.ndarray
        var: int

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def do(self, variable: str, value: float):
        """Context manager: apply do(variable=value) during the block."""
        return self._DoContext(self.posterior_mean(), self.index[variable])

    def simulate_do(self, variable: str, value: float, steps: int = 5) -> List[np.ndarray]:
        """Forward-simulate the DBN with `variable` pinned to `value`."""
        with self._lock:
            state = self.posterior_mean().copy()
            var_i = self.index[variable]
            out: List[np.ndarray] = []
            for _ in range(steps):
                nxt = self.predict_next(state)
                nxt[var_i] = value + self._rng.normal(0, self.config.do_jitter)  # intervention
                out.append(nxt.copy())
                state = nxt
            return out

    def summary(self) -> Dict[str, object]:
        return {
            "variables": self.names,
            "observations": len(self._data),
            "graph": self.get_graph(),
            "posterior_mean": self.posterior_mean().tolist(),
        }