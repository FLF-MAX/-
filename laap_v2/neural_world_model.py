"""neural_world_model.py - MLP transition predictor with heteroscedastic NLL.

Whereas ``probabilistic_world_model`` uses linear-Gaussian dynamics (which
cannot fit nonlinear transitions like Z = X * Y), this module approximates
the state-transition function with a small fully-connected network trained
by custom Adam.  The output is Gaussian with *learned, input-dependent
variance* (heteroscedastic uncertainty):

    p(y | x) = N(mu(x), sigma^2(x))

loss = 0.5 * sum( (y - mu)^2 / sigma^2 + log(sigma^2) )   (NLL)

Pure NumPy — designed for CPU-scale experiments, not production scale.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6


@dataclass
class MlpConfig:
    hidden: Tuple[int, int] = (64, 32)
    lr: float = 5e-3
    epochs: int = 300
    warmup_epochs: int = 250        # MSE phases; remainder = NLL with calibrated sigma
    tail_epochs: int = 50           # post-calibration weighted refinement
    batch_size: int = 64
    noise_sigma_min: float = 0.35   # variance floor in standardised units
    l2: float = 0.0
    grad_clip: float = 5.0          # global-norm gradient clipping


class MLPWorldModel:
    """Two-hidden-layer MLP state transition model with Adam optimiser."""

    def __init__(self, d_state: int, config: Optional[MlpConfig] = None, seed: Optional[int] = None):
        self.d_state = d_state
        self.config = config or MlpConfig()
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)
        self._init_layers()
        self._trained = False
        self._X: list = []
        self._Y: list = []

    # ------------------------------------------------------------------ #
    def _init_layers(self) -> None:
        h1, h2 = self.config.hidden
        d = self.d_state
        self.W1 = self._rng.normal(0, 1.0 / np.sqrt(d), (d, h1)).astype(np.float64)
        self.b1 = np.zeros(h1, dtype=np.float64)
        self.W2 = self._rng.normal(0, 1.0 / np.sqrt(h1), (h1, h2)).astype(np.float64)
        self.b2 = np.zeros(h2, dtype=np.float64)
        self.Wo = np.zeros((h2, 2 * d), dtype=np.float64)      # small output init
        self.bo = np.zeros(2 * d, dtype=np.float64)
        # Adam moments
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._adam_t = 0

    def _params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "Wo": self.Wo, "bo": self.bo}

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _act(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)   # bounded nonlinearity -> stable gradients

    @staticmethod
    def _act_deriv(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2

    # ------------------------------------------------------------------ #
    def add_transition(self, prev: Sequence[float], next_: Sequence[float]) -> None:
        with self._lock:
            self._X.append(np.asarray(list(prev), dtype=np.float64))
            self._Y.append(np.asarray(list(next_), dtype=np.float64))

    def fit(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None,
            epochs: Optional[int] = None, nll_phase: bool = True) -> dict:
        """Train the world model.

        Phase A (MSE):      bootstrap the mean head with Adam on standardised
                            data - identical to Gaussian NLL with fixed sigma.
        Phase B (calib):    closed-form heteroscedastic variance calibration
                            - input-dependent sigma fit on the mean residuals.
        Phase C (NLL tail): weighted refinement of the mean against the
                            calibrated per-input sigma (validates the NLL
                            objective without the mean/var coupled divergence).

        `nll_phase=False` skips Phase C.
        """
        with self._lock:
            if X is None or Y is None:
                if not self._X:
                    raise ValueError("no training data")
                Xa = np.asarray(self._X, dtype=np.float64)
                Ya = np.asarray(self._Y, dtype=np.float64)
            else:
                Xa = np.asarray(X, dtype=np.float64)
                Ya = np.asarray(Y, dtype=np.float64)
            if Xa.ndim == 1:
                Xa = Xa[None, :]
                Ya = Ya[None, :]
            n = Xa.shape[0]
            epochs = epochs or self.config.epochs
            # input standardisation to stabilise training
            self._x_mean = Xa.mean(axis=0)
            self._x_std = Xa.std(axis=0) + EPS
            self._y_mean = Ya.mean(axis=0)
            self._y_std = Ya.std(axis=0) + EPS
            Xn = (Xa - self._x_mean) / self._x_std
            Yn = (Ya - self._y_mean) / self._y_std

            warmup = min(self.config.warmup_epochs, epochs)
            losses = []
            for e in range(warmup):
                idx = self._rng.permutation(n)
                for s in range(0, n, self.config.batch_size):
                    batch = idx[s:s + self.config.batch_size]
                    losses.append(self._train_step(Xn[batch], Yn[batch], nll=False))
            mse_loss = float(np.mean(losses)) if losses else 0.0

            # Phase B: heteroscedastic variance calibration
            self._calibrate_variance(Xn, Yn)

            # Phase C: NLL-weighted tail against *fixed* calibrated sigma
            if nll_phase:
                losses_nll = []
                tail = min(self.config.tail_epochs, max(epochs - warmup, 0))
                for e in range(tail):
                    idx = self._rng.permutation(n)
                    for s in range(0, n, self.config.batch_size):
                        batch = idx[s:s + self.config.batch_size]
                        losses_nll.append(self._train_step(Xn[batch], Yn[batch], nll=True))
                nll_loss = float(np.mean(losses_nll)) if losses_nll else 0.0
            else:
                nll_loss = 0.0

            self._trained = True
            return {"epochs": epochs, "mse_loss": mse_loss, "nll_loss": nll_loss, "samples": n}

    def _calibrate_variance(self, Xn: np.ndarray, Yn: np.ndarray) -> None:
        """Closed-form calibration of an input-dependent sigma head.

        sigma(x)^2 = exp(a2(x) @ W + b) + floor,  fitted by least squares on
        log-residual^2 in the frozen penultimate feature space.  Guaranteed
        finite and positive; gives genuine heteroscedastic uncertainty.
        """
        a2, mu = self._features_and_mean(Xn)
        floor2 = self.config.noise_sigma_min ** 2
        resid2 = np.maximum((Yn - mu) ** 2, floor2)
        targets = np.log(resid2)
        design = np.concatenate([a2, np.ones((a2.shape[0], 1))], axis=1)
        W, *_ = np.linalg.lstsq(design, targets, rcond=None)
        self._sigma_w = W                                   # [h2+1, d]
        self._sigma_floor2 = floor2
        # prime the shared output head so its logvar matches the calibration
        self.Wo[:, self.d_state:] = 0.5 * W[:-1, :]         # a2 -> logvar weights
        self.bo[self.d_state:] = 0.5 * W[-1, :]
        resid_check = float(np.mean((Yn - mu) ** 2))
        self._calib_mse = resid_check

    def _features_and_mean(self, Xn: np.ndarray):
        """Penultimate features a2 and mean predictions (standardised space)."""
        params = self._params()
        z1 = Xn @ params["W1"] + params["b1"]
        a1 = self._act(z1)
        z2 = a1 @ params["W2"] + params["b2"]
        a2 = self._act(z2)
        out = a2 @ params["Wo"] + params["bo"]
        mu, _ = np.split(out, 2, axis=1)
        return a2, mu

    def _forward(self, x: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        z1 = x @ params["W1"] + params["b1"]
        a1 = self._act(z1)
        z2 = a1 @ params["W2"] + params["b2"]
        a2 = self._act(z2)
        out = a2 @ params["Wo"] + params["bo"]          # [B, 2d]
        mu, logvar = np.split(out, 2, axis=1)
        logvar = np.clip(logvar, -2.0, 2.5)
        cache = {"a1": a1, "z1": z1, "a2": a2, "z2": z2, "x": x, "logvar": logvar}
        return mu, logvar, cache

    def _reset_adam_state(self) -> None:
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._adam_t = 0

    def _adam_apply(self, grads: dict) -> None:
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = self.config.lr
        self._adam_t += 1
        for name, g in grads.items():
            m, v = self._m[name], self._v[name]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** self._adam_t)
            vhat = v / (1 - b2 ** self._adam_t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            p = getattr(self, name)
            np.nan_to_num(step, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            p -= step

    def _train_step(self, xb: np.ndarray, yb: np.ndarray, nll: bool = True) -> float:
        params = self._params()
        mu, logvar, cache = self._forward(xb, params)
        if nll:
            # NLL-tail: mean is re-weighted by the *calibrated* per-input
            # sigma (fixed variance).  No variance gradients -> no divergence.
            d = self.d_state
            a2 = cache["a2"]
            design = np.concatenate([a2, np.ones((a2.shape[0], 1))], axis=1)
            logv = design @ self._sigma_w                 # [B, d]
            var = np.exp(logv) + self._sigma_floor2
            inv = np.clip(1.0 / var, 0.1, 2.0)
            loss_val = 0.5 * ((yb - mu) ** 2 * inv).mean() + 0.5 * float(np.mean(logv))
            d_out = (mu - yb) * inv                        # dL/dmu = (mu-y)/var
            d_logvar = np.zeros_like(logvar)
        else:
            # MSE warm-up on the mean head only (variance head frozen)
            loss_val = float((0.5 * (yb - mu) ** 2).mean())
            d_out = (mu - yb)
            d_logvar = np.zeros_like(logvar)
        loss = float(loss_val)

        dz2 = d_out @ params["Wo"][:, : self.d_state].T + d_logvar @ params["Wo"][:, self.d_state:].T
        dz2 *= self._act_deriv(cache["z2"])
        gW2 = cache["a1"].T @ dz2 / xb.shape[0]
        gb2 = dz2.mean(axis=0)
        dz1 = (dz2 @ params["W2"].T) * self._act_deriv(cache["z1"])
        gW1 = cache["x"].T @ dz1 / xb.shape[0]
        gb1 = dz1.mean(axis=0)
        gWo = cache["a2"].T @ np.concatenate([d_out, d_logvar], axis=1) / xb.shape[0]
        gbo = np.concatenate([d_out, d_logvar], axis=1).mean(axis=0)

        if self.config.l2 > 0:
            gW1 += self.config.l2 * params["W1"]
            gW2 += self.config.l2 * params["W2"]
            gWo += self.config.l2 * params["Wo"]

        # clip invalid grads (NaN -> zero) and apply global-norm clipping
        grads = [gW1, gb1, gW2, gb2, gWo, gbo]
        for g in grads:
            np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if total > self.config.grad_clip and total > 0:
            scale = self.config.grad_clip / total
            grads = [g * scale for g in grads]
        gW1, gb1, gW2, gb2, gWo, gbo = grads

        self._adam_apply({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "Wo": gWo, "bo": gbo})
        return loss

    # ------------------------------------------------------------------ #
    def predict(self, x: Sequence[float], with_uncertainty: bool = False):
        """Predict the transition for a single input state.

        Returns the mean only by default, or (mu, sigma). Heteroscedastic
        per-input sigma comes from the calibrated variance head.
        """
        with self._lock:
            params = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                      "Wo": self.Wo, "bo": self.bo}
            xv = np.asarray(list(x), dtype=np.float64)
            if self._trained:
                xv = (xv - self._x_mean) / self._x_std
            mu_n, _, cache = self._forward(xv[None, :], params)
            mu = mu_n[0] * self._y_std + self._y_mean if self._trained else mu_n[0]
            if not with_uncertainty:
                return mu
            if getattr(self, "_sigma_w", None) is not None and self._trained:
                design = np.concatenate([cache["a2"], np.ones((1, 1))], axis=1)
                logv = design @ self._sigma_w
                var = np.exp(logv) + self._sigma_floor2
                sigma = np.sqrt(var)[0] * self._y_std
            else:
                sigma = np.exp(0.5 * cache["logvar"])[0] * self._y_std if self._trained \
                    else np.exp(0.5 * cache["logvar"])[0]
            return mu, np.maximum(sigma, self.config.noise_sigma_min)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        with self._lock:
            Xn = (X - self._x_mean) / self._x_std if self._trained else X
            mu_n, _, _ = self._forward(Xn,
                {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                 "Wo": self.Wo, "bo": self.bo})
            return mu_n * self._y_std + self._y_mean if self._trained else mu_n

    @property
    def trained(self) -> bool:
        return self._trained