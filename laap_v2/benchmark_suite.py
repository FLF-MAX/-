"""benchmark_suite.py - standardised laap_v2 benchmarks.

  B1 causal_structure_recovery      : DBN structure recovery (temperature->pressure/humidity)
  B2 analogical_structure_mapping   : structure-mapping consistency/coverage
  B3 drift_aware_continuous_learning: post-drift optimal-strategy rate
  B4 nonlinear_world_model_mae      : MLP world model MAE on Z = X*Y (vs mean baseline)
  B5 linucb_best_strategy_identification : best-strategy identification + exploit rate

Run all with:

    python benchmark_suite.py
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List

import numpy as np

from psi_core_v2 import PsiCoreV2
from meta_learning_engine import MetaLearningEngine
from probabilistic_world_model import ProbabilisticWorldModel
from neural_world_model import MLPWorldModel, MlpConfig
from drift_aware_meta_learning import DriftAwareMetaLearner
from deep_analogical_engine import DeepAnalogyEngine, DomainGraph, hungarian


def _b1_causal() -> Dict[str, Any]:
    rng = np.random.default_rng(3)
    N = 300
    T = [25.0]; P = [100.0]; H = [40.0]
    for _ in range(N - 1):
        t_prev = T[-1]
        T.append(0.95 * t_prev + rng.normal(0, 1.0))
        P.append(2.5 * t_prev + rng.normal(0, 1.0))
        H.append(30.0 + 0.8 * t_prev + rng.normal(0, 1.0))
    wm = ProbabilisticWorldModel(["temperature", "pressure", "humidity"], seed=3)
    wm.seed_from(list(zip(T, P, H)))
    g = wm.learn_structure()
    expected = {"pressure": {"temperature"}, "humidity": {"temperature"}}
    hits = sum(1 for var, pa in expected.items() if pa.issubset(set(g.get(var, ()))))
    score = hits / len(expected)
    return {"name": "causal_structure_recovery", "score": score, "baseline": 0.33,
            "detail": {"graph": {k: list(v) for k, v in g.items()}}}


def _b2_analogical() -> Dict[str, Any]:
    solar = DomainGraph("solar")
    solar.add_node("sun", role="central", attracts=1.0)
    for p in ("mercury", "earth", "mars"):
        solar.add_node(p, role="satellite", mass=1.0)
        solar.add_edge(p, "sun", "orbits")
    atom = DomainGraph("atom")
    atom.add_node("nucleus", role="central", charge=1.0)
    for e in ("e1", "e2", "e3"):
        atom.add_node(e, role="satellite", charge=1.0)
        atom.add_edge(e, "nucleus", "orbits")
    engine = DeepAnalogyEngine()
    ali, consistency, coverage, _ = engine.align(solar, atom)
    score = 0.5 * consistency + 0.5 * coverage
    correct = all(
        (b == "sun" and t == "nucleus") or (b != "sun" and t.startswith("e"))
        for b, t in ali.items()
    )
    if not correct:
        raise AssertionError(f"structural mapping violated: {ali}")
    return {"name": "analogical_structure_mapping", "score": score, "baseline": 0.25,
            "detail": {"alignment": ali, "consistency": consistency, "coverage": coverage}}


def _b3_continuous() -> Dict[str, Any]:
    learner = DriftAwareMetaLearner(["alpha", "beta", "gamma"], seed=7)
    rng = np.random.default_rng(7)
    seq = []
    for t in range(140):
        opt = "alpha" if t < 40 else "beta"
        s = learner.select()
        r = rng.normal(0.9, 0.1) if s == opt else rng.normal(0.2, 0.1)
        learner.observe(s, r)
        seq.append(s)
    rate = sum(1 for t in range(60, 140) if seq[t] == "beta") / 80
    # 收敛后（t=100+，探索期已过）的最优率——反映稳态真实能力，
    # 避免固定窗口从 t=60 起算时把漂移后探索期算进 score。
    rate_converged = sum(1 for t in range(100, 140) if seq[t] == "beta") / 40
    return {"name": "drift_aware_continuous_learning", "score": rate, "baseline": 0.5,
            "detail": {"alarms": learner.detected_drift(), "best": learner.best(),
                       "converged_rate": round(rate_converged, 3)}}


def _b4_prediction() -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    N = 2000
    X = rng.uniform(0, 5, N); Y = rng.uniform(0, 5, N)
    Z = X * Y + rng.normal(0, 0.1, N)
    inp = np.stack([X, Y, Z], axis=1)
    split = 1500
    m = MLPWorldModel(3, MlpConfig(epochs=300, warmup_epochs=250, tail_epochs=40), seed=0)
    started = time.time()
    m.fit(inp[:split], inp[:split])
    preds = np.array([m.predict(r) for r in inp[split:]])
    mae = float(np.abs(preds[:, 2] - Z[split:]).mean())
    bl = float(np.abs(Z[split:] - Z[:split].mean()).mean())
    score = max(0.0, 1.0 - mae / max(bl, 1e-9))
    return {"name": "nonlinear_world_model_mae", "score": score, "baseline": 0.0,
            "detail": {"mae": round(mae, 4), "baseline_mae": round(bl, 4),
                       "train_s": round(time.time() - started, 2)}}


def _b5_calibration() -> Dict[str, Any]:
    m = MetaLearningEngine(seed=1)
    feats = [0.9, 0.1, 0.8, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rng = np.random.default_rng(1)
    exploit = []
    for t in range(60):
        s, _ = m.select(feats)
        reward = 0.9 if s == "decompose" else (0.5 if rng.random() < 0.3 else 0.1)
        m.update(feats, s, reward)
        if t >= 30:
            exploit.append(float(s == "decompose"))
    rate = float(np.mean(exploit)) if exploit else 0.0
    return {"name": "linucb_best_strategy_identification", "score": rate, "baseline": 0.5,
            "detail": {"best": m.best_strategy(), "pulls": m.total_pulls}}


def _b0_psi() -> Dict[str, Any]:
    p = PsiCoreV2(seed=1)
    before = p.get_state()["needs"]["growth"]
    p.tick("帮我修复这个问题")
    p.inject_prediction_error(0.9, 0.2)
    after = p.get_state()["needs"]["growth"]
    pe_ok = after > before
    return {"name": "psi_prediction_error_injection", "score": 1.0 if pe_ok else 0.0,
            "baseline": 0.5, "detail": {"growth_before": round(before, 3),
                                          "growth_after": round(after, 3)}}


BENCHMARKS = [_b0_psi, _b1_causal, _b2_analogical, _b3_continuous, _b4_prediction, _b5_calibration]


def run_all(verbose: bool = True) -> List[Dict[str, Any]]:
    results = []
    for fn in BENCHMARKS:
        try:
            res = fn()
        except Exception as e:
            res = {"name": fn.__name__, "score": 0.0, "baseline": 0.0, "error": repr(e)}
        results.append(res)
        if verbose:
            print(f"[{res['name']}] score={res.get('score', 0.0):.3f} baseline={res.get('baseline', 0.0)}"
                  f"  {res.get('detail', res.get('error', ''))}")
    if verbose:
        avg = float(np.mean([r.get("score", 0.0) for r in results]))
        print(f"--- mean score: {avg:.3f} ---")
    return results


if __name__ == "__main__":
    run_all()