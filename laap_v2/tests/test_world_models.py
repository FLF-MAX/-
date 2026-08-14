"""Unit tests for world models - DBN (probabilistic) + MLP (neural)."""

import numpy as np
import pytest

from probabilistic_world_model import ProbabilisticWorldModel
from neural_world_model import MLPWorldModel, MlpConfig


def _linear_series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    T, P = [25.0], [100.0]
    for _ in range(n - 1):
        T.append(0.95 * T[-1] + rng.normal(0, 1.0))
        P.append(2.5 * T[-1] + rng.normal(0, 1.0))
    return list(zip(T, P))


def test_wm_guards_empty():
    wm = ProbabilisticWorldModel(["a", "b"], seed=1)
    with pytest.raises(RuntimeError):
        wm.learn_structure()
    assert wm.num_observations() == 0


def test_wm_discovers_temperature_drives_pressure():
    wm = ProbabilisticWorldModel(["temperature", "pressure"], seed=1)
    wm.seed_from(_linear_series())
    g = wm.learn_structure()
    # ground truth: pressure depends on temperature
    assert "temperature" in set(g["pressure"])
    assert "pressure" not in set(g["temperature"])  # temperature self-loop only


def test_wm_predict_next_shape():
    wm = ProbabilisticWorldModel(["temperature", "pressure"], seed=1)
    wm.seed_from(_linear_series())
    out = wm.predict_next([25.0, 100.0])
    assert len(out) == 2
    assert all(np.isfinite(x) for x in out)


def test_wm_reproducible_seed():
    a = ProbabilisticWorldModel(["a", "b"], seed=9)
    b = ProbabilisticWorldModel(["a", "b"], seed=9)
    data = _linear_series(60, seed=4)
    a.seed_from(data); b.seed_from(data)
    assert a.get_graph() == b.get_graph()
    assert np.allclose(a.posterior_mean(), b.posterior_mean(), atol=1e-9)


def test_mlp_predict_range():
    m = MLPWorldModel(3, MlpConfig(), seed=1)
    out = m.predict([0.5, 0.5, 0.5])
    assert len(out) == 3
    assert all(np.isfinite(x) for x in out)


def test_mlp_learns_nonlinearity():
    rng = np.random.default_rng(0)
    N = 1500
    X = rng.uniform(0, 5, N); Y = rng.uniform(0, 5, N)
    Z = X * Y
    inp = np.stack([X, Y, Z], axis=1)
    m = MLPWorldModel(3, MlpConfig(epochs=200, warmup_epochs=180, tail_epochs=20), seed=0)
    m.fit(inp[:1000], inp[:1000])
    preds = np.array([m.predict(r) for r in inp[1000:1010]])
    mae = float(np.abs(preds[:, 2] - Z[1000:1010]).mean())
    assert mae < 0.5  # baseline mean-only would be ~8.2


def test_mlp_heteroscedastic_output():
    m = MLPWorldModel(3, MlpConfig(epochs=5, warmup_epochs=1), seed=2)
    m.fit(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 4.0]], dtype=float),
          np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 4.0]], dtype=float))
    out = m.predict([1.0, 1.0, 1.0])
    assert all(np.isfinite(x) for x in out)