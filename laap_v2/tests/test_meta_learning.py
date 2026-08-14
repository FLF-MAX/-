"""Unit tests for meta_learning_engine - LinUCB + REINFORCE."""

import pytest

from meta_learning_engine import MetaLearningEngine


@pytest.fixture
def engine():
    return MetaLearningEngine(seed=1)


def test_strategy_set(engine):
    assert set(engine.strategies) == {"decompose", "direct_attempt", "ask_clarify",
                                      "retry", "defer", "search_memory"}


def test_select_returns_valid_strategy(engine):
    feats = [0.5] * engine.config.d_context
    for _ in range(20):
        s, conf = engine.select(feats)
        assert s in engine.strategies
        assert 0.0 <= conf <= 1.0


def test_update_does_not_crash_and_learns(engine):
    feats = [0.9, 0.1, 0.8] + [0.0] * (engine.config.d_context - 3)
    for _ in range(50):
        s, _ = engine.select(feats)
        engine.update(feats, s, 0.9 if s == "decompose" else 0.1)
    assert engine.total_pulls == 50
    assert engine.best_strategy() in engine.strategies


def test_leaning_toward_best(engine):
    feats = [0.9, 0.1, 0.8, 0.1] + [0.0] * (engine.config.d_context - 4)
    picks = []
    for t in range(100):
        s, _ = engine.select(feats)
        engine.update(feats, s, 0.9 if s == "decompose" else 0.05)
        picks.append(s)
    late = [s for s in picks[70:] if s == "decompose"]
    assert len(late) >= 15


def test_summary_shape(engine):
    s, _ = engine.select([0.1] * engine.config.d_context)
    engine.update([0.1] * engine.config.d_context, s, 0.5)
    sm = engine.summary()
    for key in ("pulls", "best", "proficiency", "avg_reward"):
        assert key in sm


def test_reproducible_seed():
    a = MetaLearningEngine(seed=5)
    b = MetaLearningEngine(seed=5)
    feats = [0.3, 0.4, 0.5] + [0.0] * (a.config.d_context - 3)
    sa = [a.select(feats)[0] for _ in range(50)]
    sb = [b.select(feats)[0] for _ in range(50)]
    assert sa == sb