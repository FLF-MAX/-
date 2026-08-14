"""Unit tests for drift_aware_meta_learning - Page-Hinkley + adaptive window."""

import pytest

from drift_aware_meta_learning import DriftAwareMetaLearner


def test_select_returns_registered_strategy():
    d = DriftAwareMetaLearner(["alpha", "beta", "gamma"], seed=1)
    for _ in range(30):
        assert d.select() in ("alpha", "beta", "gamma")


def test_observe_returns_bool():
    d = DriftAwareMetaLearner(["a", "b"], seed=1)
    r = d.observe("a", 0.5)
    assert r is True or r is False  # returns drift-detected flag


def test_drift_detected_after_switch():
    d = DriftAwareMetaLearner(["a", "b"], seed=4)
    for _ in range(40):
        s = d.select(); d.observe(s, 0.9)
    for _ in range(150):
        s = d.select(); d.observe(s, 0.1)
    assert d.detected_drift() >= 1


def test_no_false_drift_on_stable():
    d = DriftAwareMetaLearner(["a", "b"], seed=2)
    for _ in range(100):
        s = d.select(); d.observe(s, 0.8)
    assert d.detected_drift() == 0


def test_best_after_drift():
    d = DriftAwareMetaLearner(["a", "b"], seed=5)
    for _ in range(40):
        s = d.select(); d.observe(s, 0.9)               # a best
    for _ in range(300):
        s = d.select(); d.observe(s, 0.1 if s == "a" else 0.9)  # b best
    assert d.best() == "b"


def test_best_empty_never_drifts():
    d = DriftAwareMetaLearner(["a"], seed=6)
    assert d.best() in ("a",)