"""Unit tests for psi_core_v2 - competition dynamics + PAD + prediction error."""

import pytest

from psi_core_v2 import PsiCoreV2, PsiConfig


@pytest.fixture
def psi():
    p = PsiCoreV2(seed=7)
    p.tick("hi")  # heartbeat requires at least one tick
    return p


def test_initial_state(psi):
    s = psi.get_state()
    assert set(s) == {"needs", "velocities", "emotion", "pad", "attention",
                      "tick", "total_drive", "recovery_count", "alive"}
    assert set(s["needs"]) == {"competence", "relatedness", "growth", "certainty", "autonomy"}
    assert set(s["pad"]) == {"pleasure", "arousal", "dominance"}
    assert s["alive"] is True


def test_needs_stay_bounded(psi):
    for _ in range(200):
        psi.tick("持续输入，保持高强度对话，复杂任务不断涌现。很累但要坚持。")
    s = psi.get_state()
    for v in s["needs"].values():
        assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"need out of range: {v}"


def test_pad_bounded(psi):
    for _ in range(100):
        psi.tick("很好")
    s = psi.get_state()
    for v in s["pad"].values():
        assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9, f"pad out of range: {v}"


def test_prediction_error_raises_growth_need(psi):
    for _ in range(3):
        psi.tick("重复")
    before = psi.get_state()["needs"]["growth"]
    psi.inject_prediction_error(0.9, 0.1)
    after = psi.get_state()["needs"]["growth"]
    assert after > before  # PE drives growth need up


def test_competitive_attention_focuses(psi):
    psi.tick("为什么？为什么？为什么？我还是不明白为什么。")
    psi.tick("我不确定。请再解释一遍，我真的不确定。")
    psi.tick("依然不确定，完全不懂。")
    assert psi.get_state()["attention"] in PsiCoreV2.NEEDS


def test_heartbeat_and_recovery(psi):
    assert psi.heartbeat_ok() is True
    psi._last_tick_at = 0.0  # simulate heartbeat loss
    assert psi.heartbeat_ok() is False
    psi.recover()
    s = psi.get_state()
    assert s["recovery_count"] >= 1
    assert s["alive"] is True
    assert psi.heartbeat_ok() is True


def test_reproducible_seed():
    a = PsiCoreV2(seed=3)
    b = PsiCoreV2(seed=3)
    for _ in range(10):
        a.tick("same text"); b.tick("same text")
    sa, sb = a.get_state(), b.get_state()
    assert sa["needs"] == sb["needs"] and sa["pad"] == sb["pad"]


def test_config_drive():
    cfg = PsiConfig(prediction_error_scale=1.0, input_force=2.0, competition=0.0,
                    growth_rate=0.9)
    p2 = PsiCoreV2(config=cfg, seed=1)
    p2.tick("hi")
    before = p2.get_state()["needs"]["growth"]
    p2.tick("冲刺阶段，要快，快点！更多！更快！")
    assert p2.get_state()["needs"]["growth"] >= before - 1e-6