"""Unit tests for laap_integration - end-to-end cognitive loop + degraded mode."""

import pytest

from laap_integration import IntegrationConfig, LaapCognitiveSystem


@pytest.fixture
def system():
    s = LaapCognitiveSystem(integration_cfg=IntegrationConfig())
    s.bootstrap("tester")
    return s


def test_bootstrap_awakens(system):
    assert system._booted is True
    assert system.current_state()["emotion"] != "asleep"


def test_process_input_returns_response(system):
    r = system.process_input("你好，帮我分析一下")
    assert "response" in r
    assert r["degraded"] is False
    assert "strategy" in r and "cognitive_state" in r
    assert r["cognitive_state"]["emotion"] in system.psi.get_state()["emotion"]


def test_process_input_invalid(system):
    for bad in ("", "   "):
        r = system.process_input(bad)
        assert "response" in r and r.get("error")
        assert "输入无效" in r["response"]


def test_process_input_non_string(system):
    r = system.process_input(123)  # type mismatch -> validate fail, not raise
    assert "response" in r


def test_auto_bootstrap_on_first_input():
    s = LaapCognitiveSystem()
    assert s._booted is False
    s.process_input("自动唤醒")
    assert s._booted is True


def test_shutdown_reboot(system):
    system.shutdown()
    assert system._booted is False
    r = system.process_input("又回来了")  # reboots automatically
    assert r["degraded"] is False


def test_degraded_mode_never_raises(system):
    for _ in range(20):
        r = system.process_input("异常输入 %%^&# \x00")
        assert "response" in r


def test_psi_recovery_after_chaos(system):
    system.degrade_module("psi")
    assert system.psi.heartbeat_ok() is False
    r = system.process_input("还活着吗")
    assert system.psi.heartbeat_ok() is True
    assert r["degraded"] is False   # self-healed on the way through


def test_meta_fallback(system, monkeypatch):
    import laap_integration as li
    system.meta = None  # simulate corrupt meta
    r = system.process_input("好")
    assert r["response"] and r["strategy"] == "direct_attempt"


def test_current_state_keys(system):
    st = system.current_state()
    for key in ("booted", "psi", "meta", "memory_size", "emotion", "state_counter"):
        assert key in st


def test_metrics_health_readiness(system):
    assert system.metrics_snapshot()["requests"] >= 0
    assert isinstance(system.readiness(), dict)
    assert system.liveness() is True


def test_llm_bridge_local_mode(system):
    # no API key configured -> local synthesizer used
    r = system.process_input("谢谢")
    assert "本地模式" in r["response"] or "(本地模式" in r["response"]