"""Unit tests for production_infra - validator / breaker / metrics / health / config."""

import pytest

from production_infra import (
    CircuitBreaker, CircuitOpenError, HealthMonitor, MetricsCollector,
    StructuredLogger, SystemConfig, TypeValidator, ValidationError,
    safe_boundary,
)


# --------------------------------------------------------------------------- #
# TypeValidator
# --------------------------------------------------------------------------- #
def test_validate_type_ok_and_fail():
    assert TypeValidator.validate_type(5, int) == 5
    with pytest.raises(ValidationError):
        TypeValidator.validate_type(5, str)


def test_validate_non_empty():
    assert TypeValidator.validate_non_empty(" x ") == " x "
    with pytest.raises(ValidationError):
        TypeValidator.validate_non_empty("   ")
    with pytest.raises(ValidationError):
        TypeValidator.validate_non_empty(None)


def test_validate_range():
    assert TypeValidator.validate_range(0.5, 0, 1) == 0.5
    with pytest.raises(ValidationError):
        TypeValidator.validate_range(1.5, 0, 1)


def test_validate_schema():
    TypeValidator.validate_schema({"a": 1, "b": "x"}, {"a": int, "b": str})
    with pytest.raises(ValidationError):
        TypeValidator.validate_schema({"a": 1}, {"a": int, "b": str})
    with pytest.raises(ValidationError):
        TypeValidator.validate_schema({"a": "no", "b": "x"}, {"a": int, "b": str})


# --------------------------------------------------------------------------- #
# CircuitBreaker
# --------------------------------------------------------------------------- #
def test_breaker_opens_on_failures():
    cb = CircuitBreaker("t", failure_threshold=2, cooldown=0.01)
    cb.record_failure()
    assert cb.state == CircuitBreaker.CLOSED
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    with pytest.raises(CircuitOpenError):
        cb.guard()


def test_breaker_half_open_recovers():
    cb = CircuitBreaker("t", failure_threshold=2, cooldown=0.05, half_open_max=1)
    cb.record_failure(); cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    import time
    time.sleep(0.07)
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.guard()
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED


def test_breaker_reset():
    cb = CircuitBreaker("t", failure_threshold=1)
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    cb.reset()
    assert cb.state == CircuitBreaker.CLOSED
    cb.guard()  # no raise


# --------------------------------------------------------------------------- #
# MetricsCollector
# --------------------------------------------------------------------------- #
def test_metrics_snapshot_shape():
    m = MetricsCollector()
    m.record_latency(0.01); m.record_latency(0.02); m.record_latency(0.05)
    m.record_error()
    snap = m.snapshot()
    assert snap["requests"] == 3
    assert snap["errors"] == 1
    assert set(snap["latency_ms"]) == {"p50", "p95", "p99"}
    assert 0 <= snap["error_rate"] <= 1


def test_metrics_disabled():
    m = MetricsCollector(enabled=False)
    m.record_latency(0.1); m.inc("x")
    snap = m.snapshot()
    assert snap["requests"] == 0 and snap["counters"] == {}


# --------------------------------------------------------------------------- #
# HealthMonitor
# --------------------------------------------------------------------------- #
def test_health_readiness_all_ok():
    h = HealthMonitor()
    h.register("a", lambda: True)
    h.register("b", lambda: True)
    assert h.readiness()["ready"] is True


def test_health_readiness_failure():
    h = HealthMonitor()
    h.register("a", lambda: True)
    h.register("b", lambda: False)
    r = h.readiness()
    assert r["ready"] is False
    assert r["checks"]["b"] is False


def test_health_exception_is_failure():
    h = HealthMonitor()
    h.register("bad", lambda: 1 / 0)
    assert h.readiness()["ready"] is False


def test_health_liveness():
    h = HealthMonitor()
    assert h.liveness() is True
    h.set_liveness(False)
    assert h.liveness() is False


# --------------------------------------------------------------------------- #
# SystemConfig
# --------------------------------------------------------------------------- #
def test_system_config_defaults():
    c = SystemConfig()
    assert c.get("laap.port") == 11546
    assert c.get("missing", "fallback") == "fallback"


def test_system_config_env_override(monkeypatch):
    monkeypatch.setenv("LAAP_PORT", "9999")
    monkeypatch.setenv("LAAP_DEEPSEEK__API_KEY", "k123")
    c = SystemConfig()
    # LAAP_ prefix stripped; "__" becomes "."
    assert c.get("port") == "9999"
    assert c.get("deepseek.api_key") == "k123"
    assert c.int("port") == 9999


def test_system_config_require_missing():
    c = SystemConfig()
    with pytest.raises(ValidationError):
        c.require("does.not.exist")


# --------------------------------------------------------------------------- #
# StructuredLogger + safe_boundary
# --------------------------------------------------------------------------- #
def test_logger_context(tmp_path):
    StructuredLogger.clear_context()
    StructuredLogger.set_context(request_id="r1")
    logger = StructuredLogger("test_log", log_dir=str(tmp_path))
    logger.info("hello", extra=1)
    lines = (tmp_path / "test_log.log").read_text(encoding="utf-8").strip().splitlines()
    assert lines and "r1" in lines[0] and "hello" in lines[0]


def test_safe_boundary_fallback():
    calls = []

    @safe_boundary(fallback=42)
    def boom():
        raise RuntimeError("x")

    assert boom() == 42
    assert calls == []  # fallback short-circuits; no side effects here


def test_safe_boundary_reraises():
    @safe_boundary()
    def boom():
        raise ValueError("y")

    with pytest.raises(ValueError):
        boom()