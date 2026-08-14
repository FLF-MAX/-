"""production_infra.py - logging / config / validation / resilience / metrics.

P0 production infrastructure for the laap_v2 cognitive runtime:

  StructuredLogger  - JSON-lines logging to console + rotating file,
                      request-context injection via thread-local.
  SystemConfig      - dataclass of defaults with env-var overrides and
                      optional JSON file loading.
  TypeValidator     - runtime type / non-empty / range / schema checks
                      (pydantic-style, dependency-free).
  CircuitBreaker    - failure-rate based open/half-open/closed states.
  MetricsCollector  - counters, gauge, latency p50/p95/p99, error rate,
                      throughput - flushed as JSON snapshots.
  HealthMonitor     - liveness/readiness with registered checks.
  safe_boundary     - uniform exception-boundary decorator.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "StructuredLogger", "SystemConfig", "TypeValidator", "ValidationError",
    "CircuitBreaker", "CircuitOpenError", "MetricsCollector", "HealthMonitor",
    "safe_boundary",
]


# --------------------------------------------------------------------------- #
# validation                                                                  #
# --------------------------------------------------------------------------- #
class ValidationError(ValueError):
    pass


class TypeValidator:
    """Dependency-free runtime validation helpers."""

    @staticmethod
    def validate_type(value: Any, expected: type, name: str = "value") -> Any:
        if not isinstance(value, expected):
            raise ValidationError(f"{name} must be {expected.__name__}, got {type(value).__name__}")
        return value

    @staticmethod
    def validate_non_empty(value: str, name: str = "value") -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValidationError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def validate_range(value: float, lo: float, hi: float, name: str = "value") -> float:
        if not (lo <= value <= hi):
            raise ValidationError(f"{name} out of range [{lo}, {hi}]: {value}")
        return value

    @staticmethod
    def validate_schema(obj: Dict[str, Any], schema: Dict[str, type], name: str = "obj") -> Dict[str, Any]:
        missing = [k for k, t in schema.items() if k not in obj]
        if missing:
            raise ValidationError(f"{name} missing keys {missing}")
        for k, t in schema.items():
            if not isinstance(obj[k], t):
                raise ValidationError(f"{name}.{k} must be {t.__name__}, got {type(obj[k]).__name__}")
        return obj

    @staticmethod
    def validate_non_negative(value: int, name: str = "value") -> int:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
        return value


# --------------------------------------------------------------------------- #
# structured logging                                                          #
# --------------------------------------------------------------------------- #
class StructuredLogger:
    """JSON-lines logger with thread-local context injection."""

    _local = threading.local()

    def __init__(self, name: str = "laap_v2", log_dir: Optional[str] = None,
                 level: int = logging.INFO, max_bytes: int = 5_000_000, backup: int = 3):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler: logging.Handler
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                handler = logging.handlers.RotatingFileHandler(
                    str(Path(log_dir) / f"{name}.log"),
                    maxBytes=max_bytes, backupCount=backup, encoding="utf-8",
                )
            else:
                handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    @classmethod
    def set_context(cls, **kwargs) -> None:
        ctx = getattr(cls._local, "context", {})
        ctx.update(kwargs)
        cls._local.context = ctx

    @classmethod
    def clear_context(cls) -> None:
        cls._local.context = {}

    def _record(self, level: str, msg: str, **fields) -> None:
        rec = {"ts": time.time(), "level": level, "logger": self.name, "msg": msg}
        ctx = getattr(self._local, "context", {})
        if ctx:
            rec["ctx"] = dict(ctx)
        if fields:
            rec.update(fields)
        self.logger.log(
            getattr(logging, level.upper(), logging.INFO),
            json.dumps(rec, ensure_ascii=False, default=str),
        )

    def info(self, msg: str, **fields) -> None: self._record("info", msg, **fields)
    def warn(self, msg: str, **fields) -> None: self._record("warning", msg, **fields)
    def error(self, msg: str, **fields) -> None: self._record("error", msg, **fields)
    def debug(self, msg: str, **fields) -> None: self._record("debug", msg, **fields)


# --------------------------------------------------------------------------- #
# configuration                                                               #
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    "laap.app_name": "laap-v2",
    "laap.host": "0.0.0.0",
    "laap.port": 11546,
    "laap.log_level": "info",
    "deepseek.api_key": "",
    "deepseek.base_url": "https://api.deepseek.com",
    "deepseek.model": "deepseek-chat",
    "memory.embed_dim": 128,
    "memory.capacity": 5000,
    "psi.dt": 1.0,
    "meta.d_context": 16,
    "server.rate_limit": 60,
    "server.max_tokens": 2048,
}


@dataclass
class SystemConfig:
    params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    prefix: str = "LAAP_"
    file: Optional[str] = None

    def __init__(self, file: Optional[str] = None, env: bool = True):
        merged = dict(DEFAULTS)
        if file and os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                merged[str(k)] = v
        if env:
            for key in list(os.environ.keys()):
                if key.startswith(self.prefix):
                    flat = key[len(self.prefix):].lower().replace("__", ".")
                    merged[flat] = os.environ[key]
        self.params = merged
        self.file = file

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.params or self.params[key] in ("", None):
            raise ValidationError(f"missing required config: {key}")
        return self.params[key]

    def int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.params.get(key, default))
        except (TypeError, ValueError):
            return default

    def float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.params.get(key, default))
        except (TypeError, ValueError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.params)


# --------------------------------------------------------------------------- #
# circuit breaker                                                             #
# --------------------------------------------------------------------------- #
class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Failure-based circuit breaker."""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, name: str, failure_threshold: int = 5, cooldown: float = 30.0,
                 half_open_max: int = 3, on_state_change: Optional[Callable] = None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.half_open_max = half_open_max
        self._lock = threading.RLock()
        self._state = self.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._half_open_trials = 0
        self._on_state_change = on_state_change

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN and (time.time() - self._opened_at) >= self.cooldown:
                self._transition(self.HALF_OPEN)
            return self._state

    def _transition(self, new_state: str) -> None:
        if new_state != self._state:
            old = self._state
            self._state = new_state
            if self._on_state_change:
                try:
                    self._on_state_change(self.name, old, new_state)
                except Exception:
                    pass

    def guard(self) -> None:
        s = self.state
        if s == self.OPEN:
            raise CircuitOpenError(f"circuit '{self.name}' is open")

    def record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._half_open_trials += 1
                if self._half_open_trials >= self.half_open_max:
                    self._failures = 0
                    self._half_open_trials = 0
                    self._transition(self.CLOSED)
            else:
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._transition(self.OPEN)
                self._opened_at = time.time()
                self._half_open_trials = 0
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._transition(self.OPEN)
                self._opened_at = time.time()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_trials = 0
            self._transition(self.CLOSED)


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #
class MetricsCollector:
    """Thread-safe counters, histogram and derived rates."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = {}
        self._latencies: List[float] = []
        self._started = time.time()
        self._errors = 0
        self._total = 0
        self._window_start = time.time()
        self._window_total = 0
        self._window_errors = 0

    def inc(self, name: str, value: float = 1.0) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def record_latency(self, seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._latencies.append(seconds)
            if len(self._latencies) > 2000:
                self._latencies = self._latencies[-2000:]
            self._total += 1
            self._window_total += 1

    def record_error(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._errors += 1
            self._window_errors += 1

    def _pct(self, p: float) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            arr = sorted(self._latencies)
            idx = min(len(arr) - 1, int(p * len(arr)))
            return arr[idx]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            window_seconds = max(now - self._window_start, 1e-6)
            out = {
                "uptime_s": round(now - self._started, 2),
                "counters": dict(self._counters),
                "requests": self._total,
                "errors": self._errors,
                "error_rate": round(self._window_errors / max(self._window_total, 1), 4),
                "throughput_rps": round(self._window_total / window_seconds, 3),
                "latency_ms": {
                    "p50": round(1000 * self._pct(0.50), 3),
                    "p95": round(1000 * self._pct(0.95), 3),
                    "p99": round(1000 * self._pct(0.99), 3),
                },
            }
            self._window_start = now
            self._window_total = 0
            self._window_errors = 0
            return out


# --------------------------------------------------------------------------- #
# health                                                                      #
# --------------------------------------------------------------------------- #
class HealthMonitor:
    """Liveness / readiness with registered checks."""

    def __init__(self):
        self._lock = threading.RLock()
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._live = True

    def register(self, name: str, check: Callable[[], bool]) -> None:
        with self._lock:
            self._checks[name] = check

    def set_liveness(self, live: bool) -> None:
        with self._lock:
            self._live = live

    def liveness(self) -> bool:
        with self._lock:
            return self._live

    def readiness(self) -> Dict[str, Any]:
        with self._lock:
            results = {}
            for name, check in list(self._checks.items()):
                try:
                    results[name] = bool(check())
                except Exception:
                    results[name] = False
            ready = all(results.values()) if results else True
            return {"ready": ready, "checks": results}


# --------------------------------------------------------------------------- #
# exception boundary                                                          #
# --------------------------------------------------------------------------- #
def safe_boundary(logger: Optional[StructuredLogger] = None, fallback=None):
    """Decorator: never let a cognitive-module exception kill the process.

    Catches, logs, records metric and either returns `fallback` or re-raises
    so the caller can degrade gracefully.
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"boundary caught in {fn.__qualname__}",
                                 error=repr(e), trace=traceback.format_exc()[-1500:])
                if fallback is not None:
                    return fallback() if callable(fallback) else fallback
                raise
        return wrapper
    return deco
