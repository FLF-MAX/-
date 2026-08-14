"""api_server.py - FastAPI service exposing the laap_v2 cognitive runtime.

Endpoints:
  GET  /health/live            liveness probe
  GET  /health/ready           readiness probe (all module checks)
  GET  /health/modules         per-module state
  POST /v1/chat                {message, user} -> cognitive response
  POST /v1/cognitive/state     current PSI/meta/memory state
  GET  /v1/cognitive/state     same (GET convenience)
  GET  /v1/metrics             metrics snapshot (JSON)
  POST /v1/reset               reboot the cognitive system
  POST /v1/chaos/{module}      degrade a module (self-healing demo)
  POST /v1/allocation          run the multi-agent arbiter on given tasks

Run locally:
    python api_server.py                 # dev server, uvicorn if present
    uvicorn api_server:app --host 0.0.0.0 --port 11546

Dependencies (optional): fastapi, uvicorn, requests.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laap_integration import LaapCognitiveSystem, IntegrationConfig
from multi_agent_coordination import CognitiveArbiter, Agent, Task
from production_infra import StructuredLogger, SystemConfig, TypeValidator, ValidationError

app = FastAPI(title="laap_v2 cognitive runtime", version="2.0.0",
              description="PSI + meta-learning + world-model + memory + coordination")

cfg = SystemConfig()
system = LaapCognitiveSystem(config=cfg, integration_cfg=IntegrationConfig())
_system_lock = threading.RLock()


# --------------------------------------------------------------------------- #
# rate limiting (per-IP token bucket, dependency-free)
# --------------------------------------------------------------------------- #
class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


_buckets: Dict[str, TokenBucket] = {}
_bucket_lock = threading.Lock()
RATE_PER_IP = float(cfg.get("server.rate_limit", 60))


def _bucket(ip: str) -> TokenBucket:
    with _bucket_lock:
        if ip not in _buckets:
            _buckets[ip] = TokenBucket(RATE_PER_IP / 60.0, RATE_PER_IP)
        return _buckets[ip]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    if not _bucket(ip).allow():
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})
    return await call_next(request)


# --------------------------------------------------------------------------- #
# tracing: per-request trace id -> structured-log context + response header
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or f"{time.time_ns():x}-{request.client.host if request.client else 'x'}"
    StructuredLogger.set_context(trace_id=trace_id, path=request.url.path)
    start = time.time()
    response = await call_next(request)
    response.headers["x-trace-id"] = trace_id
    system.metrics.inc("http_request")
    if response.status_code >= 500:
        system.metrics.inc("http_error")
    system.logger.info("request done", path=request.url.path,
                       status=response.status_code,
                       latency_ms=round((time.time() - start) * 1000, 2))
    StructuredLogger.clear_context()
    return response


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require_booted() -> None:
    with _system_lock:
        if not system._booted:
            system.bootstrap()


@app.exception_handler(ValidationError)
async def validation_handler(_: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
@app.get("/health/live")
def liveness() -> Dict[str, Any]:
    return {"status": "alive" if system.liveness() else "dead",
            "uptime_s": round(time.time() - _start_time, 2)}


@app.get("/health/ready")
def readiness() -> Dict[str, Any]:
    _require_booted()
    r = system.readiness()
    if not r["ready"]:
        raise HTTPException(status_code=503, detail=r)
    return r


@app.get("/health/modules")
def modules() -> Dict[str, Any]:
    _require_booted()
    return {"psi": system.psi.heartbeat_ok(), "meta": system.meta is not None,
            "memory": system.memory.size() >= 0,
            "state_counter": system._state_counter}


# --------------------------------------------------------------------------- #
# cognitive
# --------------------------------------------------------------------------- #
@app.post("/v1/chat")
def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_booted()
    try:
        TypeValidator.validate_schema(payload, {"message": str}, "payload")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    user = payload.get("user", "friend")
    return system.process_input(payload["message"])


@app.post("/v1/cognitive/state")
@app.get("/v1/cognitive/state")
def cognitive_state() -> Dict[str, Any]:
    _require_booted()
    return system.current_state()


@app.get("/v1/metrics")
def metrics() -> Dict[str, Any]:
    return system.metrics_snapshot()


@app.post("/v1/reset")
def reset() -> Dict[str, Any]:
    with _system_lock:
        system.shutdown()
        system.bootstrap("friend")
        return {"status": "reset", "booted": system._booted}


@app.post("/v1/chaos/{module}")
def chaos(module: str) -> Dict[str, Any]:
    _require_booted()
    if module not in ("psi", "memory", "meta"):
        raise HTTPException(status_code=400, detail="unknown module")
    system.degrade_module(module)
    return {"degraded": module, "note": "system will self-heal on next call"}


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
@app.post("/v1/state/save")
def state_save(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Snapshot cognitive state. payload: {path: optional} (default state.json)."""
    _require_booted()
    path = (payload or {}).get("path", "state.json")
    try:
        return system.save_state(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"save failed: {e!r}") from e


@app.post("/v1/state/load")
def state_load(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Restore cognitive state. payload: {path: required}."""
    path = (payload or {}).get("path")
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    try:
        return system.load_state(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no such state file: {path}") from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load failed: {e!r}") from e


# --------------------------------------------------------------------------- #
# coordination
# --------------------------------------------------------------------------- #
@app.post("/v1/allocation")
def allocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload: {agents: [{name, capability, cost, reliability}],
                 tasks:  [{task_id, capability, reward, min_cost}],
                 budget: float}"""
    try:
        TypeValidator.validate_schema(payload, {"agents": list, "tasks": list}, "payload")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    arb = CognitiveArbiter(budget=float(payload.get("budget", 100.0)))
    for a in payload["agents"]:
        arb.register(Agent(name=a["name"], capability=a["capability"],
                           cost=float(a.get("cost", 1.0)),
                           reliability=float(a.get("reliability", 1.0))))
    tasks = [Task(task_id=t["task_id"], capability=t["capability"],
                  reward=float(t.get("reward", 1.0)),
                  min_cost=float(t.get("min_cost", 0.1))) for t in payload["tasks"]]
    return arb.allocate(tasks)


# --------------------------------------------------------------------------- #
_start_time = time.time()


def main() -> None:
    _require_booted()
    host = cfg.get("laap.host", "0.0.0.0")
    port = int(cfg.get("laap.port", 11546))
    print(f"laap_v2 api on http://{host}:{port}")
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        print("uvicorn not installed; use: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
