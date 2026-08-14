"""Unit tests for the FastAPI surface - endpoints, tracing, persistence, chaos."""

import os
import tempfile

import pytest

from fastapi.testclient import TestClient

import api_server
from laap_integration import LaapCognitiveSystem, IntegrationConfig


@pytest.fixture(scope="module")
def client():
    api_server.system = LaapCognitiveSystem(
        config=api_server.cfg,
        integration_cfg=IntegrationConfig(),
    )
    return TestClient(api_server.app)


@pytest.fixture
def state_path():
    tmp = tempfile.mkdtemp()
    return os.path.join(tmp, "api_state.json")


def test_health_live(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_health_ready(client):
    assert client.get("/health/ready").status_code == 200


def test_health_modules(client):
    m = client.get("/health/modules").json()
    assert m["psi"] is True and m["memory"] is True


def test_chat_returns_cognitive_response(client):
    r = client.post("/v1/chat", json={"message": "你好，帮我分析", "user": "t"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert "strategy" in body and "cognitive_state" in body


def test_chat_invalid_payload(client):
    assert client.post("/v1/chat", json={"message": 123}).status_code == 400
    assert client.post("/v1/chat", json={}).status_code == 400


def test_tracing_header_set(client):
    r = client.post("/v1/chat", json={"message": "hi"})
    assert "x-trace-id" in r.headers


def test_tracing_honors_custom_id(client):
    r = client.post("/v1/chat", json={"message": "hi"},
                    headers={"x-trace-id": "my-trace"})
    assert r.headers.get("x-trace-id") == "my-trace"


def test_cognitive_state_endpoint(client):
    r = client.get("/v1/cognitive/state")
    assert r.status_code == 200
    assert r.json()["booted"] is True


def test_metrics_endpoint(client):
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    assert r.json()["requests"] >= 0


def test_reset_endpoint(client):
    r = client.post("/v1/reset")
    assert r.status_code == 200 and r.json()["booted"] is True


def test_chaos_self_heal(client):
    client.post("/v1/chaos/psi")
    r = client.post("/v1/chat", json={"message": "还活着吗"})
    assert r.status_code == 200
    assert r.json()["degraded"] is False  # self-healed


def test_chaos_unknown_module(client):
    assert client.post("/v1/chaos/nope").status_code == 400


def test_allocation_endpoint(client):
    payload = {
        "agents": [{"name": "a1", "capability": "code", "cost": 0.5},
                   {"name": "a2", "capability": "doc", "cost": 0.3}],
        "tasks": [{"task_id": "t1", "capability": "code"},
                  {"task_id": "t2", "capability": "doc"}],
        "budget": 2.0,
    }
    r = client.post("/v1/allocation", json=payload)
    assert r.status_code == 200
    assert r.json()["completed"] == 2


def test_state_save_load(client, state_path):
    client.post("/v1/chat", json={"message": "保存这条"})
    r = client.post("/v1/state/save", json={"path": state_path})
    assert r.status_code == 200
    saved = r.json()["memories"]

    client.post("/v1/reset")
    r = client.post("/v1/state/load", json={"path": state_path})
    assert r.status_code == 200
    assert r.json()["memories"] == saved


def test_state_load_missing(client):
    assert client.post("/v1/state/load",
                       json={"path": "/no/such/file.json"}).status_code == 404
    assert client.post("/v1/state/load", json={}).status_code == 400