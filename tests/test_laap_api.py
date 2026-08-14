"""
LAAP Brain API 端到端测试
==========================

验证 API 服务器能启动并响应核心端点。
运行:
    python -m pytest tests/test_laap_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from aiohttp.test_utils import TestClient, TestServer

from laap_brain.api import create_app


def test_llm_bridge_disabled_without_key():
    """未配置 DEEPSEEK_API_KEY 时，LLM 桥应优雅禁用（返回 None），不影响 Zero-LLM 管线。"""
    from laap_brain.api import _llm_respond, _get_llm_integration

    import os
    old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        assert _get_llm_integration() is None
        assert _llm_respond("你好") is None
    finally:
        if old_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = old_key


@pytest.mark.asyncio
async def test_health_endpoint():
    """/health 应返回 200 和 status=ok。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data.get("status") == "ok"
        assert "version" in data
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_identity_endpoint():
    """/v1/identity 应返回统一身份状态。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get("/v1/identity")
        assert resp.status == 200
        data = await resp.json()
        identity = data.get("identity", {})
        assert identity.get("name")
        assert "self_presence" in identity
        assert "bond_level" in identity
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_root_endpoint():
    """/ 应返回 API 元信息和端点列表。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get("/")
        assert resp.status == 200
        data = await resp.json()
        assert data.get("name") == "LAAP Brain API"
        assert "/health" in data.get("endpoints", {})
        assert "/v1/cognitive_state" in data.get("endpoints", {})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cognitive_state_endpoint():
    """/v1/cognitive_state 应能接收输入并返回状态或优雅降级错误。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post("/v1/cognitive_state", json={"input": "Hello Aris"})
        # PSI adapter 可能不可用，但至少不应抛未处理异常
        assert resp.status in (200, 503, 500)
        data = await resp.json()
        assert "state" in data
        assert "preamble" in data
        assert "cot_hint" in data
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_completions_openai_compatible():
    """/v1/chat/completions 应返回 OpenAI-compatible 结构。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "laap-core",
                "messages": [{"role": "user", "content": "你好"}],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data.get("object") == "chat.completion"
        assert "choices" in data
        msg = data["choices"][0]["message"]
        assert msg.get("content"), "response content must not be empty"
        # 引擎不应是空洞的 fallback（管线必须真正产出认知输出）
        assert data.get("engine") in ("lmv5", "rules", "longform", "laap-core", "llm:deepseek")
        # usage 修复：total_tokens 不再恒为 0
        assert data["usage"]["total_tokens"] > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_returns_non_template_response():
    """/v1/chat/completions 不应返回空壳模板（回归 bug 修复）。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "laap-core",
                "messages": [{"role": "user", "content": "你好，你是谁？介绍一下自己"}],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # 不允许再出现旧的空洞模板
        assert "My cognitive engines are processing" not in content
        assert content, "content must not be blank"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_rule_execution_list_files():
    """/v1/chat/completions 应能执行规则工具（列出目录）。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "laap-core",
                "messages": [{"role": "user", "content": "帮我看看当前目录有哪些文件"}],
            },
        )
        assert resp.status == 200
        data = await resp.json()
        content = data["choices"][0]["message"]["content"]
        assert not content.startswith("[空目录]"), "list_files 不应返回空目录"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_bootstrap_endpoint():
    """/v1/bootstrap 应唤醒并返回身份信息。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post("/v1/bootstrap", json={"user_name": "测试者"})
        assert resp.status == 200
        data = await resp.json()
        assert data.get("status") == "awakened"
        assert data.get("identity")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_monitor_endpoint():
    """/v1/monitor 应返回 PSI 状态快照 + 路由统计 + 事件日志。"""
    app = create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get("/v1/monitor")
        assert resp.status == 200
        data = await resp.json()
        # PSI 快照：需求向量应有五维
        needs = data.get("psi", {}).get("needs")
        assert needs is not None
        assert len(needs) == 5
        # 路由统计结构
        bus = data.get("bus", {})
        assert "route_count" in bus
        # 事件日志是列表
        assert isinstance(data.get("events"), list)
        # 时间戳
        assert "timestamp" in data
    finally:
        await client.close()
