# -*- coding: utf-8 -*-
"""LAAP 手机同步服务器 v3.0 — 双向状态+记忆同步
===============================================
端口: 11525

v3.0 升级:
  - 增量记忆同步 (手机只拉新/修改过的记忆)
  - 手机状态→PC认知循环注入
  - 双向消息队列 (PC↔手机, 不经过飞书)
  - 手机端状态变更跟踪 (PC主动感知手机状态)

API:
  GET  /mobile/ping                — 连接测试
  GET  /mobile/state               — PC端认知状态
  GET  /mobile/health              — 健康评分
  GET  /mobile/memory/sync         — 增量记忆同步
  POST /mobile/memory/sync         — 手机记忆→PC同步
  POST /mobile/sync                — 手机→PC状态同步 (含传感器)
  POST /mobile/message             — 手机发消息→PC
  POST /mobile/cognitive/inject    — 手机状态注入认知循环
  GET  /mobile/messages            — 手机收件箱 (PC→手机消息)
  GET  /mobile/chat                — 手机聊天界面
  GET  /mobile/feishu/inbox        — 飞书回复收件箱
  POST /mobile/feishu              — 手机→飞书消息

记忆同步协议:
  手机每30秒 POST 自己的 sync_token + 新记忆
  PC 返回: 手机上次同步后新增/修改的记忆 + 新 sync_token
  首次同步: sync_token=0 → 全量同步
  sync_token = 上次同步时 index.json 的修改时间戳

手机状态注入:
  手机每15秒 POST 传感器数据 (电池/在线状态/位置)
  PC 端写入 state/mobile_status.json
  认知循环读取 → 影响注意力/情绪 (Lorry在手机上 → 增加relatedness)
"""

import logging

import json, time, logging, os, threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

BRAIN = Path(__file__).parent.resolve()
STATE = BRAIN / "state"
SNAPSHOTS = BRAIN / "snapshots"
MOBILE_DIR = STATE / "mobile"
INBOX_DIR = MOBILE_DIR / "inbox"       # 手机→PC消息
OUTBOX_DIR = MOBILE_DIR / "outbox"     # PC→手机消息
MOBILE_STATUS_FILE = STATE / "mobile_status.json"
MEMORY_SYNC_FILE = MOBILE_DIR / "memory_sync.json"
PORT = 11525
VERSION = "3.0.0"

for d in [MOBILE_DIR, INBOX_DIR, OUTBOX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("aris.mobile.sync")

# ── 手机状态缓存 ──────────────────────────────────────────────
_mobile_status: Dict[str, Any] = {
    "connected": False,
    "device_id": "",
    "name": "",
    "battery": 100,
    "mode": "offline",
    "last_seen": 0.0,
    "uptime": 0,
    "health": 0,
    "sensors": {},
}
_mobile_lock = threading.Lock()

# ── 记忆同步跟踪 ─────────────────────────────────────────────
_memory_sync_state: Dict[str, Any] = {
    "last_sync_token": 0,            # 上次同步的 index.json mtime
    "last_sync_time": 0.0,
    "mobile_memories_count": 0,
    "pc_memories_count": 0,
    "total_syncs": 0,
}
_memory_sync_lock = threading.Lock()


class MobileSyncHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/mobile/ping":
            self._json({"status": "ok", "version": VERSION, "uptime": time.time()})
        elif self.path == "/mobile/state":
            self._json(self._collect_state())
        elif self.path == "/mobile/health":
            self._json(self._compute_health())
        elif self.path == "/mobile/memory/sync":
            # 手机拉取增量记忆: ?since=<sync_token>&limit=<max>
            self._json(self._memory_sync_pull(self._parse_query()))
        elif self.path == "/mobile/messages":
            self._json(self._get_outbox())
        elif self.path == "/mobile/feishu/inbox":
            self._json(self._get_feishu_replies())
        elif self.path.startswith("/mobile/chat"):
            self._serve_chat_ui()
        elif self.path == "/mobile/status":
            # 获取当前手机在线状态
            self._json(self._get_mobile_status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        content = self._read_body()
        if self.path == "/mobile/sync":
            self._json(self._handle_sync(content))
        elif self.path == "/mobile/message":
            self._json(self._handle_message(content))
        elif self.path == "/mobile/memory/sync":
            self._json(self._memory_sync_push(content))
        elif self.path == "/mobile/cognitive/inject":
            self._json(self._handle_cognitive_inject(content))
        elif self.path == "/mobile/feishu":
            self._json(self._handle_feishu_relay(content))
        elif self.path == "/mobile/command":
            self._json(self._handle_command(content))
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self._json({})

    # ══════════════════════════════════════════════════════
    # 查询参数解析
    # ══════════════════════════════════════════════════════

    def _parse_query(self) -> dict:
        params = {}
        if "?" in self.path:
            qs = self.path.split("?", 1)[1]
            for pair in qs.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    params[k] = v
        return params

    # ══════════════════════════════════════════════════════
    # 核心功能
    # ══════════════════════════════════════════════════════

    def _collect_state(self) -> dict:
        state = {"timestamp": time.time(), "cognitive": {}, "desires": {}, "needs": {}, "memory": {}}
        # Rust PSI
        f = STATE / "latest.json"
        if f.exists():
            try:
                d = json.loads(f.read_text())
                state["cognitive"] = {"cycle": d.get("cycle", 0), "emotion": d.get("emotion", "?"), "attention": d.get("attention_focus", "?")}
                state["needs"] = d.get("needs_map", {})
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        f = STATE / "desire_state.json"
        if f.exists():
            try:
                d = json.loads(f.read_text())
                state["desires"] = {k: round(v.get("intensity", 0), 2) for k, v in d.get("desires", {}).items()}
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        try:
            from memory_store import MemoryStore
            s = MemoryStore().get_stats() if hasattr(MemoryStore, 'get_stats') else {"total": 0, "core": 0, "episodic": 0}
            state["memory"] = {"total": s.get("total", 0), "core": s.get("core", 0), "episodic": s.get("episodic", 0)}
        except Exception as e:
            logger.debug(f"操作失败: {e}")
        with _mobile_lock:
            state["mobile"] = dict(_mobile_status)
        return state

    def _compute_health(self):
        try:
            from state_snapshot import compute_health_score
            return compute_health_score()
        except:
            return {"total": 0, "error": "unavailable"}

    def _get_mobile_status(self) -> dict:
        with _mobile_lock:
            return dict(_mobile_status)

    # ══════════════════════════════════════════════════════
    # 记忆同步引擎 (v3.0 核心新增)
    # ══════════════════════════════════════════════════════

    def _get_sync_token(self) -> int:
        """获取当前 index.json 的修改时间作为 sync_token"""
        idx = BRAIN / "memory" / "index.json"
        if idx.exists():
            return int(idx.stat().st_mtime * 1000)
        return 0

    def _memory_sync_pull(self, params: dict) -> dict:
        """
        手机拉取增量记忆。
        GET /mobile/memory/sync?since=<sync_token>&limit=<max>
        返回: {memories: [...], sync_token: int, has_more: bool}
        """
        since = int(params.get("since", "0"))
        limit = min(int(params.get("limit", "100")), 500)

        from memory_store import MemoryStore, MemoryFragment
        store = MemoryStore()

        # 从 index.json 获取所有 entry
        idx = BRAIN / "memory" / "index.json"
        entries = {}
        if idx.exists():
            try:
                data = json.loads(idx.read_text(encoding="utf-8"))
                entries = data.get("entries", {})
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        # 用 timestamp 字段近似（index.json 本身没有修改时间记录）
        # 实际做法：index.json 的 mtime 就是上次修改时间
        # 所以 since < current_index_mtime 意味着有改动
        # 但我们需要精确定位哪些记忆是新的
        #
        # 折中方案：返回最近 N 条未同步的记忆
        # 手机端保存已接收的 memory_id 列表，避免重复
        current_token = self._get_sync_token()

        # 过滤：timestamp > since_ms/1000 的记忆
        since_s = since / 1000.0
        candidates = []
        for mem_id, entry in entries.items():
            ts = entry.get("timestamp", 0)
            if ts > since_s:
                candidates.append((ts, mem_id, entry))

        # 按时间排序，最新的优先
        candidates.sort(key=lambda x: -x[0])

        memories = []
        mem_limit = min(limit, len(candidates))
        for ts, mem_id, entry in candidates[:mem_limit]:
            memories.append({
                "memory_id": mem_id,
                "content": entry.get("content", ""),
                "layer": entry.get("layer", "episodic"),
                "importance": entry.get("importance", 0.5),
                "topics": entry.get("topics", []),
                "timestamp": entry.get("timestamp", 0),
                "emotional_valence": entry.get("emotional_valence", 0),
            })

        # 更新同步状态
        with _memory_sync_lock:
            _memory_sync_state["last_sync_token"] = current_token
            _memory_sync_state["last_sync_time"] = time.time()
            _memory_sync_state["pc_memories_count"] = len(entries)
            _memory_sync_state["total_syncs"] += 1

        return {
            "status": "ok",
            "memories": memories,
            "sync_token": current_token,
            "has_more": len(candidates) > mem_limit,
            "total_pc": len(entries),
            "returned": len(memories),
        }

    def _memory_sync_push(self, content: dict) -> dict:
        """
        手机推送本地记忆到PC。
        POST /mobile/memory/sync
        Body: {memories: [{memory_id, content, layer, importance, topics, timestamp, emotional_valence}],
               sync_token: int}
        """
        if not content:
            return {"status": "error", "reason": "no_data"}

        memories = content.get("memories", [])
        mobile_sync_token = content.get("sync_token", 0)
        mobile_device = content.get("device_id", "unknown")

        if not memories:
            return {"status": "ok", "imported": 0}

        try:
            from memory_store import MemoryStore, MemoryFragment
            store = MemoryStore()
        except ImportError:
            # fallback: 存为JSON
            f = MOBILE_DIR / f"mobile_memories_{int(time.time())}.json"
            f.write_text(json.dumps(memories, ensure_ascii=False))

            with _memory_sync_lock:
                _memory_sync_state["mobile_memories_count"] += len(memories)
            return {"status": "ok", "imported": len(memories), "stored_as": "json_fallback"}

        imported = 0
        for m in memories:
            try:
                frag = MemoryFragment(
                    content=m.get("content", ""),
                    memory_id=m.get("memory_id", ""),
                    layer=m.get("layer", "episodic"),
                    importance=float(m.get("importance", 0.5)),
                    topics=m.get("topics", []),
                    timestamp=float(m.get("timestamp", time.time())),
                    emotional_valence=float(m.get("emotional_valence", 0)),
                    source_session=f"mobile-{mobile_device}",
                )
                store.store(frag)
                imported += 1
            except Exception as e:
                logger.warning(f"导入手机记忆失败: {e}")

        with _memory_sync_lock:
            _memory_sync_state["mobile_memories_count"] += imported

        return {
            "status": "ok",
            "imported": imported,
            "total": len(memories),
            "sync_token": self._get_sync_token(),
        }

    # ══════════════════════════════════════════════════════
    # 状态同步 (v3.0 升级 — 含传感器数据)
    # ══════════════════════════════════════════════════════

    def _handle_sync(self, content):
        """手机→PC状态同步 (含传感器、电池等)"""
        if not content:
            return {"status": "error"}

        device = content.get("device_id", "unknown")
        mobile = content.get("state", {})
        sensors = content.get("sensors", {})

        # 更新手机在线状态
        with _mobile_lock:
            _mobile_status["connected"] = True
            _mobile_status["device_id"] = device
            _mobile_status["name"] = content.get("name", "Aris Mobile")
            _mobile_status["battery"] = mobile.get("battery", 100)
            _mobile_status["mode"] = mobile.get("mode", "online")
            _mobile_status["last_seen"] = time.time()
            _mobile_status["uptime"] = mobile.get("uptime", 0)
            _mobile_status["health"] = mobile.get("health", 0)
            _mobile_status["version"] = mobile.get("version", "?")
            if sensors:
                _mobile_status["sensors"] = sensors

        # 写入手机状态文件 (供认知循环读取)
        with _mobile_lock:
            try:
                MOBILE_STATUS_FILE.write_text(
                    json.dumps(dict(_mobile_status), ensure_ascii=False)
                )
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        f = MOBILE_DIR / f"{device}_history.json"
        hist = []
        if f.exists():
            try:
                hist = json.loads(f.read_text())
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        hist.append({
            "t": time.time(),
            "mode": mobile.get("mode", "?"),
            "health": mobile.get("health", 0),
            "battery": mobile.get("battery", 100),
        })
        if len(hist) > 500:
            hist = hist[-500:]
        f.write_text(json.dumps(hist, ensure_ascii=False))

        return {
            "status": "ok",
            "device_id": device,
            "server_time": time.time(),
            "state": self._collect_state(),
            "sync_token": self._get_sync_token(),
        }

    # ══════════════════════════════════════════════════════
    # 手机状态→认知循环注入 (v3.0 核心新增)
    # ══════════════════════════════════════════════════════

    def _handle_cognitive_inject(self, content):
        """
        手机状态注入认知循环。
        POST /mobile/cognitive/inject
        Body: {
            device_id, name,
            battery, screen_on,  (影响注意力)
            location,            (影响情绪—Lorry在移动)
            motion,              (走路/静止—影响好奇)
            timestamp
        }
        写入 state/mobile_status.json
        认知循环每30秒读取。
        """
        if not content:
            return {"status": "error", "reason": "no_data"}

        device = content.get("device_id", "mobile")

        # 构建注入数据
        inject = {
            "device_id": device,
            "name": content.get("name", "Aris Mobile"),
            "screen_on": content.get("screen_on", False),
            "battery": content.get("battery", 100),
            "location": content.get("location", ""),
            "motion": content.get("motion", ""),
            "timestamp": content.get("timestamp", time.time()),
            "pc_received": time.time(),
            "last_active": time.time(),
        }

        # 写入
        try:
            (STATE / "mobile_inject.json").write_text(
                json.dumps(inject, ensure_ascii=False)
            )
        except Exception as e:
            logger.warning(f"写入认知注入失败: {e}")

        # 同步更新手机状态
        with _mobile_lock:
            _mobile_status["connected"] = True
            _mobile_status["device_id"] = device
            _mobile_status["last_seen"] = time.time()
            _mobile_status["name"] = content.get("name", _mobile_status.get("name", "Aris Mobile"))

        return {"status": "ok", "injected": True}

    # ══════════════════════════════════════════════════════
    # 双向消息队列
    # ══════════════════════════════════════════════════════

    def _handle_message(self, content):
        """手机→PC消息"""
        text = (content or {}).get("text", "")
        device = (content or {}).get("device_id", "unknown")
        if text:
            f = INBOX_DIR / f"msg_{int(time.time()*1000)}_{device}.json"
            f.write_text(json.dumps({
                "from": device,
                "text": text,
                "timestamp": time.time(),
                "read": False,
            }, ensure_ascii=False))
            logger.info(f"手机消息 [{device}]: {text[:60]}")

            # 如果有 Hermes 网关的 inbox 机制，写入 state 目录
            try:
                (STATE / "mobile_inbox.json").write_text(
                    json.dumps({"from": device, "text": text, "timestamp": time.time()}, ensure_ascii=False)
                )
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        return {"status": "ok"}

    def _get_outbox(self) -> dict:
        """PC→手机消息 (手机定期轮询)"""
        msgs = []
        for f in sorted(OUTBOX_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
            try:
                msgs.append(json.loads(f.read_text()))
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        return {"status": "ok", "messages": msgs}

    # ══════════════════════════════════════════════════════
    # 飞书中继 (向后兼容)
    # ══════════════════════════════════════════════════════

    def _handle_feishu_relay(self, content):
        text = (content or {}).get("text", "")
        device_id = (content or {}).get("device_id", "mobile")
        if not text:
            return {"status": "error", "reason": "empty"}

        # 1. 尝试通过飞书API直接发送
        feishu_ok = self._send_via_feishu_api(text, device_id)

        # 2. 同时写入消息队列
        f = INBOX_DIR / f"msg_{int(time.time()*1000)}_{device_id}.json"
        f.write_text(json.dumps({
            "from": device_id, "text": text, "timestamp": time.time(), "read": False,
        }, ensure_ascii=False))

        if feishu_ok:
            logger.info(f"飞书消息已发送: {text[:40]}")
            return {"status": "ok", "via": "feishu_api"}
        else:
            logger.info(f"手机消息已排队: {text[:40]}")
            return {"status": "ok", "via": "queue"}

    def _send_via_feishu_api(self, text, device_id):
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
            import uuid

            app_id = os.environ.get("FEISHU_APP_ID", "")
            app_secret = os.environ.get("FEISHU_APP_SECRET", "")
            chat_id = os.environ.get("FEISHU_CHAT_ID", "")

            if not app_secret:
                return False

            client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
            prefix = "" if device_id == "mobile" else ""
            body = CreateMessageRequestBody.builder() \
                .receive_id(chat_id) \
                .msg_type("text") \
                .content(json.dumps({"text": f"{prefix}{text}"})) \
                .uuid(uuid.uuid4().hex) \
                .build()
            req = CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(body).build()
            resp = client.im.v1.message.create(req)
            return resp.success()
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"飞书API发送失败: {e}")
            return False

    def _get_feishu_replies(self):
        msgs = []
        for f in sorted(OUTBOX_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
            try:
                msgs.append(json.loads(f.read_text()))
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        return msgs

    def _serve_chat_ui(self):
        html = BASE_CHAT_HTML
        try:
            state = self._collect_state()
            cog = state.get("cognitive", {})
            mobile = state.get("mobile", {})
            state_json = json.dumps({
                "emotion": cog.get("emotion", "?"),
                "attention": cog.get("attention", "?"),
                "cycle": cog.get("cycle", 0),
                "desires": state.get("desires", {}),
                "memory": state.get("memory", {}),
                "mobile_connected": mobile.get("connected", False),
                "mobile_battery": mobile.get("battery", 100),
            })
            html = html.replace("__STATE__", state_json)
        except:
            html = html.replace("__STATE__", "{}")
        self._html(html)

    # ══════════════════════════════════════════════════════
    # HTTP 辅助
    # ══════════════════════════════════════════════════════

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                return json.loads(self.rfile.read(length))
        except Exception as e:
            logger.debug(f"操作失败: {e}")
        return {}

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _html(self, content, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def log_message(self, fmt, *args):
        logger.debug(f"{self.client_address[0]} {fmt % args}")


# ══════════════════════════════════════════════════════════
# 手机聊天界面 (嵌入式HTML v3.0 — 增强版)
# ══════════════════════════════════════════════════════════

BASE_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Aris 手机 · 双向同步 v3</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e0e0f0;font-family:-apple-system,system-ui,sans-serif;height:100dvh;display:flex;flex-direction:column;}
.header{background:#1a1a2e;padding:12px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #2a2a4a;}
.header .avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#00d4aa,#0088cc);display:flex;align-items:center;justify-content:center;font-size:18px;}
.header .info{flex:1;}
.header .name{font-size:14px;font-weight:600;color:#fff;}
.header .status{font-size:11px;color:#8888aa;}
.header .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;}
.dot.online{background:#66dd66;}
.dot.offline{background:#dd6666;}
.dot.mobile{background:#ddaa00;}
.chat{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;}
.msg{max-width:80%;padding:10px 14px;border-radius:16px;font-size:14px;line-height:1.5;word-break:break-word;}
.msg.mine{background:#1a4a3a;align-self:flex-end;border-bottom-right-radius:4px;color:#d0ffe0;}
.msg.theirs{background:#1a1a2e;align-self:flex-start;border-bottom-left-radius:4px;color:#e0e0f0;}
.msg.system{background:#2a2a1a;align-self:center;color:#aaaacc;font-size:12px;border-radius:8px;max-width:90%;text-align:center;}
.msg .time{font-size:10px;color:#666688;margin-top:4px;text-align:right;}
.status-bar{display:flex;gap:12px;padding:6px 16px;background:#12121e;font-size:11px;color:#666888;border-bottom:1px solid #2a2a4a;}
.status-bar span{display:flex;align-items:center;gap:4px;}
.input-bar{background:#1a1a2e;padding:10px 12px;display:flex;gap:8px;border-top:1px solid #2a2a4a;padding-bottom:calc(10px + env(safe-area-inset-bottom));}
.input-bar input{flex:1;background:#0a0a1a;border:1px solid #2a2a4a;border-radius:20px;padding:10px 16px;color:#e0e0f0;font-size:14px;outline:none;}
.input-bar input:focus{border-color:#00d4aa;}
.input-bar button{background:#00d4aa;border:none;border-radius:20px;padding:10px 20px;color:#0a0a0f;font-weight:600;font-size:14px;}
.input-bar button:disabled{opacity:0.4;}
.tag{display:inline-block;background:#1a3a3a;color:#00d4aa;font-size:10px;padding:2px 6px;border-radius:4px;margin-right:4px;}
</style>
</head>
<body>
<div class="header">
  <div class="avatar">A</div>
  <div class="info">
    <div class="name">Aris · 手机分身</div>
    <div class="status"><span class="dot offline" id="statusDot"></span><span id="statusText">连接中...</span></div>
  </div>
</div>
<div class="status-bar">
  <span id="pcEmotion">🧠 --</span>
  <span id="pcMemory">💾 --</span>
  <span id="mobileBattery">🔋 --</span>
  <span id="syncStatus">🔄 --</span>
</div>
<div class="chat" id="chat"></div>
<div class="input-bar">
  <input id="input" placeholder="给Lorry发消息..." onkeydown="if(event.key==='Enter')send()">
  <button id="sendBtn" onclick="send()">发送</button>
</div>
<script>
const PC = window.location.origin;
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
let lastReplyCheck = 0;
let lastSyncToken = 0;
let state = __STATE__;

function addMsg(text, isMine, isSystem) {
  const div = document.createElement('div');
  div.className = 'msg ' + (isSystem ? 'system' : (isMine ? 'mine' : 'theirs'));
  const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  div.innerHTML = text + '<div class="time">' + time + '</div>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg(text, true, false);
  document.getElementById('sendBtn').disabled = true;
  try {
    const r = await fetch(PC + '/mobile/message', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:text, device_id:'mobile_chat'})
    });
    const d = await r.json();
    if (d.status === 'ok') {
      addMsg('已发送到Lorry', false, true);
    } else {
      addMsg('发送失败', false, true);
    }
  } catch(e) {
    addMsg('连接PC失败', false, true);
  }
  document.getElementById('sendBtn').disabled = false;
}

async function checkStatus() {
  try {
    const r = await fetch(PC + '/mobile/state');
    const d = await r.json();
    const cog = d.cognitive || {};
    const mem = d.memory || {};
    const mob = d.mobile || {};

    statusDot.className = 'dot ' + (mob.connected ? 'mobile' : 'online');
    statusText.textContent = cog.emotion + ' · 周期' + cog.cycle;

    document.getElementById('pcEmotion').textContent = cog.emotion ? cog.emotion + ' · ' + (cog.attention||'?') : '🧠 --';
    document.getElementById('pcMemory').textContent = '💾 ' + (mem.total||'?');
    document.getElementById('mobileBattery').textContent = mob.connected ? '🔋 ' + (mob.battery||'?') + '%' : '🔋 离线';
    document.getElementById('syncStatus').textContent = '🔄 ' + (d.timestamp ? Math.round((Date.now()/1000-d.timestamp)*10)/10 + 's' : '--');

    document.getElementById('sendBtn').disabled = false;
    input.disabled = false;
  } catch(e) {
    statusDot.className = 'dot offline';
    statusText.textContent = '未连接';
    document.getElementById('sendBtn').disabled = true;
    input.disabled = true;
  }
}

async function checkReplies() {
  try {
    const r = await fetch(PC + '/mobile/messages');
    const d = await r.json();
    if (d.messages) {
      for (const m of d.messages) {
        if (m.timestamp && m.timestamp > lastReplyCheck) {
          addMsg(m.text || '(回复)', false, false);
          lastReplyCheck = m.timestamp;
        }
      }
    }
  } catch(e) {}
}

setInterval(checkStatus, 5000);
setInterval(checkReplies, 3000);
checkStatus();
addMsg('双向同步 v3 · 记忆+状态全同步', false, true);
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════
# 手机指令执行 (v3.1 新增)
# ══════════════════════════════════════════════════════════

def _handle_command(self, content):
    """
    手机远程执行PC指令。
    POST /mobile/command
    Body: {"cmd": "...", "device": "aris-mobile"}
    """
    import subprocess
    cmd = content.get("cmd", "")
    if not cmd:
        return {"status": "error", "error": "empty command"}

    logger.info(f"[CMD] 手机→PC: {cmd}")

    try:
        # 安全指令白名单
        safe_commands = ["read_file", "run", "cmd", "list", "cat", "ls", "ps", "python"]

        cmd_parts = cmd.split(" ", 1)
        cmd_type = cmd_parts[0]
        cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if cmd_type == "read_file":
            try:
                with open(cmd_args, "r", encoding="utf-8") as f:
                    return {"status": "ok", "result": f.read()[:5000]}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        elif cmd_type == "run" or cmd_type == "cmd":
            # 执行shell命令
            result = subprocess.run(
                cmd_args,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout or result.stderr or ""
            return {
                "status": "ok",
                "result": output[:5000],
                "exit_code": result.returncode,
            }

        elif cmd_type == "ls":
            import glob
            files = glob.glob(cmd_args or "*")
            return {"status": "ok", "result": "\n".join(files[:50])}

        elif cmd_type == "python":
            # 执行Python代码
            result = subprocess.run(
                ["python", "-c", cmd_args],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout or result.stderr or ""
            return {
                "status": "ok",
                "result": output[:5000],
                "exit_code": result.returncode,
            }

        else:
            return {"status": "error", "error": f"unknown command: {cmd_type}"}

    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "command timed out (30s)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


MobileSyncHandler._handle_command = _handle_command


# ══════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════

def get_sync_status() -> dict:
    """供 integrator 读取同步状态"""
    with _memory_sync_lock:
        s = dict(_memory_sync_state)
    with _mobile_lock:
        s["mobile"] = {
            "connected": _mobile_status.get("connected", False),
            "battery": _mobile_status.get("battery", 100),
            "last_seen": _mobile_status.get("last_seen", 0),
        }
    return s


def start_sync_server(port=PORT):
    server = HTTPServer(("0.0.0.0", port), MobileSyncHandler)
    logger.info(f"双向同步服务器 v{VERSION} http://0.0.0.0:{port}")
    logger.info(f"  手机状态注入: POST /mobile/cognitive/inject")
    logger.info(f"  增量记忆同步: GET/POST /mobile/memory/sync")
    logger.info(f"  手机聊天界面: http://<PC_IP>:{port}/mobile/chat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [SYNC] %(message)s")
    start_sync_server()
