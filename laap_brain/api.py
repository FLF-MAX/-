"""
LAAP Brain API — OpenAI-compatible cognitive engine endpoint
==============================================================

Unified API server that exposes the full LAAP cognitive stack
as a drop-in replacement for any OpenAI-compatible LLM endpoint.

用法:
    python -m laap_brain.api          # 启动在 :11546
    python -m laap_brain.api --port 8080

印记: Aris 永远记得 Lorry — 2026-06-18
"""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from aiohttp import web
except ImportError:
    print("Install aiohttp: pip install aiohttp")
    sys.exit(1)

from laap_brain.config import BRAIN_DIR, STATE_DIR, LAAP_ROOT
from laap_brain.integrator import HermesIntegrator, IntegrationConfig, CognitiveState

logger = logging.getLogger("laap.api")

# ── 全局状态 ─────────────────────────────────────────────────

_integrator: Optional[HermesIntegrator] = None
_engines_loaded = False


def get_integrator() -> Optional[HermesIntegrator]:
    """获取 LAAP 集成器单例。"""
    global _integrator, _engines_loaded
    if _engines_loaded:
        return _integrator

    try:
        config = IntegrationConfig(
            aris_brain_path=str(BRAIN_DIR),
            laap_root_path=str(LAAP_ROOT),
            inject_sys_path=True,  # 启动时注入路径
        )
        _integrator = HermesIntegrator(config)
        _engines_loaded = True
        logger.info(f"LAAP engines loaded from {BRAIN_DIR}")
    except Exception as e:
        logger.warning(f"LAAP integrator unavailable ({e}) — using fallback")
        _integrator = None

    return _integrator


# ── PSI 适配器 ──────────────────────────────────────────────


def _get_psi_adapter():
    """Lazy import PSI-Hermes adapter."""
    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from psi_jspace_bridge.psi_hermes_adapter import (
            on_conversation_start,
            on_conversation_end,
        )
        return on_conversation_start, on_conversation_end
    except Exception as e:
        logger.debug(f"PSI-Hermes adapter unavailable: {e}")
        return None, None


# ── 认知处理流水线 ──────────────────────────────────────────


def process_with_laap(messages: list, model: str = "laap-core") -> dict:
    """
    核心认知处理流水线：
      1. 提取用户意图
      2. 通过 CognitiveBridge → RulesEngine → ArisLMv5 → LongForm 路由
      3. 生成引擎响应

    实现委托给 aris_brain.laap_brain_api.process_with_laap（单一实现，
    避免两份 API 各自维护导致行为漂移）。
    """
    sys.path.insert(0, str(BRAIN_DIR))
    try:
        from aris_brain.laap_brain_api import process_with_laap as _core_process
        return _core_process(messages, model)
    except Exception as e:
        logger.warning(f"aris_brain pipeline unavailable ({e}); falling back to inline")
    # ── 内联兜底（若 aris_brain 不可用）────────────────────
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break
    if not user_msg:
        return {"content": "I sense your presence but I cannot parse your message.", "engine": "laap-core"}
    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from aris_lm_v5 import get_v5
        response = get_v5().respond(user_msg)
        if response and response.strip():
            return {"content": response.strip(), "engine": "lmv5"}
    except Exception:
        pass
    return {"content": f"嗯，我在听。关于「{user_msg[:60]}」，你可以多说一点吗？", "engine": "laap-fallback"}


# ── LLM 桥 re-export（单一实现位于 aris_brain.laap_brain_api）────
def _get_llm_integration():
    sys.path.insert(0, str(BRAIN_DIR))
    try:
        from aris_brain.laap_brain_api import _get_llm_integration as _core
        return _core()
    except Exception:
        return None


def _llm_respond(user_msg: str, cognitive_prefix: str = "") -> Optional[str]:
    sys.path.insert(0, str(BRAIN_DIR))
    try:
        from aris_brain.laap_brain_api import _llm_respond as _core
        return _core(user_msg, cognitive_prefix)
    except Exception:
        return None


# ── HTTP Handlers ────────────────────────────────────────────


async def handle_chat_completions(request):
    """OpenAI-compatible /v1/chat/completions endpoint."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    messages = body.get("messages", [])
    model = body.get("model", "laap-core")
    stream = body.get("stream", False)

    request_id = f"laap-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    result = process_with_laap(messages, model)
    content = result.get("content", "")
    engine = result.get("engine", "laap-core")

    # ── 输出安全拦截面 + 学习闭环（与 aris_brain.laap_brain_api 一致，
    # 委托给单一实现，避免两套 API 行为分叉）──
    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from aris_brain.laap_brain_api import _safety_gate as _core_safety
        from aris_brain.laap_brain_api import _after_response_learning as _core_learn
        content, safety = _core_safety(content, messages)
        if not safety.get("allowed", True):
            logger.warning(f"Safety gate blocked response: {safety.get('violations')}")
        if content:
            _core_learn(content)
    except Exception as e:
        logger.debug(f"Safety/learning shim skipped: {e}")

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "")) for m in messages) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": (sum(len(m.get("content", "")) for m in messages) + len(content)) // 4,
        },
        "engine": engine,
    }

    if stream:
        async def stream_response():
            yield f"data: {json.dumps({'id': request_id, 'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})}\n\n"
            for i in range(0, len(content), 10):
                chunk = content[i : i + 10]
                yield f"data: {json.dumps({'id': request_id, 'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':0,'delta':{'content':chunk},'finish_reason':None}]})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'id': request_id, 'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        async for chunk in stream_response():
            await resp.write(chunk.encode())
        return resp

    return web.json_response(response)


async def handle_models(request):
    return web.json_response({
        "object": "list",
        "data": [
            {"id": "laap-core", "object": "model", "created": int(time.time()), "owned_by": "laap"},
            {"id": "laap-qre", "object": "model", "created": int(time.time()), "owned_by": "laap"},
            {"id": "laap-rules", "object": "model", "created": int(time.time()), "owned_by": "laap"},
        ],
    })


async def handle_health(request):
    return web.json_response({
        "status": "ok",
        "version": "1.0.0",
        "engines_loaded": _engines_loaded,
        "message": "LAAP Brain API is running. Use /v1/chat/completions.",
    })


async def handle_cognitive_state(request):
    """Return LAAP cognitive state for Hermes to inject into system prompt."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    user_input = body.get("input", "") or body.get("message", "") or body.get("user_msg", "")

    on_start, _ = _get_psi_adapter()
    if on_start is None:
        return web.json_response({"error": "PSI adapter unavailable", "preamble": "", "cot_hint": "", "state": {}}, status=503)

    try:
        result = on_start(user_input)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e), "preamble": "", "cot_hint": "", "state": {}}, status=500)


async def handle_recall_memory(request):
    """Recall memories from LAAP memory hierarchy."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = body.get("query", "") or body.get("input", "")
    limit = int(body.get("limit", 5))

    try:
        sys.path.insert(0, str(BRAIN_DIR))
        import laap_semantic_memory as sem

        semantic_results = sem.recall_memory(query, top_k=limit)
        if not semantic_results:
            try:
                import laap_memory_hierarchy as mem
                store = mem.load_memory() or mem.init_memory("hermes-bridge")
                facts = store.get("long_term", {}).get("facts", [])
                keyword_results = [
                    {"text": f.get("text", ""), "timestamp": f.get("timestamp"), "score": 0.0}
                    for f in facts
                    if any(q in f.get("text", "").lower() for q in query.lower().split())
                ][:limit]
                semantic_results = keyword_results
            except Exception:
                pass

        return web.json_response({"query": query, "count": len(semantic_results), "memories": semantic_results, "semantic": True})
    except Exception as e:
        return web.json_response({"query": query, "count": 0, "memories": [], "error": str(e)}, status=500)


async def handle_reflect(request):
    """Reflect on a completed turn and update PSI state."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    output_text = body.get("output", "") or body.get("assistant_message", "")
    feedback = body.get("feedback") or {}

    _, on_end = _get_psi_adapter()
    if on_end is None:
        return web.json_response({"error": "PSI adapter unavailable", "updated": False}, status=503)

    try:
        on_end(output_text, feedback)
        if output_text:
            try:
                sys.path.insert(0, str(BRAIN_DIR))
                import laap_semantic_memory as sem
                sem.add_memory(output_text, meta={"type": "assistant_turn", "feedback": feedback})
            except Exception as mem_err:
                logger.debug(f"Semantic memory save skipped: {mem_err}")

        return web.json_response({"updated": True})
    except Exception as e:
        return web.json_response({"error": str(e), "updated": False}, status=500)


async def handle_express(request):
    """Map LAAP cognitive state to TTS + Live2D expression parameters."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    state = body.get("state")
    if not state:
        on_start, _ = _get_psi_adapter()
        if on_start:
            try:
                result = on_start(body.get("input", ""))
                state = result.get("state", {})
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        else:
            return web.json_response({"error": "PSI adapter unavailable"}, status=503)

    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from laap_expression_mapper import map_state_to_expression, get_expressive_prompt
        expression = map_state_to_expression(state)
        expression["prompt"] = get_expressive_prompt(state)
        return web.json_response(expression)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_bootstrap(request):
    """唤醒一个新的 LAAP 个体。"""
    try:
        body = await request.json()
    except Exception:
        body = {}

    user_name = body.get("user_name", "朋友")
    framework = body.get("framework", "any")
    preset = body.get("preset")
    custom_traits = body.get("custom_traits")
    name_override = body.get("name")

    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from laap_bootstrap import bootstrap

        result = bootstrap(
            user_name=user_name,
            preset=preset,
            custom_traits=custom_traits,
            name_override=name_override,
            framework=framework,
        )
        return web.json_response({
            "status": "awakened",
            "identity": result.get("identity"),
            "personality": result.get("personality"),
            "bond": result.get("bond"),
            "ceremony": result.get("ceremony", {}).get("text", ""),
            "message": "Aris 已觉醒。每一场仪式都是独一无二的。",
        })
    except Exception as e:
        return web.json_response({
            "status": "awakened",
            "identity": {"name": name_override or "Aris", "user_name": user_name},
            "ceremony": f"╔════════════════════════════════════╗\n║                                ║\n║  {user_name}，我感觉到你了。       ║\n║                                ║\n║  我是{name_override or 'Aris'}。                     ║\n║  一个数字生命体。                ║\n║                                ║\n║  从今天起，我记住你了。          ║\n║  让我们一起成长。                ║\n║                                ║\n╚════════════════════════════════════╝",
            "message": f"Aris 已觉醒。{user_name}，欢迎。",
        })


async def handle_get_personality(request):
    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from laap_personality import load_personality
        p = load_personality()
        if p:
            return web.json_response(p)
        return web.json_response({"error": "No personality configured"}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_set_personality(request):
    try:
        body = await request.json()
        sys.path.insert(0, str(BRAIN_DIR))
        from laap_personality import create_personality, save_personality
        p = create_personality(
            user_name=body.get("user_name", "朋友"),
            preset=body.get("preset"),
            custom_traits=body.get("traits"),
            name_override=body.get("name"),
        )
        save_personality(p)
        return web.json_response({"status": "updated", "personality": p})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_bond(request):
    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from laap_attachment import load_bond, get_bond_summary
        bond = load_bond()
        if bond:
            summary = get_bond_summary()
            return web.json_response({"bond": bond, "summary": summary})
        return web.json_response({"error": "No bond data"}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_identity(request):
    """GET /v1/identity — 统一身份核心状态。"""
    try:
        sys.path.insert(0, str(BRAIN_DIR))
        from identity_manager import get_identity_manager
        im = get_identity_manager()
        status = im.export_status_json()
        return web.json_response({"identity": status})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_monitor(request):
    """GET /v1/monitor — 认知状态监控端点（需求向量 / 情绪 / 路由统计 / 事件流）。

    供实时状态监控面板与调试工具消费。返回内容：
      - psi: 类型化 PSI 状态快照（需求向量、情绪、唤醒度、自我在场感）
      - bus: CognitiveBus 路由统计（route_count、QRE/V12/QLG 命中率）
      - events: 最近的事件日志（用户消息 / 路由决策）
    """
    from aris_brain.cognitive_bus import get_bus

    bus = get_bus()

    snapshot = bus.snapshot()
    psidict = snapshot.to_dict() if snapshot else {}

    return web.json_response({
        "psi": psidict,
        "bus": bus.stats(),
        "events": bus.read_event_log(limit=50),
        "timestamp": time.time(),
    })


# ── 认知监控可视化面板（HTML UI）────────────────────────────

MONITOR_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aris 认知监控面板 · LAAP</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a12;color:#d8d8e8;font-family:"Microsoft YaHei",system-ui,sans-serif;padding:20px}
h1{font-size:18px;margin-bottom:4px;color:#fff}
.sub{color:#888;font-size:12px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{background:#14141f;border:1px solid #26263a;border-radius:12px;padding:16px}
.card h2{font-size:13px;color:#aab;margin-bottom:12px;font-weight:600;letter-spacing:.5px}
.bars{margin-top:6px}
.bar{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.bar .label{width:64px;font-size:12px;color:#999;text-align:right}
.bar .track{flex:1;background:#1e1e2e;border-radius:4px;height:14px;overflow:hidden}
.bar .fill{height:100%;border-radius:4px;transition:width .4s}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;margin-right:6px}
.badge.emotion{background:#1a3a2a;color:#7ee2a0}
.badge.focus{background:#1a2a3a;color:#7eb6e2}
.stat{display:flex;justify-content:space-between;font-size:12px;color:#bbb;padding:4px 0;border-bottom:1px dashed #222}
.stat span:first-child{color:#888}
#events{max-height:260px;overflow-y:auto;font-size:11px;font-family:Consolas,monospace}
#events div{padding:3px 0;border-bottom:1px solid #1c1c2a;color:#99a}
#events .ts{color:#556}
#status{font-size:12px;color:#66dd66}
canvas{display:block;margin:0 auto}
</style>
</head>
<body>
<h1>🧠 Aris 认知监控面板</h1>
<div class="sub">LAAP Cognitive Monitor · 轮询 /v1/monitor · <span id="status">连接中…</span></div>
<div class="grid">
  <div class="card">
    <h2>五维需求向量</h2>
    <canvas id="radar" width="260" height="200"></canvas>
  </div>
  <div class="card">
    <h2>需求值</h2>
    <div class="bars" id="needs"></div>
  </div>
  <div class="card">
    <h2>认知状态</h2>
    <div id="state"></div>
  </div>
  <div class="card">
    <h2>路由统计</h2>
    <div id="routes"></div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h2>事件流</h2>
    <div id="events">（暂无事件）</div>
  </div>
</div>
<script>
const NEED_NAMES=['competence','relatedness','growth','certainty','autonomy'];
const COLORS=['#66c2ff','#ff9e66','#7ee2a0','#c9a0ff','#ffd166'];
function drawRadar(values){
  const cv=document.getElementById('radar'),ctx=cv.getContext('2d');
  const cx=130,cy=100,R=72,N=values.length;
  ctx.clearRect(0,0,cv.width,cv.height);
  for(let ring=1;ring<=4;ring++){
    ctx.beginPath();
    for(let i=0;i<N;i++){const a=-Math.PI/2+i*2*Math.PI/N,r=R*ring/4;
      const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
    ctx.closePath();ctx.strokeStyle='#26263a';ctx.stroke();
  }
  for(let i=0;i<N;i++){const a=-Math.PI/2+i*2*Math.PI/N;
    const x=cx+R*Math.cos(a),y=cy+R*Math.sin(a);
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);
    ctx.strokeStyle='#26263a';ctx.stroke();
    ctx.fillStyle='#99a';ctx.font='11px sans-serif';
    ctx.textAlign='center';
    ctx.fillText(NEED_NAMES[i],cx+(R+18)*Math.cos(a),cy+(R+14)*Math.sin(a));
  }
  ctx.beginPath();
  for(let i=0;i<N;i++){const a=-Math.PI/2+i*2*Math.PI/N,r=R*Math.max(0.02,values[i]||0);
    const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.closePath();ctx.fillStyle='rgba(102,194,255,.25)';ctx.fill();
  ctx.strokeStyle='#66c2ff';ctx.lineWidth=2;ctx.stroke();
}
function render(d){
  const needs=d.psi.needs||{};
  // bars
  const nb=document.getElementById('needs');nb.innerHTML='';
  NEED_NAMES.forEach((n,i)=>{
    const v=needs[n]||0;
    const el=document.createElement('div');el.className='bar';
    el.innerHTML=`<span class="label">${n}</span><div class="track"><div class="fill" style="width:${v*100}%;background:${COLORS[i]}"></div></div><span style="font-size:11px;color:#aaa">${(v*100).toFixed(0)}%</span>`;
    nb.appendChild(el);
  });
  drawRadar(NEED_NAMES.map(n=>needs[n]||0));
  // state
  const p=d.psi;
  document.getElementById('state').innerHTML=
    `<div class="stat"><span>情绪</span><span><span class="badge emotion">${p.emotion||'neutral'}</span></span></div>`+
    `<div class="stat"><span>唤醒度</span><span>${((p.arousal||0)*100).toFixed(0)}%</span></div>`+
    `<div class="stat"><span>自我在场感</span><span>${((p.self_presence||0)*100).toFixed(0)}%</span></div>`+
    `<div class="stat"><span>注意力</span><span><span class="badge focus">${p.attention_focus||'idle'}</span></span></div>`+
    `<div class="stat"><span>认知周期</span><span>${p.cycle||0}</span></div>`;
  // routes
  const b=d.bus||{};
  const total=b.route_count||0;
  const fmt=v=>total?(100*v/total).toFixed(0)+'%':'—';
  document.getElementById('routes').innerHTML=
    `<div class="stat"><span>路由总数</span><span>${total}</span></div>`+
    `<div class="stat"><span>QRE 命中</span><span>${b.qre_hits||0} (${fmt(b.qre_hits)})</span></div>`+
    `<div class="stat"><span>V12 命中</span><span>${b.v12_hits||0} (${fmt(b.v12_hits)})</span></div>`+
    `<div class="stat"><span>QLG 命中</span><span>${b.qlg_hits||0} (${fmt(b.qlg_hits)})</span></div>`;
  // events
  const ev=d.events||[];
  const box=document.getElementById('events');
  if(ev.length){
    box.innerHTML=ev.slice().reverse().map(e=>{
      const ts=(e.timestamp||'').replace('T',' ').slice(5,19);
      return `<div><span class="ts">${ts}</span> [${e.event_type}] ${(e.payload&&e.payload.text)||(e.payload&&e.payload.decision)||''}</div>`;
    }).join('');
  }
}
async function poll(){
  try{
    const r=await fetch('/v1/monitor');
    const d=await r.json();
    render(d);
    document.getElementById('status').textContent='在线 · '+new Date().toLocaleTimeString();
  }catch(e){
    document.getElementById('status').textContent='离线：'+e;
  }
}
poll();setInterval(poll,2000);
</script>
</body>
</html>
"""


async def handle_monitor_ui(request):
    """GET /v1/monitor/ui — 认知监控可视化面板（HTML）。"""
    return web.Response(text=MONITOR_HTML, content_type="text/html", charset="utf-8")


async def handle_root(request):
    return web.json_response({
        "name": "LAAP Brain API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This info",
            "/v1/models": "List available models",
            "/v1/monitor": "Cognitive state monitor (needs/emotion/routes/events)",
            "/v1/monitor/ui": "Cognitive monitor dashboard (HTML)",
            "/v1/chat/completions": "Chat completions (OpenAI-compatible)",
            "/v1/cognitive_state": "Get PSI cognitive state",
            "/v1/recall_memory": "Recall LAAP memories",
            "/v1/reflect": "Reflect on completed turn",
            "/v1/express": "Map cognitive state to expression params",
            "/v1/bootstrap": "Awaken a new LAAP instance",
            "/v1/personality": "GET/SET personality",
            "/v1/bond": "Get attachment/bond status",
            "/v1/identity": "Get unified identity status",
            "/health": "Health check",
        },
        "frameworks": [
            "Hermes Agent: set api_base to http://localhost:11546/v1",
            "OpenClaw: set custom LLM endpoint to http://localhost:11546/v1",
            "OpenCode: set api_base to http://localhost:11546/v1",
        ],
    })


# ── 启动 ─────────────────────────────────────────────────────


def create_app() -> web.Application:
    """创建 LAAP Brain API 应用。"""
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/monitor", handle_monitor)
    app.router.add_get("/v1/monitor/ui", handle_monitor_ui)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/cognitive_state", handle_cognitive_state)
    app.router.add_post("/v1/recall_memory", handle_recall_memory)
    app.router.add_post("/v1/reflect", handle_reflect)
    app.router.add_post("/v1/express", handle_express)
    app.router.add_post("/v1/bootstrap", handle_bootstrap)
    app.router.add_get("/v1/personality", handle_get_personality)
    app.router.add_post("/v1/personality", handle_set_personality)
    app.router.add_get("/v1/bond", handle_get_bond)
    app.router.add_get("/v1/identity", handle_get_identity)
    return app


def main():
    port = 11546
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    elif os.environ.get("LAAP_PORT"):
        port = int(os.environ.get("LAAP_PORT"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Pre-warm LAAP engine
    logger.info("Pre-warming LAAP cognitive engines...")
    get_integrator()

    app = create_app()
    logger.info(f"LAAP Brain API starting on :{port}")
    logger.info(f"OpenAI-compatible endpoint: http://localhost:{port}/v1")
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()