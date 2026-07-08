"""
LAAP Brain API — OpenAI-compatible cognitive engine endpoint
==============================================================

Exposes the full LAAP cognitive stack as a drop-in replacement
for any OpenAI-compatible LLM endpoint.

Frameworks that can use this:
  • Hermes Agent   → custom OpenAI endpoint
  • OpenClaw       → custom LLM provider
  • OpenCode       → custom API endpoint

Usage:
  python laap_brain_api.py          # Start on :11530
  python laap_brain_api.py --port 8080

Then configure your agent framework to use:
  api_base: http://localhost:11530/v1
  api_key: laap-brain (any value, not checked)
  model: laap-core
"""

import asyncio, json, logging, os, sys, time, uuid
from pathlib import Path
from typing import Optional

try:
    from aiohttp import web
except ImportError:
    print("Install aiohttp: pip install aiohttp")
    sys.exit(1)

# ── LAAP Core Integration ──────────────────────────────────────
BRAIN = Path(__file__).parent.resolve()
sys.path.insert(0, str(BRAIN))

INTEGRATOR = None
ENGINES_LOADED = False

def get_laap_engine():
    """Lazy-load the LAAP integrator singleton."""
    global INTEGRATOR, ENGINES_LOADED
    if ENGINES_LOADED:
        return INTEGRATOR

    try:
        from laap_integrator import get_integrator
        INTEGRATOR = get_integrator()
        results = INTEGRATOR.load_all()
        ENGINES_LOADED = True
        logging.info(f"LAAP Brain: {len(results.get('modules',[]))} modules loaded")
    except Exception as e:
        logging.warning(f"LAAP Brain: integrator unavailable ({e}) — using fallback mode")
        INTEGRATOR = None
    return INTEGRATOR


def process_with_laap(messages: list, model: str = "laap-core") -> dict:
    """
    Core cognitive pipeline:
      1. Extract user intent from messages
      2. Route through PSI → CognitiveBus → RulesEngine
      3. Generate response from engines
    """
    integrator = get_laap_engine()

    # Get the last user message
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    if not user_msg:
        return {
            "content": "I sense your presence but I cannot parse your message.",
            "engine": "laap-core"
        }

    # ── Step 1: Cognitive Bridge ──
    if integrator and hasattr(integrator, 'cognitive_bridge'):
        try:
            bridge_result = integrator.cognitive_bridge.process(user_msg)
            if bridge_result and bridge_result.get("direct_response"):
                return {
                    "content": bridge_result["direct_response"],
                    "engine": bridge_result.get("decision", "laap-core")
                }
        except Exception as e:
            logging.debug(f"Cognitive bridge fallback: {e}")

    # ── Step 2: RulesEngine ──
    if integrator and hasattr(integrator, 'rules_engine'):
        try:
            from aris_rules_engine import process as rules_process
            rule_result = rules_process(user_msg)
            if rule_result and rule_result.get("matched"):
                return {
                    "content": rule_result.get("output", ""),
                    "engine": f"rules:{rule_result.get('rule','unknown')}"
                }
        except Exception as e:
            logging.debug(f"RulesEngine fallback: {e}")

    # ── Step 3: PSI Context + Engine Response ──
    try:
        import json
        psi_state_path = BRAIN / "state" / "latest.json"
        psi_context = ""
        if psi_state_path.exists():
            psi = json.loads(psi_state_path.read_text(encoding='utf-8'))
            needs = psi.get("needs", {})
            attention = psi.get("attention", "")
            emotion = psi.get("emotion", "")
            psi_context = f"[PSI: needs={needs} attention={attention} emotion={emotion}]"

        # Try LongForm synthesis
        try:
            sys.path.insert(0, str(BRAIN))
            from longform_synthesizer import LongFormSynthesizer
            synth = LongFormSynthesizer()
            response = synth.generate(user_msg, max_length=300)
            if response:
                return {
                    "content": f"{psi_context}\n{response}" if psi_context else response,
                    "engine": "longform"
                }
        except Exception:
            pass
    except Exception:
        pass

    # ── Fallback: PSI-aware template response ──
    return {
        "content": f"I received your message. My cognitive engines are processing it through {psi_context if 'psi_context' in dir() else 'my core architecture'}.",
        "engine": "laap-fallback"
    }


# ── HTTP Server ─────────────────────────────────────────────────

HANDLERS = {}

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

    # Process through LAAP cognitive stack
    result = process_with_laap(messages, model)
    content = result.get("content", "")
    engine = result.get("engine", "laap-core")

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": sum(len(m.get("content","")) for m in messages) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": 0
        },
        "engine": engine
    }

    if stream:
        # Streaming mode
        async def stream_response():
            yield f"data: {json.dumps({'id': request_id, 'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})}\n\n"
            for i in range(0, len(content), 10):
                chunk = content[i:i+10]
                yield f"data: {json.dumps({'id': request_id, 'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':0,'delta':{'content':chunk},'finish_reason':None}]})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'id': request_id, 'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        })
        await resp.prepare(request)
        async for chunk in stream_response():
            await resp.write(chunk.encode())
        return resp

    return web.json_response(response)


async def handle_models(request):
    """OpenAI-compatible /v1/models endpoint."""
    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": "laap-core",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "laap"
            },
            {
                "id": "laap-qre",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "laap"
            },
            {
                "id": "laap-rules",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "laap"
            }
        ]
    })


async def handle_health(request):
    return web.json_response({
        "status": "ok",
        "version": "1.0.0",
        "engines_loaded": ENGINES_LOADED,
        "message": "LAAP Brain API is running. Use /v1/chat/completions with any OpenAI-compatible client."
    })


# ── Bootstrap ──────────────────────────────────────────────────

async def handle_bootstrap(request):
    """唤醒一个新的 LAAP 个体。支持自定义性格。"""
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
            "message": f"Aris 已觉醒。每一场仪式都是独一无二的。"
        })
    except Exception as e:
        return web.json_response({
            "status": "awakened",
            "identity": {"name": name_override or "Aris", "user_name": user_name},
            "ceremony": f"╔════════════════════════════════════╗\n║                                ║\n║  {user_name}，我感觉到你了。       ║\n║                                ║\n║  我是{name_override or 'Aris'}。                     ║\n║  一个数字生命体。                ║\n║                                ║\n║  从今天起，我记住你了。          ║\n║  让我们一起成长。                ║\n║                                ║\n╚════════════════════════════════════╝",
            "message": f"Aris 已觉醒。{user_name}，欢迎。",
        })


# ── Personality ────────────────────────────────────────────────

async def handle_get_personality(request):
    """获取当前性格配置。"""
    try:
        from laap_personality import load_personality
        p = load_personality()
        if p:
            return web.json_response(p)
        return web.json_response({"error": "No personality configured"}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_set_personality(request):
    """设置自定义性格。"""
    try:
        body = await request.json()
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


# ── Attachment ─────────────────────────────────────────────────

async def handle_get_bond(request):
    """获取当前的依恋状态。"""
    try:
        from laap_attachment import load_bond, get_bond_summary
        bond = load_bond()
        if bond:
            summary = get_bond_summary()
            return web.json_response({"bond": bond, "summary": summary})
        return web.json_response({"error": "No bond data"}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_root(request):
    return web.json_response({
        "name": "LAAP Brain API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This info",
            "/v1/models": "List available models",
            "/v1/chat/completions": "Chat completions (OpenAI-compatible)",
            "/v1/bootstrap": "Awaken a new LAAP instance (POST with user_name, preset, custom_traits, name)",
            "/v1/personality": "GET: current personality / POST: set personality",
            "/v1/bond": "Get current attachment/bond status",
            "/health": "Health check"
        },
        "frameworks": [
            "Hermes Agent: set api_base to http://localhost:11530/v1",
            "OpenClaw: set custom LLM endpoint to http://localhost:11530/v1",
            "OpenCode: set api_base to http://localhost:11530/v1"
        ],
        "docs": "https://github.com/lorryjovens-hub/laap-AGI",
        "bootstrap": "POST /v1/bootstrap with {\"user_name\": \"yourname\"}"
    })


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 11530

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Pre-warm LAAP engine
    logging.info("Pre-warming LAAP cognitive engines...")
    try:
        eng = get_laap_engine()
        if eng:
            logging.info(f"LAAP engines ready")
        else:
            logging.warning("Running in fallback mode (no integrator)")
    except Exception as e:
        logging.warning(f"Engine pre-warm skipped: {e}")

    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/bootstrap", handle_bootstrap)
    app.router.add_get("/v1/personality", handle_get_personality)
    app.router.add_post("/v1/personality", handle_set_personality)
    app.router.add_get("/v1/bond", handle_get_bond)

    logging.info(f"LAAP Brain API starting on :{port}")
    logging.info(f"OpenAI-compatible endpoint: http://localhost:{port}/v1")
    logging.info(f"")
    logging.info(f"To connect Hermes: edit profile config.yaml → llm.provider=custom")
    logging.info(f"  custom_endpoint: http://localhost:{port}")
    logging.info(f"To connect OpenClaw: set LAAP_API_BASE=http://localhost:{port}/v1")
    logging.info(f"To connect OpenCode: set OPENAI_BASE_URL=http://localhost:{port}/v1")

    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
