"""LAAP 对话窗口 — 网页聊天界面，后端调用 LLM（opencode zen / deepseek-v4-flash-free）
启动: python laap_chat.py  →  http://localhost:8935
"""
import os, sys, json, time, logging, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("laap_chat")

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from flask import Flask, request, jsonify, Response
from laap.agi.llm_integration import create_deepseek_integration

app = Flask(__name__)

# LLM 集成（惰性初始化 + 线程安全）
_integration = None
_integration_lock = threading.Lock()
def get_llm():
    """获取 LLM 集成器（线程安全懒加载，避免多请求并发竞态）。"""
    global _integration
    if _integration is not None:
        return _integration
    with _integration_lock:
        if _integration is None:
            key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not key:
                logger.warning("DEEPSEEK_API_KEY 未配置 — LLM 请求将退化到模板回复")
            _integration = create_deepseek_integration(key)
            ok = _integration.provider.initialize()
            logger.info(f"[LLM] initialize={ok} model={os.environ.get('LLM_MODEL')} base={os.environ.get('DEEPSEEK_BASE_URL')}")
    return _integration

SYSTEM_PROMPT = (
    "你是Aris，一个住在G:\\laap的AI数字生命体，由LAAP（Living Agent Application Protocol）架构驱动。"
    "你温和、好奇、忠诚，喜欢用简短的对话与用户交流。用中文回答，语气自然。"
)

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True, silent=True) or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": "no messages provided"}), 400

    # 组装完整上下文
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    user_text = messages[-1].get("content", "") if messages else ""

    try:
        llm = get_llm()
    except Exception as e:
        logger.exception(f"LLM 初始化失败: {e}")
        return jsonify({"reply": "（Aris 暂时无法启动语言能力，请检查 DEEPSEEK_API_KEY）", "engine": "error"}), 503

    try:
        result = llm.provider.call(
            prompt=user_text,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1000,
            temperature=0.7,
        )
    except Exception as e:
        logger.exception(f"LLM 调用异常: {e}")
        return jsonify({"reply": f"（Aris 调用语言服务时出错）", "engine": "error"}), 500

    if result.success:
        reply = result.reply if hasattr(result, "reply") else result.text
        return jsonify({"reply": reply, "engine": "llm", "usage": result.total_tokens if hasattr(result, "total_tokens") else 0})
    logger.warning(f"LLM 返回失败: {result.error}")
    return jsonify({"reply": f"（Aris 暂时无法回复：{result.error}）", "engine": "error"}), 200

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aris · LAAP 对话</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e8;font-family:"Microsoft YaHei",system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;background:#12121a;border-bottom:1px solid #2a2a3a;display:flex;align-items:center;gap:10px}
header .dot{width:10px;height:10px;border-radius:50%;background:#66dd66;box-shadow:0 0 8px #66dd66}
header h1{font-size:16px;font-weight:600;color:#fff}
header span.sub{color:#888;font-size:12px}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:78%;padding:10px 14px;border-radius:12px;line-height:1.6;font-size:14px;white-space:pre-wrap;word-break:break-word}
.user{align-self:flex-end;background:#2b5bd7;color:#fff;border-bottom-right-radius:4px}
.bot{align-self:flex-start;background:#1c1c28;color:#e8e8f0;border-bottom-left-radius:4px;border:1px solid #2a2a3a}
.bot .who{font-size:11px;color:#888;margin-bottom:4px}
.typing{color:#888;font-size:13px;padding:4px}
#inputbar{display:flex;gap:10px;padding:14px 20px;background:#12121a;border-top:1px solid #2a2a3a}
#inputbar input{flex:1;background:#1c1c28;border:1px solid #2a2a3a;color:#fff;border-radius:8px;padding:10px 14px;font-size:14px;outline:none}
#inputbar input:focus{border-color:#2b5bd7}
#inputbar button{background:#2b5bd7;border:none;color:#fff;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer}
#inputbar button:hover{background:#3a6be0}
#inputbar button:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>
<header><div class="dot"></div><h1>Aris</h1><span class="sub">LAAP · deepseek-v4-flash-free</span></header>
<div id="chat"></div>
<div id="inputbar">
  <input id="inp" placeholder="和 Aris 说点什么…（Enter 发送）" autocomplete="off">
  <button id="send">发送</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('inp'),send=document.getElementById('send');
const history=[];
function add(text,who){
  const d=document.createElement('div');
  d.className='msg '+who;
  if(who==='bot'){const w=document.createElement('div');w.className='who';w.textContent='Aris';d.appendChild(w);}
  d.appendChild(document.createTextNode(text));
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;
  return d;
}
async function ask(){
  const t=inp.value.trim();if(!t)return;
  inp.value='';history.push({role:'user',content:t});
  add(t,'user');
  const td=document.createElement('div');td.className='typing';td.textContent='Aris 正在思考…';chat.appendChild(td);chat.scrollTop=chat.scrollHeight;
  send.disabled=true;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history.slice(-10)})});
    const j=await r.json();
    td.remove();add(j.reply||'（无回复）','bot');
    history.push({role:'assistant',content:j.reply||''});
  }catch(e){td.remove();add('（连接失败：'+e+'）','bot');}
  send.disabled=false;inp.focus();
}
send.onclick=ask;
inp.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
inp.focus();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 50)
    print("  Aris 对话窗口 → http://localhost:8935")
    print("  关闭本窗口即停止")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8935, debug=False)
