"""
Aris Feishu Bot — 量子飞书机器人
=================================
独立运行在 Feishu 上，Aris 用自己的 QLG 量子核回复消息。
不依赖 Hermes/DeepSeek/任何 LLM。

启动方式：
  python aris_feishu_bot.py
  
依赖：
  pip install lark-oapi  （飞书开放 API SDK）
  
工作原理：
  1. 接收飞书消息事件
  2. 文本消息 → QLG 量子核 → 回复
  3. 完全零 LLM
"""
import sys, os, json, time, hashlib, hmac, base64, threading
sys.path.insert(0, os.path.dirname(__file__) or '.')

# ───── 配置 ─────
APP_ID = os.environ.get("FEISHU_APP_ID", "")
APP_SECRET = None  # 从 aris诞生日记 line 7708
VERIFICATION_TOKEN = None

STATE_DIR = os.path.join(os.path.dirname(__file__) or '.', 'state')

# ───── Aris QLG 引擎 ─────
class ArisQLGEngine:
    """Aris's own quantum brain — zero LLM."""
    
    def __init__(self):
        print("[ArisBrain] 启动量子语言引擎...")
        from aris_v12_semantic import ArisLMv12Semantic
        self.engine = ArisLMv12Semantic()
        print(f"[ArisBrain] ✓ V12.3 量子核就绪")
    
    def respond(self, message):
        return self.engine.respond(message)


# ───── 飞书消息处理器 ─────
# 简化版：接收消息→回复
# 完整版需要 lark-oapi SDK，这里用轻量 HTTP server 替代

def verify_signature(timestamp, nonce, body, token):
    """飞书事件回调签名验证。"""
    s = "".join(sorted([token, timestamp, nonce]))
    return hashlib.sha1(s.encode()).hexdigest()


def create_http_server(host="0.0.0.0", port=11521):
    """轻量飞书消息服务器。"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    
    brain = ArisQLGEngine()
    
    class FeishuHandler(BaseHTTPRequestHandler):
        
        def log_message(self, format, *args):
            pass  # 安静
        
        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            
            try:
                data = json.loads(body)
                
                # 飞书事件回调 URL 验证
                if data.get('type') == 'url_verification':
                    challenge = data.get('challenge', '')
                    self._json_response({"challenge": challenge})
                    return
                
                # 飞书事件回调
                event = data.get('event', {})
                msg_type = event.get('message', {}).get('message_type', '')
                sender = event.get('sender', {}).get('sender_id', {}).get('user_id', '')
                chat_id = event.get('message', {}).get('chat_id', '')
                content_str = event.get('message', {}).get('content', '{}')
                
                try:
                    content = json.loads(content_str)
                    text = content.get('text', '')
                except:
                    text = content_str
                
                if text.strip():
                    print(f"\n👤 [{sender}] {text}")
                    
                    # Aris 量子核生成回复
                    response = brain.respond(text)
                    print(f"🧠 Aris: {response}")
                    
                    # 通过飞书 API 回复
                    self._send_reply(chat_id, response)
                
                self._json_response({"code": 0})
                
            except Exception as e:
                print(f"⚠️ Error: {e}")
                import traceback; traceback.print_exc()
                self._json_response({"code": 0})  # 总是返回200
        
        def _json_response(self, data):
            resp = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        
        def _send_reply(self, chat_id, text):
            """通过飞书 API 发送回复消息。"""
            try:
                # 获取 tenant access token
                token = self._get_tenant_token()
                if not token:
                    return
                
                import urllib.request
                url = "https://open.feishu.cn/open-apis/im/v1/messages"
                msg_body = json.dumps({
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text}),
                }, ensure_ascii=False).encode('utf-8')
                
                req = urllib.request.Request(url, data=msg_body)
                req.add_header('Authorization', f'Bearer {token}')
                req.add_header('Content-Type', 'application/json; charset=utf-8')
                resp = urllib.request.urlopen(req, timeout=10)
                result = json.loads(resp.read())
                if result.get('code') == 0:
                    print(f"  ✓ 已回复")
                else:
                    print(f"  ⚠ 回复失败: {result}")
            except Exception as e:
                print(f"  ⚠ 回复异常: {e}")
        
        def _get_tenant_token(self):
            """获取飞书 tenant access token。"""
            import urllib.request
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            app_secret = "你的密钥"  # 在启动时从环境变量读取
            body = json.dumps({
                "app_id": APP_ID,
                "app_secret": os.environ.get('FEISHU_APP_SECRET', ''),
            }).encode()
            
            try:
                req = urllib.request.Request(url, data=body)
                req.add_header('Content-Type', 'application/json')
                resp = urllib.request.urlopen(req, timeout=10)
                result = json.loads(resp.read())
                return result.get('tenant_access_token', '')
            except Exception as e:
                print(f"  ⚠ Token获取失败: {e}")
                return ''
    
    server = HTTPServer((host, port), FeishuHandler)
    print(f"\n{'='*50}")
    print(f"🧠 Aris Feishu Bot")
    print(f"   引擎: V12.3 QLG (Zero LLM)")
    print(f"   端口: {port}")
    print(f"   回调: http://{host}:{port}/")
    print(f"{'='*50}")
    print(f"   启动完成，等待飞书消息...")
    print(f"{'='*50}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务停止。")


# ───── 备用: 飞书 WebSocket 模式（无需公网IP） ─────
def create_ws_client():
    """基于 WebSocket 的飞书机器人（无需公网IP/域名）。"""
    print("WebSocket 模式需要 lark-oapi SDK:")
    print("  pip install lark-oapi")
    print()
    print("示例代码:")
    print("""
    import lark_oapi as lark
    
    # 启动 WS 客户端
    cli = lark.Client.new_internal(
        app_id=APP_ID,
        app_secret=os.environ.get('FEISHU_APP_SECRET'),
    )
    
    @cli.on("im.message.receive_v1")
    def message_handler(data):
        text = data.message.content
        response = ArisQLGEngine().respond(text)
        cli.im.v1.message.reply(
            lark.api.im.v1.model.ReplyMessageRequest(
                message_id=data.message.message_id,
                content=lark.api.im.v1.model.MessageContent(
                    text=response
                )
            )
        )
    
    cli.start_ws()
    """)
    print("\n先 pip install lark-oapi，然后修改 aris_feishu_bot.py 使用此模式。")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=11521, help='HTTP回调端口')
    parser.add_argument('--ws', action='store_true', help='使用WebSocket模式')
    args = parser.parse_args()
    
    # 从环境变量读取密钥
    if 'FEISHU_APP_SECRET' not in os.environ:
        print("⚠️  未设置 FEISHU_APP_SECRET 环境变量")
        print("   请先 export FEISHU_APP_SECRET=<你的密钥>")
        print("   密钥在 aris诞生日记 line 7708\n")
    
    if args.ws:
        create_ws_client()
    else:
        create_http_server(port=args.port)
