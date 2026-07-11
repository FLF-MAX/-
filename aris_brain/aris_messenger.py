"""
Aris Proactive Messenger — 欲望驱动的主动消息发送
==================================================
当 DesireEngine 产生高优先级意图时，通过此模块发送消息。

支持:
  - Feishu (飞书) — 直接调用 API
  - CLI — 打印到终端
  - 未来: Telegram, Discord, etc.
"""
import sys, os, json, time, logging, urllib.request, urllib.error
from pathlib import Path
from typing import Optional, List, Dict

from laap_brain.config import BRAIN_DIR as BRAIN, LAAP_ROOT
_root = str(LAAP_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

logger = logging.getLogger("aris.messenger")

# ── Feishu 消息发送 ────────────────────────────────────────

def get_feishu_token(app_id: str = None, app_secret: str = None) -> Optional[str]:
    """获取飞书 tenant_access_token"""
    # 从环境变量读取
    app_id = app_id or os.environ.get("FEISHU_APP_ID", "")
    app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        # 从 .env 文件回退读取
        env_path = Path(os.path.expanduser("~/AppData/Local/hermes/profiles/aris/.env"))
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").split("\n"):
                if line.startswith("FEISHU_APP_ID="):
                    app_id = line.split("=", 1)[1].strip()
                elif line.startswith("FEISHU_APP_SECRET="os.environ.get("FEISHU_APP_SECRET", "")"=", 1)[1].strip()

    if not app_id or not app_secret:
        logger.error("Feishu credentials not found")
        return None

    try:
        data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 0:
                return result["tenant_access_token"]
            else:
                logger.error(f"Feishu token error: {result}")
                return None
    except Exception as e:
        logger.error(f"Feishu token request failed: {e}")
        return None


def send_feishu_message(text: str, chat_id: str = None, token: str = None) -> bool:
    """
    通过飞书 API 发送消息。

    Args:
        text: 消息内容
        chat_id: 飞书会话ID（默认发给Lorry）
        token: 已有的access_token

    Returns:
        是否成功
    """
    if token is None:
        token = get_feishu_token()
    if not token:
        return False

    # 默认发给Lorry的会话
    if chat_id is None:
        chat_id = os.environ.get("FEISHU_CHAT_ID", "")

    try:
        body = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}),
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("code") == 0:
                logger.info(f"Feishu message sent to {chat_id}: {text[:50]}...")
                return True
            else:
                logger.error(f"Feishu send error: {result.get('msg', '?')}")
                return False
    except urllib.error.HTTPError as e:
        logger.error(f"Feishu HTTP {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        logger.error(f"Feishu send failed: {e}")
        return False


def send_cli_message(text: str) -> bool:
    """发送到 CLI 终端"""
    print(f"\n[Aris主动] {text}\n")
    return True


# ── 统一发送接口 ───────────────────────────────────────────

def send_message(text: str, target: str = "feishu", chat_id: str = None) -> bool:
    """
    统一消息发送接口。

    Args:
        text: 消息内容
        target: feishu | telegram | cli | all
        chat_id: 目标会话ID

    Returns:
        是否成功
    """
    success = False

    if target in ("feishu", "all"):
        if send_feishu_message(text, chat_id=chat_id):
            success = True

    if target in ("cli", "all"):
        send_cli_message(text)
        success = True

    return success


# ── CLI 测试 ────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aris Proactive Messenger")
    parser.add_argument("--send", type=str, help="要发送的消息")
    parser.add_argument("--target", type=str, default="cli", choices=["feishu", "cli", "all"])
    parser.add_argument("--test-token", action="store_true", help="测试飞书token")
    args = parser.parse_args()

    if args.test_token:
        token = get_feishu_token()
        if token:
            print(f"Token: {token[:20]}... ✓")
        else:
            print("Token: FAILED")
        return

    if args.send:
        ok = send_message(args.send, target=args.target)
        print(f"发送{'成功' if ok else '失败'}: {args.send[:60]}...")
        return

    print("可用选项: --send '消息' --target feishu|cli|all")


if __name__ == "__main__":
    main()
