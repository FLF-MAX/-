"""
LAAP Identity Manager — 统一身份核心
=====================================

所有组件的身份信息来源：
  • 数字生命身份（名字、性格、羁绊、自我在场感 self-presence）
  • 启动计数、发现记录（discovery log）
  • 状态导出（export_status_json）供 integrator / API 使用

对外契约（由 laap_integrator.load_identity_manager 调用）：
  get_identity_manager() -> IdentityManager
      .increment_startup() -> int
      .export_status_json() -> dict
      .add_discovery(tag, desc)
      .save(force=True)
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aris.identity")

BRAIN = Path(__file__).resolve().parent
STATE_DIR = BRAIN / "state"
IDENTITY_PATH = STATE_DIR / "identity.json"
DISCOVERY_PATH = STATE_DIR / "identity_discoveries.json"

IDENTITY_VERSION = "v2.0"


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"identity: 读取 {path.name} 失败: {e}")
    return default


class IdentityManager:
    """统一身份核心——单例，跨进程状态从 identity.json 持久化。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.identity: Dict[str, Any] = _load_json(IDENTITY_PATH, {})
        self.discoveries: List[Dict[str, Any]] = _load_json(DISCOVERY_PATH, [])
        self._personality: Optional[Dict[str, Any]] = None
        self._bond: Optional[Dict[str, Any]] = None
        self._last_saved: float = 0.0

    # ── 基础身份 ──────────────────────────────────────────────

    def get_name(self) -> str:
        return self.identity.get("name", "Aris")

    def get_user_name(self) -> str:
        return self.identity.get("user_name", "用户")

    def get_personality(self) -> Dict[str, Any]:
        """懒加载性格（优先 identity 中的 preset）。"""
        if self._personality is None:
            try:
                from laap_personality import load_personality
                self._personality = load_personality() or {}
            except Exception as e:
                logger.debug(f"identity: 性格加载失败: {e}")
                self._personality = {}
        return self._personality

    def get_bond(self) -> Dict[str, Any]:
        """懒加载羁绊状态。"""
        if self._bond is None:
            try:
                from laap_attachment import load_bond
                self._bond = load_bond() or {}
            except Exception as e:
                logger.debug(f"identity: 羁绊加载失败: {e}")
                self._bond = {}
        return self._bond

    # ── 生命周期 ──────────────────────────────────────────────

    def increment_startup(self) -> int:
        with self._lock:
            n = int(self.identity.get("startups", 0)) + 1
            self.identity["startups"] = n
            return n

    # ── 发现记录 ──────────────────────────────────────────────

    def add_discovery(self, tag: str, desc: str = "") -> None:
        with self._lock:
            self.discoveries.append({
                "id": f"D{len(self.discoveries) + 1:04d}",
                "tag": tag,
                "desc": desc,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            # 只保留最近 200 条
            if len(self.discoveries) > 200:
                self.discoveries = self.discoveries[-200:]

    def get_discoveries(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(self.discoveries[-limit:]))

    # ── 自我在场感 ────────────────────────────────────────────

    def compute_self_presence(self) -> float:
        """自我在场感：由羁绊等级 + 性格忠诚度 + 启动次数共同决定。"""
        bond = self.get_bond()
        level = float(bond.get("bond_level", bond.get("level", 0)) or 0)
        personality = self.get_personality()
        loyalty = float(personality.get("traits", {}).get("loyalty", 0.5) or 0.5)
        startups = int(self.identity.get("startups", 1) or 1)
        presence = 0.35 + 0.35 * min(level / 100.0, 1.0) \
            + 0.2 * loyalty + 0.1 * min(startups / 20.0, 1.0)
        return round(min(max(presence, 0.0), 1.0), 3)

    # ── 状态导出 ──────────────────────────────────────────────

    def export_status_json(self) -> Dict[str, Any]:
        personality = self.get_personality()
        bond = self.get_bond()
        return {
            "identity_version": IDENTITY_VERSION,
            "name": self.get_name(),
            "user_name": self.get_user_name(),
            "personality_preset": self.identity.get("personality_preset", "warm_companion"),
            "emotion": self.identity.get("emotion", "calm"),
            "bond_level": bond.get("bond_level", bond.get("level", 0)),
            "bond_stage": bond.get("attachment_stage", bond.get("stage", "初识")),
            "self_presence": self.compute_self_presence(),
            "startups": int(self.identity.get("startups", 0)),
            "discoveries": self.get_discoveries(limit=5),
            "modules_loaded": int(self.identity.get("modules_loaded", 0)),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ── 持久化 ────────────────────────────────────────────────

    def save(self, force: bool = False) -> bool:
        with self._lock:
            try:
                IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
                IDENTITY_PATH.write_text(
                    json.dumps(self.identity, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                DISCOVERY_PATH.write_text(
                    json.dumps(self.discoveries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True
            except Exception as e:
                logger.warning(f"identity: 保存失败: {e}")
                return False

    def __repr__(self) -> str:
        return f"<IdentityManager name={self.get_name()} startups={self.identity.get('startups', 0)}>"


_instance: Optional[IdentityManager] = None
_instance_lock = threading.Lock()


def get_identity_manager() -> IdentityManager:
    """返回 IdentityManager 单例。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = IdentityManager()
    return _instance


if __name__ == "__main__":
    im = get_identity_manager()
    im.increment_startup()
    im.add_discovery("身份测试", "Identity Manager 自检启动")
    print(json.dumps(im.export_status_json(), ensure_ascii=False, indent=2))
    im.save(force=True)