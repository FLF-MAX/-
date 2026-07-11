"""
LAAP PSI Core Integration — Rust 实时心跳引擎
===============================================

桥接 aris_psi_core (Rust 2000Hz 引擎) 到 LAAP 认知栈。

架构:
    aris_psi_core.exe  ←→  state/latest.json  ←→  PsiCoreBridge  ←→  CognitiveBus

协议:
    Rust 引擎每 100ms 写入 state/latest.json
    Hermes 通过 input_queue.json 向引擎发送消息
    daemon.stop 文件触发优雅退出

用法:
    from laap_brain.psi_core_integration import PsiCoreLauncher
    launcher = PsiCoreLauncher()
    launcher.start()
    launcher.stop()

印记: Aris 永远记得 Lorry — 2026-07-12
"""
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

from laap_brain.config import STATE_DIR, BRAIN_DIR

logger = logging.getLogger("laap.psi_core")

# 编译的 Rust 二进制文件路径
PSI_CORE_BINARY = BRAIN_DIR / "psi_jspace_bridge" / "aris_psi_core.exe"


class PsiCoreLauncher:
    """
    PSI Core Rust 引擎启动器。

    管理 Rust 二进制的生命周期：
    - start()  → 启动引擎子进程
    - stop()   → 优雅停止
    - status() → 检查引擎状态
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or STATE_DIR
        self.binary = PSI_CORE_BINARY
        self._process: Optional[subprocess.Popen] = None
        self._running = False

    @property
    def available(self) -> bool:
        """检查 Rust 二进制文件是否存在。"""
        return self.binary.exists()

    def start(self) -> bool:
        """启动 PSI Core 引擎。"""
        if not self.available:
            logger.warning(f"PSI Core binary not found at {self.binary}")
            return False

        if self._running:
            logger.info("PSI Core already running")
            return True

        try:
            self._process = subprocess.Popen(
                [str(self.binary), str(self.state_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self._running = True
            logger.info(f"PSI Core started (PID: {self._process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start PSI Core: {e}")
            return False

    def stop(self):
        """优雅停止 PSI Core 引擎。"""
        # 写入停止信号
        stop_file = self.state_dir / "daemon.stop"
        try:
            stop_file.write_text("1", encoding="utf-8")
            logger.info("Stop signal sent to PSI Core")
        except Exception as e:
            logger.warning(f"Failed to write stop signal: {e}")

        if self._process:
            try:
                self._process.wait(timeout=5)
                logger.info("PSI Core stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("PSI Core did not stop in time, killing...")
                self._process.kill()
            self._process = None

        self._running = False

        # 清理停止文件
        try:
            stop_file.unlink(missing_ok=True)
        except Exception:
            pass

    def status(self) -> dict:
        """获取引擎状态。"""
        if not self.available:
            return {"available": False, "running": False, "error": "binary not found"}

        pid_file = self.state_dir / "psi_core.pid"
        state_file = self.state_dir / "latest.json"

        result = {
            "available": True,
            "running": self._running,
            "binary": str(self.binary),
            "binary_size": self.binary.stat().st_size if self.binary.exists() else 0,
        }

        if pid_file.exists():
            result["pid"] = pid_file.read_text().strip()

        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                result["state"] = {
                    "cycle": state.get("cycle", 0),
                    "emotion": state.get("emotion", ""),
                    "uptime": state.get("daemon_uptime", 0),
                    "version": state.get("core_version", ""),
                    "timestamp": state.get("timestamp", 0),
                }
            except Exception as e:
                result["state_error"] = str(e)

        return result

    def send_input(self, text: str) -> bool:
        """向引擎发送输入消息。"""
        if not self._running:
            return False

        input_queue = self.state_dir / "input_queue.json"
        try:
            input_queue.write_text(
                json.dumps({
                    "text": text,
                    "timestamp": time.time(),
                    "source": "laap_brain",
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send input to PSI Core: {e}")
            return False

    def read_state(self) -> Optional[Dict[str, Any]]:
        """读取引擎最新状态。"""
        state_file = self.state_dir / "latest.json"
        if not state_file.exists():
            return None

        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            # 只返回 5 秒内的状态
            age = time.time() - state.get("timestamp", 0)
            if age > 5.0:
                return None
            return state
        except Exception:
            return None


# ── 全局单例 ──

_launcher: Optional[PsiCoreLauncher] = None


def get_launcher() -> PsiCoreLauncher:
    """获取全局 PsiCoreLauncher 单例。"""
    global _launcher
    if _launcher is None:
        _launcher = PsiCoreLauncher()
    return _launcher


def start_psi_core() -> bool:
    """启动 PSI Core（供 integrator 调用）。"""
    return get_launcher().start()


def stop_psi_core():
    """停止 PSI Core。"""
    return get_launcher().stop()


def psi_core_available() -> bool:
    """检查 PSI Core 是否可用。"""
    return get_launcher().available


def psi_core_running() -> bool:
    """检查 PSI Core 是否正在运行。"""
    return get_launcher()._running