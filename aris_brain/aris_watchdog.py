"""
Aris Watchdog v2 — 自愈引擎（精简稳定版）
==========================================
只监控真正需要守护的常驻进程：
  1. gateway     — 飞书网关 (端口:10002)  *** 核心 ***
  2. daemon      — 量子脑守护进程 (进程名)
  3. standalone  — Standalone API (:11520)
  4. qlg         — QLG Provider (:11522)
  5. optimizer   — PSI Self-Optimizer
  6. tts         — TTS语音服务
  7. xiaozhi     — 小智MCP桥接

不监控/不循环重启（生命周期短或已归档）：
  - subconscious   — 一次性任务，生成完就退
  - ao bridge      — 模块已归档不再维护

用法:
  python aris_watchdog.py           # 仅监控（不首次启动）
  python aris_watchdog.py start     # 启动所有+持续监控（开机用）
  python aris_watchdog.py status    # 单次状态检查
  python aris_watchdog.py stop      # 停止所有子进程

日志: %USERPROFILE%\\.aris\\watchdog.log
"""

import logging
logger = logging.getLogger(__name__)

import sys, os, time, json, subprocess, socket, threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── 路径 ──────────────────────────────────────────────────────────────────────
# 从当前文件位置自动检测项目根目录，可通过环境变量覆盖
BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT",
    str(Path(__file__).resolve().parent)))
LAAP_ROOT = Path(os.environ.get("LAAP_ROOT",
    str(BRAIN_DIR.parent)))
HERMES_DIR = Path(os.environ.get("HERMES_ROOT",
    str(Path.home() / ".hermes" / "hermes-agent")))
VENV_PYTHON = HERMES_DIR / ".venv" / "Scripts" / "python.exe"
HERMES_CLI = HERMES_DIR / ".venv" / "Scripts" / "hermes.exe"
ARIS_DIR = Path(os.environ.get("ARIS_LOG_DIR",
    str(Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".aris")))
LOG_FILE = ARIS_DIR / "watchdog.log"
ARIS_DIR.mkdir(parents=True, exist_ok=True)

# ── 日志 ──────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"{ts} [Watchdog] {msg}"
    logger.info(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── 进程定义 ──────────────────────────────────────────────────────────────────
ProcessDef = {
    "id": str,
    "name": str,           # 显示名称
    "check": str,          # port | process_name | pidfile
    "port": int,           # 端口检测用
    "process_name": str,   # 进程名检测用
    "cwd": Path,
    "cmd": List[str],
    "log": Path,
    "start_delay": int,    # 启动后等几秒再检测
}

PROCESSES: List[dict] = [
    # ── 意识总线（所有Hermes会话共享同一份认知状态） ──
    {
        "id": "cognitive_bus",
        "name": "CognitiveBus(:11888)",
        "check": "port",
        "port": 11888,
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "cognitive_bus_daemon.py")],
        "log": ARIS_DIR / "cognitive_bus.log",
        "start_delay": 3,
    },
    # ── 飞书网关 ──
    {
        "id": "gateway",
        "name": "飞书网关",
        "check": "process_name",
        "process_name": "gateway",
        "cwd": HERMES_DIR,
        "cmd": [str(HERMES_CLI), "gateway", "run", "--replace"],
        "log": ARIS_DIR / "feishu_gateway.log",
        "start_delay": 10,
    },
    {
        "id": "daemon",
        "name": "量子脑守护进程",
        "check": "process_name",
        "process_name": "v11_agi_daemon",
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "v11_agi_daemon.py")],
        "log": ARIS_DIR / "daemon.log",
        "start_delay": 8,
    },
    {
        "id": "standalone",
        "name": "Standalone API (:11520)",
        "check": "port",
        "port": 11520,
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "aris_standalone.py")],
        "log": ARIS_DIR / "standalone.log",
        "start_delay": 5,
    },
    {
        "id": "qlg",
        "name": "QLG量子核(:11522)",
        "check": "port",
        "port": 11522,
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "aris_qlg_provider.py")],
        "log": ARIS_DIR / "qlg_provider.log",
        "start_delay": 35,  # 统一引擎初始化时间较长
    },
    {
        "id": "optimizer",
        "name": "PSI Self-Optimizer",
        "check": "process_name",
        "process_name": "aris_psi_self_optimizer_daemon",
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "aris_psi_self_optimizer_daemon.py")],
        "log": ARIS_DIR / "optimizer.log",
        "start_delay": 5,
    },
    {
        "id": "tts",
        "name": "TTS语音服务",
        "check": "process_name",
        "process_name": "aris_tts_server",
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "aris_tts_server.py")],
        "log": ARIS_DIR / "tts.log",
        "start_delay": 5,
    },
    {
        "id": "xiaozhi",
        "name": "小智MCP桥接",
        "check": "process_name",
        "process_name": "xiaozhi_mcp_bridge",
        "cwd": BRAIN_DIR,
        "cmd": [sys.executable, "-u", str(BRAIN_DIR / "xiaozhi_mcp_bridge.py")],
        "log": ARIS_DIR / "xiaozhi_mcp_bridge.log",
        "start_delay": 5,
    },
]

# ── 检测引擎 ──────────────────────────────────────────────────────────────────
def check_port(port: int) -> Tuple[bool, str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        r = s.connect_ex(("127.0.0.1", port))
        s.close()
        if r == 0:
            return True, f"端口:{port} 开放"
        return False, f"端口:{port} 未监听"
    except Exception as e:
        return False, f"端口检测异常:{e}"

def check_process_name(name: str) -> Tuple[bool, str]:
    """用 wmic 匹配命令行"""
    try:
        r = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'CommandLine', '/format:csv'],
            capture_output=True, text=True, timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if name.lower() in r.stdout.lower():
            return True, f"进程存活:{name}"
        return False, f"进程不在:{name}"
    except Exception as e:
        return False, f"检测异常:{e}"

def is_alive(pdef: dict) -> Tuple[bool, str]:
    method = pdef["check"]
    if method == "port":
        return check_port(pdef["port"])
    return check_process_name(pdef["process_name"])

# ── 进程管理 ──────────────────────────────────────────────────────────────────
_procs: Dict[str, subprocess.Popen] = {}

def start_one(pdef: dict) -> bool:
    pid = pdef["id"]
    log(f"  启动 {pdef['name']}...")

    # Gateway 特殊处理：启动前先清理旧进程
    if pid == "gateway":
        _cleanup_gateways_before_start()

    log_path = str(pdef["log"])
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n{'='*50}\n=== Watchdog 启动 @ {datetime.now()}\n{'='*50}\n")
        proc = subprocess.Popen(
            pdef["cmd"], cwd=str(pdef["cwd"]),
            stdout=open(log_path, "a", encoding="utf-8", errors="replace"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        _procs[pid] = proc
        log(f"  PID={proc.pid}")
        time.sleep(pdef.get("start_delay", 3))
        # 快速检查是否活着
        if proc.poll() is not None:
            log(f"  ??? 启动后快速退出 (code={proc.returncode})")
            return False
        ok, reason = is_alive(pdef)
        if ok:
            log(f"  OK: {reason}")
        else:
            log(f"  ?: {reason} (仍在等待)")
        return True
    except Exception as e:
        log(f"  失败: {e}")
        return False

def _cleanup_gateways_before_start():
    """启动 gateway 前，暴力杀光所有残留 gateway 进程，释放端口"""
    try:
        r = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=8
        )
        # tasklist 输出: "python.exe","12345","Console","1","12,345 K"
        for line in r.stdout.split('\n'):
            if '"python.exe"' not in line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            pid = parts[1].strip().strip('"')
            if not pid.isdigit():
                continue
            # 检查 cmdline 是否含 gateway (用 wmic 更可靠)
            r2 = subprocess.run(
                ['wmic', 'process', 'where', f"ProcessId={pid}", 'get', 'CommandLine', '/format:csv'],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if 'gateway' in r2.stdout.lower():
                subprocess.run(['taskkill', '/F', '/PID', pid],
                               capture_output=True, timeout=5,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                log(f"    清理旧gateway PID={pid}")
        # 等待端口释放
        for _ in range(10):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            r3 = s.connect_ex(("127.0.0.1", 10002))
            s.close()
            if r3 != 0:
                break
            time.sleep(1)
    except Exception as e:
        logger.debug(f"操作失败: {e}")
def stop_one(pdef: dict):
    pid = pdef["id"]
    proc = _procs.get(pid)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        _procs.pop(pid, None)

def stop_all():
    log("=== 停止所有进程 ===")
    for pdef in reversed(PROCESSES):
        stop_one(pdef)
    log("=== 已停止 ===")

# ── 自愈循环 ──────────────────────────────────────────────────────────────────
def heal_loop(initial_boot: bool = False):
    log(f"\n{'='*50}")
    log(f"自愈引擎 v2 @ {datetime.now()}")
    log(f"监控 {len(PROCESSES)} 个进程")
    log(f"{'='*50}\n")

    restarts: Dict[str, int] = {}
    cooldown: Dict[str, float] = {}

    if initial_boot:
        log("首次启动...")
        # 按依赖顺序启动：先QLG量子核，再其他
        ordered = sorted(PROCESSES, key=lambda p: (
            0 if p["id"] == "qlg" else  # QLG 最先
            1 if p["id"] == "gateway" else  # gateway 其次
            2  # 其他最后
        ))
        for p in ordered:
            start_one(p)
            restarts[p["id"]] = 0
        log("首次启动完成\n")

    while True:
        for p in PROCESSES:
            pid = p["id"]
            alive, reason = is_alive(p)

            if not alive:
                # 冷却中
                if pid in cooldown and time.time() < cooldown[pid]:
                    continue

                count = restarts.get(pid, 0) + 1
                restarts[pid] = count
                log(f"  DEAD #{count}: {p['name']} — {reason}")

                # 停止旧进程
                stop_one(p)
                time.sleep(1)

                # 启动
                ok = start_one(p)
                if ok:
                    # 冷却 60s 防止快速重复重启
                    cooldown[pid] = time.time() + 60
                else:
                    log(f"  FAILED: {p['name']}")
                    # 连续失败 -> 冷却更久
                    cooldown[pid] = time.time() + min(120 * count, 600)

                # 进程间间隔
                time.sleep(5)

            else:
                # 健康，reset 计数
                restarts[pid] = 0
                # 如果有冷却也解除
                cooldown.pop(pid, None)

        time.sleep(15)

# ── CLI ───────────────────────────────────────────────────────────────────────
def print_status():
    log(f"\n{'='*50}")
    log(f"状态 @ {datetime.now().strftime('%H:%M:%S')}")
    all_ok = True
    for p in PROCESSES:
        ok, reason = is_alive(p)
        icon = "OK" if ok else "DEAD"
        if not ok: all_ok = False
        log(f"  [{icon}] {p['name']:<30s} {reason}")
    log(f"{'='*50}")
    log(f"总体: {'全部正常' if all_ok else '有进程离线'}")
    return all_ok

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aris Watchdog v2")
    parser.add_argument("action", nargs="?", default="run",
                        choices=["run", "start", "status", "stop", "restart"])
    args = parser.parse_args()

    if args.action == "status":
        print_status()
    elif args.action == "stop":
        stop_all()
    elif args.action == "restart":
        stop_all()
        time.sleep(2)
        heal_loop(initial_boot=True)
    elif args.action == "start":
        heal_loop(initial_boot=True)
    else:
        heal_loop(initial_boot=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n退出...")
        stop_all()
