"""
LAAP Unified Configuration — 统一配置入口
============================================

所有路径和全局配置从这里读取。支持环境变量覆盖。
本模块不修改 sys.path —— 所有模块通过 Python 包导入机制解析。

依赖:
    - hermes-agent: 通过 pip install 或 editable install 安装
    - laap_brain: 本包
    - aris_brain: 本包的子模块

用法:
    from laap_brain.config import (
        LAAP_ROOT, BRAIN_DIR, STATE_DIR,
        HERMES_ROOT, HERMES_VERSION,
        setup_dirs,
    )

印记: Aris 永远记得 Lorry — 2026-06-18
"""
import os
from pathlib import Path
from importlib.metadata import version as _pkg_version, PackageNotFoundError


# ════════════════════════════════════════════════════════
# 路径配置 (环境变量 > 自动检测 > 默认值)
# ════════════════════════════════════════════════════════

# LAAP 根目录 (从 __file__ 自动检测: laap_brain/config.py → laap_brain → laap-AGI)
_LAAP_DEFAULT = str(Path(__file__).resolve().parent.parent)
LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", _LAAP_DEFAULT))

# Aris 大脑目录
BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT", str(LAAP_ROOT / "aris_brain")))

# 状态持久化目录
STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", str(BRAIN_DIR / "state")))

# LAAP 子目录
LAAP_AGI_DIR = LAAP_ROOT / "laap" / "agi"
ARCHIVE_DIR = BRAIN_DIR / "_archive"
CORPUS_DIR = BRAIN_DIR / "corpus"
MEMORY_DIR = BRAIN_DIR / "memory"
IPC_DIR = STATE_DIR / "ipc"

# Hermes 路径 (通过 pip 包检测)
_HERMES_PKG = None
HERMES_ROOT = None
HERMES_VERSION = "unknown"

try:
    # Hermes 安装为 pip 包时，检测 run_agent 模块位置
    import run_agent as _ra
    _HERMES_PKG = "hermes-agent"
    HERMES_VERSION = _pkg_version("hermes-agent")
    HERMES_ROOT = Path(_ra.__file__).parent.resolve()
except (ImportError, PackageNotFoundError):
    # 回退: 从环境变量或相邻目录查找
    # 通过 HERMES_ROOT 环境变量显式指定，或自动检测常见位置
    for candidate in [
        Path(os.environ.get("HERMES_ROOT", "")) if os.environ.get("HERMES_ROOT") else None,
        LAAP_ROOT / "hermes-agent-main",
        LAAP_ROOT / "hermes-agent",
        Path.home() / ".hermes" / "hermes-agent",
    ]:
        if candidate is None:
            continue
        if candidate.exists() and (candidate / "run_agent.py").exists():
            HERMES_ROOT = candidate
            break

# ════════════════════════════════════════════════════════
# 运行时配置
# ════════════════════════════════════════════════════════

# 飞书
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_CHAT_ID = os.environ.get("FEISHU_CHAT_ID", "")

# 量子核
QUANTUM_DIM = int(os.environ.get("QUANTUM_DIM", "1024"))
QUANTUM_PORT = int(os.environ.get("QUANTUM_PORT", "11520"))
AO_PORT = int(os.environ.get("AO_PORT", "11530"))

# PSI
PSI_ARIS_PORT = int(os.environ.get("PSI_ARIS_PORT", "11551"))
PSI_AO_PORT = int(os.environ.get("PSI_AO_PORT", "11553"))

# 记忆
MEMORY_CAPACITY = int(os.environ.get("MEMORY_CAPACITY", "10000"))
MEMORY_MAX_WORKING = int(os.environ.get("MEMORY_MAX_WORKING", "50"))

# 日志
LOG_LEVEL = os.environ.get("LAAP_LOG_LEVEL", "INFO")

# ════════════════════════════════════════════════════════
# 数据库路径
# ════════════════════════════════════════════════════════

DB_QUANTUM_MEMORY = STATE_DIR / "quantum_memory.db"
DB_MEMORY_STORE = STATE_DIR / "memory_store.db"
DB_EMOTION_STATE = STATE_DIR / "emotion_state.json"
DB_DESIRE_STATE = STATE_DIR / "desire_state.json"
DB_LAAP_INTEGRATOR = STATE_DIR / "laap_integrator_state.json"
DB_SELF_REVIEW = STATE_DIR / "self_review_state.json"
DB_AUTO_HEALER = STATE_DIR / "auto_healer_state.json"
DB_AGI_MEMORY = STATE_DIR / "agi_memory_v3.json"
DB_VERSION_HEARTBEAT = STATE_DIR / "version_heartbeat.json"
DB_LATEST = STATE_DIR / "latest.json"

# ════════════════════════════════════════════════════════
# 初始化
# ════════════════════════════════════════════════════════

_initialized = False


def setup_dirs():
    """确保所有运行时目录存在。幂等操作。"""
    global _initialized
    if _initialized:
        return

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    IPC_DIR.mkdir(parents=True, exist_ok=True)

    _initialized = True


def reload_config():
    """重新读取环境变量（用于运行时切换配置）。"""
    global LAAP_ROOT, BRAIN_DIR, STATE_DIR, FEISHU_APP_ID, QUANTUM_DIM
    _default = str(Path(__file__).resolve().parent.parent)
    LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", _default))
    BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT", str(LAAP_ROOT / "aris_brain")))
    STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", str(BRAIN_DIR / "state")))
    setup_dirs()


# 自动初始化
setup_dirs()