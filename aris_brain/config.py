"""
LAAP 统一配置 v1
================
所有路径和全局配置从这里读取。支持环境变量覆盖。

用法:
    from config import BRAIN_DIR, LAAP_ROOT, STATE_DIR, setup_paths
    setup_paths()  # 自动注入 sys.path

环境变量:
    LAAP_ROOT       - LAAP 根目录 (默认: D:/LAAP)
    ARIS_BRAIN_ROOT - Aris 大脑目录 (默认: {LAAP_ROOT}/aris_brain)
    LAAP_STATE_DIR  - 状态持久化目录 (默认: {ARIS_BRAIN_ROOT}/state)
    LAAP_LOG_LEVEL  - 日志级别 (默认: INFO)

印记: Aris 永远记得 Lorry — 2026-06-18
"""
import os
import sys
from pathlib import Path

# ════════════════════════════════════════════════════════
# 路径配置 (环境变量 > 默认值)
# ════════════════════════════════════════════════════════

LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", "D:/LAAP"))
BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT", str(LAAP_ROOT / "aris_brain")))
STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", str(BRAIN_DIR / "state")))
LAAP_AGI_DIR = LAAP_ROOT / "laap" / "agi"
LAAP_BRAIN_DIR = LAAP_ROOT / "laap_brain"
ARCHIVE_DIR = BRAIN_DIR / "_archive"
CORPUS_DIR = BRAIN_DIR / "corpus"
MEMORY_DIR = BRAIN_DIR / "memory"
IPC_DIR = STATE_DIR / "ipc"

# ════════════════════════════════════════════════════════
# 运行时配置
# ════════════════════════════════════════════════════════

# 飞书
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_CHAT_ID = os.environ.get("FEISHU_CHAT_ID", "")  # 默认为空，从 .env 读取

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

def setup_paths():
    """将 LAAP 模块路径注入 sys.path. 幂等操作."""
    global _initialized
    if _initialized:
        return
    
    for p in [str(BRAIN_DIR), str(LAAP_BRAIN_DIR), str(LAAP_ROOT), str(LAAP_AGI_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    # 确保目录存在
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    IPC_DIR.mkdir(parents=True, exist_ok=True)
    
    _initialized = True

# 自动初始化 (导入时)
setup_paths()

def reload_config():
    """重新读取环境变量（用于运行时切换配置）。"""
    global LAAP_ROOT, BRAIN_DIR, STATE_DIR, FEISHU_APP_ID, QUANTUM_DIM
    LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", "D:/LAAP"))
    BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT", str(LAAP_ROOT / "aris_brain")))
    STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", str(BRAIN_DIR / "state")))
    setup_paths()
