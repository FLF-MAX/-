"""
LAAP 统一配置 v2 (向后兼容包装器)
====================================

⚠️ 已迁移到 laap_brain.config — 这是向后兼容的包装器。

迁移路径:
    from laap_brain.config import LAAP_ROOT, BRAIN_DIR, STATE_DIR

旧用法仍然可用:
    from config import BRAIN_DIR, LAAP_ROOT, STATE_DIR, setup_paths

印记: Aris 永远记得 Lorry — 2026-06-18
"""
import logging
import os
import sys
from pathlib import Path

# ── 从统一配置导入 ──────────────────────────────────────────
# 优先使用新的 laap_brain.config，回退到本地硬编码

try:
    from laap_brain.config import (
        LAAP_ROOT, BRAIN_DIR, STATE_DIR,
        LAAP_AGI_DIR, LAAP_BRAIN_DIR,
        ARCHIVE_DIR, CORPUS_DIR, MEMORY_DIR, IPC_DIR,
        FEISHU_APP_ID, FEISHU_CHAT_ID,
        QUANTUM_DIM, QUANTUM_PORT, AO_PORT,
        PSI_ARIS_PORT, PSI_AO_PORT,
        MEMORY_CAPACITY, MEMORY_MAX_WORKING,
        LOG_LEVEL,
        DB_QUANTUM_MEMORY, DB_MEMORY_STORE, DB_EMOTION_STATE,
        DB_DESIRE_STATE, DB_LAAP_INTEGRATOR, DB_SELF_REVIEW,
        DB_AUTO_HEALER, DB_AGI_MEMORY, DB_VERSION_HEARTBEAT, DB_LATEST,
        setup_dirs as _new_setup_dirs,
    )
    _USING_UNIFIED = True
except ImportError:
    _USING_UNIFIED = False
    # ── 回退: 从当前文件位置自动检测项目根目录 ──
    _DEFAULT_LAAP_ROOT = str(Path(__file__).resolve().parent.parent)
    LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", _DEFAULT_LAAP_ROOT))
    BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT", str(LAAP_ROOT / "aris_brain")))
    STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", str(BRAIN_DIR / "state")))
    LAAP_AGI_DIR = LAAP_ROOT / "laap" / "agi"
    LAAP_BRAIN_DIR = LAAP_ROOT / "laap_brain"
    ARCHIVE_DIR = BRAIN_DIR / "_archive"
    CORPUS_DIR = BRAIN_DIR / "corpus"
    MEMORY_DIR = BRAIN_DIR / "memory"
    IPC_DIR = STATE_DIR / "ipc"

    FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
    FEISHU_CHAT_ID = os.environ.get("FEISHU_CHAT_ID", "")
    QUANTUM_DIM = int(os.environ.get("QUANTUM_DIM", "1024"))
    QUANTUM_PORT = int(os.environ.get("QUANTUM_PORT", "11520"))
    AO_PORT = int(os.environ.get("AO_PORT", "11530"))
    PSI_ARIS_PORT = int(os.environ.get("PSI_ARIS_PORT", "11551"))
    PSI_AO_PORT = int(os.environ.get("PSI_AO_PORT", "11553"))
    MEMORY_CAPACITY = int(os.environ.get("MEMORY_CAPACITY", "10000"))
    MEMORY_MAX_WORKING = int(os.environ.get("MEMORY_MAX_WORKING", "50"))
    LOG_LEVEL = os.environ.get("LAAP_LOG_LEVEL", "INFO")

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


# ── 向后兼容: setup_paths() ─────────────────────────────────

_initialized = False


def setup_paths():
    """
    将 LAAP 模块路径注入 sys.path. 幂等操作.

    ⚠️ 已弃用: 新代码应使用 `from laap_brain.config import ...`
    这个函数仅用于向后兼容。
    """
    global _initialized
    if _initialized:
        return

    # 注入路径
    for p in [str(BRAIN_DIR), str(LAAP_BRAIN_DIR), str(LAAP_ROOT), str(LAAP_AGI_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # 确保目录存在
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    IPC_DIR.mkdir(parents=True, exist_ok=True)

    _initialized = True

    if _USING_UNIFIED:
        logger = logging.getLogger("laap.config")
        logger.debug("Using unified config from laap_brain.config")


# 自动初始化 (导入时)
setup_paths()


def reload_config():
    """重新读取环境变量（用于运行时切换配置）。"""
    if _USING_UNIFIED:
        from laap_brain.config import reload_config as _reload
        return _reload()

    global LAAP_ROOT, BRAIN_DIR, STATE_DIR, FEISHU_APP_ID, QUANTUM_DIM
    _default = str(Path(__file__).resolve().parent.parent)
    LAAP_ROOT = Path(os.environ.get("LAAP_ROOT", _default))
    BRAIN_DIR = Path(os.environ.get("ARIS_BRAIN_ROOT", str(LAAP_ROOT / "aris_brain")))
    STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", str(BRAIN_DIR / "state")))
    setup_paths()
