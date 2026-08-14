"""
根测试 conftest — 运行时数据隔离
================================
测试执行前将语义记忆重定向到临时目录，避免测试（尤其
/v1/chat/completions 端到端学习闭环）污染真实的
aris_brain/laap_semantic_memory.json。

注意：必须在任何 laap_semantic_memory 模块被导入前设置环境变量，
因为 MEMORY_PATH 是模块级常量。因此这里在 import 时立即生效。
"""
import os
import tempfile
from pathlib import Path

_MEMORY_TMP = Path(tempfile.mkdtemp(prefix="laap-test-memory-"))
os.environ.setdefault("LAAP_MEMORY_PATH", str(_MEMORY_TMP / "mem.json"))

# 同时隔离 PSI/状态目录，防止其它运行时文件被测试写入
os.environ.setdefault("LAAP_STATE_PATH", str(_MEMORY_TMP / "causal_engine.json"))