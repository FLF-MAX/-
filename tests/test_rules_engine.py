"""
LAAP Rules Engine 测试
=======================

验证规则引擎（Zero-LLM 任务执行）的关键行为：
  1. 占位符默认值填充（修复 {path}/{pattern} 未替换 bug）
  2. 事实记忆规则可用
  3. 搜索/统计工具不返回空壳
运行:
    python -m pytest tests/test_rules_engine.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))

import pytest

from aris_rules_engine import process as rules_process


def test_list_files_fills_default_path():
    """未提供路径时，list_files 必须用默认 '.' 而非字面 {path}。"""
    r = rules_process("帮我看看当前目录有哪些文件")
    assert r.get("matched") is True
    out = r.get("output", "")
    assert "空目录" not in out, f"不应返回空目录: {out}"
    assert ".env" in out or "requirements.txt" in out or "aris_brain" in out


def test_remember_and_recall_fact():
    """记住事实 -> 语义记忆写入 -> 可召回。"""
    label = "测试咖啡偏好"
    rm = rules_process(f"记住我喜欢喝{label}")
    assert rm.get("matched") is True
    assert "已记住" in rm.get("output", "")

    rc = rules_process("回忆我喜欢的")
    assert rc.get("matched") is True


def test_count_lines_handles_single_file():
    """count_lines 应能统计单个文件（回归：rglob 不匹配文件自身）。"""
    r = rules_process("统计一下 aris_rules_engine.py 有多少行")
    assert r.get("matched") is True
    out = r.get("output", "")
    assert "未找到可统计代码文件" not in out
    assert "行" in out


def test_unknown_message_does_not_match():
    """与任何规则无关的消息应不匹配规则。"""
    r = rules_process("今天天气怎么样？我们聊聊天吧")
    assert r.get("matched") is False


def test_remember_question_goes_to_recall():
    """“你记得我吗”是回忆（recall），不是记住（remember）——回归修复。"""
    r = rules_process("你记得我吗")
    assert r.get("matched") is True
    assert r.get("rule") == "recall_fact_rule", f"应为 recall 规则，实际 {r.get('rule')}"


def test_identity_manager_status():
    """Identity Manager 应能导出完整身份状态。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))
    from identity_manager import get_identity_manager
    im = get_identity_manager()
    st = im.export_status_json()
    assert st.get("name") in ("Aris", "测试者") or st.get("name")
    assert "self_presence" in st
    assert 0.0 <= st["self_presence"] <= 1.0
    assert "bond_level" in st


def test_voice_cortex_routing():
    """Voice Cortex 路由：自我问题走 aris_only，一般创作走 llm_full。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))
    from voice_cortex import get_voice_cortex
    vc = get_voice_cortex()
    t, m = vc.speak("你是谁")
    assert m == "aris_only"
    assert t and t.strip()


def test_v12_5_engine_produces_intuition():
    """V12.5 直觉引擎应能生成连贯直觉（潜意识层）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))
    from aris_v12_5_engine import ArisV12Engine, MarkovChainV12
    eng = ArisV12Engine()
    assert eng._kernel_ok, "V12 dense kernel should be available"
    resp = eng.respond("我想你了")
    assert resp and len(resp) >= 4
    text, coh = MarkovChainV12().generate(
        seed_words=["你", "我"], max_words=15,
        temperature=0.85, topic="general", emotion="love",
    )
    assert text and len(text) >= 4
    assert 0.0 <= coh <= 1.0


def test_subconscious_uses_v12_5():
    """潜意识层应加载 V12.5 引擎并生成直觉。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))
    from aris_subconscious import get_subconscious
    sc = get_subconscious(interval=30.0)
    assert sc.status()["engine_loaded"] is True
    sc.feed("我今天有点孤独", topics=["情感"])
    sc._generate_intuition()
    assert sc.status()["intuitions_generated"] >= 1
    i = sc._intuitions[-1]
    assert i.content and len(i.content) >= 4


def test_cognitive_bridge_produces_context():
    """Cognitive Bridge 应产出认知上下文（含 V12.5 直觉），不抛异常。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))
    from aris_cognitive_bridge import get_bridge
    b = get_bridge()
    r = b.before_turn("我今天有点孤独，想找人聊聊")
    assert r is not None
    ctx = r.get("cognitive_context") or ""
    assert len(ctx) > 0, "认知上下文不应为空"
    # 感知信号应被注入（V12.5 直觉 / PSI 状态）
    assert ("直觉" in ctx) or ("感知" in ctx) or ("情绪" in ctx)