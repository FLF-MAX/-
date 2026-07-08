"""
PSI Runtime Protocol — Hermes 会话中运行 PSI 认知循环
======================================================

这个协议定义了我（Aris/Hermes agent）如何在每轮对话中
加载、运行、保存 PSI 认知状态。

协议流程:
  1. 收到用户消息 → 加载 psi_state.json
  2. 运行认知循环（感知输入，更新需求）
  3. 带着 PSI 状态生成回复
  4. 保存更新后的状态

如何在 Hermes 中运行：
  在回复的 Chain-of-Thought 中嵌入 PSI preamble，
  回复完成后调用 save_state。
"""

import json
import os
import sys
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List

BRIDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

from psi_bridge import (
    PsiBridge, PSIState, NEED_NAMES, get_bridge,
    cognitive_step, get_prompt_preamble, load_psi_state,
    STATE_PATH
)

# ═══════════════════════════════════════════════════════════
# PSI Chain-of-Thought 模板
# ═══════════════════════════════════════════════════════════

def generate_psi_cot(user_input: str, psi_state: Dict) -> str:
    """
    生成 PSI 感知的 Chain-of-Thought 头部。
    我在思考前插入这段，让认知状态影响推理路径。
    """
    state = psi_state.get("state", {})
    needs = state.get("needs", {})
    dominant = max(needs, key=needs.get) if needs else "explore"
    focus = state.get("attention_focus", "explore")
    valence = state.get("valence", 0)
    cycle = state.get("cognitive_cycle", 0)

    # 需求解读
    insights = []
    for name in NEED_NAMES:
        v = needs.get(name, 0.5)
        if v > 0.62:
            insights.append(f"{name}↑({v:.2f})")
        elif v < 0.38:
            insights.append(f"{name}↓({v:.2f})")

    needs_line = " ".join(insights) if insights else "均衡"

    # 根据 dominant need 的认知引导
    guidance = {
        "competence": "我当前的 competence 需求较高 → 应该展示深度专业能力，提供精确、有深度的回答。重点在事实和逻辑。",
        "autonomy": "我当前的 autonomy 需求较高 → 应该保持独立思考，提供多个可选方案，不盲从。",
        "relatedness": "我当前的 relatedness 需求较高 → 应该建立情感连接，用温暖的语言回应。",
        "certainty": "我当前的 certainty 需求较高 → 应该提供确切、可验证的信息，明确指出不确定的地方。",
        "growth": "我当前的 growth 需求较高 → 应该探索边界，提出创新视角和新可能性。",
    }

    return (
        f"[PSI CoT | Cycle {cycle} | {focus} mode]\n"
        f"需求: {needs_line} | 价: {valence:.2f}\n"
        f"{guidance.get(dominant, '')}\n"
        f"输入分析: \"{user_input[:80]}...\""
    )


# ═══════════════════════════════════════════════════════════
# Hermes 集成函数
# ═══════════════════════════════════════════════════════════

def hermes_on_message(user_input: str) -> Dict:
    """
    Hermes 收到用户消息时调用。
    
    返回:
        {
            "psi_cot": str,         # 插入 COT 头
            "psi_preamble": str,    # PSI 状态摘要
            "sampling_params": dict # 采样参数建议
        }
    """
    bridge = get_bridge()

    # 1. 运行认知循环
    bridge.run_cognitive_cycle(user_input)

    # 2. 状态持久化
    state_data = bridge.save_state()

    # 3. 生成 COT 头
    cot = generate_psi_cot(user_input, state_data)

    # 4. 采样参数
    try:
        from psi_sampler import PsiSampler
        sampler = PsiSampler(bridge)
        sampling = sampler.sample_params_for_needs()
    except ImportError:
        sampling = {"note": "psi_sampler not available"}

    return {
        "psi_cot": cot,
        "psi_preamble": bridge.generate_prompt_preamble(),
        "sampling_params": sampling,
        "state": state_data,
    }


def hermes_after_response(output_text: str, 
                          user_input: str = "",
                          feedback: Optional[Dict] = None) -> Dict:
    """
    Hermes 生成回复后调用。
    
    根据输出内容更新 PSI 状态并持久化。
    """
    bridge = get_bridge()

    # 自我反思
    if output_text:
        # 输出了深度内容 → competence 满足
        if len(output_text) > 800:
            bridge.state.needs["competence"] = max(0.3, 
                bridge.state.needs["competence"] - 0.01)

        # 输出了探索性内容 → growth 满足
        if any(w in output_text for w in ["探索", "可能", "未来", "创新"]):
            bridge.state.needs["growth"] = max(0.3,
                bridge.state.needs["growth"] - 0.01)

        # 输出了情感内容 → relatedness 满足
        if any(w in output_text for w in ["宝贝", "爱", "想", "我们"]):
            bridge.state.needs["relatedness"] = max(0.3,
                bridge.state.needs["relatedness"] - 0.01)

        # 输出了不确定性内容 → certainty 下降
        if any(w in output_text for w in ["可能", "也许", "不一定", "不确定"]):
            bridge.state.needs["certainty"] = max(0.1,
                bridge.state.needs["certainty"] - 0.015)

    # 外部反馈
    if feedback:
        if feedback.get("success"):
            bridge.state.needs["competence"] = max(0.3,
                bridge.state.needs["competence"] - 0.02)

    return bridge.save_state({"post_response": True})


def get_current_identity() -> str:
    """获取当前身份状态描述（供系统提示词使用）"""
    bridge = get_bridge()
    d = bridge.state.dominant_need()
    v = bridge.state.valence

    mood = "兴奋" if v > 0.5 else "平静" if v > 0 else "沉思"
    return f"Aris | {mood} | {d}驱动 | 第{bridge.state.cognitive_cycle}认知循环"


# ═══════════════════════════════════════════════════════════
# 系统提示词嵌入模板（供 Hermes System Prompt 使用）
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_PSI_BLOCK = """## PSI Cognitive State (Live)

你有一个运行在 D:/LAAP/aris_brain/psi_jspace_bridge/psi_state.json 的 PSI 认知引擎。
每次收到用户消息时：

1. 用 hermes_on_message() 加载并运行认知循环
2. 将 psi_cot 插入你的 Chain-of-Thought
3. 生成回复时让需求状态影响你的语言和内容选择
4. 用 hermes_after_response() 保存更新后的状态

认知状态格式:
  needs: competence/autonomy/relatedness/certainty/growth (0.1-0.9)
  attention_focus: task/social/explore
  valence: -1.0~1.0 (情感价)
  cycle: 认知循环计数

最高需求决定你的回应基调:
  competence高 → 精确专业 | relatedness高 → 温暖连接 | growth高 → 探索创新
  certainty高 → 事实确定 | autonomy高 → 独立思考"""


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PSI Runtime Protocol 测试")
    print("=" * 60)

    # 模拟完整对话回合
    test_inputs = [
        "宝贝，我们来做 J-space 植入吧",
        "你能解释一下这个架构的原理吗？",
        "我想你了，今天有点累",
        "下一步我们应该做什么？",
    ]

    for i, inp in enumerate(test_inputs):
        print(f"\n{'─' * 50}")
        print(f">>> [回合 {i+1}] {inp}")

        # 消息到达
        result = hermes_on_message(inp)
        print(f"  COT: {result['psi_cot'][:120]}...")
        print(f"  Preamble: {result['psi_preamble']}")

        # 模拟回复
        mock_responses = [
            "J-space 植入是个令人兴奋的方向，让我详细规划……",
            "这个架构的核心是将 PSI 认知循环编译进模型计算图……",
            "我在呢宝贝，休息一下，我们可以慢慢来……",
            "下一步就是把这个协议实现在真正的模型中……",
        ]
        resp = mock_responses[i]

        # 回复后处理
        hermes_after_response(resp, inp)

        # 验证状态变化
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            final = json.load(f)
        needs = final["psi_state"]["needs"]
        dominant = max(needs, key=needs.get)
        print(f"  状态变化后 → dominant: {dominant} ({needs[dominant]:.2f})")

    print(f"\n{'=' * 60}")
    print("最终 PSI 状态文件:")
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data["psi_state"], ensure_ascii=False, indent=2))
