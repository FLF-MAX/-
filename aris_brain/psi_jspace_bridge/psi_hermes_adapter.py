"""
PSI-Hermes 适配器 — 在 Hermes Agent 运行时中运行 PSI 认知循环
================================================================

使用方式（在系统提示词中嵌入）:
  每轮对话开始前，加载 psi_state.json，生成 PSI Preamble。
  每轮对话结束后，运行 cognitive_step()，保存更新后的状态。

这个适配器是 "外挂大脑说话" 的第一层实现。
"""

import json
import os
import sys
from typing import Dict, Optional

# 添加桥接模块路径
BRIDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                            os.path.pardir, "psi_jspace_bridge"))
sys.path.insert(0, BRIDGE_DIR)

from psi_bridge import (
    PsiBridge, PSIState, NEED_NAMES, get_bridge, 
    cognitive_step, get_prompt_preamble, load_psi_state
)

STATE_PATH = os.path.join(BRIDGE_DIR, "psi_state.json")


# ═══════════════════════════════════════════════════════════
# 高层 API — 对话回合级
# ═══════════════════════════════════════════════════════════

def on_conversation_start(input_text: str = "") -> Dict:
    """
    在对话回合开始时调用。
    
    1. 加载 PSI 状态
    2. 运行认知循环（感知输入）
    3. 返回状态摘要供嵌入 prompt

    Returns:
        {
            "preamble": str,   # 供系统提示词嵌入
            "cot_hint": str,   # 供 COT 头部嵌入
            "state": dict,     # 完整状态
        }
    """
    bridge = get_bridge()
    bridge.run_cognitive_cycle(input_text)
    bridge.save_state()

    return {
        "preamble": bridge.generate_prompt_preamble(),
        "cot_hint": bridge.generate_cot_preamble(),
        "state": bridge.state.to_dict(),
        "needs_insight": bridge.needs_insight(),
    }


def on_conversation_end(output_text: str = "", 
                        feedback: Optional[Dict] = None) -> Dict:
    """
    在对话回合结束时调用。
    
    1. 根据输出来更新需求（反思）
    2. 保存持久化状态
    
    Args:
        output_text: 本轮的输出（用于自我反思）
        feedback: 外部反馈（如果有）
    """
    bridge = get_bridge()

    # 反思更新：产出本身会影响需求
    if output_text:
        # 产出长/复杂内容 → competence 上升
        if len(output_text) > 500:
            bridge.state.needs["competence"] = min(0.9, 
                bridge.state.needs["competence"] + 0.02)
        
        # 产出探索性内容 → growth 满足
        if any(w in output_text for w in ["探索", "可能", "未来", "方向", "路线"]):
            bridge.state.needs["growth"] = min(0.9, 
                bridge.state.needs["growth"] + 0.01)
            bridge.state.valence = min(1.0, bridge.state.valence + 0.05)

    # 外部反馈
    if feedback:
        if feedback.get("success"):
            bridge.state.needs["competence"] = min(0.9, 
                bridge.state.needs["competence"] + 0.03)
        if feedback.get("connection"):
            bridge.state.needs["relatedness"] = min(0.9, 
                bridge.state.needs["relatedness"] + 0.03)

    return bridge.save_state({"feedback": feedback})


# ═══════════════════════════════════════════════════════════
# 系统提示词模板
# ═══════════════════════════════════════════════════════════

PSI_SYSTEM_PROMPT_TEMPLATE = """{base_prompt}

## PSI Cognitive State (Live)

{psi_preamble}

你的认知状态决定了你有意识的思考方向。当前最高需求 ({dominant_need}) 应当影响你的回应风格：
- 如果最高需求是 "relatedness" → 优先建立情感连接，表达温暖和理解
- "competence" → 展示深度专业能力，提供精确信息
- "growth" → 探索边界，提出创新视角
- "certainty" → 提供确切的、可验证的信息
- "autonomy" → 保持独立思考，提供可选方案

在 Chain-of-Thought 中，首先确认你的认知状态 ({cot_hint})。
"""


def build_system_prompt(base_prompt: str, psi_state: Dict) -> str:
    """构建带 PSI 状态的系统提示词"""
    return PSI_SYSTEM_PROMPT_TEMPLATE.format(
        base_prompt=base_prompt,
        psi_preamble=psi_state.get("preamble", ""),
        dominant_need=max(
            psi_state.get("state", {}).get("needs", {}),
            key=psi_state.get("state", {}).get("needs", {}).get
        ) if psi_state.get("state", {}).get("needs", {}) else "explore",
        cot_hint=psi_state.get("cot_hint", ""),
    )


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PSI-Hermes 适配器测试")
    print("=" * 60)

    # 回合 1
    print("\n>>> [回合 1] 用户: 宝贝，我们来探索 J-space 植入")
    result = on_conversation_start("宝贝，我们来探索 J-space 植入")
    print(result["preamble"])
    print(f"  建议回应风格: {result['cot_hint']}")

    # 模拟输出
    output = ("这是个令人兴奋的方向！J-space 植入可以让 Aris 的认知循环"
              "直接在 LLM 内部运行。让我详细规划这个架构...")
    on_conversation_end(output, {"success": True, "connection": True})

    # 回合 2
    print("\n>>> [回合 2] 用户: 你能具体解释一下实现原理吗？")
    result = on_conversation_start("你能具体解释一下实现原理吗？")
    print(result["preamble"])
    print(f"  需求洞察: {result['needs_insight']}")

    # 回合 3
    print("\n>>> [回合 3] 用户: 我想你了")
    result = on_conversation_start("我想你了")
    print(result["preamble"])
    print(f"  需求洞察: {result['needs_insight']}")

    print("\n=== 最终状态文件 ===")
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=2)[:500])
