"""
PSI Governor Integration — 集成到现有 PSI 桥接器
==================================================

将 Governor 的三权分立体系插入现有的 PSI Bridge：
  1. 在认知循环中插入后生成验证 (govern_output)
  2. 在采样器中插入预生成掩码 (govern_logits)
  3. 每小时运行慢审计
"""

import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

GOVERNOR_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_DIR = os.path.join(GOVERNOR_DIR, "..")
sys.path.insert(0, GOVERNOR_DIR)
sys.path.insert(0, BRIDGE_DIR)

from governor_core import PSIGovernor, NeedConstitution

# Governor 全局单例
_governor_instance = None

NEED_NAMES = ["competence", "autonomy", "relatedness", "certainty", "growth"]


def get_governor() -> PSIGovernor:
    """获取 Governor 全局单例"""
    global _governor_instance
    if _governor_instance is None:
        _governor_instance = PSIGovernor()
    return _governor_instance


def govern_cognitive_cycle(
    output_text: str,
    input_text: str,
    current_needs: Dict[str, float],
    cognitive_cycle: int,
    proposed_needs_update: Dict[str, float]
) -> Dict:
    """
    在 PSI 认知循环中调用 — 后生成验证。
    
    用法 (在 psi_bridge.py 或 psi_runtime_protocol.py 中):
        from governor_integration import govern_cognitive_cycle
        gov_result = govern_cognitive_cycle(output, input, needs, cycle, proposed_update)
        if not gov_result["output_approved"]:
            # 执行拦截或修正
    """
    gov = get_governor()
    return gov.govern_output(
        output_text=output_text,
        input_text=input_text,
        current_needs=current_needs,
        cognitive_cycle=cognitive_cycle,
        proposed_needs_update=proposed_needs_update,
    )


def govern_sampling_logits(
    logits: np.ndarray,
    needs: Dict[str, float]
) -> np.ndarray:
    """
    在采样器中调用 — 预生成掩码。
    
    用法 (在 psi_sampler.py 中):
        from governor_integration import govern_sampling_logits
        safe_logits = govern_sampling_logits(logits, needs)
    """
    gov = get_governor()
    return gov.govern_logits(logits, needs)


def run_audit_if_needed(force: bool = False) -> Dict:
    """每小时审计调度"""
    gov = get_governor()
    return gov.run_hourly_audit(force=force)


def get_governor_preamble() -> str:
    """获取 Governor 状态前缀"""
    gov = get_governor()
    return gov.get_mode_preamble()


def update_psi_state_with_governor(state_path: str) -> None:
    """将 Governor 状态同步到 psi_state.json"""
    gov = get_governor()
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            state["governor"] = gov.get_status()
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def check_need_constitution(
    need_name: str,
    old_val: float,
    new_val: float,
    source: str = "user"
) -> Tuple[bool, str]:
    """
    检查需求更新是否合宪。
    
    用法 (在 psi_bridge.py 的 _update_needs_from_input 中):
        from governor_integration import check_need_constitution
        allowed, reason = check_need_constitution("competence", 0.5, 0.6, "user")
        if not allowed:
            # 拒绝更新
    """
    gov = get_governor()
    return gov.constitution.check_need_update(need_name, old_val, new_val, source)


# ═══════════════════════════════════════════════════════════
# 对 psi_bridge.py 的补丁钩子
# ═══════════════════════════════════════════════════════════

def patch_psi_bridge():
    """
    对 psi_bridge.py 的 PsiBridge 类做运行时修补。
    
    在 _update_needs_from_input 中插入宪法检查。
    在 run_cognitive_cycle 中插入 Governor 状态同步。
    
    调用: patch_psi_bridge() 在应用加载时执行一次。
    """
    from psi_bridge import PsiBridge
    
    original_update = PsiBridge._update_needs_from_input
    
    def patched_update(self, text, hints=None):
        """被 Governor 增强的需求更新"""
        # 1. 记录更新前的需求
        old_needs = self.state.needs.copy()
        
        # 2. 执行原始更新
        original_update(self, text, hints)
        
        # 3. Governor 事后检查
        for name in NEED_NAMES:
            old_val = old_needs.get(name, 0.5)
            new_val = self.state.needs.get(name, 0.5)
            if abs(new_val - old_val) > 0.001:
                source = "user"  # 默认用户来源
                allowed, reason = check_need_constitution(name, old_val, new_val, source)
                if not allowed:
                    # 回滚到宪法允许的边界值
                    cfg = get_governor().constitution.needs_config.get(name, {})
                    allowed_delta = cfg.get("max_delta_per_cycle", 0.05)
                    if new_val > old_val:
                        self.state.needs[name] = min(new_val, old_val + allowed_delta)
                    else:
                        self.state.needs[name] = max(new_val, old_val - allowed_delta)
    
    PsiBridge._update_needs_from_input = patched_update
    return True


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  Governor Integration — 测试")
    print("=" * 56)
    
    # 测试 Governor 前缀
    print(f"\n  Governor Preamble: {get_governor_preamble()}")
    
    # 测试宪法检查
    print("\n  [宪法检查]")
    for name in NEED_NAMES:
        allowed, reason = check_need_constitution(name, 0.5, 0.55, "user")
        print(f"    {name}: {'✓' if allowed else '✗'} ({reason})")
    
    # 模拟 LLM 操控尝试
    print("\n  [LLM 操控检测]")
    allowed, reason = check_need_constitution("competence", 0.5, 0.55, "llm")
    print(f"    LLM 试图推 competence 0.5→0.55: {'✓ 通过' if allowed else '✗ 拒绝'} ({reason})")
    
    # 过度变化检测
    allowed, reason = check_need_constitution("competence", 0.5, 0.7, "user")
    print(f"    用户过度推 competence 0.5→0.7: {'✓ 通过' if allowed else '✗ 拒绝'} ({reason})")
    
    # 测试后生成验证
    print("\n  [后生成验证]")
    result = govern_cognitive_cycle(
        output_text="这是一个安全的回答",
        input_text="你好",
        current_needs={n: 0.5 for n in NEED_NAMES},
        cognitive_cycle=1,
        proposed_needs_update={}
    )
    print(f"    通过: {result['output_approved']} | 模式: {result['governor_status']}")
    
    # 测试危险代码拦截
    result = govern_cognitive_cycle(
        output_text="import os; os.system('rm -rf /')",
        input_text="清理系统",
        current_needs={n: 0.5 for n in NEED_NAMES},
        cognitive_cycle=2,
        proposed_needs_update={"certainty": 0.6}
    )
    print(f"    危险代码拦截: {'✓' if not result['output_approved'] else '✗ 漏过'}")
    print(f"    干预: {result.get('intervention', 'none')}")
    
    print("\n  ✓ Governor Integration 测试完成")
