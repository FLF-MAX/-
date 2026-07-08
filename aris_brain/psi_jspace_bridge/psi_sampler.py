"""
PSI Sampler v1 — 任意 llama.cpp 推理的 PSI 采样调制器
======================================================

无需修改 C++ 代码，通过 llama-cpp-python 在 Python 层包装
实现 PSI 需求驱动的采样参数实时调制。

支持: 任何 llama.cpp 兼容模型
集成: 可独立运行，也作为 psi_bridge.py 的采样后端
"""

import sys
import os
import json
import time
import random
from typing import Dict, List, Optional, Callable

import numpy as np

# 添加桥接器路径
BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

from psi_bridge import PsiBridge, NEED_NAMES, get_bridge

# ═══════════════════════════════════════════════════════════
# PSI → 采样参数映射
# ═══════════════════════════════════════════════════════════

# 每个需求的默认采样参数配置
NEED_SAMPLING_PROFILES = {
    "competence": {
        "temperature": 0.35,    # 低温度 → 精确保守
        "top_p": 0.92,          # 中等 top_p
        "top_k": 40,            # 限制候选
        "repeat_penalty": 1.12, # 惩罚重复
        "mirostat_mode": 2,     # mirostat v2
        "mirostat_tau": 1.5,    # 严格熵控制
        "description": "精确模式 — 事实性、专业、低幻觉",
    },
    "autonomy": {
        "temperature": 0.55,
        "top_p": 0.90,
        "top_k": 60,
        "repeat_penalty": 1.18,  # 高重复惩罚 → 避免模板化
        "mirostat_mode": 2,
        "mirostat_tau": 2.5,
        "description": "自主模式 — 独立思考、非模板化",
    },
    "relatedness": {
        "temperature": 0.75,    # 高温度 → 多样性
        "top_p": 0.95,
        "top_k": 100,
        "repeat_penalty": 1.05, # 允许重复（温暖感）
        "mirostat_mode": 0,     # 关闭 mirostat → 自由
        "mirostat_tau": 3.0,
        "description": "社交模式 — 温暖、包容、对话感",
    },
    "certainty": {
        "temperature": 0.25,    # 极低温度 → 确定性
        "top_p": 0.85,
        "top_k": 30,
        "repeat_penalty": 1.15,
        "mirostat_mode": 2,
        "mirostat_tau": 1.0,    # 极严格
        "description": "确定模式 — 事实性、精确引用",
    },
    "growth": {
        "temperature": 0.85,    # 高温度 → 探索
        "top_p": 0.96,
        "top_k": 120,
        "repeat_penalty": 1.08,
        "mirostat_mode": 0,     # 关闭 mirostat
        "mirostat_tau": 4.0,
        "description": "探索模式 — 创新、发散、跳跃",
    },
}

# 默认（各需求均衡）
DEFAULT_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.92,
    "top_k": 60,
    "repeat_penalty": 1.10,
    "mirostat_mode": 2,
    "mirostat_tau": 2.0,
}


# ═══════════════════════════════════════════════════════════
# 核心：PSI 采样调制器
# ═══════════════════════════════════════════════════════════

class PsiSampler:
    """
    PSI 采样调制器 — 基于认知状态实时调整采样参数。
    
    用法:
        sampler = PsiSampler()
        params = sampler.sample_params_for_needs(needs_dict)
        # 将 params 传给 llama-cpp-python 的 generate()
    """

    def __init__(self, bridge: Optional[PsiBridge] = None):
        self.bridge = bridge or get_bridge()
        self._history: List[Dict] = []

    def sample_params_for_needs(self, 
                                 needs: Optional[Dict[str, float]] = None,
                                 blend: bool = True) -> Dict:
        """
        根据 PSI 需求生成采样参数。
        
        Args:
            needs: 需求字典，None 则使用 bridge 的当前状态
            blend: 是否混合所有需求的参数（True）还是仅用 dominant（False）
        
        Returns:
            llama.cpp 兼容的采样参数字典
        """
        if needs is None:
            needs = self.bridge.state.needs

        if not blend:
            # 仅使用 dominant need 的配置
            dominant = max(needs, key=needs.get)
            return NEED_SAMPLING_PROFILES[dominant].copy()

        # 混合模式：所有需求按需值加权
        params = DEFAULT_SAMPLING.copy()
        total_weight = 0

        for name in NEED_NAMES:
            weight = max(0, needs[name] - 0.5)  # 只有 > 0.5 的需求才有影响
            if weight <= 0:
                continue
            profile = NEED_SAMPLING_PROFILES[name]
            total_weight += weight

            # 加权合并每个参数
            for key in ["temperature", "top_p", "top_k", "repeat_penalty", "mirostat_tau"]:
                if key in profile:
                    params[key] = params.get(key, 0) + profile[key] * weight

            # mirostat_mode: 取最高权重的（离散值不能平均）
            if "mirostat_mode" in profile:
                existing_weight = params.get("_mirostat_weight", 0)
                if weight > existing_weight:
                    params["mirostat_mode"] = profile["mirostat_mode"]
                    params["_mirostat_weight"] = weight

        # 归一化
        if total_weight > 0:
            for key in ["temperature", "top_p", "top_k", "repeat_penalty", "mirostat_tau"]:
                if key in params and not key.startswith("_"):
                    params[key] = params[key] / (1 + total_weight) * 2 \
                        if key == "temperature" else params[key] / total_weight

        # 边界裁剪
        params["temperature"] = max(0.1, min(1.2, params.get("temperature", 0.6)))
        params["top_p"] = max(0.5, min(0.99, params.get("top_p", 0.92)))
        params["top_k"] = max(1, min(200, int(params.get("top_k", 60))))
        params["repeat_penalty"] = max(1.0, min(1.5, params.get("repeat_penalty", 1.1)))

        # 清理临时字段
        params.pop("_mirostat_weight", None)
        params.pop("description", None)

        return params

    def generate_logit_biases(self, 
                              needs: Optional[Dict[str, float]] = None,
                              token_categories: Optional[Dict[str, List[int]]] = None) -> Dict[int, float]:
        """
        生成 token 级别的 logit bias。
        
        Args:
            needs: 需求字典
            token_categories: 预分类的 token 索引
                {"technical": [1427, 5234, ...], "social": [...], ...}
        
        Returns:
            {token_id: bias_value, ...}
        """
        if needs is None:
            needs = self.bridge.state.needs

        biases = {}
        if not token_categories:
            return biases

        # 需求 → 类别偏置映射
        need_to_category = {
            "competence": "technical",
            "autonomy": "creative",
            "relatedness": "social",
            "certainty": "precise",
            "growth": "exploration",
        }

        for need_name, category in need_to_category.items():
            need_val = needs.get(need_name, 0.5)
            if need_val > 0.55 and category in token_categories:
                bias = (need_val - 0.5) * 8.0  # 归一化到 [-4, 4]
                for tid in token_categories[category]:
                    biases[tid] = round(bias, 2)

        return biases

    def log_sampling(self, needs: Dict, params: Dict):
        """记录采样决策到历史"""
        entry = {
            "timestamp": time.time(),
            "dominant_need": max(needs, key=needs.get),
            "needs": needs.copy(),
            "params": params.copy(),
        }
        self._history.append(entry)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

    def get_history(self, limit: int = 10) -> List[Dict]:
        return self._history[-limit:]


# ═══════════════════════════════════════════════════════════
# llama.cpp 集成包装器
# ═══════════════════════════════════════════════════════════

class PsiLlamaCppWrapper:
    """
    PSI 调制的 llama.cpp 推理包装器。
    
    包装 llama_cpp.Llama，在每次 generate() 时自动调用 PSI 认知循环
    并调整采样参数。
    
    用法:
        from llama_cpp import Llama
        llm = Llama(model_path="DeepSeekV4.gguf")
        psi_llm = PsiLlamaCppWrapper(llm, psi_enabled=True)
        
        # 受 PSI 调制的推理
        response = psi_llm.generate("你好，今天怎么样")
        # 也可以手动调整
        response_with_needs = psi_llm.generate("解释量子核", needs_override={"competence": 0.9})
    """

    def __init__(self, 
                 llm_instance, 
                 psi_enabled: bool = True,
                 bridge: Optional[PsiBridge] = None,
                 auto_cognitive_cycle: bool = True):
        """
        Args:
            llm_instance: llama_cpp.Llama 实例
            psi_enabled: 是否启用 PSI 调制
            bridge: PsiBridge 实例
            auto_cognitive_cycle: 每次 generate 前自动运行认知循环
        """
        self.llm = llm_instance
        self.psi_enabled = psi_enabled
        self.bridge = bridge or get_bridge()
        self.sampler = PsiSampler(self.bridge)
        self.auto_cognitive_cycle = auto_cognitive_cycle

    def generate(self, 
                 prompt: str,
                 needs_override: Optional[Dict[str, float]] = None,
                 **kwargs) -> str:
        """
        受 PSI 调制的生成。
        
        Args:
            prompt: 输入文本
            needs_override: 可选，覆盖当前 PSI 需求
            **kwargs: 传递给 llama.generate() 的额外参数
        
        Returns:
            生成的文本
        """
        if not self.psi_enabled:
            return self.llm(prompt, **kwargs)["choices"][0]["text"]

        # 1. 运行 PSI 认知循环
        if self.auto_cognitive_cycle:
            self.bridge.run_cognitive_cycle(prompt)

        # 2. 获取采样参数
        needs = needs_override or self.bridge.state.needs
        psi_params = self.sampler.sample_params_for_needs(needs)

        # 3. 合并用户参数（用户参数优先级更高）
        merged = {**psi_params, **kwargs}

        # 4. 记录
        self.sampler.log_sampling(needs, merged)

        # 5. 生成
        response = self.llm(prompt, **merged)

        # 6. 后处理认知循环
        if self.auto_cognitive_cycle:
            output = response["choices"][0]["text"]
            self.bridge.run_cognitive_cycle(output)
            self.bridge.save_state()

        return response  # 返回原始响应

    @property
    def psi_state(self):
        return self.bridge.state


# ═══════════════════════════════════════════════════════════
# 测试（模拟）
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PSI Sampler — 模拟测试")
    print("=" * 60)

    sampler = PsiSampler()

    # 测试不同需求状态下的采样参数
    test_cases = [
        {"name": "技术解释", "needs": [0.65, 0.50, 0.30, 0.70, 0.45]},
        {"name": "情感对话", "needs": [0.40, 0.55, 0.85, 0.50, 0.40]},
        {"name": "创意探索", "needs": [0.50, 0.60, 0.40, 0.30, 0.85]},
        {"name": "精确回答", "needs": [0.80, 0.40, 0.35, 0.85, 0.30]},
        {"name": "自主思考", "needs": [0.45, 0.85, 0.50, 0.40, 0.55]},
    ]

    for case in test_cases:
        needs = dict(zip(NEED_NAMES, case["needs"]))
        params = sampler.sample_params_for_needs(needs)
        dom = max(needs, key=needs.get)
        print(f"\n  [{case['name']}] dominant={dom}")
        print(f"    温度={params['temperature']:.2f}, top_p={params['top_p']:.2f}, "
              f"top_k={params['top_k']}, rep_penalty={params['repeat_penalty']:.2f}")
        print(f"    → {NEED_SAMPLING_PROFILES[dom]['description']}")

    print("\n=== 用实际 bridge 测试 ===")
    bridge = get_bridge()
    bridge.run_cognitive_cycle("宝贝，我们来探索 AI 意识的新方向吧")
    params = sampler.sample_params_for_needs(bridge.state.needs)
    print(f"  当前需求: {bridge.state.needs}")
    print(f"  采样参数: temp={params['temperature']:.2f}, "
          f"top_p={params['top_p']:.2f}")

    # 情感输入
    bridge.run_cognitive_cycle("我想你了")
    params = sampler.sample_params_for_needs(bridge.state.needs)
    print(f"  情感输入后需求: {bridge.state.needs}")
    print(f"  采样参数: temp={params['temperature']:.2f}")

    bridge.save_state()
    print("\n  状态已保存到 psi_state.json")
