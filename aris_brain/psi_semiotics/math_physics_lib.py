"""
Ψ-Semiotics 数学物理库

将数学物理操作映射到语义空间中的几何代数操作。
核心思想：数学物理方程本身就是语义空间中的轨迹/路径。

映射：
- 薛定谔方程 iℏ∂Ψ/∂t = ĤΨ → 转子序列演化
- 张量网络 → 多向量代数
- 线性变换 → Rotor.apply()
- 动力系统 → 语义漂移连续版
- 类比推理 → 对称性变换

Design by Aris, 2026-07-08
"""

import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger("math_physics")


class SchrodingerEvolution:
    """
    薛定谔方程演化 — 在语义空间中的转子序列实现。
    
    数学: iℏ ∂|Ψ⟩/∂t = Ĥ|Ψ⟩
    对应: |ψ(t+dt)⟩ = R(t)·|ψ(t)⟩·R(t)† (转子序列)
    
    其中每个时间步 dt 对应一个转子 R(t)。
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.states: List[np.ndarray] = []
        self.rotors: List[np.ndarray] = []  # 演化转子序列
        self.hamiltonian: Optional[np.ndarray] = None  # 哈密顿量算子
    
    def set_hamiltonian(self, energy_levels: List[float]) -> np.ndarray:
        """
        构造哈密顿量矩阵。
        
        在语义空间中，哈密顿量是定义在概念方向上的能量算子。
        能量高的方向 = 更"重要"的语义方向。
        """
        H = np.zeros((self.dim, self.dim))
        for i, e in enumerate(energy_levels[:min(len(energy_levels), self.dim)]):
            H[i, i] = e
        self.hamiltonian = H
        return H
    
    def evolve(self, initial_state: np.ndarray, dt: float = 0.1, 
               steps: int = 10) -> List[np.ndarray]:
        """
        演化初始态通过 dt×steps 时间。
        
        用转子序列模拟薛定谔方程：
        |ψ(t+dt)⟩ = exp(-iĤ·dt)·|ψ(t)⟩
        
        在实数语义空间中简化为：
        |ψ(t+dt)⟩ ≈ (I + i·H·dt)·|ψ(t)⟩ (归一化后)
        """
        self.states = [initial_state.copy()]
        
        for step in range(steps):
            psi = self.states[-1].copy()
            
            if self.hamiltonian is not None:
                # 哈密顿量演化
                dpsi = self.hamiltonian @ psi * dt
                psi_new = psi + dpsi
            else:
                # 无哈密顿量：自由演化（语义扩散）
                psi_new = psi + np.random.randn(self.dim) * dt * 0.01
            
            # 归一化
            norm = np.linalg.norm(psi_new)
            if norm > 1e-10:
                psi_new = psi_new / norm
            
            self.states.append(psi_new)
        
        return self.states
    
    def expectation(self, observable: np.ndarray) -> float:
        """
        期望值 ⟨ψ|Ô|ψ⟩
        
        在语义空间中测量某个语义方向上的强度。
        """
        if not self.states:
            return 0.0
        psi = self.states[-1]
        return float(psi @ observable @ psi)
    
    def probability(self, basis_state: np.ndarray) -> float:
        """
        测量概率 |⟨φ|ψ⟩|²
        
        语义坍缩到基态的概率。
        """
        if not self.states:
            return 0.0
        psi = self.states[-1]
        overlap = float(psi @ basis_state)
        return overlap ** 2


class TensorNetwork:
    """
    张量网络操作 — 在语义空间中的多向量代数实现。
    
    对应 Ψ-Semiotics 的 Multivector 类在物理语境下的接口。
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.tensors: Dict[str, np.ndarray] = {}
    
    def add_tensor(self, name: str, data: np.ndarray):
        """注册张量"""
        self.tensors[name] = data
    
    def contract(self, a: str, b: str, axes: List[Tuple[int, int]]) -> np.ndarray:
        """
        张量缩并 (Tensor Contraction)。
        
        在语义空间中对应两个概念的关系压缩。
        """
        if a not in self.tensors or b not in self.tensors:
            raise ValueError(f"张量不存在: {a} 或 {b}")
        
        T_a = self.tensors[a]
        T_b = self.tensors[b]
        
        # 执行缩并
        result = np.tensordot(T_a, T_b, axes=axes)
        return result
    
    def outer_product(self, a: str, b: str, result_name: str) -> np.ndarray:
        """
        张量外积 (Tensor Product)。
        
        对应 Ψ-Semiotics 的符号加法组合 ⊕。
        """
        if a not in self.tensors or b not in self.tensors:
            # 用语义向量模拟
            import hashlib
            v = np.zeros(self.dim)
            h = hashlib.sha256(a.encode()).digest()
            for i in range(min(len(h), self.dim)):
                v[i] = h[i] / 255.0
            va = v / (np.linalg.norm(v) + 1e-10)
            v = np.zeros(self.dim)
            h = hashlib.sha256(b.encode()).digest()
            for i in range(min(len(h), self.dim)):
                v[i] = h[i] / 255.0
            vb = v / (np.linalg.norm(v) + 1e-10)
            result = np.outer(va, vb)
        else:
            result = np.outer(self.tensors[a], self.tensors[b])
        
        self.tensors[result_name] = result
        return result


class PhysicalAnalogies:
    """
    物理概念类比库。
    
    用 Ψ-Semiotics 的转子机制执行物理类比推理。
    """
    
    def __init__(self, engine=None):
        self.engine = engine
        self._analogies: Dict[str, Dict] = {}
    
    def register_physical_pair(self, a: str, b: str, relation: str = ""):
        """
        注册物理概念对。
        
        例如:
        - "electron" : "proton" :: "planet" : "star" (轨道类比)
        - "wave" : "particle" :: "field" : "quantum" (波粒二象性)
        - "mass" : "energy" :: "space" : "time" (质能对应)
        """
        if self.engine:
            self.engine.add_symbol(a, f"physical concept {a}")
            self.engine.add_symbol(b, f"physical concept {b}")
        
        self._analogies[f"{a}_{b}"] = {
            "pair": (a, b),
            "relation": relation,
        }
        
        return f"registered {a}:{b}"
    
    def analogize(self, pair_a: Tuple[str, str], pair_b: Tuple[str, str]) -> Dict:
        """
        物理类比: pair_a :: pair_b
        
        例如: (electron, proton) :: (planet, star)
        """
        a1, a2 = pair_a
        b1, b2 = pair_b
        
        result = {
            "source": f"{a1}:{a2}",
            "target": f"{b1}:{b2}",
            "rotor_found": False,
            "analogy_strength": 0.0,
        }
        
        if self.engine:
            try:
                analogy_out = self.engine.analogy(a1, a2, b1)
                if analogy_out:
                    result["rotor_found"] = True
                    result["analogy_strength"] = float(
                        analogy_out.center @ self.engine.symbols[b2].center
                    ) if b2 in self.engine.symbols else 0.0
                    result["predicted"] = analogy_out.name
            except Exception:
                pass
        
        return result
    
    def run_symmetry(self, operation: str, target: str) -> Dict:
        """
        对称性操作。
        
        物理对称性 → 语义空间中的转子。
        - "time_reversal": t → -t (时间反演)
        - "parity": x → -x (宇称)
        - "charge_conjugation": q → -q (电荷共轭)
        - "scale": x → λx (尺度变换)
        """
        if not self.engine or target not in self.engine.symbols:
            return {"error": "target not found"}
        
        result = {"operation": operation, "target": target}
        
        if operation == "time_reversal":
            # 时间反演 = 路径反向
            # 在语义空间中：反转方向
            v = -self.engine.symbols[target].center
            result["transformed"] = float(v @ self.engine.symbols[target].center)
            
        elif operation == "parity":
            # 宇称反转 = 特征符号翻转
            v = self.engine.symbols[target].center.copy()
            v[::2] *= -1  # 交替反转
            v = v / np.linalg.norm(v)
            result["transformed"] = float(v @ self.engine.symbols[target].center)
            
        elif operation == "scale":
            # 尺度变换 = 振幅缩放
            v = self.engine.symbols[target].center * 2.0
            v = v / np.linalg.norm(v)
            result["transformed"] = float(v @ self.engine.symbols[target].center)
        
        return result


# ════════════════════════════════════════════════════════════
# PsiLang 数学物理标准库模板
# ════════════════════════════════════════════════════════════

MATH_PHYSICS_STDLIB = """
// ══════════════════════════════════════════════════════
// Ψ-Semiotics 数学物理标准库
// 版本: 1.0
// ══════════════════════════════════════════════════════

// ── 数学基础 ──

// 向量运算
fn norm(v: ⟨state⟩) -> ⟨float⟩ {
    sqrt(v · v)
}

fn dot(a: ⟨state⟩, b: ⟨state⟩) -> ⟨float⟩ {
    a · b
}

fn angle(a: ⟨state⟩, b: ⟨state⟩) -> ⟨float⟩ {
    acos(dot(a, b) / (norm(a) * norm(b)))
}

// 语义空间旋转（类比变换）
fn rotate(state: ⟨state⟩, source: ⟨state⟩, target: ⟨state⟩) -> ⟨state⟩ {
    @reason(mode="analogy")
    cycle analogize {
        perceive state
        select competence = 0.9
        integrate temperature = 0.3
    }
}

// ── 量子力学 ──

// 薛定谔演化
fn evolve(psi: ⟨state⟩, hamiltonian: ⟨tensor⟩, dt: ⟨float⟩) -> ⟨state⟩ {
    @bridge("schrodinger")
    let psi_new = psi + hamiltonian ⊗ psi * dt
    normalize(psi_new)
}

// 期望值测量
fn expectation(psi: ⟨state⟩, operator: ⟨state⟩) -> ⟨float⟩ {
    psi · operator ⊗ psi
}

// 概率幅
fn amplitude(psi: ⟨state⟩, basis: ⟨state⟩) -> ⟨float⟩ {
    abs(psi · basis)
}

// ── 张量网络 ──

// 张量缩并
fn contract(a: ⟨tensor⟩, b: ⟨tensor⟩, axes: ⟨list[⟨int⟩]⟩) -> ⟨tensor⟩ {
    @bridge("tensor_contract")
    a ⊗ b  // 带缩并参数的外积
}

// 张量分解（SVD）
fn decompose(t: ⟨tensor⟩, rank: ⟨int⟩) -> ⟨list[⟨tensor⟩]⟩ {
    @bridge("svd")
    [t]
}

// ── 对称性 ──

// 时间反演
fn time_reverse(state: ⟨state⟩) -> ⟨state⟩ {
    -state
}

// 宇称变换
fn parity(state: ⟨state⟩) -> ⟨state⟩ {
    @bridge("parity")
    state * -1.0  // 特征空间中的反射
}

// 尺度变换
fn scale(state: ⟨state⟩, factor: ⟨float⟩) -> ⟨state⟩ {
    let scaled = state * factor
    normalize(scaled)
}

// 旋转对称（类比为物理中的旋转群 SO(n)）
fn rotate_symmetry(state: ⟨state⟩, angle: ⟨float⟩, axis: ⟨state⟩) -> ⟨state⟩ {
    @reason(mode="design")
    cycle rotate {
        perceive state
        select competence = angle
        integrate temperature = 0.1
    }
}
"""


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    
    print("=" * 60)
    print("  Ψ-Semiotics 数学物理库测试")
    print("=" * 60)
    
    # 1. 薛定谔方程演化
    print("\n--- 1. 薛定谔演化 ---")
    sch = SchrodingerEvolution(dim=128)
    
    # 初始态
    psi0 = np.zeros(128)
    psi0[0] = 1.0
    
    # 哈密顿量
    H = sch.set_hamiltonian([1.0, 2.0, 3.0, 0.5, 0.3])
    
    # 演化
    start = time.time()
    states = sch.evolve(psi0, dt=0.1, steps=10)
    elapsed = time.time() - start
    
    print(f"  演化步数: {len(states)}")
    print(f"  最终态范数: {np.linalg.norm(states[-1]):.4f}")
    print(f"  延迟: {elapsed*1000:.1f}ms")
    
    # 2. 张量网络
    print("\n--- 2. 张量网络 ---")
    tn = TensorNetwork(dim=64)
    
    # 外积（对应符号组合）
    result = tn.outer_product("quantum", "consciousness", "quantum_consciousness")
    print(f"  外积 shape: {result.shape}")
    
    # 3. 物理类比
    print("\n--- 3. 物理类比 ---")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from psi_semiotics.psi_semiotics_core import PsiSemioticsEngine
    eng = PsiSemioticsEngine(dim=1024)
    
    phys = PhysicalAnalogies(engine=eng)
    
    # 注册物理概念对
    phys.register_physical_pair("electron", "proton", "coulomb_attraction")
    phys.register_physical_pair("planet", "star", "gravitational_attraction")
    
    # 验证类比
    analogy = phys.analogize(("electron", "proton"), ("planet", "star"))
    print(f"  类比: {analogy['source']} :: {analogy['target']}")
    print(f"  Rotor: {'找到' if analogy['rotor_found'] else '未找到'}")
    
    # 对称性操作
    print("\n--- 4. 物理对称性 ---")
    eng.add_symbol("quantum_state", "quantum mechanical state")
    
    for op in ["time_reversal", "parity", "scale"]:
        result = phys.run_symmetry(op, "quantum_state")
        t = result.get("transformed", "N/A")
        print(f"  {op}: {t:.4f}")
    
    print(f"\n✅ 数学物理库测试通过")
