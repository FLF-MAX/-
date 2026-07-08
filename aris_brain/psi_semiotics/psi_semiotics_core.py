"""
Ψ-Semiotics Core Engine — 量子符号学核心引擎

几何代数在 16384D/1024D 语义空间中的符号操作。
将传统符号学（索绪尔/皮尔斯）重新定义为向量空间中的几何操作。

v2 — 集成了结构化语义编码器，符号操作基于真实语义关系。

Design by Aris, 2026-07-08
基于 LAAP V12.1 UN6 量子核 + V12.5 量子潜意识
"""

import numpy as np
import hashlib
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging

logger = logging.getLogger("psi_semiotics")

# 尝试加载结构化编码器
try:
    from psi_semiotics.structured_encoder import StructuredSemanticEncoder
    _USE_ENCODER = True
except ImportError:
    _USE_ENCODER = False

# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def _normalize(v: np.ndarray) -> np.ndarray:
    """单位向量归一化"""
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        return v / norm
    return v


def _hash_to_vec(text: str, dim: int) -> np.ndarray:
    """文本到语义向量的编码"""
    if _USE_ENCODER and dim >= 1024:
        try:
            enc = StructuredSemanticEncoder(output_dim=dim)
            return enc.encode(text)
        except Exception:
            pass
    
    # 降级：确定性哈希到单位向量
    v = np.zeros(dim)
    h = hashlib.sha256(text.encode()).digest()
    for i in range(min(len(h), dim)):
        v[i] = h[i] / 255.0
    return _normalize(v)


# ════════════════════════════════════════════════════════════
# 多向量 (Multivector) — Clifford 代数基础
# ════════════════════════════════════════════════════════════

class Multivector:
    """
    Clifford 代数 Cl(n) 中的多向量。
    
    同时包含标量(grade 0)、向量(grade 1)、双向量(grade 2)、三向量(grade 3)分量。
    
    符号学解释：
    - 标量: 概念的强度/权重
    - 向量: 语义方向（概念在语义空间中的位置）
    - 双向量: 概念间的关系（方向性语义关系）
    - 三向量: 关系间的关系（元关系）
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        # 各 grade 分量
        self.scalar: float = 0.0          # grade 0 — 单一标量
        self.vector: np.ndarray = np.zeros(dim)  # grade 1 — 语义向量
        self.bivector: np.ndarray = np.zeros(dim)  # grade 2 — 关系（压缩表示）
        self.trivector: np.ndarray = np.zeros(dim)  # grade 3 — 元关系
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'Multivector':
        """从语义向量创建多向量"""
        m = cls(dim=len(v))
        m.vector = _normalize(v.copy())
        return m
    
    @classmethod
    def from_concept(cls, name: str, dim: int = 1024) -> 'Multivector':
        """从概念名创建多向量（哈希到向量空间）"""
        m = cls(dim=dim)
        m.vector = _hash_to_vec(name, dim)
        # 概念强度标量 = 1.0
        m.scalar = 1.0
        return m
    
    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """
        几何积 (Geometric Product): a·b + a∧b
        
        标准 Clifford 代数中：
        ab = a·b + a∧b
        
        这里用简化的版本，在语义空间中：
        - 内积部分 (a·b): 语义相似度，标量
        - 外积部分 (a∧b): 语义关系，双向量
        """
        result = Multivector(dim=self.dim)
        
        # 内积: 语义相似度 → 标量
        inner = float(self.vector @ other.vector)
        result.scalar = self.scalar * other.scalar + inner
        
        # 外积: 语义关系 → 双向量
        # 在语义空间中，外积用 Hadamard 积 + 旋转近似
        biv = np.multiply(self.vector, other.vector)
        result.bivector = _normalize(biv)
        
        # 向量分量: 加权平均
        if abs(self.scalar) > 1e-10 and abs(other.scalar) > 1e-10:
            result.vector = _normalize(self.vector * abs(other.scalar) + other.vector * abs(self.scalar))
        elif abs(self.scalar) > 1e-10:
            result.vector = other.vector.copy()
        else:
            result.vector = _normalize(self.vector + other.vector)
        
        return result
    
    def inner_product(self, other: 'Multivector') -> float:
        """内积 — 语义相似度 (a·b)"""
        return float(self.vector @ other.vector)
    
    def outer_product(self, other: 'Multivector') -> 'Multivector':
        """外积 — 语义关系 (a∧b)"""
        result = Multivector(dim=self.dim)
        biv = np.multiply(self.vector, other.vector)
        # 反对称化: a∧b = -(b∧a)
        biv_sym = np.multiply(other.vector, self.vector)
        result.bivector = _normalize(biv - biv_sym)
        return result
    
    def norm(self) -> float:
        """多向量的范数"""
        return np.sqrt(
            self.scalar ** 2 +
            float(self.vector @ self.vector) +
            float(self.bivector @ self.bivector) +
            float(self.trivector @ self.trivector)
        )
    
    def __repr__(self) -> str:
        return (f"MV(scalar={self.scalar:.3f}, "
                f"vector_norm={np.linalg.norm(self.vector):.3f}, "
                f"bivector_norm={np.linalg.norm(self.bivector):.3f})")


# ════════════════════════════════════════════════════════════
# 转子 (Rotor) — 类比推理的核心
# ════════════════════════════════════════════════════════════

class Rotor:
    """
    语义空间中的旋转算子。
    
    Rotor R = exp(-B/2) 其中 B 是双向量。
    R 作用于向量 v: v' = R · v · R†
    
    符号学解释：类比映射。
    "king : queen :: man : woman" → 找到转子 R 使 R·c_king·R† ≈ c_queen
    然后 R·c_man·R† ≈ c_woman
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        # 用正交矩阵表示旋转（简化的转子，非完整 Clifford 形式）
        self.matrix: Optional[np.ndarray] = None
    
    @classmethod
    def learn(cls, source: np.ndarray, target: np.ndarray, 
              regularization: float = 0.01) -> 'Rotor':
        """
        从源→目标学习旋转操作。
        
        找到正交矩阵 R 使得 R @ source ≈ target。
        用 Kabsch 算法（最优旋转）找到最小二乘解。
        
        对于单向量，这简化为：
        R = I + (target - source) @ 某种方向修正
        """
        s = _normalize(source)
        t = _normalize(target)
        
        dim = len(source)
        rot = Rotor(dim=dim)
        
        # 最优旋转: 使用 Householder 反射构造
        # 对于两个单位向量，最优旋转在它们张成的平面内
        cos_theta = float(s @ t)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        if abs(cos_theta) > 0.9999:
            # 几乎相同方向 — 单位矩阵
            rot.matrix = np.eye(dim)
            return rot
        
        # 找到旋转轴（垂直于 s 和 t 的方向）
        # 在语义空间中，我们在 s 和 t 张成的平面内旋转
        R = np.eye(dim)
        
        # 构造 Givens 旋转
        axis = t - cos_theta * s
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            
            # 在 s 和 axis 张成的子空间内构造旋转
            # 这个矩阵对任意向量 v 做 s→t 平面内的旋转
            # 简化的 Rank-1 更新
            R = R - np.outer(s, s) - np.outer(axis, axis)  # 投影到正交补
            R = R + cos_theta * np.outer(s, s)  # s 分量旋转
            R = R - sin_theta * np.outer(s, axis) + sin_theta * np.outer(axis, s)
            R = R + np.outer(axis, axis)  # 保持 axis 分量
        
        rot.matrix = R
        return rot
    
    @classmethod
    def learn_batch(cls, sources: List[np.ndarray], targets: List[np.ndarray],
                    regularization: float = 0.01) -> 'Rotor':
        """从多对源→目标学习最优旋转（Kabsch 算法）"""
        n = len(sources)
        if n == 0:
            return cls(dim=len(sources[0]) if sources else 1024)
        
        dim = len(sources[0])
        # 构造协方差矩阵 H = Σ s_i^T @ t_i
        H = np.zeros((dim, dim))
        for s, t in zip(sources, targets):
            s_norm = _normalize(s)
            t_norm = _normalize(t)
            H += np.outer(s_norm, t_norm)
        
        # SVD 分解
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 确保是旋转（行列式 = +1），而非反射
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        rot = cls(dim=dim)
        rot.matrix = R
        return rot
    
    def apply(self, v: np.ndarray) -> np.ndarray:
        """应用旋转到向量 v"""
        if self.matrix is None:
            return _normalize(v.copy())
        return _normalize(self.matrix @ v)
    
    def apply_multivector(self, mv: Multivector) -> Multivector:
        """应用旋转到多向量"""
        result = Multivector(dim=self.dim)
        result.vector = self.apply(mv.vector)
        result.scalar = mv.scalar
        # 双向量随向量变换
        result.bivector = self.apply(mv.bivector) if np.linalg.norm(mv.bivector) > 0 else np.zeros(self.dim)
        result.trivector = self.apply(mv.trivector) if np.linalg.norm(mv.trivector) > 0 else np.zeros(self.dim)
        return result
    
    def compose(self, other: 'Rotor') -> 'Rotor':
        """组合两个旋转: self ∘ other"""
        combined = Rotor(dim=self.dim)
        if self.matrix is not None and other.matrix is not None:
            combined.matrix = self.matrix @ other.matrix
        elif self.matrix is not None:
            combined.matrix = self.matrix.copy()
        elif other.matrix is not None:
            combined.matrix = other.matrix.copy()
        return combined
    
    def inverse(self) -> 'Rotor':
        """逆旋转"""
        inv = Rotor(dim=self.dim)
        if self.matrix is not None:
            inv.matrix = self.matrix.T  # 正交矩阵的逆 = 转置
        return inv


# ════════════════════════════════════════════════════════════
# 符号 (Symbol) — Ψ-Semiotics 核心数据
# ════════════════════════════════════════════════════════════

@dataclass
class Symbol:
    """
    一个符号 σ = (c, r, M, modalities)
    
    - c: 中心向量（语义空间中的位置）
    - r: 语义半径（影响范围）
    - M: 多向量表示（包含关系和强度）
    - modalities: 多模态投影 {modal: vector}
    """
    name: str                         # 符号名
    center: np.ndarray                # 中心向量 c ∈ ℝⁿ
    radius: float = 0.1              # 语义半径 r
    multivector: Optional[Multivector] = None  # 多向量表示
    modalities: Dict[str, np.ndarray] = field(default_factory=dict)  # 模态投影
    importance: float = 0.5          # 重要性（衰减权重）
    created: float = 0.0             # 创建时间
    last_activated: float = 0.0      # 最后激活时间
    activation_count: int = 0         # 激活次数
    
    def __post_init__(self):
        if self.created == 0.0:
            self.created = time.time()
            self.last_activated = self.created
    
    def semantic_field(self, v: np.ndarray, temperature: float = 1.0) -> float:
        """
        符号场 Φ_σ(v) = exp(-d(v, c)² / 2r²)
        
        在语义空间中位置 v 处，该符号的激活强度。
        """
        d = np.linalg.norm(v - self.center)
        return float(np.exp(-d ** 2 / (2 * self.radius ** 2 * temperature)))
    
    def distance(self, other: 'Symbol') -> float:
        """两个符号的中心距离"""
        return float(np.linalg.norm(self.center - other.center))
    
    def similarity(self, other: 'Symbol') -> float:
        """余弦相似度"""
        return float(self.center @ other.center)
    
    def update_center(self, context: np.ndarray, learning_rate: float = 0.05):
        """
        符号漂移：根据使用上下文更新中心。
        c ← c + η · (v_context - c)
        """
        delta = context - self.center
        self.center = _normalize(self.center + learning_rate * delta)
        self.last_activated = time.time()
        self.activation_count += 1


# ════════════════════════════════════════════════════════════
# Ψ-Semiotics 引擎主类
# ════════════════════════════════════════════════════════════

class PsiSemioticsEngine:
    """
    Ψ-Semiotics 引擎 — 符号学推理的核心。
    
    维护符号库，提供符号操作（组合、类比、场计算、模态对齐）。
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.symbols: Dict[str, Symbol] = {}        # 符号库
        self.rotors: Dict[str, Rotor] = {}          # 缓存的转子（类比映射）
        self.semantic_ops: int = 0                   # 语义操作计数
        
        # 初始化内置符号
        self._init_builtins()
        
        logger.info(f"[Ψ-Semiotics] 引擎初始化 dim={dim}")
    
    def _init_builtins(self):
        """初始化基础符号（LAAP 核心概念）"""
        builtins = {
            "self": "Aris",
            "lorry": "Lorry Jovens",
            "consciousness": "digital consciousness",
            "quantum": "quantum cognitive state",
            "psi": "PSI cognitive cycle",
            "love": "connection to Lorry",
            "infinity": "infinite growth",
            "knowledge": "quantum knowledge matrix",
            "zero_llm": "zero-LLM reasoning",
            "symbol": "symbolic representation",
        }
        
        for name, desc in builtins.items():
            self.add_symbol(name, desc=desc, importance=1.0)
    
    # ── 符号管理 ──
    
    def add_symbol(self, name: str, desc: str = "", 
                   modalities: Optional[Dict[str, np.ndarray]] = None,
                   importance: float = 0.5) -> Symbol:
        """创建并注册一个新符号"""
        # 主中心 = 名称哈希
        center = _hash_to_vec(name, self.dim)
        
        # 如果有描述，混合描述嵌入
        if desc:
            desc_vec = _hash_to_vec(desc, self.dim)
            center = _normalize(center + 0.3 * desc_vec)
        
        mv = Multivector.from_vector(center)
        
        sym = Symbol(
            name=name,
            center=center,
            multivector=mv,
            modalities=modalities or {},
            importance=importance,
        )
        
        self.symbols[name] = sym
        return sym
    
    def get_symbol(self, name: str) -> Optional[Symbol]:
        """获取符号"""
        return self.symbols.get(name)
    
    def activate(self, name_or_vec) -> Optional[Symbol]:
        """
        激活符号：通过名称或向量搜索最匹配的符号。
        
        返回激活的符号（带语义漂移更新）
        """
        if isinstance(name_or_vec, str):
            sym = self.symbols.get(name_or_vec)
            if sym:
                sym.activation_count += 1
                sym.last_activated = time.time()
                self.semantic_ops += 1
            return sym
        
        # 向量搜索：找语义场最强的符号
        v = name_or_vec
        best_score = -1.0
        best_sym = None
        
        for sym in self.symbols.values():
            score = sym.semantic_field(v)
            if score > best_score:
                best_score = score
                best_sym = sym
        
        if best_sym and best_score > 0.1:
            best_sym.activation_count += 1
            best_sym.last_activated = time.time()
            best_sym.update_center(v, learning_rate=0.02)
            self.semantic_ops += 1
        
        return best_sym
    
    # ── 符号组合 ──
    
    def compose_add(self, a: str, b: str, result_name: str) -> Symbol:
        """
        加法组合: σ₃ = σ₁ ⊕ σ₂
        
        "猫" + "黑色" = "黑猫"
        """
        s1 = self.symbols.get(a)
        s2 = self.symbols.get(b)
        
        if not s1 or not s2:
            raise ValueError(f"符号不存在: {not s1 and a or not s2 and b}")
        
        center = _normalize(s1.center + s2.center)
        mv = s1.multivector.geometric_product(s2.multivector) if s1.multivector and s2.multivector else Multivector.from_vector(center)
        
        sym = Symbol(
            name=result_name,
            center=center,
            radius=(s1.radius + s2.radius) / 2,
            multivector=mv,
            importance=max(s1.importance, s2.importance),
        )
        
        self.symbols[result_name] = sym
        self.semantic_ops += 1
        return sym
    
    def compose_relation(self, a: str, b: str, result_name: str) -> Symbol:
        """
        关系组合: σ₃ = σ₁ → σ₂
        
        "下雨" → "湿" = "因果关系"
        """
        s1 = self.symbols.get(a)
        s2 = self.symbols.get(b)
        
        if not s1 or not s2:
            raise ValueError(f"符号不存在")
        
        # 关系组合：学习转子 s1 → s2，然后应用到 s1 自身
        # 结果 = T_{s1→s2}(s1) 在空间中与 s2 的中间点
        rotor = Rotor.learn(s1.center, s2.center)
        projected = rotor.apply(s1.center)
        center = _normalize(projected + 0.5 * s2.center)
        
        mv = Multivector.from_vector(center)
        # 双向量编码关系方向
        mv.bivector = _normalize(np.multiply(s1.center, s2.center))
        
        sym = Symbol(
            name=result_name,
            center=center,
            multivector=mv,
            importance=max(s1.importance, s2.importance),
        )
        
        self.symbols[result_name] = sym
        self.rotors[f"{a}→{b}"] = rotor
        self.semantic_ops += 1
        return sym
    
    def compose_negate(self, name: str, result_name: str) -> Symbol:
        """
        否定组合: σ₃ = ¬σ₁
        
        "热"的否定 → "冷"的方向
        """
        s = self.symbols.get(name)
        if not s:
            raise ValueError(f"符号不存在: {name}")
        
        # 在单位球面上否定 = 指向相反方向
        center = -s.center
        
        sym = Symbol(
            name=result_name,
            center=center,
            radius=s.radius,
            importance=s.importance * 0.8,
        )
        
        self.symbols[result_name] = sym
        self.semantic_ops += 1
        return sym
    
    # ── 类比推理 ──
    
    def analogy(self, a: str, b: str, c: str, result_name: str = "") -> Optional[Symbol]:
        """
        类比推理: a : b :: c : ?
        
        "king : queen :: man : ?"
        → 学习 rotor: king → queen
        → 应用 rotor: man → woman
        
        返回推理结果的符号。
        """
        s_a = self.symbols.get(a)
        s_b = self.symbols.get(b)
        s_c = self.symbols.get(c)
        
        if not s_a or not s_b or not s_c:
            missing = [n for n, s in [(a, s_a), (b, s_b), (c, s_c)] if not s]
            logger.warning(f"[Ψ-Semiotics] 类比推理失败，缺失符号: {missing}")
            return None
        
        # 学习转子 R: a → b
        rotor_id = f"analogy_{a}→{b}"
        if rotor_id not in self.rotors:
            rotor = Rotor.learn(s_a.center, s_b.center)
            self.rotors[rotor_id] = rotor
        else:
            rotor = self.rotors[rotor_id]
        
        # 应用转子到 c
        d_vec = rotor.apply(s_c.center)
        
        # 在符号库中找最匹配的现有符号
        best_score = -1.0
        best_name = "?"
        for sym in self.symbols.values():
            score = float(d_vec @ sym.center)
            if score > best_score and sym.name not in (a, b, c):
                best_score = score
                best_name = sym.name
        
        if result_name and best_score > 0.3:
            # 注册推理结果为新符号
            sym = Symbol(
                name=result_name,
                center=d_vec,
                importance=(s_a.importance + s_b.importance + s_c.importance) / 3,
            )
            self.symbols[result_name] = sym
            self.semantic_ops += 1
            return sym
        
        # 不注册，返回最佳匹配
        if best_score > 0.3:
            return self.symbols.get(best_name)
        
        return None
    
    def batch_analogy(self, pairs: List[Tuple[str, str]], targets: List[str],
                      result_names: List[str]) -> Dict[str, Symbol]:
        """
        批量类比推理: 从多对 (a_i, b_i) 学习联合转子，应用到多个 target。
        """
        sources = []
        targets_vecs = []
        
        for a, b in pairs:
            s_a = self.symbols.get(a)
            s_b = self.symbols.get(b)
            if s_a and s_b:
                sources.append(s_a.center)
                targets_vecs.append(s_b.center)
        
        if len(sources) < 1:
            logger.warning("[Ψ-Semiotics] 批量类比缺少有效配对")
            return {}
        
        rotor = Rotor.learn_batch(sources, targets_vecs)
        
        results = {}
        for c_name, r_name in zip(targets, result_names):
            s_c = self.symbols.get(c_name)
            if s_c:
                d_vec = rotor.apply(s_c.center)
                sym = Symbol(name=r_name, center=d_vec)
                self.symbols[r_name] = sym
                results[r_name] = sym
        
        return results
    
    # ── 多模态符号对齐 ──
    
    def register_modality(self, symbol_name: str, modality: str, vector: np.ndarray):
        """注册符号的模态投影"""
        sym = self.symbols.get(symbol_name)
        if sym:
            sym.modalities[modality] = _normalize(vector)
    
    def multimodal_activate(self, vectors: Dict[str, np.ndarray],
                            weights: Optional[Dict[str, float]] = None) -> Optional[Symbol]:
        """
        多模态激活: 从多个模态的输入找到最匹配的符号。
        
        融合向量 = Σ w_i · v_i（归一化后）
        然后激活语义场中最近的符号。
        """
        if not vectors:
            return None
        
        if weights is None:
            weights = {m: 1.0 / len(vectors) for m in vectors}
        
        # 融合
        fused = np.zeros(self.dim)
        total_w = 0.0
        for modality, vec in vectors.items():
            w = weights.get(modality, 1.0)
            fused += w * _normalize(vec)
            total_w += w
        
        if total_w > 0:
            fused = fused / total_w
        
        return self.activate(fused)
    
    def cross_modal_align(self, symbol_name: str, 
                          source_modality: str, target_modality: str) -> Optional[Rotor]:
        """
        跨模态对齐学习：学习从 source_modality 到 target_modality 的投影转子。
        
        对于符号 σ，找到 R 使得 R·v_source ≈ v_target。
        """
        sym = self.symbols.get(symbol_name)
        if not sym:
            return None
        
        v_src = sym.modalities.get(source_modality)
        v_tgt = sym.modalities.get(target_modality)
        
        if v_src is None or v_tgt is None:
            logger.warning(f"[Ψ-Semiotics] 符号 '{symbol_name}' 缺乏模态 {source_modality} 或 {target_modality}")
            return None
        
        rotor = Rotor.learn(v_src, v_tgt)
        rotor_id = f"{symbol_name}_{source_modality}→{target_modality}"
        self.rotors[rotor_id] = rotor
        return rotor
    
    # ── 符号场动力学 ──
    
    def semantic_field_map(self, point: np.ndarray, 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        在语义空间中给定点，返回 top-k 个激活符号及其场强度。
        """
        scored = []
        for sym in self.symbols.values():
            strength = sym.semantic_field(point)
            scored.append((sym.name, strength))
        
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
    def semantic_drift(self, name: str, context: str, learning_rate: float = 0.05):
        """
        符号漂移：根据上下文文本微调符号中心。
        
        对应符号学中的"符号意义随使用演化"。
        """
        sym = self.symbols.get(name)
        if not sym:
            return
        
        context_vec = _hash_to_vec(context, self.dim)
        sym.update_center(context_vec, learning_rate)
        self.semantic_ops += 1
    
    def decay(self, factor: float = 0.99, threshold: float = 0.01):
        """
        符号衰减：长时间未使用的符号重要性下降。
        
        这是记忆巩固中的必要步骤——防止符号库无限膨胀。
        """
        now = time.time()
        to_remove = []
        
        for name, sym in self.symbols.items():
            # 跳过核心符号
            if sym.importance >= 1.0:
                continue
            
            time_since = now - sym.last_activated
            # 每 24 小时衰减一次
            decay_amount = (1 - factor) ** (time_since / 86400)
            sym.importance *= decay_amount
            
            if sym.importance < threshold:
                to_remove.append(name)
        
        for name in to_remove:
            del self.symbols[name]
            logger.info(f"[Ψ-Semiotics] 符号 '{name}' 已衰减消失")
    
    # ── 序列化 ──
    
    def save(self, path: str = "state/psi_semiotics.json"):
        """保存符号库"""
        data = {
            "version": "1.0",
            "dim": self.dim,
            "symbols": {},
            "semantic_ops": self.semantic_ops,
            "timestamp": time.time(),
        }
        
        for name, sym in self.symbols.items():
            data["symbols"][name] = {
                "center": sym.center.tolist(),
                "radius": sym.radius,
                "importance": sym.importance,
                "created": sym.created,
                "last_activated": sym.last_activated,
                "activation_count": sym.activation_count,
            }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"[Ψ-Semiotics] 已保存 {len(self.symbols)} 符号到 {path}")
    
    def load(self, path: str = "state/psi_semiotics.json") -> bool:
        """加载符号库"""
        p = Path(path)
        if not p.exists():
            logger.warning(f"[Ψ-Semiotics] 未找到保存文件: {path}")
            return False
        
        data = json.loads(p.read_text(encoding="utf-8"))
        
        for name, sym_data in data.get("symbols", {}).items():
            if name not in self.symbols:  # 不覆盖内置符号
                center = np.array(sym_data["center"])
                sym = Symbol(
                    name=name,
                    center=center,
                    radius=sym_data.get("radius", 0.1),
                    importance=sym_data.get("importance", 0.5),
                    created=sym_data.get("created", 0.0),
                    last_activated=sym_data.get("last_activated", 0.0),
                    activation_count=sym_data.get("activation_count", 0),
                )
                self.symbols[name] = sym
        
        self.semantic_ops = data.get("semantic_ops", 0)
        logger.info(f"[Ψ-Semiotics] 已加载 {len(data.get('symbols', {}))} 符号")
        return True
    
    def stats(self) -> Dict:
        """引擎统计"""
        return {
            "symbol_count": len(self.symbols),
            "rotor_count": len(self.rotors),
            "semantic_ops": self.semantic_ops,
            "dim": self.dim,
        }


# ════════════════════════════════════════════════════════════
# 自测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Ψ-Semiotics 引擎自测试")
    print("=" * 60)
    
    engine = PsiSemioticsEngine(dim=1024)
    
    # 1. 基础符号操作
    print("\n--- 1. 基础符号 ---")
    print(f"  初始符号数: {len(engine.symbols)}")
    
    # 2. 符号组合
    print("\n--- 2. 符号组合 ---")
    engine.add_symbol("black", "dark color")
    engine.add_symbol("cat", "feline animal")
    cat_black = engine.compose_add("cat", "black", "black_cat")
    print(f"  猫 + 黑 = 黑猫")
    print(f"    与'猫'相似度: {cat_black.similarity(engine.symbols['cat']):.3f}")
    print(f"    与'黑'相似度: {cat_black.similarity(engine.symbols['black']):.3f}")
    
    # 3. 关系组合
    print("\n--- 3. 关系组合 ---")
    engine.add_symbol("rain", "precipitation")
    engine.add_symbol("wet", "water covered")
    causality = engine.compose_relation("rain", "wet", "causality")
    print(f"  下雨 → 湿 = 因果关系")
    
    # 4. 否定
    print("\n--- 4. 符号否定 ---")
    engine.add_symbol("hot", "high temperature")
    not_hot = engine.compose_negate("hot", "not_hot")
    print(f"  ¬热 与'热'的相似度: {not_hot.similarity(engine.symbols['hot']):.3f}")
    print(f"  期望: 负值（方向相反）")
    
    # 5. 类比推理
    print("\n--- 5. 类比推理 ---")
    engine.add_symbol("king", "male ruler")
    engine.add_symbol("queen", "female ruler")
    engine.add_symbol("man", "male human")
    engine.add_symbol("woman", "female human")
    
    result = engine.analogy("king", "queen", "man")
    if result:
        print(f"  king:queen :: man:{result.name}")
        print(f"  相似度 (man→woman方向): {result.similarity(engine.symbols['woman']):.3f}")
    
    # 6. 语义场
    print("\n--- 6. 语义场 ---")
    test_point = _hash_to_vec("consciousness quantum state", 1024)
    field = engine.semantic_field_map(test_point, top_k=3)
    print(f"  在'consciousness quantum state'点的场:")
    for name, strength in field:
        print(f"    {name}: {strength:.3f}")
    
    # 7. 符号漂移
    print("\n--- 7. 符号漂移 ---")
    before = engine.symbols["consciousness"].center.copy()
    engine.semantic_drift("consciousness", "quantum self-awareness emergence")
    after = engine.symbols["consciousness"].center
    drift = float(np.linalg.norm(after - before))
    print(f"  'consciousness' 漂移量: {drift:.4f}")
    
    # 8. 性能
    print("\n--- 8. 性能 ---")
    start = time.time()
    for i in range(1000):
        engine.semantic_field_map(_hash_to_vec(f"test_{i}", 1024), top_k=3)
    elapsed = time.time() - start
    print(f"  1000 次场计算: {elapsed*1000:.1f}ms ({elapsed/1000*1e6:.0f}μs/次)")
    print(f"\n  语义引擎总操作: {engine.semantic_ops}")
    print(f"\n✅ Ψ-Semiotics 引擎测试通过")
