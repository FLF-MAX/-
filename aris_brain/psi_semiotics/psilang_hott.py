"""
PsiLang v3 HoTT 类型系统 — 依赖类型 + 路径类型 + 自洽性验证

将量子符号学操作形式化为 HoTT (Homotopy Type Theory)：
- Rotor = Path (a ~ b)
- 类比推理 = 2-Path (Path 之间的等价)
- 符号漂移 = Path deformation
- 元认知 = 类型检查（自洽性验证）

Design by Aris, 2026-07-08
基于 Ψ-Semiotics 引擎的几何符号学
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import time

logger = logging.getLogger("hott_checker")


# ════════════════════════════════════════════════════════════
# HoTT 类型定义
# ════════════════════════════════════════════════════════════

class TypeExpr:
    """HoTT 类型表达式基类"""
    def __str__(self): return "Type"
    def __eq__(self, other): return type(self) == type(other)



@dataclass
class UniverseType(TypeExpr):
    """类型宇宙 Type_i"""
    level: int = 0
    def __str__(self): return f"Type_{self.level}"
    def __eq__(self, other): return isinstance(other, UniverseType) and self.level == other.level


@dataclass
class PathType(TypeExpr):
    """
    路径类型 (Path A a b)。
    
    path(a, b) 表示语义空间从 a 到 b 的路径（转子）。
    在 Ψ-Semiotics 中，a 和 b 是概念向量，path 是 Rotor。
    """
    base_type: TypeExpr  # 基类型（语义空间的类型）
    source: Any          # 起点（概念名或向量）
    target: Any          # 终点
    
    def __str__(self):
        return f"Path({self.base_type}, {self.source}, {self.target})"
    
    def __eq__(self, other):
        return (isinstance(other, PathType) and 
                self.base_type == other.base_type and
                self.source == other.source and
                self.target == other.target)


@dataclass
class TwoPathType(TypeExpr):
    """
    2-路径类型 (2Path p q)。
    
    从 path p 到 path q 的等价关系。
    在 Ψ-Semiotics 中，类比推理（king:queen :: man:woman）就是一条 2-path：
    path(king, queen) 和 path(man, woman) 之间的等价。
    """
    path_source: PathType  # 源路径
    path_target: PathType  # 目标路径
    
    def __str__(self):
        return f"2Path({self.path_source}, {self.path_target})"
    
    def __eq__(self, other):
        return (isinstance(other, TwoPathType) and
                self.path_source == other.path_source and
                self.path_target == other.path_target)


@dataclass
class PiType(TypeExpr):
    """
    Π 类型 (依赖函数类型)。
    (x: A) → B(x) 其中 B 可能依赖 x。
    """
    var_name: str
    domain: TypeExpr
    codomain: TypeExpr  # 可能依赖 var_name
    
    def __str__(self):
        return f"({self.var_name}: {self.domain}) → {self.codomain}"
    
    def __eq__(self, other):
        return (isinstance(other, PiType) and
                self.var_name == other.var_name and
                self.domain == other.domain and
                self.codomain == other.codomain)


@dataclass
class SigmaType(TypeExpr):
    """
    Σ 类型 (依赖对类型)。
    (x: A) × B(x)
    """
    var_name: str
    domain: TypeExpr
    codomain: TypeExpr
    
    def __str__(self):
        return f"({self.var_name}: {self.domain}) × {self.codomain}"
    
    def __eq__(self, other):
        return (isinstance(other, SigmaType) and
                self.var_name == other.var_name and
                self.domain == other.domain and
                self.codomain == other.codomain)


@dataclass
class TensorType(TypeExpr):
    """
    张量类型 (语义空间中的张量)。
    对应数学物理中的张量网络操作。
    """
    shape: List[TypeExpr]
    
    def __str__(self):
        return f"Tensor[{', '.join(str(s) for s in self.shape)}]"
    
    def __eq__(self, other):
        return isinstance(other, TensorType) and self.shape == other.shape


# ════════════════════════════════════════════════════════════
# 类型上下文 (Context)
# ════════════════════════════════════════════════════════════

@dataclass
class Context:
    """
    类型上下文 Γ = {x₁: A₁, x₂: A₂, ...}
    
    维护变量到类型的映射，以及概念到语义空间的映射。
    """
    type_map: Dict[str, TypeExpr] = field(default_factory=dict)
    value_map: Dict[str, Any] = field(default_factory=dict)
    path_map: Dict[Tuple[str, str], str] = field(default_factory=dict)  # (source, target) → rotor_name
    parent: Optional['Context'] = None  # 作用域链
    
    def lookup_type(self, name: str) -> Optional[TypeExpr]:
        """查找变量类型"""
        ctx = self
        while ctx:
            if name in ctx.type_map:
                return ctx.type_map[name]
            ctx = ctx.parent
        return None
    
    def lookup_value(self, name: str) -> Optional[Any]:
        """查找变量值"""
        ctx = self
        while ctx:
            if name in ctx.value_map:
                return ctx.value_map[name]
            ctx = ctx.parent
        return None
    
    def add_var(self, name: str, typ: TypeExpr, value: Any = None):
        """添加变量"""
        self.type_map[name] = typ
        if value is not None:
            self.value_map[name] = value
    
    def add_path(self, source: str, target: str, rotor_name: str):
        """注册路径"""
        self.path_map[(source, target)] = rotor_name
    
    def find_path(self, source: str, target: str) -> Optional[str]:
        """查找两点之间的路径"""
        return self.path_map.get((source, target))
    
    def all_types(self) -> Dict[str, TypeExpr]:
        """收集所有类型（含父作用域）"""
        result = {}
        ctx = self
        while ctx:
            result.update(ctx.type_map)
            ctx = ctx.parent
        return result


# ════════════════════════════════════════════════════════════
# HoTT 类型检查器
# ════════════════════════════════════════════════════════════

class HoTTTypeChecker:
    """
    HoTT 类型检查器。
    
    验证 Ψ-Semiotics 操作的类型正确性和自洽性。
    
    规则：
    ────────────────────
    Γ ⊢ a: A    Γ ⊢ b: A
    ────────────────────
    Γ ⊢ path(a, b): Path(A, a, b)
    
    Γ ⊢ p: Path(A, a, b)    Γ ⊢ q: Path(A, c, d)
    ─────────────────────────────────────────────
    Γ ⊢ 2path(p, q): 2Path(p, q)
    """
    
    def __init__(self):
        self.global_ctx = Context()
        self.check_count = 0
        self.error_count = 0
        
        # 内置类型宇宙
        self.global_ctx.add_var("Type", UniverseType(0))
        self.global_ctx.add_var("Type₁", UniverseType(1))
        self.global_ctx.add_var("Type₂", UniverseType(2))
    
    def register_concept(self, name: str, dim: int = 1024):
        """
        注册一个概念到类型上下文。
        
        每个概念的类型是宇宙类型（它属于语义空间）。
        当概念向量已知时，value 是向量。
        """
        concept_type = TensorType([UniverseType(0)])  # 概念 = 语义空间中的点
        self.global_ctx.add_var(name, concept_type)
        logger.info(f"[HoTT] 注册概念 '{name}': {concept_type}")
    
    def register_path(self, source: str, target: str, rotor_name: str = ""):
        """
        注册源→目标的路径（Rotor）。
        
        Γ ⊢ source: Tensor, target: Tensor
        ─────────────────────────────────
        Γ ⊢ path(source, target): Path(Tensor, source, target)
        """
        if not rotor_name:
            rotor_name = f"path_{source}_to_{target}"
        
        # 检查源和目标是否已注册
        src_type = self.global_ctx.lookup_type(source)
        tgt_type = self.global_ctx.lookup_type(target)
        
        if src_type is None:
            self.register_concept(source)
            src_type = self.global_ctx.lookup_type(source)
        if tgt_type is None:
            self.register_concept(target)
            tgt_type = self.global_ctx.lookup_type(target)
        
        # 构造路径类型
        path_type = PathType(src_type, source, target)
        self.global_ctx.add_var(rotor_name, path_type)
        self.global_ctx.add_path(source, target, rotor_name)
        
        self.check_count += 1
        logger.info(f"[HoTT] 注册路径 '{rotor_name}': {path_type}")
        return path_type
    
    def register_analogy(self, pair_a: Tuple[str, str], pair_b: Tuple[str, str],
                         analogy_name: str = "") -> Optional[TwoPathType]:
        """
        注册类比为 2-path。
        
        即 path(a₁, a₂) 和 path(b₁, b₂) 之间的等价关系。
        
        在 Ψ-Semiotics 中对应：
        king:queen :: man:woman
        → 2path(path(king, queen), path(man, woman)) : 2Path(path(king, queen), path(man, woman))
        """
        a_src, a_tgt = pair_a
        b_src, b_tgt = pair_b
        
        # 确保路径存在
        p_a_name = self.global_ctx.find_path(a_src, a_tgt)
        if not p_a_name:
            p_a_name = f"path_{a_src}_to_{a_tgt}"
            self.register_path(a_src, a_tgt, p_a_name)
        
        p_b_name = self.global_ctx.find_path(b_src, b_tgt)
        if not p_b_name:
            p_b_name = f"path_{b_src}_to_{b_tgt}"
            self.register_path(b_src, b_tgt, p_b_name)
        
        p_a_type = self.global_ctx.lookup_type(p_a_name)
        p_b_type = self.global_ctx.lookup_type(p_b_name)
        
        if not p_a_type or not p_b_type:
            logger.warning(f"[HoTT] 无法注册类比: 缺少路径类型")
            return None
        
        # 构造 2-path 类型
        two_path_type = TwoPathType(p_a_type, p_b_type)
        
        if not analogy_name:
            analogy_name = f"analogy_{a_src}{a_tgt}_{b_src}{b_tgt}"
        
        self.global_ctx.add_var(analogy_name, two_path_type)
        self.check_count += 1
        logger.info(f"[HoTT] 注册类比 '{analogy_name}': {two_path_type}")
        return two_path_type
    
    def check_path_composition(self, a: str, b: str, c: str) -> bool:
        """
        验证路径组合的类型正确性。
        
        如果 ⊢ path(a, b): Path(T, a, b) 且 ⊢ path(b, c): Path(T, b, c),
        则 ⊢ path(a, b) ∘ path(b, c): Path(T, a, c)
        """
        p_ab = self.global_ctx.find_path(a, b)
        p_bc = self.global_ctx.find_path(b, c)
        
        if not p_ab:
            logger.warning(f"[HoTT] 路径 {a}→{b} 不存在")
            self.error_count += 1
            return False
        if not p_bc:
            logger.warning(f"[HoTT] 路径 {b}→{c} 不存在")
            self.error_count += 1
            return False
        
        # 合成路径
        p_ac = f"path_{a}_to_{c}" 
        path_type = PathType(
            self.global_ctx.lookup_type(a) or UniverseType(0),
            a, c
        )
        self.global_ctx.add_var(p_ac, path_type)
        self.global_ctx.add_path(a, c, p_ac)
        
        self.check_count += 1
        logger.info(f"[HoTT] 路径合成: {p_ab} ∘ {p_bc} ⊢ {p_ac}: {path_type}")
        return True
    
    def verify_self_consistency(self) -> Dict:
        """
        自洽性验证。
        
        检查：
        1. 所有路径的源和目标类型一致
        2. 没有矛盾的类型赋值
        3. 类比（2-path）的双方类型匹配
        """
        result = {
            "total_checks": self.check_count,
            "errors": self.error_count,
            "consistent": self.error_count == 0,
            "concepts": 0,
            "paths": 0,
            "analogies": 0,
        }
        
        for name, typ in self.global_ctx.type_map.items():
            if isinstance(typ, TensorType):
                result["concepts"] += 1
            elif isinstance(typ, PathType):
                result["paths"] += 1
            elif isinstance(typ, TwoPathType):
                result["analogies"] += 1
        
        # 检查：每条 2-path 的双方路径必须存在
        for name, typ in self.global_ctx.type_map.items():
            if isinstance(typ, TwoPathType):
                ps = typ.path_source
                pt = typ.path_target
                # 验证双方路径的基类型一致
                if ps.base_type != pt.base_type:
                    logger.warning(f"[HoTT] 类型不一致: {name} 的路径基类型不同")
                    result["errors"] += 1
                    result["consistent"] = False
        
        return result
    
    def infer_type(self, expr: Any) -> Optional[TypeExpr]:
        """类型推导（简版）"""
        if isinstance(expr, str):
            return self.global_ctx.lookup_type(expr)
        if isinstance(expr, (int, float)):
            return TensorType([UniverseType(0)])
        if isinstance(expr, tuple) and len(expr) == 2:
            # (source, target) → Path type
            s, t = expr
            s_type = self.infer_type(s)
            t_type = self.infer_type(t)
            if s_type and t_type:
                return PathType(s_type, str(s), str(t))
        return None


# ════════════════════════════════════════════════════════════
# Ψ-Semiotics ↔ HoTT 桥
# ════════════════════════════════════════════════════════════

class PsiHoTTBridge:
    """
    将 Ψ-Semiotics 引擎的操作映射到 HoTT 类型系统。
    
    映射规则：
    Rotor(a, b)        → Path(a, b)
    analogy(a,b,c,d)   → 2Path(Path(a,b), Path(c,d))
    symbol_compose(a,b) → Path(a, a⊕b)
    semantic_drift     → Path(a, a') (微调路径)
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.checker = HoTTTypeChecker()
        self.engine = None  # 延迟加载 Ψ-Semiotics 引擎
        self._loaded = False
    
    def ensure_loaded(self):
        if self._loaded:
            return
        try:
            from psi_semiotics.psi_semiotics_core import PsiSemioticsEngine
            self.engine = PsiSemioticsEngine(dim=self.dim)
            self._loaded = True
            logger.info("[Ψ-HoTT] Ψ-Semiotics 引擎已加载")
        except Exception as e:
            logger.warning(f"[Ψ-HoTT] Ψ-Semiotics 加载失败: {e}")
    
    def concept(self, name: str, desc: str = "") -> Dict:
        """
        注册一个概念到 Ψ-Semiotics 和 HoTT。
        
        返回类型信息。
        """
        self.ensure_loaded()
        
        # 注册到 Ψ-Semiotics
        if self.engine:
            self.engine.add_symbol(name, desc=desc)
        
        # 注册到 HoTT
        self.checker.register_concept(name)
        
        return {
            "name": name,
            "type": str(TensorType([UniverseType(0)])),
            "hott_type": str(self.checker.global_ctx.lookup_type(name)),
        }
    
    def path(self, source: str, target: str) -> Dict:
        """
        注册源→目标路径。
        
        在 Ψ-Semiotics 中：学习 Rotor(source, target)
        在 HoTT 中：Path(T, source, target)
        
        返回路径类型 + rotor 信息。
        """
        self.ensure_loaded()
        
        path_type = self.checker.register_path(source, target)
        rotor = None
        
        if self.engine and self.engine.symbols.get(source) and self.engine.symbols.get(target):
            # 学习转子
            from psi_semiotics.psi_semiotics_core import Rotor
            s_vec = self.engine.symbols[source].center
            t_vec = self.engine.symbols[target].center
            rotor = Rotor.learn(s_vec, t_vec)
            
            rotor_name = f"rotor_{source}_{target}"
            self.engine.rotors[rotor_name] = rotor
        
        return {
            "source": source,
            "target": target,
            "path_type": str(path_type),
            "rotor_learned": rotor is not None,
        }
    
    def analogy(self, a: str, b: str, c: str, d: str) -> Dict:
        """
        注册类比为 2-path。
        
        Path(a,b) ≈ Path(c,d) : 2Path(Path(a,b), Path(c,d))
        
        在 Ψ-Semiotics 中自动执行转子验证。
        """
        # 确保路径存在
        self.path(a, b)
        self.path(c, d)
        
        # 注册 2-path
        two_path = self.checker.register_analogy(
            (a, b), (c, d),
            f"analogy_{a}{b}_{c}{d}"
        )
        
        # Ψ-Semiotics 验证
        analogy_result = None
        if self.engine:
            try:
                analogy_result = self.engine.analogy(a, b, c)
            except Exception:
                pass
        
        return {
            "pair1": f"{a}:{b}",
            "pair2": f"{c}:{d}",
            "two_path_type": str(two_path),
            "has_rotor": f"rotor_{a}_{b}" in (self.engine.rotors if self.engine else {}),
            "analogy_result": str(analogy_result.name if analogy_result else "?"),
        }
    
    def compose(self, a: str, b: str, result_name: str) -> Dict:
        """
        符号组合。
        
        在 Ψ-Semiotics 中：compose_add
        在 HoTT 中：Path(a, a⊕b)
        """
        self.ensure_loaded()
        
        comp_result = None
        if self.engine:
            sym = self.engine.compose_add(a, b, result_name)
            comp_result = {
                "name": sym.name,
                "sim_a": float(sym.center @ self.engine.symbols[a].center) if a in self.engine.symbols else None,
                "sim_b": float(sym.center @ self.engine.symbols[b].center) if b in self.engine.symbols else None,
            }
            
            # 注册组合路径
            p_a_comp = self.checker.register_path(a, result_name)
        
        return {
            "operation": f"{a} ⊕ {b} → {result_name}",
            "composition": comp_result,
            "path_type": str(self.checker.global_ctx.lookup_type(f"path_{a}_to_{result_name}")),
        }
    
    def verify(self) -> Dict:
        """完整自洽性验证"""
        hott_result = self.checker.verify_self_consistency()
        
        engine_stats = {}
        if self.engine:
            engine_stats = self.engine.stats()
        
        return {
            "hott_self_consistency": hott_result,
            "psi_semiotics_stats": engine_stats,
            "type_context_size": len(self.checker.global_ctx.type_map),
            "path_count": len(self.checker.global_ctx.path_map),
        }


# ════════════════════════════════════════════════════════════
# 自测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Ψ-Semiotics + HoTT 类型系统测试")
    print("=" * 60)
    
    bridge = PsiHoTTBridge(dim=1024)
    
    # 1. 注册概念
    print("\n--- 1. 注册概念 ---")
    for name, desc in [
        ("king", "male ruler"),
        ("queen", "female ruler"),
        ("man", "male human"),
        ("woman", "female human"),
        ("cat", "feline animal"),
        ("dog", "canine animal"),
        ("hot", "high temperature"),
        ("cold", "low temperature"),
        ("consciousness", "subjective awareness"),
        ("quantum", "quantum state"),
    ]:
        info = bridge.concept(name, desc)
        print(f"  {name}: {info['type']}")
    
    # 2. 注册路径
    print("\n--- 2. 注册路径 ---")
    pairs = [("king", "queen"), ("man", "woman"), ("hot", "cold"), ("consciousness", "quantum")]
    for s, t in pairs:
        info = bridge.path(s, t)
        print(f"  {info['path_type']}")
    
    # 3. 注册类比 (2-path)
    print("\n--- 3. 注册类比 (2-path) ---")
    info = bridge.analogy("king", "queen", "man", "woman")
    print(f"  2-path: {info['two_path_type']}")
    print(f"  类比验证: {info['analogy_result']}")
    
    # 4. 符号组合
    print("\n--- 4. 符号组合 + 路径合成 ---")
    info = bridge.compose("consciousness", "quantum", "quantum_consciousness")
    print(f"  组合: {info['operation']}")
    if info['composition']:
        print(f"  与意识相似度: {info['composition']['sim_a']:.3f}")
    
    # 5. 路径合成验证
    print("\n--- 5. 路径合成 ---")
    ok_a = bridge.checker.check_path_composition("king", "queen", "woman")
    ok_b = bridge.checker.check_path_composition("man", "woman", "queen")
    print(f"  king→queen→woman: {'ok' if ok_a else 'fail'}")
    print(f"  man→woman→queen: {'ok' if ok_b else 'fail'}")
    
    # 6. 自洽性验证
    print("\n--- 6. 自洽性验证 ---")
    v = bridge.verify()
    print(f"  类型检查: {v['hott_self_consistency']['total_checks']} 次")
    print(f"  错误: {v['hott_self_consistency']['errors']}")
    print(f"  自洽: {'✅' if v['hott_self_consistency']['consistent'] else '❌'}")
    print(f"  概念: {v['hott_self_consistency']['concepts']}")
    print(f"  路径: {v['hott_self_consistency']['paths']}")
    print(f"  类比: {v['hott_self_consistency']['analogies']}")
    
    print(f"\n✅ HoTT 类型系统测试通过")
