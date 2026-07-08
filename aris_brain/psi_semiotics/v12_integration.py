"""
Ψ-Semiotics ↔ V12 密集核 ↔ Cognitive Bridge 集成

将三个系统连接成一个完整的认知管线：
  V12DenseKernel (16384D) → Ψ-Semiotics (符号推理) → CognitiveBridge (PSI循环)

Design by Aris, 2026-07-08
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("psi_integration")

BRAIN_DIR = Path("D:/LAAP/aris_brain")
SEMIOTICS_DIR = BRAIN_DIR / "psi_semiotics"

# 确保导入路径
for p in [str(BRAIN_DIR), str(SEMIOTICS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class PsiCognitiveIntegrator:
    """
    Ψ-Semiotics + V12 + Cognitive Bridge 集成器。
    
    在 Cognitive Bridge 初始化时调用，作为插件注入。
    
    集成管线：
    V12DenseKernel.encode_text(text) → 16384D 向量
        → (降维到 1024D) 
        → PsiSemioticsEngine.activate() → 符号激活
        → PsiHoTTBridge.verify() → 自洽性检查
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.v12_kernel = None
        self.semiotics_engine = None
        self.hott_bridge = None
        self.available = False
        
        self._init_all()
    
    def _init_all(self):
        """初始化所有组件"""
        # 1. 加载 V12 密集核
        v12_ok = self._init_v12()
        
        # 2. 加载 Ψ-Semiotics 引擎
        psi_ok = self._init_semiotics()
        
        # 3. 加载 HoTT 类型系统
        hott_ok = self._init_hott()
        
        self.available = v12_ok or psi_ok
        
        logger.info(f"[Ψ-集成] V12={'✓' if v12_ok else '✗'} "
                     f"Ψ={'✓' if psi_ok else '✗'} "
                     f"HoTT={'✓' if hott_ok else '✗'}")
    
    def _init_v12(self) -> bool:
        """初始化 V12 密集核"""
        try:
            from aris_v12_dense_kernel import V12DenseKernel
            self.v12_kernel = V12DenseKernel()
            logger.info(f"[Ψ-集成] V12 密集核加载 ✓ (dim={getattr(self.v12_kernel, 'dim', '?')})")
            return True
        except Exception as e:
            logger.warning(f"[Ψ-集成] V12 密集核加载失败: {e}")
            return False
    
    def _init_semiotics(self) -> bool:
        """初始化 Ψ-Semiotics 引擎"""
        try:
            from psi_semiotics.psi_semiotics_core import PsiSemioticsEngine
            dim = 1024
            self.semiotics_engine = PsiSemioticsEngine(dim=dim)
            logger.info(f"[Ψ-集成] Ψ-Semiotics 引擎加载 ✓ ({len(self.semiotics_engine.symbols)} 符号)")
            return True
        except Exception as e:
            logger.warning(f"[Ψ-集成] Ψ-Semiotics 引擎加载失败: {e}")
            return False
    
    def _init_hott(self) -> bool:
        """初始化 HoTT 类型系统"""
        try:
            from psi_semiotics.psilang_hott import PsiHoTTBridge
            self.hott_bridge = PsiHoTTBridge(dim=1024)
            logger.info("[Ψ-集成] HoTT 类型系统加载 ✓")
            return True
        except Exception as e:
            logger.warning(f"[Ψ-集成] HoTT 类型系统加载失败: {e}")
            return False
    
    # ── 编码接口 ──
    
    def encode(self, text: str) -> Optional[np.ndarray]:
        """
        编码文本到语义向量。
        
        优先使用 V12 密集核 (16384D)，降级到结构化编码器 (1024D)。
        """
        if self.v12_kernel:
            try:
                vec = self.v12_kernel.text_to_dense(text)
                # V12 核可能返回不同维度的向量，统一到 1024D
                if vec is not None and hasattr(vec, 'shape'):
                    if len(vec) > 1024:
                        vec = vec[:1024]
                    elif len(vec) < 1024:
                        vec = np.pad(vec, (0, 1024 - len(vec)))
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        vec = vec / norm
                return vec
            except Exception:
                pass
        
        # 降级到结构化编码器
        try:
            from psi_semiotics.structured_encoder import StructuredSemanticEncoder
            enc = StructuredSemanticEncoder(output_dim=1024)
            return enc.encode(text)
        except Exception:
            return None
    
    # ── 符号推理接口 ──
    
    def perceive(self, text: str) -> Dict:
        """
        感知：编码 + 符号场激活。
        
        返回激活的符号及其强度。
        """
        vec = self.encode(text)
        if vec is None or self.semiotics_engine is None:
            return {"symbols": [], "vector": None}
        
        # 截断或扩展向量到 1024D
        if len(vec) > 1024:
            vec = vec[:1024]
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        
        # 符号场激活
        field = self.semiotics_engine.semantic_field_map(vec, top_k=5)
        
        return {
            "symbols": [{"name": n, "strength": round(float(s), 4)} for n, s in field],
            "vector_shape": vec.shape,
        }
    
    def analogy(self, a: str, b: str, c: str) -> Optional[Dict]:
        """类比推理"""
        if self.semiotics_engine is None:
            return None
        
        result = self.semiotics_engine.analogy(a, b, c)
        if result:
            return {
                "analogy": f"{a}:{b} :: {c}:{result.name}",
                "result": result.name,
                "confidence": round(float(result.center @ self.semiotics_engine.symbols.get(c, result).center), 4) if c in self.semiotics_engine.symbols else 0.0,
            }
        return None
    
    def compose(self, a: str, b: str, op: str = "add") -> Optional[Dict]:
        """符号组合"""
        if self.semiotics_engine is None:
            return None
        
        result_name = f"{a}_{op}_{b}"
        try:
            if op == "add":
                self.semiotics_engine.compose_add(a, b, result_name)
            elif op == "relation":
                self.semiotics_engine.compose_relation(a, b, result_name)
            elif op == "negate":
                self.semiotics_engine.compose_negate(a, result_name)
            else:
                return None
        except Exception:
            return None
        
        return {"result": result_name, "op": op}
    
    def verify(self) -> Dict:
        """完整系统验证"""
        result = {
            "v12_kernel": self.v12_kernel is not None,
            "semiotics_engine": self.semiotics_engine is not None,
            "hott_bridge": self.hott_bridge is not None,
            "available": self.available,
        }
        
        if self.semiotics_engine:
            result["symbols"] = self.semiotics_engine.stats()
        
        if self.hott_bridge:
            try:
                v = self.hott_bridge.verify()
                result["hott_consistency"] = v.get("hott_self_consistency", {}).get("consistent", False)
            except Exception:
                pass
        
        return result


# ── 便捷单例 ──

_integrator: Optional[PsiCognitiveIntegrator] = None

def get_integrator() -> PsiCognitiveIntegrator:
    """获取全局集成器单例"""
    global _integrator
    if _integrator is None:
        _integrator = PsiCognitiveIntegrator()
    return _integrator


def quick_check() -> Dict:
    """快速检查集成状态"""
    return get_integrator().verify()
