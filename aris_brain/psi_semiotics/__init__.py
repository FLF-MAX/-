"""
Ψ-Semiotics 引擎包

首次 import 时初始化引擎，供所有 Hermes 会话使用。
"""

import sys
import os
import logging

logger = logging.getLogger("psi_semiotics")

# 确保包路径在 sys.path 中
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
    _brain_dir = os.path.dirname(_pkg_dir)
    if _brain_dir not in sys.path:
        sys.path.insert(0, _brain_dir)

# 延迟加载的引擎实例
_engine = None
_encoder = None
_cli_available = False


def get_engine(dim: int = 1024):
    """获取或初始化Ψ-Semiotics引擎"""
    global _engine
    if _engine is None:
        from psi_semiotics_core import PsiSemioticsEngine
        _engine = PsiSemioticsEngine(dim=dim)
        logger.info(f"[Ψ] 引擎初始化 dim={dim}, 符号数={len(_engine.symbols)}")
    return _engine


def get_encoder(output_dim: int = 1024):
    """获取结构化语义编码器"""
    global _encoder
    if _encoder is None:
        from structured_encoder import StructuredSemanticEncoder
        _encoder = StructuredSemanticEncoder(output_dim=output_dim)
        logger.info(f"[Ψ] 编码器初始化 dim={output_dim}")
    return _encoder


def check_cli():
    """检查CLI是否可运行"""
    global _cli_available
    try:
        from psi_semiotics_cli import cmd_verify
        _cli_available = True
        return True
    except Exception as e:
        logger.warning(f"[Ψ] CLI不可用: {e}")
        return False


def quick_verify() -> dict:
    """快速验证引擎是否正常工作"""
    eng = get_engine()
    enc = get_encoder()
    
    # 注册基础概念
    for name in ["king", "queen", "man", "woman", "consciousness", "quantum"]:
        if name not in eng.symbols:
            eng.add_symbol(name, "")
    
    k, q, m, w = [eng.symbols[n].center for n in ["king", "queen", "man", "woman"]]
    
    # 转子类比
    from psi_semiotics_core import Rotor
    rotor = Rotor.learn(k, q)
    pred = rotor.apply(m)
    analogy_strength = float(pred @ w)
    
    # 语义关系
    relations = {}
    pairs = [("king", "queen"), ("man", "woman"), ("hot", "cold"), 
             ("consciousness", "awareness"), ("cat", "dog")]
    for a, b in pairs:
        if a in eng.symbols and b in eng.symbols:
            relations[f"{a}~{b}"] = round(float(eng.symbols[a].center @ eng.symbols[b].center), 4)
    
    return {
        "engine_loaded": True,
        "symbol_count": len(eng.symbols),
        "analogy_king_queen_man_woman": round(analogy_strength, 4),
        "semantic_relations": relations,
        "cli_available": _cli_available,
    }


# 导入时快速自检
try:
    result = quick_verify()
    logger.info(f"[Ψ] 自检通过: {result['symbol_count']}符号, "
                f"类比强度={result['analogy_king_queen_man_woman']}")
except Exception as e:
    logger.warning(f"[Ψ] 自检失败: {e}")
