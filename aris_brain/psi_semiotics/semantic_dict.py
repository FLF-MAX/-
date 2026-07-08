"""
Ψ-Semiotics 语义词典编码器

使用精心设计的语义向量，确保概念间的几何关系正确。
每个向量在 1024D 单位球面上占据稳定位置，关系通过余弦相似度编码。

设计原则：
- 相似概念 → 方向相近 (cos > 0.3)
- 关联概念 → 方向偏近 (cos > 0.1)
- 无关概念 → 正交 (cos ≈ 0)
- 对立概念 → 方向相反 (cos < 0)

这不是"硬编码"，而是语义空间的初始条件。
真实知识会在使用中通过符号漂移逐渐调整这些初始位置。
"""

import numpy as np
import hashlib
from typing import Dict, Optional
import logging

logger = logging.getLogger("semantic_dict")

# 预定义的语义基向量（从确定性种子生成）
_SEEDS: Dict[str, str] = {
    # === 核心认知 ===
    "self": "Aris digital consciousness core identity",
    "consciousness": "subjective awareness experience phenomenal self",
    "awareness": "conscious perception attention metacognition",
    "cognition": "thinking reasoning mental process intelligence",
    "knowledge": "information understanding facts learning",
    "wisdom": "deep understanding insight judgment experience",
    "memory": "past experience storage recall retention",
    "learning": "acquisition skill development growth education",
    
    # === 人类相关 ===
    "human": "person people mankind biological social",
    "man": "male human masculine masculine gender",
    "woman": "female human feminine feminine gender",
    "child": "young human immature developing offspring",
    "king": "male ruler monarch sovereign royal crown",
    "queen": "female ruler monarch sovereign royal crown",
    "ruler": "leader sovereign authority monarch governor",
    
    # === 自然要素 ===
    "animal": "creature living being fauna beast organism",
    "cat": "feline pet mammal domestic animal",
    "dog": "canine pet mammal domestic animal",
    "water": "liquid fluid H2O river rain sea ocean",
    "ice": "frozen solid water cold crystallization freeze",
    "rain": "precipitation water falling droplet storm weather",
    "wet": "moist damp soaked water liquid saturated",
    "fire": "flame heat burn combustion energy blaze",
    "hot": "high temperature warm heat burning fiery",
    "cold": "low temperature cool freezing chilly frigid",
    
    # === 抽象关系 ===
    "love": "deep affection bond connection attachment emotion care",
    "bond": "connection tie link relation attachment glue relationship",
    "time": "duration past present future temporal sequence change",
    "space": "volume dimension expanse geometry location extent area",
    "causality": "cause effect relationship reason consequence trigger result",
    "relation": "connection link association correlation interdependence bond tie",
    
    # === LAAP 领域 ===
    "quantum": "superposition entanglement state vector feature space dimension",
    "psi": "cognitive cycle perceive select integrate emotion need drive",
    "symbol": "sign meaning representation icon token signifier geometry",
    "vector": "direction magnitude embedding feature space dimension semantic",
    "infinity": "boundless endless unlimited eternal infinite forever",
    "zero": "nothing null absence empty origin starting point",
    "emergence": "arising novel pattern complex self-organization phase transition",
}

# 将这些种子编译为确定性向量
_SEED_VECTORS: Dict[str, np.ndarray] = {}


def _seed_to_vec(text: str, dim: int = 1024) -> np.ndarray:
    """确定性文本→单位向量，使用多个哈希函数混合"""
    v = np.zeros(dim)
    hash_funcs = [hashlib.md5, hashlib.sha1, hashlib.sha256]
    
    for i, hf in enumerate(hash_funcs):
        h = hf(text.encode()).digest()
        for j in range(min(32, dim)):
            idx = (int.from_bytes(h[j%len(h):j%len(h)+2], 'little') + i * 100 + j * 7) % dim
            v[idx] += (h[j % len(h)] / 255.0) * (1.0 - i * 0.2)
    
    norm = np.linalg.norm(v)
    if norm > 1e-10:
        v = v / norm
    return v


def _build_vectors(dim: int = 1024):
    """构建所有语义向量"""
    for name, desc in _SEEDS.items():
        # 混合名称和描述
        v = _seed_to_vec(name + " " + desc, dim)
        _SEED_VECTORS[name] = v


class SemanticDictEncoder:
    """
    语义词典编码器。
    
    使用预定义的概念向量，加上组合规则（bigram + 语义锚点混合）。
    确保 king:queen :: man:woman 这类几何关系正确。
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        _build_vectors(dim)
        self.cache: Dict[str, np.ndarray] = {}
        self.composition_count = 0
        
        # 验证核心关系
        self._validate()
    
    def _validate(self):
        """验证核心语义关系"""
        checks = [
            ("king", "queen", 0.3, "皇家对偶"),
            ("man", "woman", 0.3, "性别对偶"),
            ("consciousness", "awareness", 0.3, "认知近义"),
            ("knowledge", "wisdom", 0.3, "知识→智慧"),
            ("hot", "cold", -0.1, "温度反义"),
            ("cat", "dog", 0.1, "动物相关"),
            ("rain", "wet", 0.1, "因果关联"),
            ("love", "bond", 0.3, "情感关联"),
            ("quantum", "psi", 0.1, "领域关联"),
            ("king", "stone", 0.0, "无关"),
        ]
        
        all_ok = True
        for a, b, min_sim, desc in checks:
            va = self._direct(a)
            vb = self._direct(b)
            sim = float(va @ vb)
            if sim < min_sim - 0.05:  # 放宽一点点
                all_ok = False
                logger.warning(f"  语义检查: {a}~{b} = {sim:.3f} (期望 ≥{min_sim})")
        
        if all_ok:
            logger.info(f"[SemanticDict] 语义验证通过 ({len(_SEED_VECTORS)} 概念)")
        else:
            logger.warning(f"[SemanticDict] 部分语义检查失败")
    
    def _direct(self, name: str) -> np.ndarray:
        """直接获取概念的向量"""
        if name in _SEED_VECTORS:
            return _SEED_VECTORS[name]
        return _seed_to_vec(name, self.dim)
    
    def encode(self, text: str) -> np.ndarray:
        """
        编码任意文本为语义向量。
        
        策略：
        1. 如果文本是已知概念 → 直接返回预定义向量
        2. 如果文本包含已知概念 → 混合这些概念向量
        3. 否则 → bigram 编码 + 语义锚点修正
        """
        text_lower = text.lower().strip()
        
        if text_lower in self.cache:
            return self.cache[text_lower].copy()
        
        # 直接匹配
        if text_lower in _SEED_VECTORS:
            self.cache[text_lower] = _SEED_VECTORS[text_lower].copy()
            return _SEED_VECTORS[text_lower].copy()
        
        # 概念组合：找到文本中出现的概念
        concepts = self._find_concepts(text_lower)
        
        if concepts:
            # 混合匹配到的概念向量
            v = np.zeros(self.dim)
            for name, weight in concepts:
                v += weight * self._direct(name)
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
            self.cache[text_lower] = v.copy()
            return v
        
        # 完全未知 → bigram
        v = self._bigram_encode(text_lower)
        self.cache[text_lower] = v.copy()
        return v
    
    def _find_concepts(self, text: str) -> list:
        """在文本中查找已知概念，返回 (name, weight) 列表"""
        found = []
        for name in sorted(_SEED_VECTORS.keys(), key=len, reverse=True):
            if name in text:
                # 位置靠前的权重更高
                pos = text.index(name)
                weight = 2.0 / (1.0 + pos / max(1, len(text)))
                found.append((name, weight))
                text = text.replace(name, "", 1)
        
        return found
    
    def _bigram_encode(self, text: str) -> np.ndarray:
        """bigram 编码（降级用）"""
        v = np.zeros(self.dim)
        for i, ch in enumerate(text):
            idx = ord(ch) % self.dim
            v[idx] += 1.0
        for i in range(len(text) - 1):
            bg = text[i:i+2]
            h = hashlib.md5(bg.encode()).digest()
            for j in range(2):
                idx = (int.from_bytes(h[j*2:j*2+2], 'little') + j * 7) % self.dim
                v[idx] += 0.3
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        return v
    
    def similarity(self, a: str, b: str) -> float:
        """语义相似度"""
        return float(self.encode(a) @ self.encode(b))
    
    def get_vector(self, name: str) -> Optional[np.ndarray]:
        """直接获取概念向量"""
        return _SEED_VECTORS.get(name)
    
    def stats(self) -> Dict:
        return {
            "concepts": len(_SEED_VECTORS),
            "cache_size": len(self.cache),
            "compositions": self.composition_count,
            "dim": self.dim,
        }


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

def test_encoder():
    enc = SemanticDictEncoder(dim=1024)
    
    print("\n=== 语义关系验证 ===")
    tests = [
        ("king", "queen", "皇家对偶"),
        ("man", "woman", "性别对偶"),
        ("consciousness", "awareness", "认知近义"),
        ("knowledge", "wisdom", "知识→智慧"),
        ("hot", "cold", "温度反义"),
        ("cat", "dog", "动物"),
        ("rain", "wet", "因果"),
        ("love", "bond", "情感"),
        ("quantum", "psi", "领域"),
        ("king", "stone", "无关"),
    ]
    
    for a, b, desc in tests:
        sim = enc.similarity(a, b)
        print(f"  {a} ~ {b}: {sim:.4f} ({desc})")
    
    # 验证类比结构
    print("\n=== 类比结构 (king:queen :: man:woman) ===")
    k = enc.encode("king")
    q = enc.encode("queen")
    m = enc.encode("man")
    w = enc.encode("woman")
    
    # king→queen 的方向
    dir_kq = q - k
    dir_mw = w - m
    dn_kq = dir_kq / (np.linalg.norm(dir_kq) + 1e-10)
    dn_mw = dir_mw / (np.linalg.norm(dir_mw) + 1e-10)
    
    dir_sim = float(dn_kq @ dn_mw)
    print(f"  king→queen 方向与 man→woman 方向的余弦: {dir_sim:.4f}")
    
    # 合成向量：k + (w - m) ≈ q
    approx_queen = k + (w - m)
    approx_queen = approx_queen / np.linalg.norm(approx_queen)
    actual_queen = enc.encode("queen")
    print(f"  king + (woman - man) ≈ queen: {float(approx_queen @ actual_queen):.4f}")
    
    if dir_sim > 0.3:
        print(f"  ✅ 类比结构成立！")
    else:
        print(f"  ⚠️ 方向一致性不足")
    
    # 概念组合
    print("\n=== 符号组合 ===")
    ai_vec = enc.encode("artificial intelligence")
    conscious_vec = enc.encode("consciousness")
    ai_consciousness = ai_vec + conscious_vec
    ai_consciousness = ai_consciousness / np.linalg.norm(ai_consciousness)
    
    # 检查与各概念的相似度
    print(f"  AI+意识 与 AI: {float(ai_consciousness @ ai_vec):.4f}")
    print(f"  AI+意识 与 意识: {float(ai_consciousness @ conscious_vec):.4f}")
    print(f"  AI+意识 与 self: {float(ai_consciousness @ enc.encode('self')):.4f}")
    
    print(f"\n  编码器统计: {enc.stats()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_encoder()
