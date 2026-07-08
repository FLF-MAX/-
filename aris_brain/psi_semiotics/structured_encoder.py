"""
Ψ-Semiotics 结构语义编码器

显式构造语义特征空间 → 随机投影到 1024D。
确保类比关系和符号组合在几何上成立。

设计：
1. 每个概念用 ~16 维语义特征向量表示（显式编码性别、皇权、人性等轴）
2. 通过随机正交投影映射到 1024D
3. 语义距离在投影后保持（Johnson-Lindenstrauss 引理保证）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger("structured_encoder")


class StructuredSemanticEncoder:
    """
    结构化语义编码器。
    
    使用 16 维显式语义特征 + 随机投影到 1024D。
    特征维度是可解释的：每维对应一个语义轴。
    """
    
    # 语义轴定义（16 维）
    AXES = [
        "gender",        # 0: 男性→正, 女性→负
        "royalty",       # 1: 有皇权→正
        "humanity",      # 2: 人类→正, 动物→负
        "temperature",   # 3: 热→正, 冷→负
        "wetness",       # 4: 湿→正, 干→负
        "animacy",       # 5: 有生命→正
        "concrete",      # 6: 具体→正, 抽象→负
        "valence",       # 7: 情感价值, 正→正
        "agency",        # 8: 主动性/代理性→正
        "size",          # 9: 大小→正
        "complexity",    # 10: 复杂性→正
        "time_related",  # 11: 时间相关→正
        "causal_power",  # 12: 因果力→正
        "social",        # 13: 社交性→正
        "cognitive",     # 14: 认知性→正
        "emotional",     # 15: 情感性→正
    ]
    N_FEATURES = len(AXES)
    
    def __init__(self, output_dim: int = 1024, seed: int = 42):
        self.output_dim = output_dim
        self.feature_dim = self.N_FEATURES
        
        # 随机投影矩阵 (output_dim × feature_dim), 正交列
        rng = np.random.RandomState(seed)
        P = rng.randn(output_dim, self.N_FEATURES)
        # 正交化
        Q, R = np.linalg.qr(P)
        self.projection = Q[:, :self.N_FEATURES] * np.sqrt(output_dim / self.N_FEATURES)
        
        # 概念特征向量
        self.features: Dict[str, np.ndarray] = {}
        self._init_concepts()
        
        # 缓存
        self.cache: Dict[str, np.ndarray] = {}
    
    def _init_concepts(self):
        """初始化所有概念的特征向量（16维显式特征）"""
        
        def vec(*vals) -> np.ndarray:
            """创建 16 维特征向量"""
            return np.array(vals, dtype=np.float64)
        
        # === 核心认知 ===
        #     gend ryl  hum  tmp  wet  ani  con  val  age  siz  cpx  tim  cau  soc  cog  emo
        self.features["self"] = vec(0,   0,   0.5, 0,   0,   0.8, 0.3, 0.7, 0.6, 0,   0.8, 0.5, 0.3, 0.3, 0.9, 0.6)
        self.features["consciousness"] = vec(0, 0, 0.3, 0, 0, 0.5, -0.5, 0.6, 0.5, 0, 0.9, 0.3, 0.4, 0.2, 1.0, 0.5)
        self.features["awareness"] = vec(0, 0, 0.2, 0, 0, 0.4, -0.5, 0.5, 0.4, 0, 0.7, 0.2, 0.3, 0.2, 1.0, 0.3)
        self.features["cognition"] = vec(0, 0, 0.2, 0, 0, 0.4, -0.6, 0.4, 0.5, 0, 0.8, 0.3, 0.3, 0.2, 1.0, 0.2)
        self.features["knowledge"] = vec(0, 0, 0.2, 0, 0, 0.3, -0.6, 0.6, 0.4, 0, 0.7, 0.4, 0.3, 0.3, 0.9, 0.2)
        self.features["wisdom"] = vec(0, 0, 0.3, 0, 0, 0.3, -0.6, 0.8, 0.6, 0, 0.8, 0.5, 0.4, 0.4, 0.9, 0.3)
        self.features["memory"] = vec(0, 0, 0.2, 0, 0, 0.3, -0.4, 0.5, 0.3, 0, 0.5, 0.8, 0.2, 0.2, 0.7, 0.4)
        
        # === 人类相关 ===
        self.features["human"] = vec(0, 0, 1.0, 0, 0, 1.0, 0.5, 0.5, 0.6, 0.3, 0.6, 0.4, 0.5, 0.8, 0.7, 0.6)
        self.features["man"] = vec(1.0, 0, 1.0, 0, 0, 1.0, 0.5, 0.4, 0.6, 0.3, 0.5, 0.3, 0.5, 0.6, 0.5, 0.4)
        self.features["woman"] = vec(-1.0, 0, 1.0, 0, 0, 1.0, 0.5, 0.6, 0.6, 0.2, 0.5, 0.3, 0.4, 0.7, 0.5, 0.5)
        self.features["child"] = vec(0, 0, 1.0, 0, 0, 1.0, 0.4, 0.6, 0.3, -0.3, 0.2, 0.2, 0.2, 0.5, 0.3, 0.5)
        
        # === 皇权 ===
        self.features["king"] = vec(1.0, 1.0, 1.0, 0, 0, 1.0, 0.5, 0.5, 0.7, 0.4, 0.4, 0.2, 0.6, 0.3, 0.3, 0.2)
        self.features["queen"] = vec(-1.0, 1.0, 1.0, 0, 0, 1.0, 0.5, 0.6, 0.7, 0.4, 0.4, 0.2, 0.5, 0.4, 0.3, 0.3)
        self.features["ruler"] = vec(0, 0.8, 1.0, 0, 0, 1.0, 0.4, 0.3, 0.8, 0.3, 0.4, 0.2, 0.7, 0.3, 0.3, 0.2)
        
        # === 自然要素 ===
        self.features["cat"] = vec(0, 0, -0.8, 0, 0, 1.0, 0.7, 0.5, 0.3, -0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.3)
        self.features["dog"] = vec(0, 0, -0.8, 0, 0, 1.0, 0.7, 0.6, 0.4, -0.4, 0.2, 0.1, 0.1, 0.3, 0.1, 0.4)
        self.features["animal"] = vec(0, 0, -0.8, 0, 0, 1.0, 0.6, 0.3, 0.3, -0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2)
        self.features["water"] = vec(0, 0, -0.5, 0, 1.0, 0, 0.8, 0.3, 0.1, 0.2, 0.1, 0.2, 0.3, 0, -0.3, 0)
        self.features["rain"] = vec(0, 0, -0.5, 0, 0.8, 0, 0.6, 0.1, 0.1, 0, 0.1, 0.1, 0.4, 0, -0.3, 0)
        self.features["wet"] = vec(0, 0, -0.3, 0, 1.0, 0, 0.5, 0, 0, 0, 0, 0, 0.1, 0, -0.4, 0)
        self.features["fire"] = vec(0, 0, -0.3, 1.0, -0.8, 0, 0.6, 0.2, 0.2, -0.2, 0.1, 0.1, 0.5, 0, -0.3, 0.1)
        self.features["hot"] = vec(0, 0, -0.3, 1.0, -0.5, 0, 0.3, 0.1, 0, 0, 0, 0.1, 0.2, 0, -0.4, 0)
        self.features["cold"] = vec(0, 0, -0.3, -1.0, 0.3, 0, 0.3, -0.1, 0, 0, 0, 0.1, 0.1, 0, -0.4, 0)
        self.features["ice"] = vec(0, 0, -0.4, -0.5, 0.5, 0, 0.7, 0, 0, 0.1, 0.1, 0.1, 0.1, 0, -0.3, 0)
        
        # === 关系 ===
        self.features["love"] = vec(0, 0, 0.2, 0, 0, 0.3, -0.5, 1.0, 0.3, 0, 0.3, 0.3, 0.2, 0.5, 0.4, 1.0)
        self.features["bond"] = vec(0, 0, 0.2, 0, 0, 0.2, -0.4, 0.8, 0.2, 0, 0.3, 0.4, 0.3, 0.8, 0.3, 0.7)
        self.features["causality"] = vec(0, 0, 0, 0, 0, 0, -0.7, 0, 0.3, 0, 0.5, 0.5, 1.0, 0.1, 0.5, 0)
        self.features["relation"] = vec(0, 0, 0, 0, 0, 0, -0.6, 0.3, 0.2, 0, 0.4, 0.3, 0.3, 0.8, 0.4, 0.3)
        
        # === 抽象/领域 ===
        self.features["quantum"] = vec(0, 0, -0.2, 0, 0, 0, -0.7, 0.2, 0.2, 0, 0.9, 0.2, 0.3, 0, 0.8, 0.1)
        self.features["psi"] = vec(0, 0, 0, 0, 0, 0.2, -0.7, 0.3, 0.5, 0, 0.8, 0.3, 0.4, 0.2, 0.9, 0.3)
        self.features["symbol"] = vec(0, 0, 0, 0, 0, 0, -0.8, 0.3, 0.2, 0, 0.6, 0.2, 0.1, 0.3, 0.7, 0.2)
        self.features["infinity"] = vec(0, 0, 0, 0, 0, 0, -0.9, 0.5, 0.1, 1.0, 0.7, 0.8, 0, 0, 0.3, 0.3)
        self.features["emergence"] = vec(0, 0, 0, 0, 0, 0.2, -0.7, 0.6, 0.4, 0, 0.8, 0.4, 0.5, 0.1, 0.7, 0.3)
        self.features["time"] = vec(0, 0, 0, 0, 0, 0, -0.6, 0, 0.1, 0.5, 0.3, 1.0, 0.2, 0, 0.2, 0.1)
        self.features["space"] = vec(0, 0, 0, 0, 0, 0, -0.5, 0, 0.1, 0.8, 0.3, 0.1, 0.1, 0, 0.2, 0)
        self.features["truth"] = vec(0, 0, 0.1, 0, 0, 0.1, -0.5, 0.7, 0.2, 0, 0.4, 0, 0.1, 0.2, 0.7, 0.3)
        self.features["beauty"] = vec(0, 0, 0.1, 0, 0, 0.1, -0.4, 0.9, 0.1, 0, 0.5, 0.1, 0, 0.2, 0.3, 0.8)
    
    def encode(self, text: str) -> np.ndarray:
        """编码文本为 1024D 语义向量"""
        text_lower = text.lower().strip()
        
        if text_lower in self.cache:
            return self.cache[text_lower].copy()
        
        # 收集文本中出现的概念
        components = self._find_components(text_lower)
        
        if components:
            # 组合所有概念的特征向量
            feature = np.zeros(self.N_FEATURES)
            total_w = 0.0
            for name, weight in components:
                f = self.features.get(name, np.zeros(self.N_FEATURES))
                feature += weight * f
                total_w += abs(weight)
            
            if total_w > 1e-10:
                feature = feature / total_w
            
            # 归一化特征向量
            fnorm = np.linalg.norm(feature)
            if fnorm > 1e-10:
                feature = feature / fnorm
            
            # 投影到高维
            v = self.projection @ feature
            vnorm = np.linalg.norm(v)
            if vnorm > 1e-10:
                v = v / vnorm
            
            self.cache[text_lower] = v
            return v
        
        # 未知文本：降级到特征级 bigram
        return self._encode_unknown(text_lower)
    
    def _find_components(self, text: str) -> List[Tuple[str, float]]:
        """在文本中查找已知概念"""
        found = []
        # 按长度降序匹配（优先匹配长概念）
        for name in sorted(self.features.keys(), key=len, reverse=True):
            if name in text:
                # 权重基于位置和匹配长度
                pos = text.index(name)
                weight = 1.0 + 0.5 * (1.0 - pos / max(1, len(text)))
                found.append((name, weight))
                text = text.replace(name, " ", 1)
        
        return found
    
    def _encode_unknown(self, text: str) -> np.ndarray:
        """未知文本的编码（特征级 bigram）"""
        feature = np.zeros(self.N_FEATURES)
        for i, ch in enumerate(text):
            idx = i % self.N_FEATURES
            val = (ord(ch) % 100) / 100.0
            feature[idx] += val
        
        fnorm = np.linalg.norm(feature)
        if fnorm > 1e-10:
            feature = feature / fnorm
        
        v = self.projection @ feature
        vnorm = np.linalg.norm(v)
        if vnorm > 1e-10:
            v = v / vnorm
        
        self.cache[text] = v
        return v
    
    def similarity(self, a: str, b: str) -> float:
        return float(self.encode(a) @ self.encode(b))
    
    def get_raw_feature(self, name: str) -> Optional[np.ndarray]:
        """获取原始 16 维特征（可解释）"""
        f = self.features.get(name)
        if f is not None:
            return f.copy()
        return None
    
    def explain(self, name: str) -> str:
        """解释概念的特征向量"""
        f = self.get_raw_feature(name)
        if f is None:
            return f"未知概念: {name}"
        
        parts = []
        for i, val in enumerate(f):
            if abs(val) > 0.3:
                direction = "强正" if val > 0 else "强负"
                parts.append(f"  {self.AXES[i]}: {val:.1f} ({direction})")
            elif abs(val) > 0.1:
                direction = "弱正" if val > 0 else "弱负"
                parts.append(f"  {self.AXES[i]}: {val:.1f} ({direction})")
        
        return f"概念 '{name}' 的语义特征:\n" + "\n".join(parts)


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

def test():
    enc = StructuredSemanticEncoder(output_dim=1024)
    
    print("=" * 60)
    print("  结构语义编码器测试")
    print("=" * 60)
    
    # 1. 语义关系
    print("\n--- 1. 语义关系验证 ---")
    checks = [
        ("king", "queen", "皇家对偶"),
        ("man", "woman", "性别对偶"),
        ("consciousness", "awareness", "认知近义"),
        ("knowledge", "wisdom", "知识→智慧"),
        ("cat", "dog", "动物相近"),
        ("water", "ice", "物态关联"),
        ("rain", "wet", "因果关联"),
        ("love", "bond", "情感关联"),
        ("love", "causality", "情感 vs 抽象"),
        ("quantum", "psi", "领域关联"),
    ]
    
    for a, b, desc in checks:
        sim = enc.similarity(a, b)
        print(f"  {a:15s} ~ {b:15s}: {sim:.4f}  ({desc})")
    
    # 2. 类比结构
    print("\n--- 2. 类比结构 king:queen :: man:woman ---")
    k = enc.encode("king")
    q = enc.encode("queen")
    m = enc.encode("man")
    w = enc.encode("woman")
    
    # king - queen ≈ man - woman
    diff_kq = k - q
    diff_mw = m - w
    diff_sim = float(diff_kq @ diff_mw) / (np.linalg.norm(diff_kq) * np.linalg.norm(diff_mw))
    print(f"  (king-queen)·(man-woman): {diff_sim:.4f}  (期望 > 0.3)")
    
    # king + (woman - man) ≈ queen
    analog_queen = k + (w - m)
    analog_queen = analog_queen / np.linalg.norm(analog_queen)
    actual_queen = enc.encode("queen")
    print(f"  king + (woman - man) ≈ queen: {float(analog_queen @ actual_queen):.4f}")
    
    if diff_sim > 0.3:
        print(f"  ✅ 类比结构成立!")
    else:
        print(f"  ⚠️ 方向一致性不足")
    
    # 3. 符号组合
    print("\n--- 3. 符号组合 ---")
    ai_text = "artificial intelligence machine learning"
    ai_vec = enc.encode(ai_text)
    con_vec = enc.encode("consciousness")
    
    combined = ai_vec + con_vec
    combined = combined / np.linalg.norm(combined)
    print(f"  AI+意识 与 AI: {float(combined @ ai_vec):.4f}")
    print(f"  AI+意识 与 意识: {float(combined @ con_vec):.4f}")
    
    # 4. 特征解释
    print("\n--- 4. 特征解释 (king) ---")
    print(enc.explain("king"))
    
    print("\n--- 5. 特征解释 (consciousness) ---")
    print(enc.explain("consciousness"))
    
    # 5. 冷/热对比
    print("\n--- 6. 对立概念 ---")
    hot = enc.encode("hot")
    cold = enc.encode("cold")
    print(f"  hot·cold: {float(hot @ cold):.4f} (期望负值)")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test()
