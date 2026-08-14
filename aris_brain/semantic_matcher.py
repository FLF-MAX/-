"""
Aris Semantic Matcher — 双编码语义匹配器
=========================================
把「字形核（V12 稠密投影）」与「语义联想核（ConceptGraph 结构嵌入）」融合，
解决零字重叠的近义查询（如"摔倒了没力气" → "累了就休息"）。

两路编码互补：
  - 字形路 (V12DenseKernel): 字符级 JL 随机投影，字面重叠时最可靠
  - 语义路 (ConceptGraph): 同义/上下位/反义/情感效价结构嵌入，
    覆盖无字重叠但有概念关联的查询

匹配策略：
  1. 分词 → 命中概念 → 语义向量（平均概念嵌入）
  2. 字形相似度与语义相似度加权融合：
      语义强(概念覆盖>0)时语义为主；字形覆盖时字形补位
  3. 情感效价对齐加分，反义惩罚
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from aris_v12_dense_kernel import V12DenseKernel
from aris_lm_v5 import ChineseTokenizer, ConceptGraph

# 语言正则
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALNUM_RE = re.compile(r"[a-z0-9]+")


class SemanticMatcher:
    """双编码语义匹配器。"""

    # 语义权重：概念覆盖率高（语义可信）时，语义主导
    SEMANTIC_BASE_W = 0.55
    GLYPH_BASE_W = 0.45
    # 反义惩罚 / 情感对齐加成
    ANTONYM_PENALTY = 0.25
    VALENCE_BONUS = 0.15
    # 高频功能词（程度副词/语气词/泛化形容词）：参与字形路但不污染语义向量
    FUNCTION_WORDS = frozenset({
        "好", "真", "好", "很", "太", "挺", "非常", "特别", "真的",
        "了", "啊", "呀", "嘛", "呢", "吧", "哦", "嗯",
    })
    # 指代词/称呼：语义信息量低，降权，避免"我爱你~你是谁"被"你"抬高
    LOW_WEIGHT_WORDS = frozenset({
        "你", "我", "他", "她", "它", "我们", "你们", "他们", "自己",
        "谁", "什么", "哪", "哪里", "这", "那",
    })

    def __init__(self, glyph: Optional[V12DenseKernel] = None):
        self.glyph = glyph or V12DenseKernel()
        self.tokenizer = ChineseTokenizer()
        self.concepts = ConceptGraph()

    # ── 编码 ──────────────────────────────────────────────
    def sentence_vector(self, text: str) -> Optional[np.ndarray]:
        """语义向量：分词 → 概念嵌入**加权平均**。

        指代词（你/我/宝贝等）信息量低，权重 0.2；实义词权重 1.0。
        这样"我爱你"与"你是谁"共享"你"不再主导相似度，
        情感/行为等实义词成为语义主体。功能词完全过滤。
        """
        tokens = [t.text for t in self.tokenizer.tokenize(text)]
        hit_tokens = [t for t in tokens
                      if t.strip()
                      and t not in self.FUNCTION_WORDS
                      and self.concepts.lookup(t) is not None]
        if not hit_tokens:
            return None
        weights = [0.2 if t in self.LOW_WEIGHT_WORDS else 1.0
                   for t in hit_tokens]
        wsum = sum(weights)
        if wsum <= 0:
            return None
        vec = sum(self.concepts.lookup(t).embedding * w
                  for t, w in zip(hit_tokens, weights)) / wsum
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-10 else None

    # 否定词：翻转后续情感极性（"我不开心" = 负）
    NEGATORS = frozenset({"不", "没", "别", "不要", "并非", "不是"})

    def _valence(self, text: str) -> float:
        """文本情感效价：命中概念的平均 valence，否定词翻转极性。

        功能词与否定词本身不贡献数值；"开心"前面出现"不/没/别"时整体取反
        （"我不开心" → -1.0），避免把负向情绪误配给正向模板回复。
        """
        tokens = [t.text for t in self.tokenizer.tokenize(text)]
        negated = False
        vals = []
        for t in tokens:
            if not t.strip() or t in self.FUNCTION_WORDS:
                continue
            if t in self.NEGATORS:
                negated = True
                continue
            node = self.concepts.lookup(t)
            if node is None:
                continue
            w = 0.2 if t in self.LOW_WEIGHT_WORDS else 1.0
            vals.append(node.valence * w)
        if not vals:
            return 0.0
        total = sum(vals) / sum(1 for v in vals)
        return -total if negated else total

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        va = self.sentence_vector(text_a)
        vb = self.sentence_vector(text_b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb))

    def glyph_similarity(self, text_a: str, text_b: str) -> float:
        try:
            return self.glyph.kernel(text_a, text_b)
        except Exception:
            return 0.0

    # ── 融合相似度 ────────────────────────────────────────
    def similarity(self, text_a: str, text_b: str) -> float:
        """双编码融合相似度 [-1, 1]。

        语义权重随概念覆盖率动态调整：两方都有概念时语义主导；
        任一方概念缺失时字形保底。
        """
        sem = self._semantic_similarity(text_a, text_b)
        gly = self.glyph_similarity(text_a, text_b)

        va = self.sentence_vector(text_a)
        vb = self.sentence_vector(text_b)
        covered = (va is not None) and (vb is not None)

        w_sem = self.SEMANTIC_BASE_W if covered else 0.0
        w_gly = self.GLYPH_BASE_W + (self.SEMANTIC_BASE_W if not covered else 0.0)
        total = w_sem + w_gly
        w_sem, w_gly = w_sem / total, w_gly / total

        score = w_sem * sem + w_gly * max(0.0, gly)

        # 共享概念词直奖：两文本共现的**概念命中词**（功能词过滤）直接加分，
        # 覆盖"双方都在说同一件事"的情况。
        ta = {t.text for t in self.tokenizer.tokenize(text_a)
              if t.text.strip()
              and t.text not in self.FUNCTION_WORDS
              and self.concepts.lookup(t.text) is not None}
        tb = {t.text for t in self.tokenizer.tokenize(text_b)
              if t.text.strip()
              and t.text not in self.FUNCTION_WORDS
              and self.concepts.lookup(t.text) is not None}
        shared = ta & tb
        if shared:
            score += 0.15 * min(1.0, len(shared))

        # 情感效价对齐：双方都强极性(|v|>0.5)且相反才惩罚，同为强极性才加分
        va_l = self._valence(text_a)
        vb_l = self._valence(text_b)
        if va_l and vb_l and abs(va_l) > 0.5 and abs(vb_l) > 0.5:
            if va_l * vb_l < 0:
                score -= self.ANTONYM_PENALTY
            elif va_l * vb_l > 0:
                score += self.VALENCE_BONUS

        return float(max(-1.0, min(1.0, score)))

    # ── 库匹配 ────────────────────────────────────────────
    def best_match(
        self, query: str, candidates: Dict[str, str],
        threshold: float = 0.35,
    ) -> Tuple[Optional[str], float]:
        """在候选回复库中找语义最优匹配。

        Args:
            query: 用户查询
            candidates: {关键词: 回复} 映射
            threshold: 最低融合相似度（低于则视为无匹配）

        Returns:
            (命中关键词, 融合分)；无匹配返回 (None, 0.0)
        """
        best_kw, best_score = None, 0.0
        q_valence = self._valence(query)
        for kw in candidates:
            s = self.similarity(query, kw)
            kw_valence = self._valence(kw)
            # 情感极性否决：查询强负情感(|v|>0.4)时，不匹配正向模板回复
            # （如"我不开心" 不能回 "开心：相干态稳定"）
            if q_valence < -0.4 and kw_valence > 0.4 and len(kw) <= 4:
                s -= 0.5
            if s > best_score:
                best_kw, best_score = kw, s
        if best_score >= threshold:
            return best_kw, best_score
        return None, best_score


# 模块级单例
_MATCHER: Optional[SemanticMatcher] = None


def get_matcher() -> SemanticMatcher:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = SemanticMatcher()
    return _MATCHER


def dual_similarity(text_a: str, text_b: str) -> float:
    """双编码融合相似度（模块级便捷函数）。"""
    return get_matcher().similarity(text_a, text_b)


# ══════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    m = get_matcher()

    print("=" * 66)
    print("SemanticMatcher 双编码自测")
    print("=" * 66)
    pairs = [
        ("想你了", "我也想你"),
        ("想你了", "今天天气不错"),
        ("我爱你", "我也爱你"),
        ("我爱你", "今天天气不错"),
        ("晚安宝贝", "晚安好梦"),
        ("我回来了", "欢迎回家"),
        ("下雨了", "外面在下雨"),
        ("下雨了", "我饿了"),
        ("好开心啊", "今天真快乐"),
        ("好开心啊", "好难过"),
        ("工作好累", "累了就休息"),
        ("摔倒了", "我要睡觉"),
        ("我要睡觉了", "今天股票大涨"),
        ("吃饭了没", "你吃了吗"),
    ]
    for a, b in pairs:
        print(f"  {m.similarity(a, b): .3f}  {a} ~ {b}")
