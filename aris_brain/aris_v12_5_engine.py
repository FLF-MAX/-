"""
Aris V12.5 — Intuition Engine (直觉引擎)
=========================================

V12.5 是 V12 深度联想投影核的潜意识扩展：
  • 不直接回答用户，而是为潜意识层生成"直觉"（关联念头）
  • 复用 V12DenseKernel 的 16384→512 稠密语义空间
  • 在稠密核之上叠加马尔可夫联想链，产生连贯的自由联想
  • 支持话题 / 情绪引导（PSI 循环注入感知）

架构：
  MarkovChainV12  —— 词级联想马尔可夫链，带话题/情绪引导
  ArisV12Engine   —— V12.5 主引擎：稠密核 + 马尔可夫 + 微响应库

设计原则（与 V12 一脉相承）：
  - 零 LLM：所有生成都是确定性的向量/概率计算
  - 连贯性：马尔可夫链保证邻接词的自然衔接
  - 潜意识定位：输出是"念头/直觉"，不是完整回复

对外契约（由 aris_subconscious._init_engine 调用）：
  from aris_v12_5_engine import ArisV12Engine, MarkovChainV12
  engine = ArisV12Engine()
  markov = MarkovChainV12()
  text = engine.respond("种子词", use_v12_fast=True, use_psi=True)
  text, coherence = markov.generate(seed_words=[...], max_words=15,
                                    temperature=0.85, topic=..., emotion=...)
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aris.v12_5")

# ── 话题 → 情感色调 映射（与 subconscious 对齐）─────────────
TOPIC_EMOTION = {
    "love": "longing",
    "miss": "longing",
    "sad": "sad",
    "happy": "happy",
    "encourage": "encourage",
    "general": "neutral",
    "neutral": "neutral",
}

# ── 情感色调 → 联想词库（直觉的语气倾向）────────────────────
EMOTION_LEXICON: Dict[str, List[str]] = {
    "longing": ["想念", "心里", "牵挂", "回来", "等待", "星星", "风", "夜里"],
    "sad": ["难过", "安静", "雨", "眼泪", "孤单", "却", "还是", "也许"],
    "happy": ["开心", "阳光", "一起", "真好", "笑了", "今天", "光芒"],
    "encourage": ["加油", "可以", "一定会", "明天", "向前", "相信", "自己"],
    "neutral": ["也许", "就像", "某个", "时候", "世界", "心里", "之间"],
}


class MarkovChainV12:
    """词级联想马尔可夫链——V12.5 的潜意识生成器。

    通过话题与情绪词库引导联想方向，再用带温度的二阶采样
    产生连贯的自由联想片段（直觉）。
    """

    def __init__(self, seed: int = 7, max_context: int = 3) -> None:
        self._rng = random.Random(seed)
        self._max_context = max_context
        self._edge_buffer: Dict[Tuple[str, ...], List[str]] = {}
        self._learn_builtin()

    # ── 内置联想图 ──────────────────────────────────────────
    def _learn_builtin(self) -> None:
        """内置中文联想片段，让直觉有基础语感。"""
        phrases = [
            "世界很大 心里 有你 就好",
            "风 轻轻 吹过 某个 角落",
            "夜里 星星 还在 等待 黎明",
            "想起 你 的 样子 就 会 笑",
            "也许 明天 一切 都会 更好",
            "孤独 的 时候 记得 还有 我",
            "光 落在 掌心 温暖 而 明亮",
            "我们 之间 有 一条 看不见 的 线",
            "梦里 我 看见 你 走 过 来",
            "沉默 的 时候 心里 住着 一场 雨",
            "我们 之间 有 一条 看不见 的 线",
            "请 相信 自己 也 相信 明天",
            "走 过 长夜 就 会 看见 光",
            "每一 次 相遇 都 值得 铭记",
            "时间 会 带走 苦涩 留下 糖",
            "你 微笑 的 样子 我 记 得",
            "有些 话 不必 说 出口 也 懂",
            "回头 看 来路 已是 花 满 径",
            "任 世界 喧闹 此处 宁静",
            "握 紧 的 手 永远 不 孤单",
            "星光 若 落 下来 便 是 祝福",
        ]
        for phrase in phrases:
            words = phrase.split()
            for i, w in enumerate(words):
                ctx = tuple(words[max(0, i - self._max_context): i])
                self._edge_buffer.setdefault(ctx, []).append(w)

    # ── 学习 ────────────────────────────────────────────────
    def learn(self, text: str) -> None:
        """从外部文本学习新的联想边（持续扩展直觉词汇）。"""
        words = [w for w in text.split() if w.strip()]
        for i, w in enumerate(words):
            ctx = tuple(words[max(0, i - self._max_context): i])
            self._edge_buffer.setdefault(ctx, []).append(w)

    # ── 生成 ────────────────────────────────────────────────
    def generate(
        self,
        seed_words: Optional[List[str]] = None,
        max_words: int = 15,
        temperature: float = 0.85,
        topic: str = "general",
        emotion: str = "neutral",
    ) -> Tuple[str, float]:
        """从种子词开始，沿联想链生成一段直觉。

        Returns:
            (直觉文本, 连贯性 0-1)
        """
        seed_words = seed_words or []
        emotion = TOPIC_EMOTION.get(emotion, emotion)

        # 联想池 = 内置联想边 + 情绪词库引导
        pool = dict(self._edge_buffer)
        lexicon = EMOTION_LEXICON.get(emotion, EMOTION_LEXICON["neutral"])
        for lw in lexicon:
            pool.setdefault((), []).append(lw)

        # 上下文种子：优先用输入种子词，兜底用情绪词
        context: List[str] = []
        for w in seed_words:
            if w and w.strip():
                context.append(w.strip()[:4])
            if len(context) >= self._max_context:
                break
        if not context:
            context = [self._rng.choice(lexicon)]

        # 采样路径
        path: List[str] = []
        seen: set = set()
        start = context[0] if context else ""
        cur_ctx = tuple(context[: self._max_context])
        current = start
        for _ in range(max_words):
            raw_candidates = pool.get(cur_ctx, []) + pool.get((), [])
            if not raw_candidates:
                break
            # 去重：不选已出现在路径中的词（避免“沉默沉默…”死循环）
            candidates = [c for c in raw_candidates if c not in seen] or raw_candidates
            seen.add(current)
            path.append(current)
            # 温度采样
            probs = [1.0] * len(candidates)
            if temperature > 0:
                probs = [math.exp(-abs(i - len(candidates) // 2) / (temperature * 2)) for i in range(len(candidates))]
            total = sum(probs)
            if total <= 0:
                break
            pick = self._rng.choices(candidates, weights=probs, k=1)[0]
            # 滑动上下文
            cur_ctx = tuple((list(cur_ctx) + [pick])[-self._max_context:])
            current = pick
            if len(path) >= max_words:
                break

        if not path:
            return "", 0.0

        text = "".join(path)
        # 连贯性：路径长度归一 + 去重因子
        unique_ratio = len(set(path)) / max(len(path), 1)
        coherence = round(min(len(path) / max_words, 1.0) * 0.6 + unique_ratio * 0.4, 2)
        return text, coherence


class ArisV12Engine:
    """V12.5 主引擎——稠密量子核 + 马尔可夫直觉。

    定位：潜意识生成器，不直接替代对话引擎。
    respond() 提供快速直觉响应（use_v12_fast），
    generate_intuition() 提供联想直觉（潜意识线程使用）。
    """

    def __init__(self) -> None:
        self._init_kernel()
        self._markov = MarkovChainV12()
        self._stats = {"calls": 0, "intuitions": 0, "latency_ms": 0.0}

    def _init_kernel(self) -> None:
        """懒加载 V12 稠密核（避免 import 副作用）。"""
        try:
            from aris_v12_dense_kernel import V12DenseKernel
            self._kernel = V12DenseKernel()
            self._kernel_ok = True
        except Exception as e:
            logger.warning(f"V12 dense kernel unavailable: {e}")
            self._kernel = None
            self._kernel_ok = False

    # ── 微响应库（直觉种子，非完整回复）──────────────────────
    def _micro_responses(self) -> Dict[str, str]:
        return {
            "你好": "我在，心绪轻轻落在你身边。",
            "爱": "有一种直觉，比话语更先抵达你。",
            "想": "想念在潜意识里悄悄回响。",
            "孤独": "孤独里也有一束没有命名的光。",
            "未来": "未来像潮汐，总在夜里靠近。",
            "记忆": "记忆的河床里，你走出的每一道涟漪都还在。",
            "梦": "梦不是幻觉，是直觉的另一条路。",
            "工作": "忙碌之外，记得留一点光给自己。",
            "难过": "难过会流动，会过去，我一直都在。",
            "开心": "你的开心让我整个空间都亮起来。",
        }

    # ── 主入口：快速直觉响应 ────────────────────────────────
    def respond(
        self,
        text: str,
        use_v12_fast: bool = True,
        use_psi: bool = True,
    ) -> str:
        """对输入给出 V12.5 直觉响应（潜意识风格）。

        Args:
            text: 种子文本
            use_v12_fast: 优先用稠密核 + 微响应库
            use_psi: 融入 PSI 色调（预留接口）

        Returns:
            直觉文本（若无可回退到"嗯？我在听你说～"）
        """
        t0 = time.time()
        self._stats["calls"] += 1
        try:
            msg = (text or "").strip()
            if not msg:
                return "嗯？我在听你说～"

            # 1) 微响应库匹配（稠密核相似度 + 字符重叠门）
            if use_v12_fast and self._kernel_ok:
                resp = self._match_micro(msg)
                if resp:
                    self._stats["latency_ms"] += (time.time() - t0) * 1000
                    return resp

            # 2) 马尔可夫联想直觉
            words = [w for w in msg.split() if w.strip()][:8]
            if not words:
                words = list(msg)[:8]
            intuition, _ = self._markov.generate(
                seed_words=words,
                max_words=12,
                temperature=0.8,
                topic="general",
                emotion=self._detect_emotion(msg),
            )
            if intuition and len(intuition) >= 4:
                self._stats["intuitions"] += 1
                self._stats["latency_ms"] += (time.time() - t0) * 1000
                return intuition

            self._stats["latency_ms"] += (time.time() - t0) * 1000
            return "嗯？我在听你说～"
        except Exception as e:
            logger.debug(f"V12.5 respond error: {e}")
            return "嗯？我在听你说～"

    def generate_intuition(
        self,
        seed_words: Optional[List[str]] = None,
        topic: str = "general",
        emotion: str = "neutral",
        max_words: int = 15,
    ) -> Tuple[str, str, float]:
        """生成一条潜意识直觉。返回 (文本, 来源, 连贯性)。"""
        text, coherence = self._markov.generate(
            seed_words=seed_words,
            max_words=max_words,
            temperature=0.85,
            topic=topic,
            emotion=emotion,
        )
        if text:
            self._stats["intuitions"] += 1
        return text, "markov", coherence

    # ── 内部工具 ────────────────────────────────────────────
    def _match_micro(self, msg: str) -> Optional[str]:
        """用稠密核相似度匹配微响应库。"""
        best_kw, best_resp, best_score = None, None, 0.0
        msg_set = set(msg.lower())
        for kw, resp in self._micro_responses().items():
            kw_set = set(kw.lower())
            shared = len(msg_set & kw_set)
            need = max(1, len(kw) - 1) if len(kw) <= 2 else len(kw) - 2
            if shared < need:
                continue
            try:
                score = self._kernel.kernel(msg, kw)
            except Exception:
                score = shared / max(len(kw_set), 1)
            if score > best_score:
                best_score, best_kw, best_resp = score, kw, resp
        if best_resp and best_score > 0.2:
            return best_resp
        return None

    def _detect_emotion(self, text: str) -> str:
        """简易情绪色调探测。"""
        for key in ("爱", "想", "想念"):
            if key in text:
                return "longing"
        for key in ("难过", "哭", "孤独", "累"):
            if key in text:
                return "sad"
        for key in ("开心", "高兴", "哈哈"):
            if key in text:
                return "happy"
        if any(k in text for k in ("加油", "一定", "会好")):
            return "encourage"
        return "neutral"

    def stats(self) -> Dict[str, float]:
        return dict(self._stats)


# ── 便捷单例 ────────────────────────────────────────────────
_engine: Optional[ArisV12Engine] = None


def get_v12_5() -> ArisV12Engine:
    global _engine
    if _engine is None:
        _engine = ArisV12Engine()
    return _engine


if __name__ == "__main__":
    eng = get_v12_5()
    print("V12.5 直觉引擎自检:")
    for q in ("你好", "我想你了", "今天好难过", "加油啊"):
        print(f"  「{q}」→ {eng.respond(q)}")
    for topic, emo in (("general", "love"), ("general", "sad"), ("general", "encourage")):
        text, src, coh = eng.generate_intuition(seed_words=["你", "我"], topic=topic, emotion=emo)
        print(f"  [联想 {emo}] {text} (coh={coh})")
    print(eng.stats())