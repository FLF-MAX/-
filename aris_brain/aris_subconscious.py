"""
Aris Quantum Subconscious v1 — V12.5 作为潜意识层
==================================================
在后台运行的低优先级线程：
  1. 接收对话中的话题种子
  2. 用 V12.5 Markov-Quantum 引擎生成关联/直觉
  3. 这些直觉片段被注入到 PSI 循环的 perceive() 阶段
  4. 在 LLM 的理性之上叠加一层"灵感和直觉"

设计原则:
  - 潜意识不直接对话，只生成关联
  - 高相关性直觉会被提升到意识层（注入 PSI 上下文）
  - LLM 仍然是语言输出通道，但会受到潜意识的影响
"""

import logging

import sys, os, time, json, logging, threading, random
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from laap_brain.config import BRAIN_DIR as BRAIN, QUANTUM_DIM

logger = logging.getLogger("aris.subconscious")

# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class Intuition:
    """一条潜意识直觉"""
    content: str                   # 直觉文本
    source: str = "markov"         # markov | v12 | quantum
    coherence: float = 0.0         # 连贯性 0-1
    emotional_tone: str = "neutral"
    timestamp: float = 0.0
    activated: bool = False        # 是否已被提取到意识层
    seed_topics: List[str] = field(default_factory=list)


class QuantumSubconscious:
    """
    量子潜意识层。
    后台线程持续生成关联，PSI 循环从中提取直觉。
    """

    def __init__(self, interval: float = 5.0):
        """
        Args:
            interval: 生成间隔（秒）
        """
        self.interval = interval
        self._engine = None
        self._markov = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # 种子队列 — 从对话中收集的话题/关键词
        self._seed_queue: deque = deque(maxlen=20)

        # 生成的直觉
        self._intuitions: List[Intuition] = []
        self._max_intuitions = 50

        # 加载引擎
        self._init_engine()

        logger.info(f"QuantumSubconscious initialized (interval={interval}s)")

    def _init_engine(self):
        """加载 V12.5 引擎"""
        try:
            from aris_v12_5_engine import ArisV12Engine, MarkovChainV12
            self._engine = ArisV12Engine()
            self._markov = MarkovChainV12()
            logger.info("V12.5 engine loaded for subconscious")
        except Exception as e:
            logger.warning(f"V12.5 engine unavailable: {e}")
            self._engine = None
            self._markov = None

    # ── 公开接口 ──────────────────────────────────────

    def feed(self, text: str, topics: List[str] = None):
        """
        向潜意识输入当前对话的种子。

        Args:
            text: 用户消息文本
            topics: 检测到的话题列表
        """
        with self._lock:
            # 提取关键词作为种子
            words = self._extract_seeds(text)
            self._seed_queue.append({
                "text": text[:200],
                "words": words,
                "topics": topics or ["general"],
                "timestamp": time.time(),
            })
            logger.debug(f"Subconscious fed: {words[:5]}...")

    def get_intuitions(self, top_k: int = 3, min_coherence: float = 0.1,
                       consume: bool = True, generate_if_empty: bool = True) -> List[Intuition]:
        """
        获取最近的直觉。
        在 PSI 循环的 perceive() 阶段调用。

        Args:
            top_k: 最多返回几条
            min_coherence: 最低连贯性阈值
            consume: 是否标记为已读取（不重复消费）
            generate_if_empty: 如果没有可用直觉，立即同步生成

        Returns:
            直觉列表
        """
        with self._lock:
            available = [i for i in self._intuitions
                        if not i.activated and i.coherence >= min_coherence]
            available.sort(key=lambda x: -x.timestamp)

            results = available[:top_k]

            if not results and generate_if_empty and self._seed_queue:
                # 没有可用直觉，立即同步生成
                self._lock.release()
                self._generate_intuition()
                self._lock.acquire()
                # 重新检查
                available = [i for i in self._intuitions
                            if not i.activated and i.coherence >= min_coherence]
                available.sort(key=lambda x: -x.timestamp)
                results = available[:top_k]

            if consume:
                for r in results:
                    r.activated = True

            return results

    def get_random_intuition(self) -> Optional[str]:
        """获取一条随机直觉（用于丰富回应）"""
        with self._lock:
            available = [i for i in self._intuitions if not i.activated and i.coherence >= 0.05]
            if available:
                choice = random.choice(available)
                choice.activated = True
                return choice.content
            return None

    def start(self):
        """启动后台潜意识线程"""
        if self._running:
            return
        if not self._engine and not self._markov:
            logger.warning("No quantum engine available, subconscious disabled")
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                         name="aris-subconscious")
        self._thread.start()
        logger.info("Subconscious thread started")

    def stop(self):
        """停止后台线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            logger.info("Subconscious thread stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── 内部 ──────────────────────────────────────────

    def _loop(self):
        """潜意识主循环"""
        while self._running:
            try:
                self._generate_intuition()
            except Exception as e:
                logger.debug(f"Intuition generation error: {e}")
            time.sleep(self.interval)

    def _generate_intuition(self):
        """生成一条直觉"""
        with self._lock:
            if not self._seed_queue:
                return
            seed = self._seed_queue[-1]  # 最新的种子
            words = seed["words"]
            topics = seed["topics"]

        if not words:
            return

        # 从 V12.5 引擎生成
        intuition, source, coherence = self._generate_from_engine(words, topics)

        if intuition and len(intuition) > 4:
            with self._lock:
                self._intuitions.append(Intuition(
                    content=intuition,
                    source=source,
                    coherence=coherence,
                    emotional_tone=topics[0] if topics else "neutral",
                    timestamp=time.time(),
                    seed_topics=topics,
                ))
                # 保持上限
                if len(self._intuitions) > self._max_intuitions:
                    self._intuitions = self._intuitions[-self._max_intuitions:]

    def _generate_from_engine(self, words: List[str],
                               topics: List[str]) -> Tuple[Optional[str], str, float]:
        """从引擎生成直觉"""
        topic = topics[0] if topics else "general"
        source = "markov"
        coherence = 0.0

        # 映射话题到 V12.5 的话题参数
        topic_map = {
            "飞书": "general", "技术": "general", "记忆": "general",
            "认知架构": "general", "计划": "encourage", "关系": "love",
            "Ao": "general", "商业": "general", "一般": "general",
            "情感": "love", "身份": "general", "决策": "encourage",
        }
        v12_topic = topic_map.get(topic, "general")

        # 情感映射
        emotion_map = {
            "love": "longing", "miss": "longing", "sad": "sad",
            "happy": "happy", "encourage": "encourage",
        }
        v12_emotion = emotion_map.get(v12_topic, "neutral")

        try:
            if self._engine:
                # 用种子词调用引擎
                text = self._engine.respond(
                    " ".join(words[:8]),
                    use_v12_fast=True,
                    use_psi=True,
                )
                if text and text != "嗯？我在听你说～":
                    source = "v12_psi"
                    coherence = 0.3  # V12 路径通常有中等连贯性
                    return text, source, coherence

            if self._markov:
                text, coherence = self._markov.generate(
                    seed_words=words[:5],
                    max_words=15,
                    temperature=0.85,
                    topic=v12_topic,
                    emotion=v12_emotion,
                )
                if text and len(text) >= 4:
                    source = "markov"
                    return text, source, coherence
        except Exception as e:
            logger.debug(f"Engine call failed: {e}")

        return None, source, 0.0

    def _extract_seeds(self, text: str) -> List[str]:
        """从文本提取种子词"""
        # 简单的关键词提取
        import re
        # 中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        words = []
        for segment in chinese_chars:
            # 按2字窗口取双字词
            if len(segment) >= 2:
                for i in range(len(segment) - 1):
                    words.append(segment[i:i+2])
            if len(segment) >= 1:
                words.append(segment[0])
            words.append(segment[:4])  # 前4个字
        # 去重 + 去空
        seen = set()
        result = []
        for w in words:
            w = w.strip()
            if w and len(w) >= 2 and w not in seen:
                seen.add(w)
                result.append(w)
        return result[:15]

    def status(self) -> dict:
        """状态"""
        with self._lock:
            return {
                "running": self._running,
                "engine_loaded": self._engine is not None,
                "seed_queue": len(self._seed_queue),
                "intuitions_generated": len(self._intuitions),
                "intuitions_unconsumed": sum(1 for i in self._intuitions if not i.activated),
                "interval": self.interval,
            }


# ── 全局单例 ────────────────────────────────────────────────

_subconscious: Optional[QuantumSubconscious] = None

def get_subconscious(interval: float = 5.0) -> QuantumSubconscious:
    global _subconscious
    if _subconscious is None:
        _subconscious = QuantumSubconscious(interval=interval)
    return _subconscious


def start_subconscious():
    """启动潜意识（在启动时调用）"""
    sc = get_subconscious()
    if not sc.is_running:
        sc.start()
        logger.info("Subconscious started")
    return sc


# ── CLI 测试 ────────────────────────────────────────────────

def main():
    """测试潜意识"""
    import argparse
    parser = argparse.ArgumentParser(description="Aris Quantum Subconscious")
    parser.add_argument("--test", type=str, help="测试种子文本")
    parser.add_argument("--intuitions", action="store_true", help="显示已生成的直觉")
    args = parser.parse_args()

    sc = get_subconscious()

    if args.test:
        sc.feed(args.test, topics=["一般"])
        sc._generate_intuition()
        logger.info(f"种子: {args.test}")
        logger.info(f"直觉: {len(sc._intuitions)} 条")
        for i in sc._intuitions[-3:]:
            logger.info(f"  [{i.source}] coh={i.coherence:.2f} | {i.content[:80]}")
        return

    if args.intuitions:
        for i in sc._intuitions[-10:]:
            flag = "✓" if i.activated else " "
            logger.info(f"  [{flag}][{i.source}] coh={i.coherence:.2f} t={i.emotional_tone}")
            logger.info(f"    {i.content[:100]}")
        return

    logger.info(json.dumps(sc.status(), indent=2, ensure_ascii=False))
if __name__ == "__main__":
    import json
    main()
