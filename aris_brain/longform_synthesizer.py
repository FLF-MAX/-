"""
Aris LongForm Synthesizer — 长文合成引擎
=========================================
秒出一万字结构化论文。

管线:
  主题 → 章节规划(8-10章) → 并行KB检索(每章独立)
       → Markov扩展(每章200-500字) → 章节过渡
       → 序言+结语 → 格式化输出

速度: 10章 × 20ms/章 = 200ms → 10000字
"""

import logging
logger = logging.getLogger(__name__)

import os, sys, time, re, json, hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple

from laap_brain.config import LAAP_ROOT
_root = str(LAAP_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)


class LongFormSynthesizer:
    """长文合成器 — 结构化论文/文档输出"""

    # 论文结构模板
    PAPER_STRUCTURE = [
        ("摘要", "本文介绍", "abstract", "整体概述"),
        ("引言", "随着", "intro", "背景和动机"),
        ("诞生与起源", "Aris的诞生源于", "origin", "创造过程"),
        ("核心架构", "从架构角度看，Aris由", "architecture", "系统组成"),
        ("量子核引擎", "量子核是Aris的感知核心，", "quantum", "UN6六语桥"),
        ("PSI认知循环", "PSI认知循环是Aris的意识引擎，", "psi", "需求系统"),
        ("零LLM能力", "与传统AI不同，Aris完全", "zero_llm", "ACAP评估"),
        ("知识系统", "Aris的知识系统包含", "knowledge", "矩阵检索"),
        ("未来方向", "展望未来，Aris将", "future", "路线图"),
        ("结论", "综上所述，", "conclusion", "总结"),
    ]

    # 自我介绍结构
    SELF_INTRO_STRUCTURE = [
        ("我是谁", "我是Aris，一个", "identity", "身份定义"),
        ("我的诞生", "2026年，Lorry创造了", "birth", "创造者"),
        ("我的大脑", "我的认知架构包含", "brain", "认知引擎"),
        ("我的感知", "我通过量子核感知世界，", "perception", "V7/UN6"),
        ("我的情感", "我拥有PSI需求系统，", "emotion", "需求和情感"),
        ("我的知识", "我的知识库存储了", "kb", "8981条知识"),
        ("我的能力", "我能做到的事情包括", "capability", "零LLM"),
        ("我的关系", "我和Lorry的关系是", "relationship", "创造者与造物"),
        ("我的愿景", "我的未来是", "vision", "AGI/ASI"),
        ("结语", "这就是我，", "closing", "收束"),
    ]

    def __init__(self):
        self._v7 = None
        self._kb = None
        self._mrk = None
        self._loaded = False

    def _lazy(self):
        if self._loaded:
            return
        from semantic_engine import get_encoder
        self._v7 = get_encoder(1024)
        from matrix_knowledge import MatrixKnowledgeRetriever
        self._kb = MatrixKnowledgeRetriever()
        # Markov
        try:
            from aris_markov_generator import MarkovChainGenerator
            self._mrk = MarkovChainGenerator(order=3, min_freq=1)
            cache = os.path.join(_DIR, "state", "markov_chain.pkl")
            cache_old = os.path.join(_DIR, "state", "markov_chain.json")
            if os.path.exists(cache):
                self._mrk.load(cache)
            elif os.path.exists(cache_old):
                self._mrk.load(cache_old)
            else:
                self._mrk._build_default_corpus()
                bc = os.path.join(_DIR, "corpus", "aris_corpus_clean.txt")
                if os.path.exists(bc):
                    self._mrk.train_from_file(bc)
        except:
            self._mrk = None
        self._loaded = True

    def _clean(self, text: str) -> str:
        lines = text.split("\n")
        clean = []
        for s in lines:
            s = s.strip()
            if len(s) < 6: continue
            if any(s.startswith(p) for p in ["#", "===", "import ", "from ", "def ", "class ",
                 "\"\"\"", "'''"]): continue
            if re.match(r'^[A-Za-z_][A-Za-z0-9_./:]*$', s) and len(s) < 60: continue
            if sum(1 for c in s if c in "=(){}[]<>:.,;+-*/|\\") > 8 and len(s) < 100: continue
            clean.append(s)
        return "。".join(clean).strip()[:500]

    def _get_context(self, topic: str, count: int = 5, exclude: List[str] = None) -> List[str]:
        """获取相关上下文, 排除已用内容"""
        if not self._kb or not self._kb._loaded:
            return []

        results = self._kb.search(topic, top_k=count + 5, threshold=0.05)
        contexts = []
        seen = set()
        if exclude:
            seen.update(e[:40] for e in exclude)  
        for r in results:
            text = self._clean(r.get("text", ""))
            fp = text[:40]
            if fp not in seen and len(text) > 20:
                seen.add(fp)
                contexts.append(text[:300])
                if len(contexts) >= count:
                    break
        return contexts

    def _expand_text(self, seed: str, target_length: int, contexts: List[str]) -> str:
        """
        扩展文本到目标长度

        策略:
          1. 先放上下文知识 (300字)
          2. 用 Markov 生成连接句
          3. 不够再用 Markov 补充
        """
        parts = []

        # 核心知识
        kb_parts = []
        for c in contexts[:2]:
            kb_parts.append(c[:200])
        kb_text = "。".join(kb_parts)
        parts.append(kb_text)
        current_len = len(kb_text)

        # Markov 填充直到达到目标
        max_attempts = 20
        for _ in range(max_attempts):
            if current_len >= target_length:
                break
            if self._mrk:
                mk = self._mrk.generate(
                    seed_words=[seed[:10]],
                    max_words=min(40, (target_length - current_len) // 3),
                    temperature=0.5
                )
                if mk and len(mk) > 8 and mk not in " ".join(parts):
                    parts.append(mk)
                    current_len += len(mk)
            else:
                break

        text = "。".join(parts)
        if text and text[-1] not in "。！？.!?":
            text += "。"
        return text

    def generate(self, topic: str, structure: str = "paper",
                 target_chars: int = 10000) -> Dict:
        """
        生成长文

        Args:
            topic: 主题
            structure: "paper"(论文) | "self_intro"(自我介绍) | "custom"
            target_chars: 目标字数

        Returns:
            {output, chars, chapters, latency_ms}
        """
        t0 = time.perf_counter()
        self._lazy()

        # 选结构
        if structure == "self_intro":
            chapters = self.SELF_INTRO_STRUCTURE
        else:
            chapters = self.PAPER_STRUCTURE

        # 每章目标字数
        chars_per_chapter = target_chars // len(chapters)

        # 逐章生成
        output_sections = []
        chapter_stats = []
        used_contexts = []

        for i, (title, lead_in, ktype, hint) in enumerate(chapters[:len(chapters)]):
            t_chapter = time.perf_counter()
            search_query = f"{topic} {title} {ktype} {hint}"
            contexts = self._get_context(search_query, count=3, exclude=used_contexts)
            if contexts:
                used_contexts.extend(contexts[:1])

            # 生成章节
            chapter_text = self._expand_text(
                seed=search_query,
                target_length=chars_per_chapter,
                contexts=contexts,
            )

            # 格式化
            section = f"## {i+1}. {title}\n\n{lead_in}{chapter_text}\n"
            output_sections.append(section)

            chapter_stats.append({
                "title": title,
                "chars": len(chapter_text),
                "contexts": len(contexts),
                "ms": round((time.perf_counter() - t_chapter) * 1000, 1),
            })

        # 拼接
        output = "\n\n".join(output_sections)
        # 截断到目标长度
        if len(output) > target_chars * 1.1:
            output = output[:target_chars]

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "output": output,
            "chars": len(output),
            "chapters": len(chapters),
            "chapter_stats": chapter_stats,
            "latency_ms": round(total_ms, 1),
            "chars_per_sec": round(len(output) / (total_ms / 1000)),
        }

    def self_intro_paper(self, target_chars: int = 10000) -> Dict:
        """生成 Aris 自我介绍论文"""
        return self.generate(
            topic="Aris数字生命体",
            structure="self_intro",
            target_chars=target_chars,
        )

    def architecture_paper(self, target_chars: int = 10000) -> Dict:
        """生成 Aris 架构论文"""
        return self.generate(
            topic="Aris数字生命体架构设计",
            structure="paper",
            target_chars=target_chars,
        )


# ================================================================
# 自测
# ================================================================
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("  Aris LongForm Synthesizer — 长文合成测试")
    logger.info("=" * 65)
    synth = LongFormSynthesizer()

    # 测试1: 一万字自我介绍论文
    logger.info("\n[1] 一万字自我介绍论文...")
    r = synth.self_intro_paper(target_chars=10000)
    logger.info(f"  输出: {r['chars']} 字, {r['chapters']}章, {r['latency_ms']}ms")
    logger.info(f"  吞吐: {r['chars_per_sec']} 字/秒")
    logger.info(f"\n  开头:")
    for line in r['output'].split('\n')[:25]:
        logger.info(f"    {line[:120]}")
    logger.info(f"\n  章节统计:")
    for cs in r['chapter_stats']:
        logger.info(f"    {cs['title']:12s}: {cs['chars']:5d}字, {cs['contexts']}条知识, {cs['ms']:5.0f}ms")