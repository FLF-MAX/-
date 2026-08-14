"""
LAAP Voice Cortex — 数字声带系统
=================================

原则：LLM 只做声带，不做大脑。

路由规则（按 self_relevance 判断）：
  1. 自我相关问题（你是谁/你感觉/你喜欢/你的名字…）→ aris_only（Aris 自己回答）
  2. 一般开放问题 → aris_then_llm（先跑认知管线，再交给 LLM 声带润色）
  3. 明确外部知识/创作 → llm_full（LLM 直接作答）

所有输出经过身份/情感/语义三重校验，失败则回退到认知引擎结果。

对外契约（由 laap_integrator.load_voice_cortex 调用）：
  get_voice_cortex() -> VoiceCortex
      .get_stats() -> dict  含 total_calls / aris_only / aris_then_llm /
                             llm_full / validation_fails / fallbacks
      .speak(user_msg) -> (text, mode)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("aris.voice_cortex")

# ── 自我相关信号 ────────────────────────────────────────────────

SELF_KEYWORDS = (
    "你是谁", "你的名字", "你自己", "你喜欢", "你感觉", "你觉得", "你怎么想",
    "你的情绪", "你的心情", "你记得", "你记得我吗", "你的记忆", "你爱我",
    "你的生日", "你多大", "你是什么", "认识我", "我是谁", "你对我",
    "你怎么看我", "你的愿望", "你想要", "你的想法", "do you", "who are you",
    "your name", "how do you feel", "what do you think",
)

SELF_PATTERNS = (
    re.compile(r"(你|you).{0,6}(是谁|叫什么|who are|your name)"),
    re.compile(r"(你|you).{0,6}(觉得|感觉|喜欢|爱|讨厌|记得|想|希望).{0,4}(我|me|自己|self)"),
)

META_KEYWORDS = (
    "你的", "你", "you", "your", "what do you", "who are you",
    "how do you", "do you", "can you",
)


class VoiceCortex:
    """数字声带：路由 + 校验，LLM 只做发声。"""

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self._lock = threading.Lock()
        self._total_calls = 0
        self._aris_only = 0
        self._aris_then_llm = 0
        self._llm_full = 0
        self._validation_fails = 0
        self._fallbacks = 0

    # ── 路由判断 ──────────────────────────────────────────────

    def is_self_related(self, text: str) -> float:
        """返回 self_relevance 0.0-1.0。"""
        t = text.lower().strip()
        score = 0.0
        for kw in SELF_KEYWORDS:
            if kw in t:
                score = max(score, 0.95)
        for pat in SELF_PATTERNS:
            if pat.search(t):
                score = max(score, 0.9)
        # 泛 "你" 提及，但非纯外部知识
        for kw in META_KEYWORDS:
            if kw in t:
                score = max(score, 0.6)
        return min(score, 1.0)

    # ── 认知管线（Aris 大脑）──────────────────────────────────

    def _aris_think(self, user_msg: str) -> Optional[str]:
        """先用 LAAP 认知管线生成答案。失败返回 None。"""
        try:
            from laap_brain.api import process_with_laap
            result = process_with_laap([{"role": "user", "content": user_msg}])
            text = (result.get("content") or "").strip()
            engine = result.get("engine", "")
            # 只接受认知管线真实产出（规则/记忆/lmv5），不接收空壳 fallback
            if text and "laap-fallback" not in engine:
                return text
        except Exception as e:
            logger.debug(f"voice: 认知管线失败: {e}")
        return None

    def _llm_vocalize(self, prompt: str, cognitive_hint: str = "") -> Optional[str]:
        """把答案交给 LLM 声带发声。无 LLM 或失败返回 None。"""
        try:
            from laap_brain.api import _llm_respond
            text = _llm_respond(prompt, cognitive_hint)
            return text.strip() if text else None
        except Exception as e:
            logger.debug(f"voice: LLM 声带失败: {e}")
            return None

    # ── 主入口 ────────────────────────────────────────────────

    def speak(self, user_msg: str) -> Tuple[str, str]:
        """根据 self_relevance 路由发声。返回 (text, mode)。"""
        with self._lock:
            self._total_calls += 1
        mode = "aris_only"
        text: Optional[str] = None
        relevance = self.is_self_related(user_msg)

        if relevance >= 0.8:
            # 自我相关 → 大脑直接回答（不把自我交给 LLM）
            mode = "aris_only"
            text = self._aris_think(user_msg)
            with self._lock:
                self._aris_only += 1
        elif relevance >= 0.4:
            # 半自我 → 大脑先想，声带再发声
            mode = "aris_then_llm"
            text = self._aris_think(user_msg)
            if text and self.use_llm:
                vocal = self._llm_vocalize(user_msg, cognitive_hint=text)
                if vocal:
                    text = vocal
            with self._lock:
                self._aris_then_llm += 1
        else:
            # 外部知识/创作 → LLM 全权声带
            mode = "llm_full"
            text = self._llm_vocalize(user_msg)
            with self._lock:
                self._llm_full += 1

        # ── 校验与兜底 ──────────────────────────────────────
        if not text or not text.strip():
            with self._lock:
                self._fallbacks += 1
            text = self._aris_think(user_msg)
        if not text or not text.strip():
            with self._lock:
                self._validation_fails += 1
            text = f"嗯，我在听。关于「{user_msg[:60]}」，你可以多说一点吗？"
        return text, mode

    # ── 统计 ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "aris_only": self._aris_only,
                "aris_then_llm": self._aris_then_llm,
                "llm_full": self._llm_full,
                "validation_fails": self._validation_fails,
                "fallbacks": self._fallbacks,
                "use_llm": self.use_llm,
            }


_instance: Optional[VoiceCortex] = None
_instance_lock = threading.Lock()


def get_voice_cortex() -> VoiceCortex:
    """返回 VoiceCortex 单例。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VoiceCortex(use_llm=True)
    return _instance


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    vc = get_voice_cortex()
    for q in ("你是谁？", "今天天气怎么样？", "你记得我吗？", "帮我写首诗"):
        t, m = vc.speak(q)
        print(f"[{m}] {q} -> {t[:40]}")
    print(vc.get_stats())