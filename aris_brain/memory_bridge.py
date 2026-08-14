"""
LAAP Memory Bridge — three-layer memory wiring.

P1 wiring: when laap/agi memory_system is available, this bridge uses the full
hierarchical memory (episodic + semantic + procedural + MemoryConsolidator)
with JSON persistence. Otherwise it falls back to the minimal MemoryStore.

Public interface is unchanged:
  - get_memory_context(max_core, max_recent, max_working) -> str
  - recall_related(query, top_k) -> List[MemoryFragment]
  - store_important(content, layer, importance, topics) -> None
  - memory_stats() -> Dict
  - consolidate_memory() -> None
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from aris_brain.memory_store import MemoryFragment, MemoryStore

logger = logging.getLogger("aris.memory_bridge")

_BRIDGE_STATE_DIR = Path(__file__).resolve().parent / "state"
_AGI_DB_PATH = _BRIDGE_STATE_DIR / "agi_memory.json"

_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


# ── AGI three-layer backend ──────────────────────────────────────────────

class _AgiMemoryBackend:
    """Wraps laap/agi memory_system with JSON persistence."""

    def __init__(self) -> None:
        from laap.agi.memory_system import (
            EpisodicMemory,
            SemanticMemory,
            ProceduralMemory,
            MemoryConsolidator,
            MemoryTrace,
            MemoryType,
            MemoryPriority,
        )
        self._MemoryTrace = MemoryTrace
        self._MemoryType = MemoryType
        self._MemoryPriority = MemoryPriority
        self.episodic = EpisodicMemory(capacity=1000)
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.consolidator = MemoryConsolidator(self.episodic, self.semantic, self.procedural)
        self._BRIDGE_STATE_DIR = _BRIDGE_STATE_DIR
        self._AGI_DB_PATH = _AGI_DB_PATH
        self._load()

    # ── serialization ──

    def _serialize_trace(self, t) -> Dict[str, Any]:
        return {
            "trace_id": t.trace_id,
            "memory_type": t.memory_type.value,
            "content": t.content,
            "timestamp": t.timestamp,
            "emotional_valence": t.emotional_valence,
            "emotional_arousal": t.emotional_arousal,
            "rehearsal_count": t.rehearsal_count,
            "last_accessed": t.last_accessed,
            "associations": t.associations,
            "source_episode": t.source_episode,
            "confidence": t.confidence,
            "decay_rate": t.decay_rate,
        }

    def _deserialize_trace(self, d: Dict[str, Any]):
        return self._MemoryTrace(
            trace_id=d["trace_id"],
            memory_type=self._MemoryType(d["memory_type"]),
            content=d["content"],
            timestamp=d.get("timestamp", time.time()),
            emotional_valence=d.get("emotional_valence", 0.0),
            emotional_arousal=d.get("emotional_arousal", 0.0),
            rehearsal_count=d.get("rehearsal_count", 0),
            last_accessed=d.get("last_accessed", time.time()),
            associations=d.get("associations", []),
            source_episode=d.get("source_episode"),
            confidence=d.get("confidence", 0.5),
            decay_rate=d.get("decay_rate", 0.01),
        )

    def save(self) -> None:
        try:
            self._BRIDGE_STATE_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "episodic": [self._serialize_trace(t) for t in self.episodic.episodes],
                "concepts": self.semantic.concepts,
                "relations": {
                    k: v for k, v in self.semantic.relations.items()
                },
                "hierarchy": dict(self.semantic.hierarchy),
                "skills": self.procedural.skills,
                "habits": self.procedural.habits,
                "automated_responses": self.procedural.automated_responses,
                "dream_reports": self.consolidator.dream_reports,
            }
            # 原子写：临时文件 + os.replace，防止多进程并发读半截 agi_memory.json
            tmp = self._AGI_DB_PATH.with_name(self._AGI_DB_PATH.name + ".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
            )
            os.replace(tmp, self._AGI_DB_PATH)
        except Exception as e:
            logger.error(
                "MemoryBridge.save: 记忆持久化失败 %s: %s",
                self._AGI_DB_PATH, e,
            )

    def _backup_corrupt(self) -> None:
        try:
            backup = self._AGI_DB_PATH.with_suffix(".corrupt.json")
            if self._AGI_DB_PATH.exists():
                backup.write_bytes(self._AGI_DB_PATH.read_bytes())
                logger.warning(
                    "MemoryBridge: 损坏记忆文件已备份 → %s（原始记忆未覆盖）",
                    backup,
                )
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if not self._AGI_DB_PATH.exists():
                return
            data = json.loads(self._AGI_DB_PATH.read_text(encoding="utf-8"))
            self.episodic.episodes = [
                self._deserialize_trace(d) for d in data.get("episodic", [])
            ]
            # 重建时间/情感索引：_load 只回填 episodes 会导致
            # retrieve_by_time / retrieve_by_emotion 在重启后恒空。
            self.episodic._time_index.clear()
            self.episodic._emotion_index.clear()
            for tr in self.episodic.episodes:
                self.episodic._time_index[tr.timestamp].append(tr.trace_id)
                ek = self.episodic._emotion_key(tr.emotional_valence, tr.emotional_arousal)
                self.episodic._emotion_index[ek].append(tr.trace_id)
            self.semantic.concepts = dict(data.get("concepts", {}))
            from collections import defaultdict as _dd
            self.semantic.relations = _dd(list, {
                k: list(v) for k, v in data.get("relations", {}).items()
            })
            self.semantic.hierarchy = _dd(list, {
                k: list(v) for k, v in data.get("hierarchy", {}).items()
            })
            self.procedural.skills = dict(data.get("skills", {}))
            self.procedural.habits = dict(data.get("habits", {}))
            self.procedural.automated_responses = dict(
                data.get("automated_responses", {})
            )
            self.consolidator.dream_reports = list(
                data.get("dream_reports", [])
            )
        except Exception as e:
            logger.error(
                "MemoryBridge._load: 记忆文件损坏，备份后跳过加载 %s: %s",
                self._AGI_DB_PATH, e,
            )
            self._backup_corrupt()

    # ── ops ──

    def store_important(self, content: str, layer: str = "episodic",
                        importance: float = 0.7, topics: Optional[List[str]] = None) -> None:
        priority_map = {
            "core": self._MemoryPriority.CORE,
            "episodic": self._MemoryPriority.IMPORTANT,
            "working": self._MemoryPriority.RELEVANT,
        }
        priority = priority_map.get(layer, self._MemoryPriority.RELEVANT)
        trace = self.episodic.encode_episode(
            content=content,
            emotional_valence=max(-1.0, min(1.0, (importance - 0.5))),
            emotional_arousal=importance,
            associations=topics or [],
            priority=priority,
        )
        self.consolidator.queue_for_consolidation(trace.trace_id)
        self.save()

    def recall_related(self, query: str, top_k: int = 3) -> List[MemoryFragment]:
        fragments: List[MemoryFragment] = []

        for t in self.episodic.retrieve_similar(query, max_results=top_k):
            fragments.append(MemoryFragment(
                content=t.content,
                layer="episodic",
                importance=t.confidence,
                topics=t.associations,
                timestamp=t.timestamp,
            ))

        semantic_hits = self.semantic.query(query.split(), max_results=top_k)
        for concept, score in semantic_hits:
            fragments.append(MemoryFragment(
                content=concept,
                layer="semantic",
                importance=min(1.0, score / 3.0),
                timestamp=time.time(),
            ))

        return fragments[:top_k]

    def get_memory_context(self, max_core: int = 3, max_recent: int = 3,
                           max_working: int = 2) -> str:
        parts: List[str] = []

        narrative = self.episodic.get_life_narrative(recent_hours=24.0)
        if narrative:
            parts.append("[人生叙事] " + narrative[:300])

        core = [t for t in self.episodic.episodes
                if t.confidence >= 0.8][-max_core:]
        if core:
            parts.append("[核心记忆] " + "；".join(t.content[:80] for t in core))

        recent = sorted(self.episodic.episodes,
                        key=lambda t: t.timestamp, reverse=True)[:max_recent]
        if recent:
            parts.append("[最近经历] " + "；".join(t.content[:80] for t in recent))

        if self.semantic.concepts:
            top = sorted(self.semantic.concepts.items(),
                         key=lambda kv: kv[1].get("access_count", 0),
                         reverse=True)[:max_working]
            parts.append("[概念知识] " + "；".join(k for k, _ in top))

        return "\n".join(parts)

    def stats(self) -> Dict[str, int]:
        return {
            "episodic": len(self.episodic.episodes),
            "semantic_concepts": len(self.semantic.concepts),
            "skills": len(self.procedural.skills),
            "habits": len(self.procedural.habits),
            "consolidations": self.consolidator.consolidation_count,
            "dreams": len(self.consolidator.dream_reports),
        }

    def consolidate(self) -> None:
        self.consolidator.consolidate()
        self.save()


_agi_backend: _AgiMemoryBackend | None = None


def _get_agi_backend() -> Optional[_AgiMemoryBackend]:
    global _agi_backend
    if _agi_backend is not None:
        return _agi_backend
    try:
        _agi_backend = _AgiMemoryBackend()
        return _agi_backend
    except Exception:
        return None


def get_memory_context(max_core: int = 3, max_recent: int = 3, max_working: int = 2) -> str:
    """Return a short memory context string for prompt injection."""
    backend = _get_agi_backend()
    if backend is not None:
        return backend.get_memory_context(max_core, max_recent, max_working)

    store = _get_store()
    parts: List[str] = []

    core = store.query(layer="core", top_k=max_core)
    if core:
        parts.append("[核心记忆] " + "；".join(f.content[:80] for f in core))

    recent = store.query(layer="episodic", top_k=max_recent)
    if recent:
        parts.append("[最近经历] " + "；".join(f.content[:80] for f in recent))

    working = store.query(layer="working", top_k=max_working)
    if working:
        parts.append("[当前工作记忆] " + "；".join(f.content[:60] for f in working))

    return "\n".join(parts)


def recall_related(query: str, top_k: int = 3) -> List[MemoryFragment]:
    """Return memory fragments related to the query."""
    backend = _get_agi_backend()
    if backend is not None:
        return backend.recall_related(query, top_k)

    store = _get_store()
    query_words = set(query.lower().split())
    scored = []
    for f in store._fragments:
        frag_words = set(f.content.lower().split())
        score = len(query_words & frag_words)
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_k]]


def store_important(content: str, layer: str = "episodic", importance: float = 0.7,
                    topics: List[str] | None = None) -> None:
    """Store an important memory fragment."""
    backend = _get_agi_backend()
    if backend is not None:
        backend.store_important(content, layer, importance, topics)
        return

    store = _get_store()
    fragment = MemoryFragment(
        content=content,
        layer=layer,
        importance=importance,
        topics=topics or [],
    )
    store.store(fragment)


def memory_stats() -> Dict[str, Any]:
    """Return memory statistics from whichever backend is active."""
    backend = _get_agi_backend()
    if backend is not None:
        return {"backend": "agi", **backend.stats()}
    store = _get_store()
    return {"backend": "fallback", **store.get_stats()}


def consolidate_memory() -> None:
    """Run memory consolidation (dream generation) on the AGI backend."""
    backend = _get_agi_backend()
    if backend is not None:
        backend.consolidate()
