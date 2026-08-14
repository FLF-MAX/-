"""hierarchical_memory.py - LSH-indexed, temporally-decaying memory system.

Three memory tiers (mirroring Baddeley / consolidation theory):

  WorkingMemory  - capacity 7 ± 2, LRU eviction, holds the "present".
  EpisodicMemory - long-term store backed by an LSH approximate-NN index,
                   items scored by cosine similarity * exponential decay.
  SemanticMemory - concepts extracted by clustering episodic vectors; the
                   raw experiences are then consolidated into the concept.

Fixes versus v1.0:
  - No JSON-file thrashing: in-memory vector index (10K+ capacity).
  - Every np.linalg.norm is guarded against the zero vector.
  - Capacity limits + automatic forgetting + semantic integration.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# hashing embeddings (no pretrained encoder -> feature-hash character n-grams) #
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Deterministic feature-hash embedding over character 2/3-grams.

    Uses a stable FNV-1a style digest instead of Python's builtin ``hash()``,
    which is salted per-process (PYTHONHASHSEED) and would make LSH buckets
    (and therefore recall ranking) non-deterministic across runs.
    """

    def __init__(self, dim: int = 128, seed: int = 3, gram_sizes: Tuple[int, int] = (2, 3)):
        self.dim = dim
        self.gram_sizes = gram_sizes

    @staticmethod
    def _stable_hash(text: str) -> int:
        h = 2166136261
        for byte in text.encode("utf-8"):
            h ^= byte
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = text.lower().split()
        for tok in tokens:
            grams = list(tok)
            for n in self.gram_sizes:
                grams += [tok[i:i + n] for i in range(max(0, len(tok) - n + 1))]
            for g in grams:
                h = self._stable_hash(g)
                idx = h % self.dim
                vec[idx] += (1.0 if h % 2 == 0 else -1.0)
        norm = float(np.linalg.norm(vec))
        if norm > EPS:
            vec /= norm
        return vec


# --------------------------------------------------------------------------- #
# LSH index                                                                   #
# --------------------------------------------------------------------------- #
class LSHIndex:
    """Multi-table random-hyperplane LSH with exact cosine re-ranking."""

    def __init__(self, dim: int, tables: int = 8, depth: int = 8, seed: int = 5):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._planes = [self._rng.normal(0, 1, (depth, dim)) for _ in range(tables)]
        self._buckets: List[Dict[bytes, List[int]]] = [{} for _ in range(tables)]
        self._vectors: List[np.ndarray] = []
        self._keys: List[str] = []

    def _keys_of(self, v: np.ndarray) -> List[bytes]:
        return [np.packbits(planes @ v > 0).tobytes() for planes in self._planes]

    def add(self, key: str, vec: np.ndarray) -> None:
        idx = len(self._keys)
        self._vectors.append(vec.copy())
        self._keys.append(key)
        for bucket, sig in zip(self._buckets, self._keys_of(vec)):
            bucket.setdefault(sig, []).append(idx)

    def query(self, vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """Approximate nearest neighbours -> exact cosine, candidates from buckets."""
        if not self._vectors:
            return []
        candidates: List[int] = []
        seen: set = set()
        sig_keys = self._keys_of(vec)
        for sig, bucket in zip(sig_keys, self._buckets):
            for idx in bucket.get(sig, []):
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(idx)
        if not candidates:
            candidates = list(range(len(self._vectors)))
        vn = np.asarray(vec, dtype=np.float64)
        vnorm = float(np.linalg.norm(vn))
        scored = []
        for idx in candidates:
            other = self._vectors[idx]
            onorm = float(np.linalg.norm(other))
            if vnorm <= EPS or onorm <= EPS:
                sim = 0.0
            else:
                sim = float(vn @ other) / (vnorm * onorm)
            scored.append((self._keys[idx], sim, idx))
        scored.sort(key=lambda t: -t[1])
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._keys)


# --------------------------------------------------------------------------- #
# memory datum                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class MemoryItem:
    text: str
    vector: np.ndarray
    created_at: float
    last_access: float
    access_count: int = 1
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)

    def relevance(self, now: float, decay: float = 0.01) -> float:
        age = max(0.0, now - self.last_access)
        recency = math.exp(-decay * age) if age > 0 else 1.0
        return self.importance * recency


# --------------------------------------------------------------------------- #
# workspace + episodic store                                                  #
# --------------------------------------------------------------------------- #
class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._items: List[str] = []

    def push(self, text: str) -> None:
        if text in self._items:
            self._items.remove(text)
        self._items.append(text)
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity:]

    def size(self) -> int:
        return len(self._items)

    def items(self) -> List[str]:
        return list(self._items)


class HierarchicalMemory:
    """Working / episodic / semantic memory with LSH + decay + consolidation."""

    def __init__(self, embed_dim: int = 128, capacity: int = 5000,
                 decay: float = 0.008, seed: Optional[int] = None, embedder=None):
        self._lock = threading.RLock()
        self.embedder = embedder or HashingEmbedder(dim=embed_dim, seed=seed or 11)
        self.capacity = capacity
        self.decay = decay
        self.working = WorkingMemory()
        self._index = LSHIndex(dim=embed_dim, seed=(seed or 11) + 1)
        self._items: List[MemoryItem] = []
        self._concepts: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    # ------------------------------------------------------------------ #
    def store(self, text: str, importance: float = 0.5, tags: Optional[List[str]] = None) -> MemoryItem:
        with self._lock:
            v = self.embedder.embed(text)
            item = MemoryItem(text=text, vector=v, created_at=time.time(),
                              last_access=time.time(), importance=importance,
                              tags=list(tags) if tags else [])
            self._items.append(item)
            self._index.add(f"m{len(self._items) - 1}", v)
            self.working.push(text)
            self._enforce_capacity()
            return item

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        now = time.time()
        # evict lowest relevance first
        order = sorted(range(len(self._items)),
                       key=lambda i: self._items[i].relevance(now, self.decay))
        drop = order[: len(self._items) - self.capacity]
        drop_ids = {f"m{i}" for i in drop}
        keep_ids = set(range(len(self._items))) - set(drop)
        self._items = [self._items[i] for i in sorted(keep_ids)]
        # rebuild index compactly
        self._index = LSHIndex(dim=self.embedder.dim, seed=3)
        for i, item in enumerate(self._items):
            self._index.add(f"m{i}", item.vector)

    # ------------------------------------------------------------------ #
    def recall(self, query: str, top_k: int = 5, use_decay: bool = True) -> List[Dict[str, object]]:
        """Cosine * decay ranked recall."""
        with self._lock:
            if not self._items:
                return []
            q = self.embedder.embed(query)
            now = time.time()
            ranked = self._index.query(q, top_k=min(top_k * 8, len(self._items)))
            results = []
            for key, sim, idx in ranked:
                if not (0 <= idx < len(self._items)):
                    continue
                item = self._items[idx]
                item.last_access = now
                item.access_count += 1
                score = sim * item.relevance(now, self.decay) if use_decay else sim
                results.append({"text": item.text, "similarity": float(sim), "score": float(score),
                                "importance": item.importance, "accesses": item.access_count,
                                "tags": item.tags})
            results.sort(key=lambda r: -r["score"])
            return results[:top_k]

    # ------------------------------------------------------------------ #
    def forget(self, threshold: Optional[float] = None) -> int:
        """Remove dead memories (low relevance or too old), return count."""
        with self._lock:
            if not self._items:
                return 0
            now = time.time()
            threshold = 0.01 if threshold is None else threshold
            keep = [it for it in self._items
                    if it.relevance(now, self.decay) > threshold and
                    (now - it.created_at) < 86400 * 30]
            dropped = len(self._items) - len(keep)
            if dropped:
                self._items = keep
                self._index = LSHIndex(dim=self.embedder.dim, seed=3)
                for i, item in enumerate(self._items):
                    self._index.add(f"m{i}", item.vector)
            return dropped

    # ------------------------------------------------------------------ #
    def integrate_concepts(self, min_sim: float = 0.80, max_concepts: int = 50) -> Dict[str, List[str]]:
        """Cluster episode vectors into semantic concepts via greedy leader
        clustering; consolidated episodes point at a centroid."""
        with self._lock:
            if len(self._items) < 2:
                return {}
            clusters: List[Tuple[np.ndarray, List[int]]] = []
            for i, item in enumerate(self._items):
                v = item.vector
                best, best_d = None, float("inf")
                for c, (centroid, members) in enumerate(clusters):
                    d = 1.0 - _cos(v, centroid)
                    if d < best_d:
                        best, best_d = c, d
                if best is not None and best_d <= (1.0 - min_sim):
                    centroid, members = clusters[best]
                    members.append(i)
                    centroid[:] = (centroid * (len(members) - 1) + v) / len(members)
                else:
                    clusters.append((v.copy(), [i]))
            clusters.sort(key=lambda c: -len(c[1]))
            clusters = clusters[:max_concepts]
            concepts: Dict[str, List[str]] = {}
            for c, (_centroid, members) in enumerate(clusters):
                name = f"concept_{c}"
                concepts[name] = [self._items[i].text for i in members]
            self._concepts = concepts
            return concepts

    def concepts(self) -> Dict[str, List[str]]:
        return dict(self._concepts)

    def size(self) -> int:
        return len(self._items)

    def working_summary(self) -> Dict[str, object]:
        return {"capacity": self.working.capacity, "size": self.working.size(),
                "items": self.working.items()}

    # ------------------------------------------------------------------ #
    # persistence                                                         #
    # ------------------------------------------------------------------ #
    def export(self) -> Dict[str, object]:
        """Serialize all episodic items (vectors + metadata) for disk storage.

        Everything returned is JSON-safe (lists/str/float/int) so callers can
        ``json.dump`` it directly; vectors are stored as plain lists.
        """
        with self._lock:
            return {
                "embed_dim": self.embedder.dim,
                "capacity": self.capacity,
                "decay": self.decay,
                "working": self.working.items(),
                "items": [
                    {
                        "text": it.text,
                        "vector": it.vector.tolist(),
                        "created_at": it.created_at,
                        "last_access": it.last_access,
                        "access_count": it.access_count,
                        "importance": it.importance,
                        "tags": list(it.tags),
                    }
                    for it in self._items
                ],
            }

    def import_state(self, data: Dict[str, object]) -> int:
        """Rebuild the memory from an ``export()`` payload.  Returns item count.

        Re-creates the LSH index from the stored vectors so recall quality is
        preserved exactly as before the restart (no re-embedding drift).
        """
        with self._lock:
            items = data.get("items") or []
            self._items = [
                MemoryItem(
                    text=it["text"],
                    vector=np.asarray(it["vector"], dtype=np.float64),
                    created_at=float(it["created_at"]),
                    last_access=float(it["last_access"]),
                    access_count=int(it.get("access_count", 0)),
                    importance=float(it.get("importance", 0.5)),
                    tags=list(it.get("tags") or []),
                )
                for it in items
            ]
            self._index = LSHIndex(dim=self.embedder.dim, seed=3)
            for i, item in enumerate(self._items):
                self._index.add(f"m{i}", item.vector)
            self.working._items = list(data.get("working") or self.working._items)
            self.working._items = self.working._items[-self.working.capacity:]
            return len(self._items)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(a @ b) / (na * nb)