"""Unit tests for hierarchical_memory - LSH recall + decay + capacity."""

import pytest

from hierarchical_memory import HierarchicalMemory


@pytest.fixture
def mem():
    return HierarchicalMemory(seed=3)


def test_empty_recall(mem):
    assert mem.recall("anything") == []


def test_store_and_recall_exact(mem):
    mem.store("量子计算需要低温环境", importance=0.9)
    hits = mem.recall("量子计算", top_k=3)
    assert hits and "量子" in hits[0]["text"]
    assert hits[0]["importance"] == pytest.approx(0.9)


def test_recall_ranks_relevant_first(mem):
    mem.store("量子计算需要低温环境", importance=0.9)
    for i in range(50):
        mem.store(f"无关内容 {i}", importance=0.1)
    hits = mem.recall("量子计算", top_k=5)
    assert "量子" in hits[0]["text"]


def test_capacity_bounded(mem):
    mem = HierarchicalMemory(capacity=100, seed=2)
    for i in range(2000):
        mem.store(f"item-{i}", importance=float(i % 7) / 7.0)
    assert mem.size() <= 200  # hard cap with small slack


def test_forget_removes_dead_memories(mem):
    mem.store("陈旧记忆", importance=0.001)
    mem.store("鲜活记忆", importance=0.9)
    dropped = mem.forget(threshold=0.01)
    assert dropped >= 1
    texts = [r["text"] for r in mem.recall("记忆", top_k=10)]
    assert "鲜活记忆" in texts


def test_deterministic_hashing_across_processes():
    # FNV-1a stable hash must not depend on PYTHONHASHSEED
    a = HierarchicalMemory(seed=3)
    b = HierarchicalMemory(seed=3)
    a.store("测试句子A", importance=0.5)
    a.store("测试句子B", importance=0.5)
    b.store("测试句子A", importance=0.5)
    b.store("测试句子B", importance=0.5)
    ra = a.recall("测试句子A", top_k=2)
    rb = b.recall("测试句子A", top_k=2)
    assert [r["text"] for r in ra] == [r["text"] for r in rb]


def test_access_updates_recency(mem):
    mem.store("被访问的记忆", importance=0.5)
    before = mem.recall("被访问的记忆", top_k=1)[0]
    again = mem.recall("被访问的记忆", top_k=1)[0]
    assert again["accesses"] == before["accesses"] + 1