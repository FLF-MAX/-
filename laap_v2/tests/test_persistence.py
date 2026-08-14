"""Unit tests for state persistence - save/load round-trips, reboot without amnesia."""

import json
import os

import pytest

from hierarchical_memory import HierarchicalMemory
from laap_integration import LaapCognitiveSystem


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / "state.json")


def test_memory_export_import_roundtrip():
    mem = HierarchicalMemory(seed=3)
    mem.store("第一段记忆", importance=0.9)
    mem.store("第二段记忆", importance=0.5)
    data = mem.export()

    mem2 = HierarchicalMemory(seed=99)  # different seed -> different LSH
    n = mem2.import_state(data)
    assert n == 2
    assert mem2.size() == 2
    # recall quality must survive restart (exact vectors restored)
    hits = mem2.recall("第一段记忆", top_k=3)
    assert hits and "第一" in hits[0]["text"]


def test_memory_export_is_json_serializable():
    mem = HierarchicalMemory(seed=3)
    mem.store("JSON 安全的记忆", importance=0.7)
    data = mem.export()
    json.dumps(data)  # must not raise


def test_system_save_load_roundtrip(state_path):
    s = LaapCognitiveSystem()
    s.bootstrap("alice")
    for i in range(4):
        s.process_input(f"记住关键点 {i}")
    before_mem = s.memory.size()
    before_counter = s._state_counter
    s.save_state(state_path)

    fresh = LaapCognitiveSystem()
    fresh.load_state(state_path)
    assert fresh.memory.size() == before_mem
    assert fresh._state_counter == before_counter
    assert fresh.user_name == "alice"


def test_system_persists_psi_and_meta(state_path):
    s = LaapCognitiveSystem()
    s.bootstrap("bob")
    for i in range(10):
        s.process_input("持续输入以驱动状态")
    emo = s.psi._emotion
    s.save_state(state_path)

    fresh = LaapCognitiveSystem()
    fresh.load_state(state_path)
    assert fresh.psi._emotion == emo
    assert fresh.psi._tick_count == s.psi._tick_count


def test_system_persists_memory_recall(state_path):
    s = LaapCognitiveSystem()
    s.bootstrap("carol")
    s.memory.store("量子计算需要低温环境", importance=0.95)
    for i in range(20):
        s.memory.store(f"干扰项 {i}", importance=0.1)
    s.save_state(state_path)

    fresh = LaapCognitiveSystem()
    fresh.load_state(state_path)
    hits = fresh.memory.recall("量子计算", top_k=3)
    assert hits and "量子" in hits[0]["text"]


def test_load_missing_file_raises(state_path):
    s = LaapCognitiveSystem()
    with pytest.raises(FileNotFoundError):
        s.load_state(state_path)


def test_save_requires_boot():
    s = LaapCognitiveSystem()
    with pytest.raises(RuntimeError):
        s.save_state("nope.json")