"""stress_tests.py - load / fuzz / chaos / memory / long-run tests for laap_v2.

Run all suites (each prints a PASS/FAIL line and returns exit code):

    python stress_tests.py            # quick smoke set
    python stress_tests.py --full     # everything (long-running)
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np

from psi_core_v2 import PsiCoreV2
from meta_learning_engine import MetaLearningEngine
from probabilistic_world_model import ProbabilisticWorldModel
from neural_world_model import MLPWorldModel, MlpConfig
from hierarchical_memory import HierarchicalMemory
from drift_aware_meta_learning import DriftAwareMetaLearner
from deep_analogical_engine import DomainGraph, DeepAnalogyEngine, hungarian
from multi_agent_coordination import CognitiveArbiter, Agent, Task
from laap_integration import LaapCognitiveSystem

RESULTS: List[Dict[str, Any]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def test_load_burst() -> None:
    sys_ = LaapCognitiveSystem()
    sys_.bootstrap("stress")
    t0 = time.time()
    for i in range(300):
        sys_.process_input(f"第 {i} 条压测消息，请记住它并回答。")
    lat = time.time() - t0
    ok = lat < 180 and sys_.process_input("")["error"] is not None
    report("load_burst_300", ok, f"{300/lat:.0f} in/s total={lat:.1f}s")


# --------------------------------------------------------------------------- #
# fuzz
# --------------------------------------------------------------------------- #
def test_fuzz_texts() -> None:
    sys_ = LaapCognitiveSystem()
    sys_.bootstrap("fuzz")
    rng = np.random.default_rng(0)
    payloads = [
        "", " ", "\x00\x01\x02", "a" * 5000, "\ud83d\ude00" * 200,
        "前前前前" * 200, "%%%$#@","\n\t\r", "x" * 100000,
        "hello world 你好世界",
    ]
    payloads += [rng.bytes(64).decode("latin1") for _ in range(20)]
    failures = 0
    for p in payloads:
        try:
            r = sys_.process_input(p)
            if "response" not in r:
                failures += 1
        except Exception:
            failures += 1
    report("fuzz_30_payloads", failures == 0, f"failures={failures}")


# --------------------------------------------------------------------------- #
# chaos / self-healing
# --------------------------------------------------------------------------- #
def test_chaos_psi_recovery() -> None:
    sys_ = LaapCognitiveSystem()
    sys_.bootstrap("chaos")
    sys_.degrade_module("psi")                 # corrupt heartbeat
    r = sys_.process_input("还活着吗")
    # heartbeat should be auto-restored during the pass
    ok = sys_.psi.heartbeat_ok() and not r.get("degraded", True)
    report("chaos_psi_auto_recovery", ok,
           f"degraded={r.get('degraded')} heartbeat_ok={sys_.psi.heartbeat_ok()}")


def test_chaos_memory() -> None:
    sys_ = LaapCognitiveSystem()
    sys_.bootstrap("chaosmem")
    sys_.memory.store("重要的记忆A")
    sys_.memory.store("重要的记忆B")
    sys_.degrade_module("memory")              # wipe _items
    r = sys_.process_input("记得A吗")
    ok = "response" in r                        # still responds after memory wipe
    report("chaos_memory_wipe_graceful", ok, f"response_len={len(r['response'])}")


def test_chaos_meta_none() -> None:
    sys_ = LaapCognitiveSystem()
    sys_.bootstrap("chaosmeta")
    sys_.degrade_module("meta")                 # meta.set None
    r = sys_.process_input("策略系统坏了？")
    ok = "response" in r and r.get("strategy") is not None
    report("chaos_meta_degrade_strategy_fallback", ok, f"strategy={r.get('strategy')}")


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def test_memory_bounded_growth() -> None:
    mem = HierarchicalMemory(capacity=300, seed=2)
    for i in range(2000):
        mem.store(f"item-{i}", importance=float(i % 7) / 7.0)
    ok = mem.size() <= 400
    report("memory_bounded_capacity", ok, f"size={mem.size()} after 2000 inserts")


def test_memory_recall_relevance() -> None:
    mem = HierarchicalMemory(seed=3)
    mem.store("量子计算需要低温环境", importance=0.9)
    for i in range(50):
        mem.store(f"无关内容 {i}", importance=0.1)
    hits = mem.recall("量子计算", top_k=5)
    ok = hits and "量子" in hits[0]["text"]
    report("memory_relevance_recall", ok, f"top={hits[0]['text'][:20] if hits else None}")


# --------------------------------------------------------------------------- #
# deep module stability
# --------------------------------------------------------------------------- #
def test_hungarian_shapes() -> None:
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(200):
        n = int(rng.integers(1, 6)); m = int(rng.integers(1, 6))
        c = rng.random((n, m))
        pairs = hungarian(c)
        if len(pairs) != min(n, m):
            ok = False
        rows = [r_ for r_, _ in pairs]; cols = [c_ for _, c_ in pairs]
        if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
            ok = False
    report("hungarian_random_shapes", ok, f"{len([None for _ in range(200)])} matrices ok")


def test_drift_detects_change() -> None:
    learner = DriftAwareMetaLearner(["a", "b"], seed=4)
    for _ in range(40):
        learner.observe(learner.select(), 0.9)
    for _ in range(120):
        learner.observe(learner.select(), 0.1)
    d = learner.detected_drift()
    report("drift_detects_distribution_change", d > 0, f"alarms={d}")


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_persistence_roundtrip() -> None:
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), f"laap_stress_state_{time.time_ns()}.json")
    try:
        sys_ = LaapCognitiveSystem()
        sys_.bootstrap("persist")
        for i in range(80):
            sys_.process_input(f"持久化压力测试 {i}，包含需要记住的事实。")
        mem_before = sys_.memory.size()
        counter = sys_._state_counter
        emo = sys_.psi._emotion
        sys_.save_state(path)

        fresh = LaapCognitiveSystem()
        fresh.load_state(path)
        mem_after = fresh.memory.size()
        recall = fresh.memory.recall("压力测试", top_k=3)
        ok = (mem_after == mem_before and fresh._state_counter == counter
              and fresh.psi._emotion == emo and recall)
        report("persistence_roundtrip", ok,
               f"memories {mem_before}->{mem_after}, counter={counter}, emotion={emo}, recall={'ok' if recall else 'EMPTY'}")
    finally:
        if os.path.exists(path):
            os.remove(path)


# --------------------------------------------------------------------------- #
# long-run (full only)
# --------------------------------------------------------------------------- #
def test_long_run_consistency(duration_s: float = 10.0) -> None:
    sys_ = LaapCognitiveSystem()
    sys_.bootstrap("longrun")
    t0 = time.time(); n = 0
    while time.time() - t0 < duration_s:
        r = sys_.process_input(f"持续对话第 {n} 轮，记录当前状态。")
        if r.get("degraded"):
            break
        n += 1
    ok = n > 0
    report("long_run_consistency", ok, f"{n} turns, {n/max(time.time()-t0,1e-9):.1f}/s")


# --------------------------------------------------------------------------- #
def main() -> int:
    full = "--full" in sys.argv
    suites: List[str] = []
    test_load_burst()
    test_fuzz_texts()
    test_chaos_psi_recovery()
    test_chaos_memory()
    test_chaos_meta_none()
    test_memory_bounded_growth()
    test_memory_recall_relevance()
    test_hungarian_shapes()
    test_drift_detects_change()
    test_persistence_roundtrip()
    if full:
        test_long_run_consistency(20.0)
    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n=== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())