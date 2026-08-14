"""
CognitiveBus 压力/并发测试套件
===============================
验证认知总线在真实并发场景（多事件源、多请求、多周期）下的稳定性：

  1. emit_event 高并发 → JSONL 日志行不交织、无损坏
  2. route 并发 → _stats 计数不丢失（原子性）
  3. send_to_psi_core 并发 → input_queue 原子写，reader 永见完整 JSON

运行: python -m pytest tests/test_stress.py -q -v
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

# 确保仓库根目录在 sys.path 上（直接运行/其他启动方式一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from aris_brain.cognitive_bus import CognitiveBus
from aris_brain.schemas.events import CognitiveEvent, EventSource


@pytest.fixture()
def bus(tmp_path):
    """每个测试独立 state_dir，避免互相污染。"""
    return CognitiveBus(state_dir=str(tmp_path))


# ════════════════════════════════════════════════════════════
# 1. emit_event 并发写入 → JSONL 完整性
# ════════════════════════════════════════════════════════════

def test_concurrent_emit_event_jsonl_integrity(bus):
    """200 线程 × 50 事件并发追加，日志行必须逐行完整 JSON。"""
    n_threads, n_events = 200, 50

    def worker(tid):
        for i in range(n_events):
            ev = CognitiveEvent(
                event_type="stress",
                source=EventSource.COGNITIVE_BUS,
                payload={"tid": tid, "i": i, "pad": "x" * 64},
            )
            bus.emit_event(ev)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log_file = bus.state_dir / "agi_events.jsonl"
    assert log_file.exists()

    total = 0
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            # 每行必须可独立解析为完整 JSON —— 行交织会在这里抛错
            obj = json.loads(line)
            assert obj["payload"]["pad"] == "x" * 64
            assert obj["event_type"] == "stress"

    assert total == n_threads * n_events, (
        f"事件行数不符: 期望 {n_threads * n_events}, 实际 {total}"
    )


def test_concurrent_read_event_log_consistent(bus):
    """并发写同时读事件日志，读到的每一行都必须是完整 JSON。"""
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            ev = CognitiveEvent(
                event_type="rw",
                source=EventSource.COGNITIVE_BUS,
                payload={"i": i},
            )
            bus.emit_event(ev)
            i += 1

    def reader():
        while not stop.is_set():
            for item in bus.read_event_log(limit=50):
                if not isinstance(item, dict) or "event_type" not in item:
                    errors.append(item)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    readers = [threading.Thread(target=reader) for _ in range(2)]
    all_threads = threads + readers
    for t in all_threads:
        t.start()

    import time
    time.sleep(1.5)
    stop.set()
    for t in all_threads:
        t.join()

    assert not errors, f"读到损坏的事件行: {errors[:5]}"


# ════════════════════════════════════════════════════════════
# 2. route 并发 → _stats 计数原子性
# ════════════════════════════════════════════════════════════

def _fake_state(engine: str, response: str = "hello", latency_us: float = 10.0) -> dict:
    return {
        "psi_cycle": 1,
        "quantum_engine": engine,
        "quantum_response": response,
        "quantum_latency_us": latency_us,
        "emotion": "neutral",
        "arousal": 0.5,
        "self_presence": 0.5,
    }


def test_concurrent_route_stats_no_loss(bus):
    """300 次并发 route（全部命中 qre）→ route_count 必须精确等于 300。"""
    n_routes = 300

    def worker(_):
        # 预置 state 文件，route 直接命中 qre 分支
        with open(bus.state_file, "w", encoding="utf-8") as f:
            json.dump(_fake_state("qre_synth"), f)
        bus.route("hi", timeout_ms=1000)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_routes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = bus.stats()
    assert s["route_count"] == n_routes, (
        f"route_count 丢失: 期望 {n_routes}, 实际 {s['route_count']}"
    )
    assert s["qre_hits"] == n_routes, (
        f"qre_hits 丢失: 期望 {n_routes}, 实际 {s['qre_hits']}"
    )


# ════════════════════════════════════════════════════════════
# 3. send_to_psi_core 并发 → input_queue 原子写
# ════════════════════════════════════════════════════════════

def test_concurrent_send_to_psi_core_atomic(bus):
    """50 线程并发写 input_queue，最终文件必须是完整 JSON（不撕裂）。"""
    n_threads = 50

    def worker(tid):
        for _ in range(10):
            assert bus.send_to_psi_core(f"msg from {tid}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert bus.input_queue.exists()
    with open(bus.input_queue, "r", encoding="utf-8") as f:
        data = json.load(f)  # 撕裂会在这里抛 JSONDecodeError
    assert "text" in data
    assert "timestamp" in data
    # 临时文件不应残留
    assert not (bus.state_dir / "input_queue.json.tmp").exists()


# ════════════════════════════════════════════════════════════
# 4. 吞吐冒烟（性能基线锚点）
# ════════════════════════════════════════════════════════════

def test_emit_throughput_smoke(bus):
    """5000 次事件写入 ≤ 6s。

    这是 Windows 文件系统下的宽松吞吐基线（实测 ~1500 事件/s），
    仅作版本间趋势对比锚点，避免 CI 抖动导致误报。
    """
    import time

    t0 = time.perf_counter()
    for i in range(5000):
        ev = CognitiveEvent(
            event_type="throughput",
            source=EventSource.COGNITIVE_BUS,
            payload={"i": i},
        )
        bus.emit_event(ev)
    elapsed = time.perf_counter() - t0

    log_file = bus.state_dir / "agi_events.jsonl"
    count = sum(1 for _ in open(log_file, encoding="utf-8"))
    assert count == 5000
    assert elapsed < 6.0, f"5000 次事件写入耗时 {elapsed:.2f}s，超基线 6s"
