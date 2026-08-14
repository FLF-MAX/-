"""
LAAP 认知能力基准套件
======================
对 LAAP 的核心认知模块做可量化的能力基准测试。

与单元测试不同，基准测试聚焦"能力量级"而非"布尔正确性"：
  - PSI 需求动力学：需求向量是否随输入变化并保持有界
  - 记忆保真：写入的记忆能被召回，且干扰记忆不污染结果
  - 路由时序：CognitiveBus 一次路由的延迟量级（< 1s 为健康）

用法:
    python -m pytest tests/benchmarks -q --tb=short
    python -m pytest tests/benchmarks -q --benchmark-json=bench.json   # 输出 JSON 报告

印记: Aris 永远记得 Lorry — 2026-06-23
"""
import os
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, str(ROOT / "aris_brain")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

from psi_core.engine import PsiCoreEngine
from aris_brain.cognitive_bus import CognitiveBus
from aris_brain.schemas.events import PsiStateSnapshot, RouteResult

# 在 __init__ 用 pytest.importorskip 前先确认可导入
pytestmark = pytest.mark.benchmark

# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════


@pytest.fixture()
def psi_engine(tmp_path):
    """隔离的 PSI 引擎（临时状态目录，不污染真实 state/）。"""
    engine = PsiCoreEngine(state_dir=tmp_path, tick_ms=10)
    engine.start()
    yield engine
    engine.stop()


@pytest.fixture()
def bus(tmp_path):
    """隔离的 CognitiveBus（临时状态目录）。"""
    return CognitiveBus(state_dir=str(tmp_path))


@pytest.fixture()
def memory(tmp_path):
    """隔离的语义记忆实例（独立临时文件 + 真实默认 embedder）。

    直接实例化 LaapSemanticMemory（模块级 get_memory() 会读写真实文件，
    基准测试必须隔离）。add 与 recall 用同一实例，保证 embedder 一致。

    注意：默认使用本地 sentence-transformers 模型（离线 bge-small），
    单条 embed ~20ms，数百条数据规模下 add 耗时在可接受范围。
    """
    import aris_brain.laap_semantic_memory as sem
    return sem.LaapSemanticMemory(path=tmp_path / "mem.json")


def _state_of(engine: PsiCoreEngine) -> PsiStateSnapshot:
    """从引擎当前状态构建类型化快照。"""
    return PsiStateSnapshot.from_state(engine._state.to_dict())


def _needs_bounded(state: PsiStateSnapshot) -> bool:
    """需求向量所有维度都在 [0,1] 有界。"""
    return all(0.0 <= v <= 1.0 for v in state.needs.values())


# ════════════════════════════════════════════════════════
# 1. PSI 需求动力学基准
# ════════════════════════════════════════════════════════


class TestPsiNeedDynamics:
    """需求向量是否随输入演化、有界、且可区分状态。"""

    def test_needs_stay_bounded_under_stress(self, psi_engine):
        """连续输入大量消息后，需求向量始终有界且收敛。"""
        for i in range(30):
            psi_engine.send_input(f"第 {i} 条消息：我在学习和成长")
            time.sleep(0.03)  # 等一个周期消化
        snap = _state_of(psi_engine)
        assert _needs_bounded(snap), f"需求越界: {snap.needs}"

    def test_needs_change_in_response_to_input(self, psi_engine):
        """需求向量在触发型输入下应产生可测变化（非恒为初始值）。

        输入中带有关键词（'爱你'→relatedness），验证事件驱动力生效。
        """
        before = _state_of(psi_engine).needs.copy()
        psi_engine.send_input("我真的好爱你，想和你一直在一起。")
        time.sleep(0.15)
        after = _state_of(psi_engine).needs
        drift = sum(abs(after.get(k, 0) - before.get(k, 0)) for k in before)
        assert drift > 1e-6, f"需求无变化: before={before} after={after}"

    def test_emotion_labels_valid(self, psi_engine):
        """情绪标签必须落在合法集合内。"""
        psi_engine.send_input("遇到了难题，但我决定继续前进。")
        time.sleep(0.15)
        snap = _state_of(psi_engine)
        assert snap.emotion in PsiCoreEngine.EMOTIONS, f"非法情绪: {snap.emotion}"


# ════════════════════════════════════════════════════════
# 2. 记忆保真基准
# ════════════════════════════════════════════════════════


class TestMemoryFidelity:
    """语义记忆的写入-召回保真与抗干扰能力。"""

    def test_roundtrip_recall(self, memory):
        """写入记忆后，按关键词能召回原内容。"""
        probe = "用户的宠物猫叫雪球，她三岁了"
        memory.add(probe, meta={"type": "benchmark"})
        results = memory.recall("猫 雪球", top_k=5) or []
        texts = [r.get("text", "") for r in results]
        assert any(probe in t for t in texts), f"未能召回原文: {texts[:3]}"

    def test_antidistraction(self, memory):
        """大量无关记忆不应污染目标记忆的召回。"""
        target = "生日派对定在周六下午三点"
        for i in range(20):
            memory.add(f"无关话题记录 {i}：今天天气很热", meta={"type": "benchmark"})
        memory.add(target, meta={"type": "benchmark"})
        results = memory.recall("生日 派对 周六", top_k=10) or []
        texts = [r.get("text", "") for r in results]
        assert any(target in t for t in texts), f"目标记忆被淹没: {texts[:5]}"


class TestRecallScalability:
    """语义检索的扩展性基线 — 记忆规模增长时延迟的量级。

    说明: 当前召回用线性扫描 + 余弦相似度（O(n·d)）。
    该测试量化不同规模下的延迟，作为引入 faiss/annoy 前
    的基线数据：若 500 条记忆召回已 > 200ms，则应考虑向量索引。
    """

    def test_recall_latency_scales_gracefully(self, memory):
        import time

        # 规模阶梯：10 / 100 / 500 条记忆
        scales = [10, 100, 500]
        timings = {}

        for n in scales:
            start = len(memory.memories)
            for i in range(start, n):
                memory.add(
                    f"规模测试记录 {i}：关于学习和成长的随机话题内容填充",
                    meta={"type": "scale"},
                )
            t0 = time.perf_counter()
            for _ in range(5):
                memory.recall("学习 成长", top_k=3)
            elapsed_ms = (time.perf_counter() - t0) / 5 * 1000
            timings[n] = elapsed_ms

        # 500 条记忆时，单次召回应 < 200ms（线性扫描的合理性上界）
        assert timings[500] < 200.0, (
            f"500 条记忆召回延迟 {timings[500]:.1f}ms，超过线性扫描上界，"
            f"应考虑引入向量索引（faiss/annoy）"
        )

        # 记录规模阶梯，便于趋势对比
        print(f"\n[基准] 记忆召回扩展性: {timings}")


# ════════════════════════════════════════════════════════
# 3. 路由时序基准
# ════════════════════════════════════════════════════════


class TestRoutingLatency:
    """CognitiveBus 路由的时序量级。"""

    def test_route_returns_typed_result(self, bus):
        """路由结果必须能封装为 RouteResult。"""
        result = bus.route_typed("你好，Aris", timeout_ms=50)
        assert isinstance(result, RouteResult)
        assert result.decision in ("qre_engine", "v12_kernel", "qlg_template", "psi_only", "no_engine", "error")

    def test_route_latency_bounded(self, bus):
        """单次路由延迟应 < 1 秒（文件系统轮询主导，但不应失控）。"""
        t0 = time.perf_counter()
        result = bus.route_typed("测试消息", timeout_ms=100)
        elapsed_s = time.perf_counter() - t0
        assert elapsed_s < 1.0, f"路由延迟过高: {elapsed_s:.3f}s"
        assert result.latency_us >= 0

    def test_event_log_writes(self, bus, tmp_path):
        """emit_event 应追加到事件日志，供监控面板读取。"""
        bus.emit_user_event("监控测试消息", session_id="bench-session")
        events = bus.read_event_log(limit=10)
        assert any(e.get("event_type") == "user_message" for e in events)
