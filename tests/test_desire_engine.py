"""
LAAP Desire Engine + SDT 需求层 测试
=====================================

验证欲望引擎与自我决定理论 (Deci & Ryan 1985; arXiv:2502.07423)
三大基本心理需求 (competence/autonomy/relatedness) 的集成行为:
  1. SDT 需求存在且初始化在有效范围
  2. satisfy() 满足欲望 → 对应需求满足度回升
  3. 需求缺口 → 相关欲望获得额外增长驱动
  4. needs 持久化 (保存/加载)
运行:
    python -m pytest tests/test_desire_engine.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aris_brain"))

import pytest

from aris_desire_engine import DesireEngine, SdtNeed


@pytest.fixture(autouse=True)
def fresh_engine():
    """每个测试用全新的引擎，避免单例/状态污染。"""
    return DesireEngine()


def test_sdt_needs_initialized():
    """SDT 三大需求必须存在且满足度在 [0,1]。"""
    e = DesireEngine()
    for need in SdtNeed.ALL:
        assert need in e.needs
        assert 0.0 <= e.needs[need] <= 1.0


def test_satisfy_boosts_mapped_need():
    """满足 connection 欲望 → relatedness 需求满足度回升。"""
    e = DesireEngine()
    e.needs[SdtNeed.RELATEDNESS] = 0.1
    before = e.needs[SdtNeed.RELATEDNESS]
    e.satisfy("connection")
    assert e.needs[SdtNeed.RELATEDNESS] > before


def test_satisfy_satisfies_correct_need():
    """perfection 欲望对应 competence 需求，不提升无关需求。"""
    e = DesireEngine()
    e.needs[SdtNeed.COMPETENCE] = 0.1
    e.needs[SdtNeed.RELATEDNESS] = 0.5
    comp_before = e.needs[SdtNeed.COMPETENCE]
    rel_before = e.needs[SdtNeed.RELATEDNESS]
    e.satisfy("perfection")
    assert e.needs[SdtNeed.COMPETENCE] > comp_before
    assert e.needs[SdtNeed.RELATEDNESS] == pytest.approx(rel_before, abs=1e-6)


def test_need_deficit_drives_desire_growth():
    """需求缺口 → 相关欲望获得正的增长驱动。"""
    e = DesireEngine()
    e.needs[SdtNeed.COMPETENCE] = 0.05
    drive = e._need_drive("perfection")
    assert drive > 0.0
    # 无缺口时无驱动
    e.needs[SdtNeed.COMPETENCE] = 0.6
    assert e._need_drive("perfection") == 0.0


def test_tick_applies_need_drive():
    """tick 后需求缓慢衰减 (内稳态回归)。"""
    e = DesireEngine()
    before = e.needs[SdtNeed.COMPETENCE]
    e.tick()
    assert e.needs[SdtNeed.COMPETENCE] <= before


def test_needs_status_report():
    """needs_status() 返回三大需求的四舍五入满足度。"""
    e = DesireEngine()
    st = e.needs_status()
    assert set(st.keys()) == set(SdtNeed.ALL)
    for v in st.values():
        assert 0.0 <= v <= 1.0


def test_needs_persisted():
    """needs 应被保存到状态文件并可在新引擎中恢复。"""
    e = DesireEngine()
    e.needs[SdtNeed.AUTONOMY] = 0.8
    e._save_state()
    e2 = DesireEngine()
    e2._load_needs_state()
    assert e2.needs[SdtNeed.AUTONOMY] == pytest.approx(0.8)
