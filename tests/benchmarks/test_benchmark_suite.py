"""
laap_v2 B0-B5 基准收敛测试
=========================
把 laap_v2 的认知能力基准（B0-B5）接入主测试体系，产出一致、可积累的量化结果。

来源（真身，避免双份代码）:
    G:\\laap\\laap_v2\\benchmark_suite.py

本测试的职责不是重写基准，而是:
  1. 每个基准可独立验证（随机种子固定 → 结果可复现）
  2. 得分必须超过 naive naive baseline（证明架构确实有效，而非碰巧）
  3. 把结果写入 docs/benchmark_results.json，积累你自己的实验数据

运行:
    python -m pytest tests/benchmarks/test_benchmark_suite.py -q -s

印记: Aris 永远记得 Lorry — 2026-06-23
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LAAP_V2 = ROOT / "laap_v2"
if str(LAAP_V2) not in sys.path:
    sys.path.insert(0, str(LAAP_V2))

import pytest

from benchmark_suite import BENCHMARKS, run_all

RESULTS_FILE = ROOT / "docs" / "benchmark_results.json"
DESCRIPTION = {
    "b0_psi": "PSI 预测误差注入：growth 需求应随 PE 上升",
    "b1_causal": "因果结构恢复：从时序恢复温度→气压/湿度",
    "b2_analogical": "类比结构映射：太阳系↔原子的一致性/覆盖",
    "b3_continuous": "漂移感知连续学习：概念漂移后的最优策略率",
    "b4_prediction": "非线性世界模型 MAE：score=1-MAE/baseline",
    "b5_calibration": "LinUCB 最优策略识别：exploit 率",
}


# ════════════════════════════════════════════════════════
# 每个基准独立测试
# ════════════════════════════════════════════════════════

@pytest.mark.parametrize("fn", BENCHMARKS, ids=[f.__name__ for f in BENCHMARKS])
def test_benchmark_scores(fn):
    res = fn()
    assert "error" not in res, f"{fn.__name__} 抛异常: {res.get('error')}"
    assert res["score"] >= res["baseline"], (
        f"{fn.__name__}: score={res['score']:.3f} < baseline={res['baseline']}。"
        f"引擎在该任务上不如随机/朴素基线。"
    )


# ════════════════════════════════════════════════════════
# 汇总 + 实验数据积累
# ════════════════════════════════════════════════════════

def test_run_all_records_results():
    """跑全量 B0-B5 并把结果（含时间和 git commit）追加进实验结果 JSON。

    这是"自我实验笔记本"的数据层：每次跑都留痕，方便追踪引擎
    能力随迭代的变化（目前是强/弱/平的判断依据）。
    """
    results = run_all(verbose=False)
    assert len(results) == len(BENCHMARKS)
    assert all("score" in r for r in results)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        import subprocess
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    entry = {
        "timestamp": now,
        "commit": commit,
        "scores": {r["name"]: round(r["score"], 4) for r in results},
        "mean": round(sum(r["score"] for r in results) / len(results), 4),
    }

    history = []
    if RESULTS_FILE.exists():
        try:
            history = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(entry)
    RESULTS_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 打印当前得分（-s 查看）
    print(f"\n[基准] {now} @{commit} mean={entry['mean']}")
    for r in results:
        print(f"  {r['name']:<34} score={r['score']:.3f}  (baseline={r['baseline']})")