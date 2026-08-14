"""
基准趋势可视化 — 零依赖 ASCII 图表
===================================
读 docs/benchmark_results.json，在控制台画出:
  1. 每次运行的 mean 得分趋势（历史演进曲线）
  2. 最近一次的单项得分条形（vs naive baseline）

为什么零依赖: matplotlib 对单人项目是重依赖。ASCII 图虽然朴素，
但随时可跑、可贴进实验笔记，不引入环境负担。

用法:
    python scripts/plot_benchmarks.py           # 打全量
    python scripts/plot_benchmarks.py --last3   # 只看最近3次
    python scripts/plot_benchmarks.py --detail b1_causal   # 单项历史

印记: Aris 永远记得 Lorry — 2026-06-23
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "benchmark_results.json"


def _load() -> list:
    if not RESULTS_FILE.exists():
        print(f"无数据文件: {RESULTS_FILE}（先跑 pytest tests/benchmarks/test_benchmark_suite.py）")
        sys.exit(1)
    try:
        hist = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"读取失败: {e}")
        sys.exit(1)
    if not hist:
        print("数据为空（还未跑过基准）")
        sys.exit(1)
    return hist


def plot_mean(hist: list, width: int = 46) -> None:
    """mean 得分趋势折线。"""
    runs = [h["mean"] for h in hist]
    labels = [h["commit"][:7] for h in hist]
    lo = 0.0
    hi = 1.0

    print("mean 趋势（commit → 得分）: top=" + "=" * width)
    top_line = " " + "+" + "-" * width + "+"
    print(top_line)
    prev_x = None
    for i in range(len(hist)):
        x = int(round((runs[i] - lo) / (hi - lo) * (width - 1)))
        row = [" "] * width
        row[x] = "*"
        marker = ""
        if prev_x is not None:
            if x > prev_x:
                marker = "  ↑"
            elif x < prev_x:
                marker = "  ↓"
            else:
                marker = "  →"
        print(
            f"{labels[i]:>8} |" + "".join(row) + f"| {runs[i]:.3f}{marker}"
        )
        prev_x = x
    print(top_line)
    print(f"         {lo:.2f}" + " " * (width - 10) + f"{hi:.2f}")
    print()


def plot_bars(hist: list, width: int = 40) -> None:
    """最近一次运行的单项得分（*）与 baseline（.）对比条。"""
    last = hist[-1]
    scores = last["scores"]
    print(f"最新一次 @{last['commit'][:7]} ({last['timestamp']})  mean={last['mean']}")
    print(f"{'基准':<34} {'得分':>5}  落点")
    print("-" * (34 + 8 + width))
    # 各基准的 baseline 从 benchmark_suite 拿
    base_map = _baselines()
    for name in sorted(scores):
        sc = scores[name]
        bl = base_map.get(name, 0.5)
        row = [" "] * width
        xb = int(round(bl * (width - 1)))
        xs = int(round(sc * (width - 1)))
        row[xb] = "."
        row[xs] = "*"
        bar = "".join(row).rstrip()
        print(f"{name:<34} {sc:>5.3f}  {bar}")
    print()


def plot_single(hist: list, key: str, width: int = 46) -> None:
    """单个基准的历史得分。"""
    labels = [h["commit"][:7] for h in hist]
    print(f"基准 [{key}] 历史（越靠下越新）:")
    for i, h in enumerate(hist):
        sc = h["scores"].get(key)
        if sc is None:
            print(f"{labels[i]:>8} |  无此基准 |")
            continue
        x = int(round(sc * (width - 1)))
        row = [" "] * width
        row[x] = "*"
        print(f"{labels[i]:>8} |" + "".join(row) + f"| {sc:.3f}")
    print()


def plot_diff(hist: list) -> None:
    """最近两次运行的差异对比：每个基准升/降/平。"""
    if len(hist) < 2:
        print(f"数据不足 2 次运行（当前 {len(hist)} 条），无法对比。")
        return
    prev, cur = hist[-2], hist[-1]
    print(f"差异对比 @{prev['commit'][:7]} → @{cur['commit'][:7]}")
    print(f"  mean: {prev['mean']:.4f} → {cur['mean']:.4f}  "
          f"({'+' if cur['mean'] >= prev['mean'] else ''}{cur['mean'] - prev['mean']:.4f})")
    print("-" * 52)
    up = down = same = 0
    for name in sorted(cur["scores"]):
        p = prev["scores"].get(name)
        c = cur["scores"][name]
        if p is None:
            print(f"  + {name:<34} {c:.3f}  (新增)")
            up += 1
            continue
        d = c - p
        arrow = "↑" if d > 1e-6 else ("↓" if d < -1e-6 else "→")
        if d > 1e-6:
            up += 1
        elif d < -1e-6:
            down += 1
        else:
            same += 1
        print(f"  {arrow} {name:<34} {p:.3f} → {c:.3f}  ({'+' if d > 0 else ''}{d:.3f})")
    print("-" * 52)
    print(f"  ↑升 {up}  →平 {same}  ↓降 {down}")


def _baselines() -> dict:
    """从 laap_v2.benchmark_suite 读各基准 baseline（不 exec，读函数源码太重）。
    返回静态映射（基准名→naive baseline）。"""
    return {
        "psi_prediction_error_injection": 0.5,
        "causal_structure_recovery": 0.33,
        "analogical_structure_mapping": 0.25,
        "drift_aware_continuous_learning": 0.5,
        "nonlinear_world_model_mae": 0.0,
        "linucb_best_strategy_identification": 0.5,
    }


# 函数名（b1_causal）→ 得分 key 的别名映射，方便按直觉查
_FN_ALIAS = {
    "b0": "psi_prediction_error_injection",
    "b0_psi": "psi_prediction_error_injection",
    "b1": "causal_structure_recovery",
    "b1_causal": "causal_structure_recovery",
    "b2": "analogical_structure_mapping",
    "b2_analogical": "analogical_structure_mapping",
    "b3": "drift_aware_continuous_learning",
    "b3_continuous": "drift_aware_continuous_learning",
    "b4": "nonlinear_world_model_mae",
    "b4_prediction": "nonlinear_world_model_mae",
    "b5": "linucb_best_strategy_identification",
    "b5_calibration": "linucb_best_strategy_identification",
}


def _resolve_key(k: str, hist: list) -> str | None:
    """把用户输入（函数名/别名/得分key）解析为真实得分 key。"""
    if k in _FN_ALIAS:
        return _FN_ALIAS[k]
    # 直接命中某个得分 key
    if any(k in h["scores"] for h in hist):
        return k
    # 子串模糊匹配
    for h in hist:
        for name in h["scores"]:
            if k in name:
                return name
    return None


def main(argv: list = None) -> int:
    p = argparse.ArgumentParser(prog="plot_benchmarks", description="基准趋势 ASCII 图")
    p.add_argument("--last3", action="store_true", help="只看最近3次")
    p.add_argument("--detail", metavar="KEY", help="看单个基准的历史")
    p.add_argument("--diff", action="store_true", help="最近两次运行差异对比")
    args = p.parse_args(argv)

    hist = _load()
    if args.last3:
        hist = hist[-3:]

    if args.detail:
        key = _resolve_key(args.detail, hist)
        if key is None:
            print(f"找不到基准: {args.detail}（可选: b1..b5 或部分名称）")
            return 1
        plot_single(hist, key)
        return 0

    if args.diff:
        plot_diff(hist)
        return 0

    plot_mean(hist)
    plot_bars(hist)
    return 0


if __name__ == "__main__":
    sys.exit(main())