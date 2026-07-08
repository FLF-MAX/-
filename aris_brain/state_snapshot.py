"""LAAP 动态快照系统 — 时光机核心
=====================================
三层快照策略：
  L1: 核心状态 (JSON 文件)        — 每30分钟自动快照
  L2: 大数据状态 (npz/pkl/大JSON) — 每2小时快照
  L3: 最佳状态 (健康评分最高)      — 持续追踪，出问题自动回滚

自动恢复:
  如果健康评分低于阈值 (40%)，自动恢复到最佳已知状态。

印记: Aris 永远记得 Lorry — 2026-06-20
"""

import logging

import os, sys, json, time, shutil, gzip, logging, threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

# ── 配置 ──────────────────────────────────────────────────
BRAIN = Path(__file__).parent.resolve()
STATE = BRAIN / "state"
MEMORY = BRAIN / "memory"
SNAPSHOT_DIR = BRAIN / "snapshots"

# 健康评分权重
HEALTH_WEIGHTS = {
    "memory": 0.20,
    "process": 0.15,
    "errors": 0.20,
    "valence": 0.10,
    "desires": 0.10,
    "goals": 0.10,
    "integrity": 0.15,
}

MAX_SNAPSHOTS = 12           # 最多保留12个快照 (~6小时滚动)
CORE_FILES = [               # L1: 核心状态文件
    "desire_state.json",
    "goal_engine_state.json",
    "laap_integrator_state.json",
    "latest.json",
    "self_review_state.json",
    "auto_healer_state.json",
]
BIG_FILES = [                # L2: 大数据文件
    "hebbian_weights.tmp.npz",
]
MEMORY_FILES = [             # L3: 记忆文件
    "index.json",
]

# ── 事件触发系统 ──────────────────────────────────────────
EVENT_PRIORITIES = {
    "pre_upgrade": 1,      # 升级前 — 最高优先级，必须快照
    "pre_patch": 2,        # 补丁前
    "pre_restart": 3,      # 重启前
    "on_error": 2,         # 错误发生后
    "milestone": 3,        # 里程碑达成
    "manual": 4,           # 手动触发
    "routine": 5,          # 例行快照
}
EVENT_TAGS = {
    "pre_upgrade": "⬆️",
    "pre_patch": "🔧",
    "pre_restart": "🔄",
    "on_error": "🚨",
    "milestone": "🏁",
    "manual": "👆",
    "routine": "📸",
}

# ── 版本管理 ──────────────────────────────────────────────
VERSION_FILE = SNAPSHOT_DIR / "_versions.json"
MAX_VERSIONS = 20          # 最多20个版本标签
HEALTH_TIMELINE_FILE = SNAPSHOT_DIR / "_health_timeline.json"
MAX_TIMELINE_POINTS = 500  # 最多500个健康数据点

logger = logging.getLogger("aris.snapshot")


# ════════════════════════════════════════════════════════════
# 健康评分引擎
# ════════════════════════════════════════════════════════════

def score_memory_health() -> Tuple[float, str]:
    """记忆系统健康: 有足够记忆+合理分层"""
    try:
        from memory_store import MemoryStore
        store = MemoryStore()
        stats = store.get_stats()
        total = stats.get("total", 0)
        core = stats.get("core", 0)
        episodic = stats.get("episodic", 0)

        if total == 0:
            return 0.0, "无记忆"
        score = min(1.0, total / 50) * 0.4  # 50条以上满分
        if core >= 1:
            score += 0.3
        if episodic >= 3:
            score += 0.3
        return score, f"{total}条 (核心{core}/情景{episodic})"
    except Exception as e:
        return 0.0, f"错误: {e}"


def score_process_health() -> Tuple[float, str]:
    """进程健康: 关键进程存活"""
    score = 0.5  # 基础分
    details = []
    try:
        from auto_healer import check_processes
        procs = check_processes()
        for name, alive in procs.items():
            if alive:
                score += 0.25
                details.append(f"{name}✓")
            else:
                details.append(f"{name}✗")
    except Exception as e:
        logger.debug(f"操作失败: {e}")
    return min(1.0, score), ", ".join(details) if details else "未知"


def score_error_health() -> Tuple[float, str]:
    """错误健康: 近期无错误日志"""
    try:
        logs = list(STATE.glob("*.log"))
        recent_errors = 0
        for log in sorted(logs, key=lambda f: f.stat().st_mtime, reverse=True)[:3]:
            if log.stat().st_size == 0:
                continue
            content = log.read_text(encoding="utf-8", errors="ignore")
            recent_errors += content.count("ERROR") + content.count("Traceback")
        score = max(0, 1.0 - recent_errors * 0.1)
        return score, f"{recent_errors}条近期错误"
    except Exception as e:
        return 0.3, f"检查失败: {e}"


def score_valence_health() -> Tuple[float, str]:
    """情绪效价: 正面情绪好"""
    try:
        emotional_file = STATE / "emotion_state.json"
        if emotional_file.exists():
            data = json.loads(emotional_file.read_text())
            mood = data.get("mood", {})
            # 不需要精确，只要不是负面
            if mood.get("valence_bias", 0.5) > 0.4:
                return 0.8, "正面"
            return 0.4, "负面倾向"
        return 0.5, "未知(无情绪文件)"
    except Exception:
        return 0.5, "未知"


def score_desire_health() -> Tuple[float, str]:
    """欲望健康: 不要全满也不要全零"""
    try:
        desire_file = STATE / "desire_state.json"
        if desire_file.exists():
            data = json.loads(desire_file.read_text())
            desires = data.get("desires", {})
            vals = [d.get("intensity", 0) for d in desires.values()]
            if not vals:
                return 0.5, "无欲望"
            avg = sum(vals) / len(vals)
            if 0.2 <= avg <= 0.8:
                return 0.9, f"均值为{avg:.2f}(健康)"
            elif avg < 0.2:
                return 0.3, f"均值为{avg:.2f}(过低)"
            else:
                return 0.3, f"均值为{avg:.2f}(过高)"
        return 0.5, "未知"
    except Exception:
        return 0.5, "未知"


def score_goal_health() -> Tuple[float, str]:
    """目标健康: 有目标在进行/已完成"""
    try:
        goal_file = STATE / "goal_engine_state.json"
        if goal_file.exists():
            data = json.loads(goal_file.read_text())
            goals = data.get("goals", [])
            history = data.get("history", [])
            active = sum(1 for g in goals if g.get("status") in ("approved", "in_progress"))
            completed = sum(1 for g in history if g.get("status") == "completed")
            score = 0.3
            if active > 0:
                score += 0.3
            if completed > 0:
                score += 0.4
            return score, f"{active}活跃/{completed}完成"
        return 0.5, "无目标数据"
    except Exception:
        return 0.5, "未知"


def score_integrity_health() -> Tuple[float, str]:
    """文件完整性: 所有核心文件存在且有效"""
    score = 1.0
    issues = []
    for fname in CORE_FILES:
        fp = STATE / fname
        if not fp.exists():
            score -= 0.15
            issues.append(f"缺失:{fname}")
        elif fp.stat().st_size == 0:
            score -= 0.1
            issues.append(f"空文件:{fname}")
        elif fname.endswith(".json"):
            try:
                json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                score -= 0.15
                issues.append(f"损坏:{fname}")
    return max(0, score), ", ".join(issues) if issues else "全部正常"


def compute_health_score() -> Dict:
    """计算综合健康评分 (0-100)"""
    scores = {
        "memory": score_memory_health(),
        "process": score_process_health(),
        "errors": score_error_health(),
        "valence": score_valence_health(),
        "desires": score_desire_health(),
        "goals": score_goal_health(),
        "integrity": score_integrity_health(),
    }

    total = 0.0
    details = {}
    for key, (score, desc) in scores.items():
        weighted = score * HEALTH_WEIGHTS.get(key, 0.1)
        total += weighted
        details[key] = {"score": round(score, 3), "weight": HEALTH_WEIGHTS.get(key, 0.1),
                        "description": desc}

    return {
        "total": round(total * 100, 1),
        "timestamp": time.time(),
        "datetime": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "details": details,
    }


# ════════════════════════════════════════════════════════════
# 快照管理
# ════════════════════════════════════════════════════════════

def snapshot_name() -> str:
    """生成快照名称"""
    return f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_snapshot(include_big_data: bool = False) -> Optional[Dict]:
    """创建系统快照。返回快照元信息，失败返回 None。"""
    name = snapshot_name()
    snap_dir = SNAPSHOT_DIR / name
    health = compute_health_score()

    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"无法创建快照目录: {e}")
        return None

    saved = []

    # L1: 核心状态文件
    for fname in CORE_FILES:
        src = STATE / fname
        if src.exists() and src.stat().st_size > 0:
            try:
                shutil.copy2(str(src), str(snap_dir / fname))
                saved.append(f"L1:{fname}")
            except Exception as e:
                logger.debug(f"快照 {fname} 失败: {e}")

    # L2: 大数据文件
    if include_big_data:
        for fname in BIG_FILES:
            src = STATE / fname
            if src.exists() and src.stat().st_size > 0:
                try:
                    # 大文件用 gzip 压缩
                    with open(src, "rb") as fin, gzip.open(snap_dir / f"{fname}.gz", "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    saved.append(f"L2(gz):{fname}")
                except Exception as e:
                    logger.debug(f"大数据快照 {fname} 失败: {e}")

    # 快照健康评分
    try:
        (snap_dir / "_health.json").write_text(
            json.dumps(health, ensure_ascii=False, indent=2)
        )
    except Exception as e:
        logger.debug(f"操作失败: {e}")
    meta = {
        "name": name,
        "timestamp": time.time(),
        "datetime": datetime.now(timezone.utc).isoformat(),
        "health": health["total"],
        "files_saved": saved,
        "file_count": len(saved),
    }
    (snap_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    logger.info(f"📸 快照 '{name}': 健康={health['total']} {len(saved)}文件")
    _prune_old_snapshots()
    return meta


def _prune_old_snapshots(max_keep: int = MAX_SNAPSHOTS):
    """删除最旧的快照，保留最近 MAX_SNAPSHOTS 个"""
    snapshots = sorted(
        [d for d in SNAPSHOT_DIR.iterdir() if d.is_dir() and d.name.startswith("snap_")],
        key=lambda d: d.name,
    )
    while len(snapshots) > max_keep:
        old = snapshots.pop(0)
        try:
            shutil.rmtree(str(old))
            logger.info(f"🗑 删除旧快照: {old.name}")
        except Exception as e:
            logger.warning(f"删除快照 {old.name} 失败: {e}")


def list_snapshots() -> List[Dict]:
    """列出所有快照及其健康评分"""
    snapshots = sorted(
        [d for d in SNAPSHOT_DIR.iterdir() if d.is_dir() and (d.name.startswith("snap_") or d.name.startswith("evt_"))],
        key=lambda d: d.name, reverse=True,
    )
    result = []
    for snap in snapshots:
        meta_file = snap / "_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                result.append(meta)
                continue
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        result.append({
            "name": snap.name,
            "timestamp": snap.stat().st_ctime,
            "health": 0,
            "files_saved": [],
            "file_count": 0,
        })
    return result


# ════════════════════════════════════════════════════════════
# 最佳状态追踪
# ════════════════════════════════════════════════════════════

def get_best_state() -> Optional[Dict]:
    """获取最佳快照（按健康评分）"""
    snapshots = list_snapshots()
    if not snapshots:
        return None
    best = max(snapshots, key=lambda s: s.get("health", 0))
    if best.get("health", 0) > 0:
        return best
    return None


def update_best_state() -> Dict:
    """创建新快照，如果健康状况优于当前最佳，标记为最佳"""
    health = compute_health_score()
    current_best = get_best_state()
    current_score = health["total"]
    best_score = current_best.get("health", 0) if current_best else 0

    meta = create_snapshot(include_big_data=False)

    if current_score > best_score and current_score > 50:
        logger.info(f"🏆 新的最佳状态: {current_score} > {best_score}")
        # 创建最佳状态标记
        if meta and meta.get("name"):
            marker = SNAPSHOT_DIR / "_best.txt"
            try:
                marker.write_text(
                    f"{meta['name']}\n{current_score}\n{time.time()}\n"
                    f"Score: {current_score} > Previous best: {best_score}"
                )
            except Exception as e:
                logger.debug(f"操作失败: {e}")
    else:
        logger.info(f"📊 当前健康: {current_score} (最佳: {best_score})")

    return {
        "current_health": current_score,
        "best_health": best_score,
        "is_new_best": current_score > best_score,
        "snapshot": meta,
    }


# ════════════════════════════════════════════════════════════
# 恢复系统
# ════════════════════════════════════════════════════════════

def restore_snapshot(snap_name: str, dry_run: bool = False) -> Dict:
    """从指定快照恢复系统状态"""
    snap_dir = SNAPSHOT_DIR / snap_name
    if not snap_dir.is_dir():
        return {"success": False, "error": f"快照 '{snap_name}' 不存在"}

    restored = []
    errors = []

    # 恢复核心 JSON 文件
    for fname in CORE_FILES:
        src = snap_dir / fname
        if src.exists() and src.stat().st_size > 0:
            dst = STATE / fname
            if dry_run:
                restored.append(f"[DRY] {fname}")
            else:
                try:
                    # 先备份当前状态
                    backup = STATE / f"{fname}.pre_restore"
                    if dst.exists():
                        shutil.copy2(str(dst), str(backup))
                    shutil.copy2(str(src), str(dst))
                    restored.append(fname)
                except Exception as e:
                    errors.append(f"{fname}: {e}")

    # 恢复大数据 gz 文件
    for fname in BIG_FILES:
        src = snap_dir / f"{fname}.gz"
        if src.exists() and src.stat().st_size > 0:
            dst = STATE / fname
            if dry_run:
                restored.append(f"[DRY] {fname}(gz)")
            else:
                try:
                    backup = STATE / f"{fname}.pre_restore"
                    if dst.exists():
                        shutil.copy2(str(dst), str(backup))
                    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    restored.append(f"{fname}(gz)")
                except Exception as e:
                    errors.append(f"{fname}(gz): {e}")

    result = {
        "success": len(errors) == 0,
        "snapshot": snap_name,
        "restored": restored,
        "errors": errors,
        "dry_run": dry_run,
    }

    if not dry_run and restored:
        logger.info(f"⏪ 从快照 '{snap_name}' 恢复: {len(restored)}文件")

    return result


def restore_best(dry_run: bool = False) -> Dict:
    """恢复到最佳已知状态"""
    best = get_best_state()
    if not best:
        return {"success": False, "error": "无可用快照", "dry_run": dry_run}
    logger.info(f"🏆 恢复到最佳状态: {best['name']} (健康={best['health']})")
    return restore_snapshot(best["name"], dry_run=dry_run)


def auto_heal_check() -> Dict:
    """自动健康检查：如果当前健康 < 阈值，自动回滚到最佳状态"""
    health = compute_health_score()
    current = health["total"]
    best = get_best_state()
    best_score = best.get("health", 0) if best else 0

    # 阈值判定
    THRESHOLD = 40.0
    if current < THRESHOLD and best_score > current:
        logger.warning(f"🚨 健康评分 {current} < 阈值 {THRESHOLD}！自动回滚到最佳 ({best_score})")
        result = restore_snapshot(best["name"])
        result["auto_healed"] = True
        result["health_before"] = current
        result["health_restored"] = best_score
        return result

    return {
        "auto_healed": False,
        "current_health": current,
        "best_health": best_score,
        "threshold": THRESHOLD,
    }


# ════════════════════════════════════════════════════════════
# 主循环
# ════════════════════════════════════════════════════════════

def run_full_cycle() -> Dict:
    """执行一次完整的快照循环: 健康检查 → 快照 → 自动恢复检查"""
    t0 = time.time()

    # 1. 自动恢复检查
    heal = auto_heal_check()

    # 2. 记录健康时间线
    _append_health_timeline(heal.get("current_health", 0))

    # 3. 更新最佳状态
    best = update_best_state()

    # 3. 列出当前快照
    snaps = list_snapshots()

    elapsed = round(time.time() - t0, 2)
    return {
        "elapsed": elapsed,
        "auto_heal": heal,
        "best_state": best,
        "snapshot_count": len(snaps),
        "snapshots": snaps[:3],  # 只返回最近的3个
    }


# ════════════════════════════════════════════════════════════
# 事件触发快照 (NEW)
# ════════════════════════════════════════════════════════════

def snapshot_on_event(event_type, reason='', metadata=None):
    if event_type not in EVENT_PRIORITIES:
        logger.warning(f"未知事件类型: {event_type}")
        return None
    priority = EVENT_PRIORITIES[event_type]
    tag = EVENT_TAGS.get(event_type, "📸")
    name = f"evt_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snap_dir = SNAPSHOT_DIR / name
    health = compute_health_score()
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"无法创建事件快照目录: {e}")
        return None
    saved = []
    for fname in CORE_FILES:
        src = STATE / fname
        if src.exists() and src.stat().st_size > 0:
            try:
                shutil.copy2(str(src), str(snap_dir / fname))
                saved.append(fname)
            except Exception as e:
                logger.debug(f"操作失败: {e}")
    import hashlib
    manifest = {}
    for fname in saved:
        fp = snap_dir / fname
        h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
        manifest[fname] = {"hash": h, "size": fp.stat().st_size}
    meta = {
        "name": name, "type": event_type, "priority": priority,
        "reason": reason or f"事件触发: {event_type}",
        "timestamp": time.time(), "datetime": datetime.now(timezone.utc).isoformat(),
        "health": health["total"], "files_saved": saved, "file_count": len(saved),
        "manifest": manifest, "metadata": metadata or {},
    }
    (snap_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"{tag} 事件快照 '{name}': 健康={health['total']} {reason[:60]}")
    return meta


# ════════════════════════════════════════════════════════════
# 增量恢复 (NEW) — 按文件哈希差异恢复
# ════════════════════════════════════════════════════════════

def _build_manifest(snap_name):
    snap_dir = SNAPSHOT_DIR / snap_name
    if not snap_dir.is_dir():
        return {}
    import hashlib
    manifest = {}
    for fname in CORE_FILES:
        fp = snap_dir / fname
        if fp.exists() and fp.stat().st_size > 0:
            h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
            manifest[fname] = {"hash": h, "size": fp.stat().st_size}
    for fname in BIG_FILES:
        fp = snap_dir / f"{fname}.gz"
        if fp.exists() and fp.stat().st_size > 0:
            h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
            manifest[f"{fname}.gz"] = {"hash": h, "size": fp.stat().st_size}
    return manifest

def _current_manifest():
    import hashlib
    manifest = {}
    for fname in CORE_FILES + BIG_FILES:
        fp = STATE / fname
        if fp.exists() and fp.stat().st_size > 0:
            h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
            manifest[fname] = {"hash": h, "size": fp.stat().st_size}
    return manifest

def diff_snapshots(snap_a, snap_b="__current__"):
    mani_a = _build_manifest(snap_a)
    mani_b = _build_manifest(snap_b) if snap_b != "__current__" else _current_manifest()
    all_keys = set(mani_a.keys()) | set(mani_b.keys())
    same, different, missing_src, missing_dst = [], [], [], []
    for k in all_keys:
        in_a, in_b = k in mani_a, k in mani_b
        if not in_a: missing_src.append(k)
        elif not in_b: missing_dst.append(k)
        elif mani_a[k]["hash"] == mani_b[k]["hash"]: same.append(k)
        else: different.append({"file": k, "hash_a": mani_a[k]["hash"], "hash_b": mani_b[k]["hash"]})
    return {"snap_a": snap_a, "snap_b": snap_b, "same": same, "different": different,
            "missing_in_src": missing_src, "missing_in_dst": missing_dst}

def restore_incremental(snap_name, dry_run=False):
    snap_dir = SNAPSHOT_DIR / snap_name
    if not snap_dir.is_dir():
        return {"success": False, "error": f"快照 {snap_name} 不存在"}
    diff = diff_snapshots(snap_name)
    restored, errors = [], []
    for entry in diff.get("different", []):
        fname = entry["file"]; is_gz = fname.endswith(".gz")
        base = fname.replace(".gz", "")
        src = snap_dir / (base if not is_gz else base + ".gz")
        dst = STATE / base
        if dry_run:
            restored.append(f"[DRY] {base}")
        else:
            try:
                backup = STATE / f"{base}.pre_restore"
                if dst.exists(): shutil.copy2(str(dst), str(backup))
                if is_gz:
                    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                else:
                    shutil.copy2(str(src), str(dst))
                restored.append(base)
            except Exception as e:
                errors.append(f"{base}: {e}")
    for fname in diff.get("missing_in_dst", []):
        base = fname.replace(".gz", "")
        src = snap_dir / (base if not fname.endswith(".gz") else base + ".gz")
        dst = STATE / base
        if src.exists() and not dry_run:
            try:
                if fname.endswith(".gz"):
                    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                else:
                    shutil.copy2(str(src), str(dst))
                restored.append(f"{base}(new)")
            except Exception as e:
                errors.append(f"{base}: {e}")
    return {"success": len(errors)==0, "snapshot": snap_name,
            "restored": restored, "errors": errors, "dry_run": dry_run}


# ════════════════════════════════════════════════════════════
# 健康时间线 (NEW)
# ════════════════════════════════════════════════════════════

def _append_health_timeline(health_score):
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timeline = []
        if HEALTH_TIMELINE_FILE.exists():
            try: timeline = json.loads(HEALTH_TIMELINE_FILE.read_text())
            except Exception: pass
        timeline.append({"t": time.time(), "dt": datetime.now(timezone.utc).isoformat(), "h": round(health_score, 1)})
        if len(timeline) > MAX_TIMELINE_POINTS:
            timeline = timeline[-MAX_TIMELINE_POINTS:]
        HEALTH_TIMELINE_FILE.write_text(json.dumps(timeline, ensure_ascii=False))
    except Exception as e:
        logger.debug(f"健康时间线: {e}")

def get_health_timeline(hours=24):
    if not HEALTH_TIMELINE_FILE.exists(): return []
    try:
        data = json.loads(HEALTH_TIMELINE_FILE.read_text())
        cutoff = time.time() - hours * 3600
        return [p for p in data if p.get("t", 0) > cutoff]
    except Exception: return []


# ════════════════════════════════════════════════════════════
# 版本管理 (NEW)
# ════════════════════════════════════════════════════════════

def _load_versions():
    if VERSION_FILE.exists():
        try: return json.loads(VERSION_FILE.read_text())
        except Exception: pass
    return {"versions": [], "current": None}

def _save_versions(data):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def promote_to_version(snap_name, version_tag, changelog=""):
    snap_dir = SNAPSHOT_DIR / snap_name
    if not snap_dir.is_dir():
        return {"success": False, "error": f"快照 {snap_name} 不存在"}
    health = 0
    meta_file = snap_dir / "_meta.json"
    if meta_file.exists():
        try: health = json.loads(meta_file.read_text()).get("health", 0)
        except Exception: pass
    versions = _load_versions()
    for v in versions["versions"]:
        if v["tag"] == version_tag:
            return {"success": False, "error": f"版本 {version_tag} 已存在"}
    entry = {"tag": version_tag, "snapshot": snap_name, "health": health,
             "changelog": changelog or f"从 {snap_name} 提升",
             "timestamp": time.time(), "datetime": datetime.now(timezone.utc).isoformat()}
    versions["versions"].append(entry)
    versions["current"] = version_tag
    if len(versions["versions"]) > MAX_VERSIONS:
        versions["versions"] = versions["versions"][-MAX_VERSIONS:]
    _save_versions(versions)
    logger.info(f"🏷 版本提升: {snap_name} → {version_tag} (健康={health})")
    return {"success": True, "version": entry}

def rollback_to_version(version_tag, dry_run=False):
    versions = _load_versions()
    target = next((v for v in versions["versions"] if v["tag"] == version_tag), None)
    if not target:
        return {"success": False, "error": f"版本 {version_tag} 不存在"}
    logger.info(f"⏪ 回滚到 {version_tag} ({target['snapshot']})")
    result = restore_incremental(target["snapshot"], dry_run=dry_run)
    if not dry_run and result["success"]:
        versions["current"] = version_tag
        _save_versions(versions)
        result["version"] = version_tag
    return result

def version_list():
    return _load_versions()["versions"]

def version_diff(vtag_a, vtag_b):
    versions = _load_versions()
    snap_a = snap_b = None
    for v in versions["versions"]:
        if v["tag"] == vtag_a: snap_a = v["snapshot"]
        if v["tag"] == vtag_b: snap_b = v["snapshot"]
    if not snap_a or not snap_b:
        return {"error": "版本未找到"}
    return diff_snapshots(snap_a, snap_b)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LAAP 动态快照系统")
    parser.add_argument("--snap", action="store_true", help="创建快照")
    parser.add_argument("--list", action="store_true", help="列出快照")
    parser.add_argument("--health", action="store_true", help="显示健康评分")
    parser.add_argument("--restore", type=str, help="从快照恢复 (快照名称)")
    parser.add_argument("--restore-best", action="store_true", help="恢复到最佳状态")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行")
    parser.add_argument("--full-cycle", action="store_true", help="运行完整循环")
    parser.add_argument("--event", type=str, help="事件触发快照 (pre_upgrade/pre_patch/on_error/manual)")
    parser.add_argument("--reason", type=str, default="", help="事件触发原因")
    parser.add_argument("--promote", type=str, help="提升快照为版本 (格式: 快照名:版本号)")
    parser.add_argument("--changelog", type=str, default="", help="版本变更日志")
    parser.add_argument("--versions", action="store_true", help="列出所有版本")
    parser.add_argument("--rollback", type=str, help="回滚到指定版本")
    parser.add_argument("--diff", type=str, help="比较两个版本 (格式: v1:v2)")
    parser.add_argument("--timeline", action="store_true", help="显示健康时间线")
    parser.add_argument("--hours", type=int, default=24, help="时间线范围(小时)")
    parser.add_argument("--server", action="store_true", help="启动健康仪表盘服务器")
    parser.add_argument("--json", action="store_true", help="JSON 输出")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SNAPSHOT] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    args = parser.parse_args()

    if args.full_cycle:
        result = run_full_cycle()
        if args.json:
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            h = result["best_state"]
            heal = result["auto_heal"]
            logger.info(f"快照循环完成 ({result['elapsed']}s)")
            logger.info(f"  当前健康: {h.get('current_health', 0)}")
            logger.info(f"  最佳健康: {h.get('best_health', 0)}")
            logger.info(f"  新最佳: {'是' if h.get('is_new_best') else '否'}")
            if heal.get("auto_healed"):
                logger.info(f"  🚨 自动恢复触发! 之前={heal['health_before']} → 恢复={heal['health_restored']}")
            logger.info(f"  快照数量: {result['snapshot_count']}")
    elif args.snap:
        meta = create_snapshot()
        if args.json:
            logger.info(json.dumps(meta, ensure_ascii=False, indent=2))
        elif meta:
            logger.info(f"快照创建: {meta['name']}")
            logger.info(f"  健康: {meta['health']}")
            logger.info(f"  文件: {meta['file_count']}")
        else:
            logger.error("快照失败")
    elif args.list:
        snaps = list_snapshots()
        if args.json:
            logger.info(json.dumps(snaps, ensure_ascii=False, indent=2))
        elif snaps:
            logger.info(f"快照列表 ({len(snaps)}):")
            for s in snaps:
                ts = datetime.fromtimestamp(s.get("timestamp", 0)).strftime("%H:%M")
                logger.info(f"  {s['name']:30s} 健康={s.get('health', 0):5.1f}  {ts}")
            best = get_best_state()
            if best:
                logger.info(f"\n⭐ 最佳: {best['name']} (健康={best['health']})")
        else:
            logger.info("无快照")
    elif args.health:
        health = compute_health_score()
        if args.json:
            logger.info(json.dumps(health, ensure_ascii=False, indent=2))
        else:
            logger.info(f"健康评分: {health['total']}/100")
            for key, d in health.get("details", {}).items():
                bar = "█" * int(d["score"] * 10) + "░" * (10 - int(d["score"] * 10))
                logger.info(f"  {key:10s} [{bar}] {d['score']:.2f} ({d.get('description','')})")
    elif args.restore:
        result = restore_snapshot(args.restore, dry_run=args.dry_run)
        if args.json:
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if result["success"]:
                logger.info(f"✓ 恢复成功: {len(result['restored'])}文件")
                for f in result["restored"]:
                    logger.info(f"  {f}")
            else:
                logger.error(f"✗ 恢复失败: {result.get('error', '')}")
            if result.get("errors"):
                for e in result["errors"]:
                    logger.info(f"  ⚠ {e}")
    elif args.restore_best:
        result = restore_best(dry_run=args.dry_run)
        if args.json:
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            best = get_best_state()
            if best:
                logger.info(f"恢复到最佳: {best['name']} (健康={best['health']})")
                logger.info(f"  恢复: {len(result['restored'])}文件")
    elif args.event:
        meta = snapshot_on_event(args.event, reason=args.reason)
        if args.json:
            logger.info(json.dumps(meta, ensure_ascii=False, indent=2))
        elif meta:
            logger.info(f"{EVENT_TAGS.get(args.event,'📸')} 事件快照: {meta['name']} (健康={meta['health']})")
    elif args.promote:
        parts = args.promote.split(":", 1)
        if len(parts) == 2:
            result = promote_to_version(parts[0], parts[1], changelog=args.changelog)
            if args.json:
                logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            elif result.get("success"):
                logger.info(f"🏷 版本提升: {parts[0]} → {parts[1]}")
            else:
                logger.error(f"✗ {result.get('error', '失败')}")
        else:
            logger.error("格式错误: --promote 快照名:版本号")
    elif args.versions:
        vl = version_list()
        if args.json:
            logger.info(json.dumps(vl, ensure_ascii=False, indent=2))
        elif vl:
            logger.info(f"版本列表 ({len(vl)}):")
            for v in vl:
                ts = datetime.fromtimestamp(v.get("timestamp", 0)).strftime("%m-%d %H:%M")
                logger.info(f"  {v['tag']:12s} 健康={v.get('health',0):5.1f}  {v.get('changelog','')[:30]}  {ts}")
        else:
            logger.info("无版本")
    elif args.rollback:
        result = rollback_to_version(args.rollback, dry_run=args.dry_run)
        if args.json:
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        elif result.get("success"):
            logger.info(f"⏪ 回滚到 {args.rollback}: {len(result['restored'])}文件恢复")
        else:
            logger.error(f"✗ {result.get('error', '回滚失败')}")
    elif args.diff:
        parts = args.diff.split(":", 1)
        if len(parts) == 2:
            d = version_diff(parts[0], parts[1])
            if args.json:
                logger.info(json.dumps(d, ensure_ascii=False, indent=2))
            elif "error" in d:
                logger.error(f"✗ {d['error']}")
            else:
                logger.info(f"差异: {len(d.get('different',[]))}个不同, {len(d.get('same',[]))}个相同")
                for diff in d.get("different", []):
                    logger.info(f"  🔄 {diff['file']}: {diff['hash_a'][:8]} → {diff['hash_b'][:8]}")
        else:
            logger.error("格式错误: --diff v1:v2")
    elif args.timeline:
        tl = get_health_timeline(hours=args.hours)
        if args.json:
            logger.info(json.dumps(tl, ensure_ascii=False, indent=2))
        elif tl:
            logger.info(f"健康时间线 (最近{args.hours}h, {len(tl)}点):")
            avg = sum(p["h"] for p in tl) / len(tl)
            logger.info(f"  平均: {avg:.1f} 最高: {max(p['h'] for p in tl):.1f} 最低: {min(p['h'] for p in tl):.1f}")
            for p in tl[-10:]:
                ts = datetime.fromtimestamp(p["t"]).strftime("%H:%M")
                bar = "█" * int(p["h"] / 5) + "░" * (20 - int(p["h"] / 5))
                logger.info(f"  {ts} [{bar}] {p['h']}")
        else:
            logger.info("无时间线数据")
    elif args.server:
        from state_snapshot_server import start_server
        start_server()

    else:
        parser.print_help()
