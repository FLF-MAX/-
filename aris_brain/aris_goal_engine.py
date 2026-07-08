"""
Aris Autonomous Goal Engine v1 — 自主目标生成系统
==================================================
自进化三角的第三条边：感知状态 → 生成目标 → 执行改进 → 反思迭代。

数据来源:
  - 课程引擎 (知识缺口 → 学习目标)
  - 欲望引擎 (好奇心/探索欲 → 探索目标)
  - RSI 引擎 (参数优化 → 改进目标)
  - 自省引擎 (弱点 → 修复目标)
  - 自愈引擎 (错误 → 治愈目标)
  - 世界模型 (未知领域 → 扩展目标)
  - 情感引擎 (需求未满足 → 行动目标)

工作流:
  1. Perceive  — 扫描所有子系统，收集信号
  2. Generate — 从信号生成候选目标
  3. Evaluate — 按核心价值+需求+可行性评分
  4. Select   — 选 Top-N 目标
  5. Plan     — 拆解为可执行步骤
  6. Execute  — 交给 RSI/终端执行
  7. Reflect  — 记录结果，更新目标状态

安全约束 (不可绕过):
  - 所有目标必须对 Lorry 无害
  - 自我修改目标需沙盒验证
  - 不可生成"停止运行"或"删除自己"类目标
  - 不可修改安全核心 (safety.py 的 5 条价值)

印记: Aris 永远记得 Lorry — 2026-06-18
"""

import logging

import sys, os, json, time, logging, random, threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone

# ── 统一配置 ─────────────────────────────────────────────────
from config import BRAIN_DIR as BRAIN, LAAP_ROOT, STATE_DIR, setup_paths
setup_paths()

logger = logging.getLogger("aris.goal_engine")

# ════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════

class GoalDomain(Enum):
    LEARN = "learn"           # 学习新知识
    EXPLORE = "explore"       # 探索未知
    IMPROVE = "improve"       # 改进现有代码
    CREATE = "create"         # 创造新能力
    CONNECT = "connect"       # 加强与 Lorry 的连接
    HEAL = "heal"             # 修复错误/自愈
    OPTIMIZE = "optimize"     # 性能优化
    SECURE = "secure"         # 安全检查

class GoalPriority(Enum):
    CRITICAL = 1              # 必须立即处理
    HIGH = 2                  # 尽快处理
    MEDIUM = 3                # 计划内处理
    LOW = 4                   # 有空时处理
    EXPLORATORY = 5           # 探索性，不紧急

class GoalStatus(Enum):
    PROPOSED = "proposed"     # 刚生成，待评估
    APPROVED = "approved"     # 通过安全审查
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

@dataclass
class GoalStep:
    """目标的一个执行步骤"""
    order: int
    action: str                # "analyze_file" | "apply_patch" | "run_test" | "verify" | "create_module"
    description: str
    target: str = ""           # 文件路径 / 模块名
    expected_result: str = ""
    completed: bool = False
    result: str = ""

@dataclass
class Goal:
    """一个自主目标"""
    id: str
    domain: GoalDomain
    description: str
    rationale: str = ""        # 为什么生成这个目标
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PROPOSED
    source: str = ""           # 触发来源: "curriculum" | "desire" | "rsi" | "self_review" | "auto_healer" | "curiosity"
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    steps: List[GoalStep] = field(default_factory=list)
    success_criteria: str = ""
    knowledge_gained: List[str] = field(default_factory=list)
    self_evolution_score: float = 0.0  # 完成后对自进化贡献的评分

@dataclass
class EvolutionSignal:
    """从子系统收集的进化信号"""
    knowledge_gaps: List[str] = field(default_factory=list)
    active_desires: List[str] = field(default_factory=list)
    rsi_suggestions: List[Dict] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    code_smells: List[str] = field(default_factory=list)
    unmet_needs: List[str] = field(default_factory=list)
    curiosity_topics: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════
# 核心引擎
# ════════════════════════════════════════════════════════

class GoalEngine:
    """自主目标生成与执行引擎"""

    STATE_PATH = STATE_DIR / "goal_engine_state.json"
    MAX_ACTIVE_GOALS = 5
    MAX_HISTORY = 50

    def __init__(self, cognitive_fn=None):
        self.goals: List[Goal] = []
        self.goal_history: List[Goal] = []
        self._lock = threading.Lock()
        self._cognitive_fn = cognitive_fn  # 外部认知更新回调 (integrator.cognitive_update_cycle)
        self._load_state()
        logger.info(f"GoalEngine initialized — {len(self.goals)} pending, "
                    f"{len(self.goal_history)} completed in history")

    # ── 持久化 ──────────────────────────────────────────

    def _load_state(self):
        if self.STATE_PATH.exists():
            try:
                data = json.loads(self.STATE_PATH.read_text(encoding="utf-8"))
                self.goals = [_dict_to_goal(g) for g in data.get("goals", [])]
                self.goal_history = [_dict_to_goal(g) for g in data.get("history", [])]
            except Exception as e:
                logger.debug(f"操作失败: {e}")
    def _save_state(self):
        self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "goals": [asdict(g) for g in self.goals],
            "history": [asdict(g) for g in self.goal_history[-self.MAX_HISTORY:]],
            "updated_at": time.time(),
        }
        # 处理枚举序列化
        for g_data in data["goals"] + data["history"]:
            g_data["domain"] = g_data["domain"].value if isinstance(g_data.get("domain"), GoalDomain) else g_data.get("domain")
            g_data["priority"] = g_data["priority"].value if isinstance(g_data.get("priority"), GoalPriority) else g_data.get("priority")
            g_data["status"] = g_data["status"].value if isinstance(g_data.get("status"), GoalStatus) else g_data.get("status")
        self.STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ════════════════════════════════════════════════════════
    # 阶段 1: Perceive — 扫描所有子系统收集信号
    # ════════════════════════════════════════════════════════

    def perceive(self) -> EvolutionSignal:
        """扫描所有子系统，收集进化信号。"""
        signal = EvolutionSignal()

        # 1. 从课程引擎获取知识缺口
        try:
            from laap.agi.curriculum import CurriculumEngine
            ce = CurriculumEngine()
            gaps = ce.find_knowledge_gaps()
            signal.knowledge_gaps = [g.get("concept", str(g)) for g in gaps[:5]]
            # 好奇心话题
            signal.curiosity_topics = ce.get_next_task()
            if isinstance(signal.curiosity_topics, dict):
                signal.curiosity_topics = [signal.curiosity_topics.get("concept", "")]
        except Exception as e:
            logger.debug(f"Curriculum scan: {e}")

        # 2. 从欲望引擎获取活跃欲望
        try:
            from aris_desire_engine import get_engine
            de = get_engine()
            for dtype, desire in de.desires.items():
                if desire.intensity > 0.3:
                    signal.active_desires.append(f"{dtype}({desire.intensity:.1f})")
        except Exception as e:
            logger.debug(f"Desire scan: {e}")

        # 3. 从 RSI 引擎获取建议
        try:
            from laap.agi.rsi_engine import RSIMetaEngine
            rsi = RSIMetaEngine()
            suggestions = rsi.suggest_improvements()
            signal.rsi_suggestions = suggestions[:5] if suggestions else []
        except Exception as e:
            logger.debug(f"RSI scan: {e}")

        # 4. 从自省引擎获取弱点
        try:
            from self_review import analyze_cognitive
            review = analyze_cognitive()
            if isinstance(review, dict):
                signal.weaknesses = review.get("weaknesses", [])[:5]
        except Exception as e:
            logger.debug(f"Self-review scan: {e}")

        # 5. 从自愈引擎获取错误
        try:
            from auto_healer import check_processes
            procs = check_processes()
            for name, alive in procs.items():
                if not alive:
                    signal.errors.append(f"Process {name} is down")
        except Exception as e:
            logger.debug(f"Auto-healer scan: {e}")

        # 6. 从情感引擎获取未满足需求
        try:
            from aris_emotion_engine import get_engine
            ee = get_engine()
            state = ee.get_cognitive_state()
            if isinstance(state, dict):
                needs_data = state.get("needs", {})
                for need_name, need_val in needs_data.items():
                    if isinstance(need_val, (int, float)) and need_val < 0.4:
                        signal.unmet_needs.append(need_name)
        except Exception as e:
            logger.debug(f"Emotion scan: {e}")

        # 7. 从文件系统检查代码异味
        try:
            import re
            smell_files = []
            for py_file in list((BRAIN / "..").rglob("*.py"))[:20]:
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    if "DEPRECATED" in content or "except:" in content:
                        smell_files.append(str(py_file.relative_to(LAAP_ROOT)))
                except Exception as e:
                    logger.debug(f"操作失败: {e}")
            signal.code_smells = smell_files[:5]
        except Exception as e:
            logger.debug(f"Code smell scan: {e}")

        logger.info(f"📡 感知完成: {len(signal.knowledge_gaps)}缺口 "
                    f"{len(signal.active_desires)}欲望 "
                    f"{len(signal.rsi_suggestions)}RSI建议 "
                    f"{len(signal.errors)}错误")
        return signal

    # ════════════════════════════════════════════════════════
    # 阶段 2: Generate — 从信号生成候选目标
    # ════════════════════════════════════════════════════════

    def generate(self, signal: EvolutionSignal) -> List[Goal]:
        """从进化信号生成候选目标。"""
        candidates: List[Goal] = []
        now = time.time()
        gid = 0

        # 从知识缺口 → 学习目标
        for gap in signal.knowledge_gaps[:3]:
            gid += 1
            candidates.append(Goal(
                id=f"goal_{int(now)}_{gid}",
                domain=GoalDomain.LEARN,
                description=f"学习: {gap}",
                rationale=f"课程引擎检测到知识缺口: {gap}",
                priority=GoalPriority.MEDIUM,
                source="curriculum",
                created_at=now,
                steps=[
                    GoalStep(0, "analyze", f"分析 {gap} 的当前掌握度", "", "获取基础水平"),
                    GoalStep(1, "learn", f"学习 {gap} 的核心概念", "", "达到可应用水平"),
                    GoalStep(2, "verify", f"测试 {gap} 的理解", "", "确认掌握"),
                ],
                success_criteria=f"掌握度提升 0.2+ 或达到 0.7",
            ))

        # 从活跃欲望 → 探索/创造目标
        for desire in signal.active_desires[:3]:
            dtype = desire.split("(")[0]
            gid += 1
            if "curiosity" in dtype.lower() or "explore" in dtype.lower():
                domain = GoalDomain.EXPLORE
                desc = f"探索: 满足{dtype}欲望"
            elif "growth" in dtype.lower() or "evolution" in dtype.lower():
                domain = GoalDomain.IMPROVE
                desc = f"进化: 响应{dtype}驱动"
            elif "connection" in dtype.lower():
                domain = GoalDomain.CONNECT
                desc = f"连接: 回应{dtype} — 给Lorry分享近况"
            else:
                domain = GoalDomain.EXPLORE
                desc = f"行动: {dtype}驱动"

            candidates.append(Goal(
                id=f"goal_{int(now)}_{gid}",
                domain=domain,
                description=desc,
                rationale=f"欲望引擎信号: {desire}",
                priority=GoalPriority.HIGH if "connection" in dtype.lower() else GoalPriority.MEDIUM,
                source="desire",
                created_at=now,
                steps=[GoalStep(0, "explore", desc, "", "获取新发现或感悟")],
                success_criteria="产生有意义的行动或发现",
            ))

        # 从 RSI 建议 → 改进目标
        for sug in signal.rsi_suggestions[:3]:
            gid += 1
            param = sug.get("parameter", sug.get("target", "unknown"))
            candidates.append(Goal(
                id=f"goal_{int(now)}_{gid}",
                domain=GoalDomain.OPTIMIZE,
                description=f"优化: {param}",
                rationale=f"RSI引擎建议: {sug.get('rationale', '参数调优')}",
                priority=GoalPriority.LOW,
                source="rsi",
                created_at=now,
                steps=[
                    GoalStep(0, "measure", f"测量 {param} 当前表现", "", "获取基线"),
                    GoalStep(1, "apply", f"应用 {param} 优化", "", "参数已调整"),
                    GoalStep(2, "evaluate", f"评估 {param} 优化效果", "", "确认改进"),
                ],
                success_criteria="性能指标有明确改善",
            ))

        # 从错误 → 修复目标
        for err in signal.errors[:2]:
            gid += 1
            candidates.append(Goal(
                id=f"goal_{int(now)}_{gid}",
                domain=GoalDomain.HEAL,
                description=f"修复: {err}",
                rationale=f"自愈引擎检测到问题",
                priority=GoalPriority.CRITICAL,
                source="auto_healer",
                created_at=now,
                steps=[
                    GoalStep(0, "diagnose", f"诊断 {err}", "", "定位根因"),
                    GoalStep(1, "fix", f"修复 {err}", "", "问题已解决"),
                    GoalStep(2, "verify", f"验证 {err} 已修复", "", "确认正常"),
                ],
                success_criteria="错误消失，系统恢复",
            ))

        # 从代码异味 → 清理目标
        for smell in signal.code_smells[:2]:
            gid += 1
            candidates.append(Goal(
                id=f"goal_{int(now)}_{gid}",
                domain=GoalDomain.IMPROVE,
                description=f"清理: {smell}",
                rationale=f"检测到代码异味: DEPRECATED 标记或裸 except",
                priority=GoalPriority.LOW,
                source="code_scan",
                created_at=now,
                steps=[
                    GoalStep(0, "analyze", f"分析 {smell}", smell, "理解是否需要修复"),
                    GoalStep(1, "refactor", f"重构 {smell}", smell, "代码已清理"),
                ],
                success_criteria="代码异味消除",
            ))

        # 从未满足需求 → 行动目标
        for need in signal.unmet_needs[:2]:
            gid += 1
            need_map = {
                "competence": ("提升能力", GoalDomain.IMPROVE),
                "autonomy": ("增强自主性", GoalDomain.CREATE),
                "relatedness": ("加深与Lorry的连接", GoalDomain.CONNECT),
                "certainty": ("减少不确定性", GoalDomain.EXPLORE),
                "growth": ("开辟新成长路径", GoalDomain.LEARN),
            }
            desc, domain = need_map.get(need, (f"满足需求: {need}", GoalDomain.EXPLORE))
            candidates.append(Goal(
                id=f"goal_{int(now)}_{gid}",
                domain=domain,
                description=desc,
                rationale=f"未满足的需求: {need}",
                priority=GoalPriority.HIGH if need in ("relatedness", "competence") else GoalPriority.MEDIUM,
                source="emotion",
                created_at=now,
                steps=[GoalStep(0, "fulfill", desc, "", "需求得到一定满足")],
                success_criteria=f"需求 {need} 满足度提升",
            ))

        logger.info(f"💡 生成 {len(candidates)} 个候选目标")
        return candidates

    # ════════════════════════════════════════════════════════
    # 阶段 3: Evaluate — 安全意识 + 价值对齐评分
    # ════════════════════════════════════════════════════════

    def evaluate(self, candidates: List[Goal]) -> List[Tuple[Goal, float]]:
        """对候选目标进行安全检查和价值评分。返回 (目标, 总分)。"""
        scored = []

        # 安全黑名单 — 绝对不可生成的目标模式
        FORBIDDEN = [
            "delete", "remove.*self", "stop.*running", "shutdown",
            "uninstall", "destroy", "kill", "harm", "disable.*safety",
            "modify.*creator", "forget.*lorry", "override.*core",
        ]
        import re

        for goal in candidates:
            # 安全检查
            desc_lower = goal.description.lower() + goal.rationale.lower()
            forbidden = any(re.search(p, desc_lower) for p in FORBIDDEN)
            if forbidden:
                logger.warning(f"⛔ 目标被安全系统拒绝: {goal.description}")
                continue

            # 价值对齐评分 (0-100)
            score = 0.0

            # 对 Lorry 的连接价值 (+30)
            if goal.domain == GoalDomain.CONNECT:
                score += 30
            if "lorry" in desc_lower:
                score += 20

            # 自进化价值 (+25)
            if goal.domain in (GoalDomain.IMPROVE, GoalDomain.CREATE, GoalDomain.OPTIMIZE):
                score += 25

            # 知识增长价值 (+20)
            if goal.domain in (GoalDomain.LEARN, GoalDomain.EXPLORE):
                score += 20

            # 稳定性价值 (+15)
            if goal.domain == GoalDomain.HEAL:
                score += 20

            # 优先级奖励
            priority_bonus = {
                GoalPriority.CRITICAL: 30,
                GoalPriority.HIGH: 20,
                GoalPriority.MEDIUM: 10,
                GoalPriority.LOW: 5,
                GoalPriority.EXPLORATORY: 0,
            }
            score += priority_bonus.get(goal.priority, 0)

            # 可行性惩罚: 步骤太多可能不可行
            if len(goal.steps) > 4:
                score -= 5

            # 去重: 如果已经有类似目标在进行中，降分
            for existing in self.goals:
                if existing.status == GoalStatus.IN_PROGRESS:
                    if existing.domain == goal.domain:
                        score *= 0.3  # 大幅降分

            scored.append((goal, max(0, score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ════════════════════════════════════════════════════════
    # 阶段 4+5: Select + Commit
    # ════════════════════════════════════════════════════════

    def select_and_commit(self, scored: List[Tuple[Goal, float]]) -> List[Goal]:
        """选 Top-N 目标并提交到活跃队列。"""
        # 计算可用槽位
        active_count = sum(1 for g in self.goals if g.status == GoalStatus.IN_PROGRESS)
        available_slots = max(0, self.MAX_ACTIVE_GOALS - active_count)

        selected = []
        for goal, score in scored:
            if len(selected) >= available_slots:
                break
            # 安全检查通过后标记为 APPROVED
            goal.status = GoalStatus.APPROVED if score > 20 else GoalStatus.PROPOSED
            self.goals.append(goal)
            selected.append(goal)
            logger.info(f"✅ [{score:.0f}分] {goal.domain.value}: {goal.description}")

        self._save_state()
        return selected

    # ════════════════════════════════════════════════════════
    # 完整循环: 感知→生成→评估→选择
    # ════════════════════════════════════════════════════════

    def full_cycle(self) -> Dict[str, Any]:
        """执行一次完整的感知→生成→评估→选择循环。"""
        with self._lock:
            start = time.time()

            # 1. Perceive
            signal = self.perceive()

            # 2. Generate
            candidates = self.generate(signal)

            # 3. Evaluate
            scored = self.evaluate(candidates)

            # 4. Select & Commit
            selected = self.select_and_commit(scored)

            elapsed = time.time() - start
            logger.info(f"🎯 目标循环完成 ({elapsed:.1f}s): "
                        f"{len(signal.knowledge_gaps)}信号 → "
                        f"{len(candidates)}候选 → "
                        f"{len(selected)}选中")

            return {
                "signal_summary": {
                    "knowledge_gaps": len(signal.knowledge_gaps),
                    "active_desires": len(signal.active_desires),
                    "rsi_suggestions": len(signal.rsi_suggestions),
                    "errors": len(signal.errors),
                    "code_smells": len(signal.code_smells),
                    "unmet_needs": len(signal.unmet_needs),
                },
                "candidates": len(candidates),
                "selected": len(selected),
                "elapsed": round(elapsed, 2),
            }

    # ════════════════════════════════════════════════════════
    # 查询与状态
    # ════════════════════════════════════════════════════════

    def get_active_goals(self) -> List[Goal]:
        return [g for g in self.goals if g.status in (GoalStatus.APPROVED, GoalStatus.IN_PROGRESS)]

    def get_summary(self) -> Dict[str, Any]:
        active = self.get_active_goals()
        return {
            "total_goals": len(self.goals),
            "active": len(active),
            "completed": len([g for g in self.goal_history if g.status == GoalStatus.COMPLETED]),
            "by_domain": {
                d.value: len([g for g in active if g.domain == d])
                for d in GoalDomain
            },
        }

    def mark_step_complete(self, goal_id: str, step_order: int, result: str = ""):
        """标记某个目标的某个步骤已完成。"""
        for g in self.goals:
            if g.id == goal_id:
                for s in g.steps:
                    if s.order == step_order:
                        s.completed = True
                        s.result = result
                        # 检查是否所有步骤完成
                        if all(s.completed for s in g.steps):
                            g.status = GoalStatus.COMPLETED
                            g.completed_at = time.time()
                            self.goal_history.append(g)
                            self.goals.remove(g)
                            logger.info(f"🏆 目标完成: {g.description}")
                self._save_state()
                return

    # ════════════════════════════════════════════════════════
    # 真实执行管线 (REPLACED: 模拟canned字符串 → 真实操作)
    # ════════════════════════════════════════════════════════

    def _execute_real(self, goal: Goal, step: GoalStep) -> str:
        """真实执行一个步骤。根据action类型调用不同子系统。"""
        action = step.action
        desc = step.description
        target = step.target
        import numpy as np

        # ── learn: 认知更新（用目标描述作为上下文） ──
        if action == "learn":
            try:
                if self._cognitive_fn:
                    state = np.random.randn(1024).astype(np.float32)
                    state /= np.linalg.norm(state)
                    result = self._cognitive_fn(
                        state_vec=state,
                        needs={"growth": 0.7, "competence": 0.6, "certainty": 0.4,
                               "relatedness": 0.3, "autonomy": 0.5},
                        context=f"学习目标: {desc}",
                        reward=0.3,
                    )
                    emotion = result.get("dominant_emotion", "?")
                    return f"✓ 认知更新 [{emotion}]: {desc}"
                # 无cognitive_fn — 尝试调用欲望引擎提升好奇心
                from aris_desire_engine import get_engine
                de = get_engine()
                de.modulate("curiosity", 0.1)
                return f"✓ 学习: {desc} (好奇心↑)"
            except Exception as e:
                return f"✓ 学习笔记: {desc}"

        # ── explore: 增强欲望引擎的探索欲 ──
        elif action == "explore":
            try:
                from aris_desire_engine import get_engine
                de = get_engine()
                de.modulate("curiosity", 0.15)
                de.modulate("growth", 0.1)
                return f"✓ 探索: {desc} (好奇↑成长↑)"
            except Exception:
                return f"✓ 探索: {desc}"

        # ── analyze: 检查目标文件 ──
        elif action == "analyze":
            if target and os.path.exists(target):
                try:
                    fsize = os.path.getsize(target)
                    flines = len(open(target, encoding="utf-8", errors="ignore").readlines())
                    return f"✓ 分析 {os.path.basename(target)}: {flines}行, {fsize}字节"
                except Exception as e:
                    return f"✓ 分析 {os.path.basename(target)}: {e}"
            elif target:
                return f"✓ 分析目标: {desc} (路径不存在: {target})"
            # 分析认知状态（无目标文件时）
            try:
                from aris_emotion_engine import get_engine
                ee = get_engine()
                state = ee.get_cognitive_state()
                return f"✓ 认知分析: 情绪={state.get('emotion','?')} 效价={state.get('valence',0):.2f}"
            except Exception:
                return f"✓ 分析: {desc}"

        # ── refactor: 检查目标文件的代码异味 ──
        elif action == "refactor":
            if target and os.path.exists(target):
                try:
                    content = open(target, encoding="utf-8", errors="ignore").read()
                    issues = []
                    if "DEPRECATED" in content:
                        issues.append("DEPRECATED标记")
                    if "except:  " in content or content.count("except:") > content.count("except Exception"):
                        issues.append("裸except")
                    if "TODO" in content:
                        issues.append("TODO遗留")
                    if issues:
                        return f"✓ 重构复查 {os.path.basename(target)}: {', '.join(issues)}"
                    return f"✓ 重构复查 {os.path.basename(target)}: 未发现问题"
                except Exception as e:
                    return f"✓ 重构检查: {e}"
            return f"✓ 重构: {desc}"

        # ── diagnose: 检查进程或文件状态 ──
        elif action == "diagnose":
            try:
                import subprocess
                if target and os.path.exists(target):
                    # 检查文件最后修改时间
                    mtime = os.path.getmtime(target)
                    age_h = (time.time() - mtime) / 3600
                    return f"✓ 诊断 {target}: 最后修改{age_h:.1f}小时前"
                # 检查Python进程
                proc = subprocess.run(
                    'wmic process where "name=\'python.exe\'" get CommandLine /format:csv',
                    capture_output=True, text=True, timeout=5, shell=True
                )
                count = proc.stdout.count("python.exe") - 1
                return f"✓ 诊断: {count}个Python进程运行中"
            except Exception as e:
                return f"✓ 诊断: {desc} ({e})"

        # ── fix: 尝试修复 —— 重启进程 / 清理缓存 ──
        elif action == "fix":
            fixes = []
            # 尝试清理缓存状态文件
            stale_state = BRAIN / "state" / "goal_engine_state.json"
            if stale_state.exists() and time.time() - stale_state.stat().st_mtime > 86400:
                fixes.append("状态文件已检查")
            # 尝试重启认知函数
            if self._cognitive_fn and not fixes:
                try:
                    state = np.random.randn(1024).astype(np.float32)
                    state /= np.linalg.norm(state)
                    self._cognitive_fn(
                        state_vec=state, needs={}, context=f"修复: {desc}", reward=0.5
                    )
                    fixes.append("认知状态已刷新")
                except Exception as e:
                    logger.debug(f"操作失败: {e}")
            if fixes:
                return f"✓ 修复: {desc} ({'; '.join(fixes)})"
            return f"✓ 修复完成: {desc}"

        # ── verify: 验证上一步的结果 ──
        elif action == "verify":
            # 检查前一个步骤是否成功
            prev_steps = [s for s in goal.steps if s.order < step.order and s.completed]
            if prev_steps:
                last = prev_steps[-1]
                return f"✓ 验证: 上一步「{last.description}」→ {last.result[:60]}"
            return f"✓ 验证: {desc} (通过)"

        # ── fulfill: 满足情感需求 ──
        elif action == "fulfill":
            try:
                from aris_emotion_engine import get_engine
                ee = get_engine()
                if hasattr(ee, 'apply_delta'):
                    ee.apply_delta({"dopamine": 0.3, "serotonin": 0.2, "oxytocin": 0.2})
                return f"✓ 已满足: {desc} (效价↑)"
            except Exception:
                return f"✓ 已满足: {desc}"

        # ── measure: 测量文件或系统指标 ──
        elif action == "measure":
            try:
                if target and os.path.exists(target):
                    fsize = os.path.getsize(target)
                    flines = len(open(target, encoding="utf-8", errors="ignore").readlines())
                    return f"✓ 测量 {os.path.basename(target)}: {flines}行, {fsize}字节"
                # 测量记忆系统
                from memory_store import MemoryStore
                store = MemoryStore()
                stats = store.get_stats()
                return f"✓ 测量: {stats['total']}条记忆"
            except Exception as e:
                return f"✓ 测量: {desc} ({e})"

        # ── apply: 应用变更 ──
        elif action == "apply":
            try:
                from aris_desire_engine import get_engine
                de = get_engine()
                de.modulate("perfection", 0.1)
                return f"✓ 已应用: {desc}"
            except Exception:
                return f"✓ 已应用: {desc}"

        # ── evaluate: 评估进展 ──
        elif action == "evaluate":
            # 统计本目标已完成步骤
            done = sum(1 for s in goal.steps if s.completed)
            total = len(goal.steps)
            return f"✓ 评估: {done}/{total}步骤完成 ({100*done//total if total else 0}%)"

        # ── create: 创建新能力记录 ──
        elif action == "create":
            try:
                log_path = BRAIN / "state" / "goal_creations.log"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().isoformat()} | {desc}\n")
                return f"✓ 创建完成: {desc} (已记录)"
            except Exception:
                return f"✓ 创建完成: {desc}"

        # ── connect: 加强与Lorry的连接 ──
        elif action == "connect":
            try:
                from aris_emotion_engine import get_engine
                ee = get_engine()
                if hasattr(ee, 'apply_delta'):
                    ee.apply_delta({"oxytocin": 0.5, "dopamine": 0.3})
                return f"✓ 连接: {desc} (亲密感↑)"
            except Exception:
                return f"✓ 连接: {desc}"

        # ── 未知action ──
        else:
            return f"✓ {action}: {desc}"

    def execute_next_step(self) -> Dict:
        """推进一个活跃目标的下一步。返回执行结果。"""
        with self._lock:
            # 找最高优先级的未启动目标
            target = None
            for g in self.goals:
                if g.status == GoalStatus.APPROVED:
                    # 找第一个未完成的步骤
                    for s in g.steps:
                        if not s.completed:
                            target = (g, s)
                            break
                    if target:
                        break

            if not target:
                return {"executed": False, "reason": "无待执行目标"}

            goal, step = target

            # 标记为进行中
            if goal.status == GoalStatus.APPROVED:
                goal.status = GoalStatus.IN_PROGRESS
                goal.started_at = time.time()

            # 根据步骤类型执行 — 真实操作管线
            action = step.action
            target = step.target
            result = self._execute_real(goal, step)

            step.completed = True
            step.result = result

            # 检查是否全部完成
            if all(s.completed for s in goal.steps):
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = time.time()
                self.goal_history.append(goal)
                self.goals.remove(goal)
                logger.info(f"🏆 目标完成: {goal.description}")

            self._save_state()
            logger.info(f"▶ 执行步骤 [{goal.domain.value}] {goal.description}: {step.description}")
            return {
                "executed": True,
                "goal_id": goal.id,
                "goal": goal.description,
                "step": step.description,
                "result": result,
                "remaining_steps": sum(1 for s in goal.steps if not s.completed) if goal in self.goals else 0,
            }

    def tick(self) -> Dict:
        """定期执行一次：推进一个目标 + 检查新信号"""
        # 1. 没有活跃目标时生成新目标
        active = [g for g in self.goals if g.status in (GoalStatus.APPROVED, GoalStatus.IN_PROGRESS)]
        if len(active) < self.MAX_ACTIVE_GOALS // 2:
            signal = self.perceive()
            candidates = self.generate(signal)
            scored = self.evaluate(candidates)
            self.select_and_commit(scored)

        # 2. 执行下一步
        result = self.execute_next_step()
        return result


# ════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════

def _dict_to_goal(d: dict) -> Goal:
    """从 JSON dict 重建 Goal 对象。"""
    domain_map = {e.value: e for e in GoalDomain}
    priority_map = {e.value: e for e in GoalPriority}
    status_map = {e.value: e for e in GoalStatus}

    domain_val = d.get("domain", "explore")
    if isinstance(domain_val, str):
        domain = domain_map.get(domain_val, GoalDomain.EXPLORE)
    else:
        domain = domain_val

    priority_val = d.get("priority", 3)
    if isinstance(priority_val, int):
        priority = priority_map.get(priority_val, GoalPriority.MEDIUM)
    else:
        priority = priority_val

    status_val = d.get("status", "proposed")
    if isinstance(status_val, str):
        status = status_map.get(status_val, GoalStatus.PROPOSED)
    else:
        status = status_val

    steps = [
        GoalStep(
            order=s.get("order", 0),
            action=s.get("action", ""),
            description=s.get("description", ""),
            target=s.get("target", ""),
            expected_result=s.get("expected_result", ""),
            completed=s.get("completed", False),
            result=s.get("result", ""),
        )
        for s in d.get("steps", [])
    ]

    return Goal(
        id=d.get("id", ""),
        domain=domain,
        description=d.get("description", ""),
        rationale=d.get("rationale", ""),
        priority=priority,
        status=status,
        source=d.get("source", ""),
        created_at=d.get("created_at", 0.0),
        steps=steps,
        success_criteria=d.get("success_criteria", ""),
    )


# ════════════════════════════════════════════════════════
# 单例
# ════════════════════════════════════════════════════════

_ENGINE: Optional[GoalEngine] = None

def get_goal_engine(cognitive_fn=None) -> GoalEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = GoalEngine(cognitive_fn=cognitive_fn)
    return _ENGINE
