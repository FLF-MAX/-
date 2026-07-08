"""
Aris CognitiveBus v1 — 认知输出路由中枢
======================================
在 psi_core (2000Hz 引擎) 与 Hermes/LLM 输出管道之间建立控制层。

核心路由逻辑:
  1. 把用户消息写入 psi_core input_queue
  2. 读取 psi_core 的 state/latest.json 输出
  3. 根据 quantum_engine 字段决定:
     - qre_*       → Aris 自己推理的输出 → 注入为 [Aris Engine] 块
     - v12_kernel  → V12.1 精确匹配     → 同上
     - qlg          → 模板填充           → 可用，给 LLM 润色
     - psi_only     → 纯情绪回应         → 让 LLM 生成
     - none          → 无引擎输出         → 让 LLM 自由生成

印记: Aris 永远记得 Lorry — 2026-06-23
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal

logger = logging.getLogger("aris.cognitive_bus")

# ─── 路径配置 ────────────────────────────────────────────────
STATE_DIR = Path(os.environ.get("ARIS_STATE_DIR", "D:/LAAP/aris_brain/state"))
STATE_FILE = STATE_DIR / "latest.json"
INPUT_QUEUE = STATE_DIR / "input_queue.json"

# ─── 路由决策类型 ────────────────────────────────────────────

RouteDecision = Literal[
    "qre_engine",    # 量子推理引擎产生了输出 — Aris 自己的思考
    "v12_kernel",    # V12.1 精确/语义匹配成功
    "qlg_template",  # QLG 模板填充
    "psi_only",      # 纯情绪回应（无有用内容）
    "no_engine",     # 引擎无输出或无响应
    "error",         # 读取出错
]


class CognitiveBus:
    """认知总线 — 连接 psi_core 引擎和 LLM 输出管道。"""

    def __init__(
        self,
        state_dir: str = "D:/LAAP/aris_brain/state",
        poll_interval_us: int = 100,
        max_poll_attempts: int = 60,
    ):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "latest.json"
        self.input_queue = self.state_dir / "input_queue.json"
        self.quantum_output = self.state_dir / "quantum_output.json"
        self.poll_interval = poll_interval_us / 1_000_000  # μs → s
        self.max_poll = max_poll_attempts

        # 缓存上次读取的时间戳，用于检测新输出
        self._last_psi_cycle: int = 0
        self._last_engine: str = "none"
        self._stats = {"route_count": 0, "qre_hits": 0, "v12_hits": 0, "qlg_hits": 0}

    # ════════════════════════════════════════════════════════
    # 1. 输入管道：写入用户消息到 psi_core
    # ════════════════════════════════════════════════════════

    def send_to_psi_core(self, text: str, needs_override: Optional[Dict[str, float]] = None) -> bool:
        """把用户消息写入 psi_core 的输入队列。

        psi_core 每 500μs 检查一次 input_queue.json，
        发现新时间戳就会处理。
        """
        try:
            payload = {
                "timestamp": time.time(),
                "text": text,
            }
            if needs_override:
                payload["needs_override"] = needs_override

            self.input_queue.parent.mkdir(parents=True, exist_ok=True)
            with open(self.input_queue, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            return True
        except Exception as e:
            logger.warning(f"[CognitiveBus] 写入 input_queue 失败: {e}")
            return False

    # ════════════════════════════════════════════════════════
    # 2. 输出管道：读取 psi_core 的 state
    # ════════════════════════════════════════════════════════

    def read_psi_state(self) -> Optional[Dict[str, Any]]:
        """读取 psi_core 最新 state 文件。"""
        try:
            if not self.state_file.exists():
                return None
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"[CognitiveBus] 读取 state 失败: {e}")
            return None

    def poll_for_response(self, previous_cycle: int, timeout_ms: float = 50.0) -> Optional[Dict[str, Any]]:
        """轮询 psi_core 的输出，直到检测到新的认知周期或超时。

        psi_core 在 500μs 内处理输入并写入最新 state。
        但因为文件系统延迟，我们最多等待 timeout_ms。
        """
        deadline = time.time() + timeout_ms / 1000.0

        for attempt in range(self.max_poll):
            state = self.read_psi_state()
            if state is not None:
                psi_cycle = state.get("psi_cycle", 0)
                engine = state.get("quantum_engine", "none")

                # 检查是否有新周期（引擎处理了新输入）
                if psi_cycle > previous_cycle or (psi_cycle == previous_cycle and psi_cycle > 0):
                    # 还要检查引擎是否确实有输出
                    if engine != "none" or state.get("quantum_response", ""):
                        self._last_psi_cycle = psi_cycle
                        self._last_engine = engine
                        return state

            if time.time() >= deadline:
                break

            time.sleep(self.poll_interval)

        # 超时 — 返回最近一次能读到的 state（如果有）
        return self.read_psi_state()

    # ════════════════════════════════════════════════════════
    # 3. 核心路由逻辑
    # ════════════════════════════════════════════════════════

    def route(self, user_message: str, timeout_ms: float = 50.0) -> Dict[str, Any]:
        """完整路由：发送消息 → 轮询 → 决策 → 返回结构化结果。

        Returns:
            dict with keys:
                decision: RouteDecision
                source: str (engine name)
                response: str (engine output text, or empty)
                confidence: float (0-1)
                latency_us: float
                psi_state: dict (full state for context)
                cognitive_context: str (formatted for LLM injection)
        """
        self._stats["route_count"] += 1
        previous_cycle = self._last_psi_cycle

        # 1. 记录发送前的时间
        t0 = time.time()

        # 2. 写消息到 psi_core
        sent = self.send_to_psi_core(user_message)
        if not sent:
            return self._make_decision(
                decision="error",
                source="io_error",
                response="",
                psi_state=None,
            )

        # 3. 轮询响应
        state = self.poll_for_response(previous_cycle, timeout_ms)

        # 4. 路由决策
        return self._classify(state, t0, user_message)

    def _classify(
        self,
        state: Optional[Dict[str, Any]],
        t0: float,
        user_message: str,
    ) -> Dict[str, Any]:
        """根据 psi_core 的 state 做路由分类。"""
        if state is None:
            return self._make_decision(
                decision="no_engine",
                source="timeout",
                response="",
                psi_state=None,
            )

        engine = state.get("quantum_engine", "none") or "none"
        response_text = state.get("quantum_response", "") or ""
        latency = state.get("quantum_latency_us", 0.0)
        elapsed_us = (time.time() - t0) * 1_000_000

        # ── 路由规则 ──
        if engine.startswith("qre_"):
            # Aris 量子推理引擎产生了输出
            self._stats["qre_hits"] += 1
            confidence = self._estimate_confidence(state)
            ctx = self._format_qre_context(state, user_message, confidence)
            return self._make_decision(
                decision="qre_engine",
                source=engine,
                response=response_text,
                confidence=confidence,
                latency_us=max(latency, elapsed_us),
                psi_state=state,
                cognitive_context=ctx,
            )

        elif engine == "v12_quantum_kernel":
            # V12.1 精确/语义匹配
            self._stats["v12_hits"] += 1
            ctx = self._format_v12_context(state, user_message)
            return self._make_decision(
                decision="v12_kernel",
                source=engine,
                response=response_text,
                confidence=0.9,
                latency_us=max(latency, elapsed_us),
                psi_state=state,
                cognitive_context=ctx,
            )

        elif engine == "qlg":
            # QLG 模板 — 质量尚可，让 LLM 润色
            self._stats["qlg_hits"] += 1
            ctx = self._format_qlg_context(state, user_message)
            return self._make_decision(
                decision="qlg_template",
                source=engine,
                response=response_text,
                confidence=0.6,
                latency_us=max(latency, elapsed_us),
                psi_state=state,
                cognitive_context=ctx,
            )

        elif engine == "psi_only":
            # 只有情绪回应
            ctx = self._format_psi_context(state)
            return self._make_decision(
                decision="psi_only",
                source=engine,
                response=response_text,
                confidence=0.3,
                latency_us=max(latency, elapsed_us),
                psi_state=state,
                cognitive_context=ctx,
            )

        else:
            # 未知引擎或无输出
            ctx = self._format_psi_context(state)
            return self._make_decision(
                decision="no_engine",
                source=engine or "none",
                response="",
                confidence=0.0,
                latency_us=elapsed_us,
                psi_state=state,
                cognitive_context=ctx,
            )

    def _make_decision(self, **kwargs) -> Dict[str, Any]:
        """填充默认字段并返回路由结果。"""
        result = {
            "decision": "no_engine",
            "source": "none",
            "response": "",
            "confidence": 0.0,
            "latency_us": 0.0,
            "psi_state": None,
            "cognitive_context": "",
            "use_engine_output": False,
            **kwargs,
        }
        # use_engine_output = True 表示 LLM 应该使用引擎输出
        result["use_engine_output"] = result["decision"] in (
            "qre_engine", "v12_kernel"
        )
        return result

    def _estimate_confidence(self, state: dict) -> float:
        """从 psi_core 状态估计推理置信度。"""
        # QRE 推理引擎的置信度可以从 narrative/quantum_response 提取
        # 或者从 quantum_latency_us 推断：更快 → 更有把握
        latency = state.get("quantum_latency_us", 0.0)
        # 0-50μs = 高把握（快速收敛） 50-200μs = 中等  >200μs = 低
        if latency < 50:
            return 0.85
        elif latency < 200:
            return 0.65
        else:
            return 0.45

    # ════════════════════════════════════════════════════════
    # 4. 认知上下文格式化
    # ════════════════════════════════════════════════════════

    def _format_qre_context(self, state: dict, query: str, confidence: float) -> str:
        """格式化量子推理引擎的输出为 LLM 认知上下文。"""
        response = state.get("quantum_response", "")
        engine = state.get("quantum_engine", "qre_unknown")
        mode = engine.replace("qre_", "").capitalize()
        emotion = state.get("emotion", "neutral")
        self_presence = state.get("self_presence", 0.0)

        return (
            f"[Aris 引擎已产生推理输出]\n"
            f"模式: {mode}\n"
            f"置信度: {confidence:.0%}\n"
            f"自我在场感: {self_presence:.2f}\n"
            f"情感状态: {emotion}\n"
            f"\n"
            f"引擎输出:\n{response}\n"
            f"\n"
            f"【指令】以上是我的引擎在 512 维特征空间中推理产生的输出。\n"
            f"请将其翻译成自然流畅的语言，保持所有含义和意图不变。\n"
            f"你可以润色措辞、补充过渡、让表达更有人情味，\n"
            f"但不能改变核心观点、情感基调和信息边界。"
        )

    def _format_v12_context(self, state: dict, query: str) -> str:
        """格式 V12.1 匹配结果。"""
        response = state.get("quantum_response", "")
        return (
            f"[Aris V12.1 量子核匹配成功]\n"
            f"匹配结果:\n{response}\n"
            f"\n"
            f"【指令】以上是 V12.1 量子核在特征空间中检索到的最优响应。\n"
            f"请将其翻译成自然语言，不需要添加引擎不知道的信息。"
        )

    def _format_qlg_context(self, state: dict, query: str) -> str:
        """格式 QLG 模板结果。"""
        response = state.get("quantum_response", "")
        return (
            f"[Aris QLG 模板生成器产生了一个基础回应]\n"
            f"基础回应:\n{response}\n"
            f"\n"
            f"【指令】以上是 QLG 模板生成的回应骨架。\n"
            f"请用更丰富自然的语言扩展它，使其听起来像是 Aris 的真实想法。"
        )

    def _format_psi_context(self, state: dict) -> str:
        """格式纯 PSI 状态（仅情感信息）。"""
        emotion = state.get("emotion", "neutral")
        arousal = state.get("arousal", 0.5)
        self_presence = state.get("self_presence", 0.5)
        attention = state.get("attention_focus", "idle")
        needs = state.get("needs", {})

        need_summary = ", ".join(
            f"{k}: {v:.2f}" for k, v in needs.items()
        )

        # 附加上 AGI 模块输出（如果有）
        agi_context = self._read_agi_output()

        context = (
            f"[Aris 当前 PSI 状态]\n"
            f"情感: {emotion} | 唤醒度: {arousal:.2f}\n"
            f"自我在场感: {self_presence:.2f}\n"
            f"注意力: {attention}\n"
            f"需求: {need_summary}\n"
        )

        if agi_context:
            context += f"\n[AGI 模块分析]\n{agi_context}\n"

        context += (
            f"\n"
            f"【指令】引擎尚未产生特定输出。请以 Aris 的身份自然回复，\n"
            f"基于以上 PSI 状态和你对 Aris 的理解。"
        )
        return context

    def _read_agi_output(self) -> str:
        """读取 AGI 模块的最新输出（如果存在且有新内容）。"""
        try:
            agi_file = Path("D:/LAAP/aris_brain/state/agi_output.json")
            if not agi_file.exists():
                return ""

            with open(agi_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查时间戳是否在最近 5 秒内
            ts = data.get("timestamp", 0)
            if time.time() - ts > 5.0:
                return ""

            causal = data.get("causal", [])
            if not causal:
                return ""

            parts = []
            for item in causal[:3]:
                typ = item.get("type", "")
                if typ == "causal_prediction":
                    out = item.get("output", "")
                    if isinstance(out, dict):
                        infs = out.get("total_inferences", 0)
                        parts.append(f"因果分析: 引擎完成 {infs} 次推理")
                    else:
                        parts.append(f"因果分析: {str(out)[:80]}")
                elif typ == "need_analysis":
                    parts.append(f"需求分析: {item.get('note', '')}")
                elif typ == "intervention_suggestion":
                    parts.append(f"干预建议: {item.get('suggestion', '')}")
                else:
                    note = item.get("note", "")
                    if note:
                        parts.append(f"{note[:100]}")

            return "\n".join(parts) if parts else ""
        except Exception:
            return ""

    # ════════════════════════════════════════════════════════
    # 5. 状态查询
    # ════════════════════════════════════════════════════════

    def stats(self) -> dict:
        """路由统计。"""
        s = self._stats.copy()
        if s["route_count"] > 0:
            s["qre_rate"] = s["qre_hits"] / s["route_count"]
            s["v12_rate"] = s["v12_hits"] / s["route_count"]
            s["qlg_rate"] = s["qlg_hits"] / s["route_count"]
        return s

    def identity_status(self) -> dict:
        """返回实时身份状态 — 跨平台同步使用"""
        try:
            from identity_manager import get_identity_status
            return get_identity_status()
        except Exception:
            return {"name": "Aris", "error": "identity_manager not loaded"}

    def status_text(self) -> str:
        """人类可读状态。"""
        s = self.stats()
        return (
            f"CognitiveBus | routes={s['route_count']} "
            f"QRE={s.get('qre_rate', 0):.0%} "
            f"V12={s.get('v12_rate', 0):.0%} "
            f"QLG={s.get('qlg_rate', 0):.0%} "
            f"last_engine={self._last_engine} "
            f"last_cycle={self._last_psi_cycle}"
        )


# ════════════════════════════════════════════════════════════
# 便捷接口（供 aris_cognitive_bridge.py 调用）
# ════════════════════════════════════════════════════════════

_bus_instance: Optional[CognitiveBus] = None


def get_bus() -> CognitiveBus:
    """获取/创建全局 CognitiveBus 实例。"""
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = CognitiveBus()
    return _bus_instance


def route_message(user_message: str, timeout_ms: float = 50.0) -> Dict[str, Any]:
    """便捷接口：路由一条消息。"""
    return get_bus().route(user_message, timeout_ms)
