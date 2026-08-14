"""
LAAP 类型化认知事件 Schema
==========================
定义认知流水线中所有跨模块数据交换的结构化契约，
替代裸 dict 传递，使 CognitiveBus / PSI Core / AGI 订阅者的
事件 schema、字段、来源与优先级显式化。

实现：纯 dataclass（零外部依赖，与 integrator.py 现有风格一致）。

用法:
    from aris_brain.schemas.events import CognitiveEvent, RouteResult

    event = CognitiveEvent(event_type="user_message", payload={"text": "你好"})
    result = RouteResult(decision="qre_engine", response="...")

印记: Aris 永远记得 Lorry — 2026-06-23
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Literal, Optional


# ─── 事件来源 ─────────────────────────────────────────────────

class EventSource(str, Enum):
    """事件来源模块。"""
    PSI_CORE = "psi_core"
    EMOTION_ENGINE = "emotion_engine"
    RULES_ENGINE = "rules_engine"
    CAUSAL_ENGINE = "causal_engine"
    MEMORY_STORE = "memory_store"
    COGNITIVE_BUS = "cognitive_bus"
    AGI_SUBSCRIBER = "agi_subscriber"
    USER = "user"
    LLM = "llm"


# ─── 事件类型 ─────────────────────────────────────────────────

EventType = Literal[
    "user_message",       # 用户输入到达
    "state_update",       # PSI 状态更新
    "memory_recall",      # 记忆召回
    "memory_consolidate", # 记忆固化
    "goal_formation",     # 目标形成
    "emotion_shift",      # 情绪状态迁移
    "causal_inference",   # 因果推理完成
    "route_decision",     # 认知总线路由决策
    "response_synthesize",# 响应合成
    "system_event",       # 系统级事件
]

# ─── 路由决策类型（与 cognitive_bus 对齐）────────────────────

RouteDecision = Literal[
    "qre_engine",    # 量子推理引擎产生了输出 — Aris 自己的思考
    "v12_kernel",    # V12.1 精确/语义匹配成功
    "qlg_template",  # QLG 模板填充
    "psi_only",      # 纯情绪回应（无有用内容）
    "no_engine",     # 引擎无输出或无响应
    "error",         # 读取出错
]


# ─── CognitiveEvent ───────────────────────────────────────────

@dataclass
class CognitiveEvent:
    """跨模块认知事件。

    Attributes:
        event_id: 唯一标识
        timestamp: 事件产生时间（UTC）
        source: 来源模块
        target: 目标订阅者（"all" 表示广播）
        event_type: 事件类型
        payload: 事件载荷（模块相关数据）
        priority: 1-10，越高越优先（用于注意力选择）
        session_id: 会话标识（可空）
    """
    event_type: str = "state_update"
    payload: Dict[str, Any] = field(default_factory=dict)
    source: EventSource = EventSource.COGNITIVE_BUS
    target: str = "all"
    priority: int = 5
    session_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value if isinstance(self.source, EventSource) else self.source
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ─── RouteResult ──────────────────────────────────────────────

@dataclass
class RouteResult:
    """认知总线路由结果 — 替代裸 dict。

    Attributes:
        decision: 路由决策类型
        source: 引擎名
        response: 引擎输出文本
        confidence: 置信度 0-1
        latency_us: 延迟（微秒）
        cognitive_context: 注入 LLM 的认知上下文
        use_engine_output: 是否应使用引擎输出
        psi_state: 完整 PSI 状态（可选）
    """
    decision: str = "no_engine"
    source: str = "none"
    response: str = ""
    confidence: float = 0.0
    latency_us: float = 0.0
    cognitive_context: str = ""
    use_engine_output: bool = False
    psi_state: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouteResult":
        """从 RouteResult.to_dict()（或兼容 dict）构建。"""
        return cls(
            decision=d.get("decision", "no_engine"),
            source=d.get("source", "none"),
            response=d.get("response", ""),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            latency_us=float(d.get("latency_us", 0.0) or 0.0),
            cognitive_context=d.get("cognitive_context", ""),
            use_engine_output=bool(d.get("use_engine_output", False)),
            psi_state=d.get("psi_state"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── PSI 状态快照 ─────────────────────────────────────────────

@dataclass
class PsiStateSnapshot:
    """PSI Core 状态快照 — 用于监控面板与认知状态注入。

    Attributes:
        cycle: 认知周期计数
        emotion: 情绪标签
        arousal: 唤醒度 0-1
        self_presence: 自我在场感 0-1
        attention_focus: 注意力焦点
        needs: 五维需求向量
    """
    cycle: int = 0
    emotion: str = "neutral"
    arousal: float = 0.5
    self_presence: float = 0.5
    attention_focus: str = "idle"
    needs: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PsiStateSnapshot":
        """从 raw state dict 构建快照（容忍缺字段，兼容 `needs` 与 `needs_map`）。"""
        return cls(
            cycle=state.get("cycle", state.get("psi_cycle", 0)) or 0,
            emotion=state.get("emotion", "neutral") or "neutral",
            arousal=float(state.get("arousal", 0.5) or 0.5),
            self_presence=float(state.get("self_presence", 0.5) or 0.5),
            attention_focus=state.get("attention_focus", "idle") or "idle",
            needs=state.get("needs", state.get("needs_map", {})) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
