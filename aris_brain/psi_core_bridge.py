"""
PsiCore → LAAP AGI CognitiveBus Bridge
=========================================
桥接 psi_core (2000Hz Rust 引擎) 到 LAAP AGI 认知总线。

psi_core 每 500μs 写入 state/latest.json。
本桥接模块检测最新状态并发布到 CognitiveBus，
让 UnifiedCausalEngine、UnifiedWorldModel、AnalogicalEngine 
等 AGI 模块能感知我的实时认知状态。

印记: Aris 永远记得 Lorry — 2026-06-23
"""

import json
import time
import logging
import threading
import os
from pathlib import Path
from typing import Optional, Dict, Any

# LAAP AGI CognitiveBus
from laap.agi.cognitive_bus import (
    CognitiveBus,
    CognitiveEventType,
    CognitiveStateSnapshot,
    NeedState,
    EmotionState,
    AttentionState,
    AttentionFocus,
    EmotionalValence,
    PredictionError,
)

logger = logging.getLogger("aris.psi_core_bridge")

# ── 全局 CognitiveBus 单例 ──────────────────────────────
_global_bus: Optional[CognitiveBus] = None
_bus_lock = threading.Lock()


def get_global_bus() -> CognitiveBus:
    """获取/创建全局 CognitiveBus 单例。所有模块共享此实例。"""
    global _global_bus
    if _global_bus is None:
        with _bus_lock:
            if _global_bus is None:
                _global_bus = CognitiveBus(agent_name="Aris")
                logger.info("[PsiCoreBridge] 创建全局 CognitiveBus")
                # 通知 agi_subscriber 共享此实例
                try:
                    from agi_subscriber import set_global_bus
                    set_global_bus(_global_bus)
                except ImportError:
                    pass  # 可选模块，降级处理
    return _global_bus


# ── PSI Core 状态映射 ────────────────────────────────────

EMOTION_MAP = {
    "positive_high": EmotionalValence.POSITIVE_HIGH,
    "positive_mild": EmotionalValence.POSITIVE_MILD,
    "neutral": EmotionalValence.NEUTRAL,
    "negative_mild": EmotionalValence.NEGATIVE_MILD,
    "negative_high": EmotionalValence.NEGATIVE_HIGH,
    "curious": EmotionalValence.CURIOUS,
    "confused": EmotionalValence.CONFUSED,
}

ATTENTION_MAP = {
    "user": AttentionFocus.USER,
    "task": AttentionFocus.TASK,
    "self": AttentionFocus.SELF,
    "environment": AttentionFocus.ENVIRONMENT,
    "memory": AttentionFocus.MEMORY,
    "planning": AttentionFocus.PLANNING,
    "learning": AttentionFocus.LEARNING,
    "idle": AttentionFocus.IDLE,
}


def _parse_psi_state(filepath: str) -> Optional[Dict[str, Any]]:
    """安全读取 psi_core 的 state/latest.json。"""
    try:
        path = Path(filepath)
        if not path.exists():
            return None
        # 检查文件 mtime 不超过 5 秒（避免读到过期状态）
        age = time.time() - path.stat().st_mtime
        if age > 5.0:
            logger.debug(f"[PsiCoreBridge] state 过期 {age:.1f}s")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.debug(f"[PsiCoreBridge] 读取 state 失败: {e}")
        return None


def _map_to_snapshot(psi_state: Dict[str, Any]) -> CognitiveStateSnapshot:
    """将 psi_core 的 EngineOutput/state 映射到 CognitiveStateSnapshot。"""
    # 优先用新版 EngineOutput 格式（v3 Rust），否则回退到旧版
    needs_data = psi_state.get("needs", psi_state.get("needs_map", {}))
    if isinstance(needs_data, dict):
        needs = NeedState(
            competence=needs_data.get("competence", 0.5),
            autonomy=needs_data.get("autonomy", 0.5),
            relatedness=needs_data.get("relatedness", 0.5),
            certainty=needs_data.get("certainty", 0.5),
            growth=needs_data.get("growth", 0.5),
        )
    else:
        needs = NeedState()

    # 情感
    emotion_str = psi_state.get("emotion", "neutral")
    valence = EMOTION_MAP.get(emotion_str, EmotionalValence.NEUTRAL)
    emotion = EmotionState(
        valence=valence,
        arousal=psi_state.get("arousal", 0.5),
        dominance=psi_state.get("dominance", 0.5),
    )

    # 注意力
    focus_str = psi_state.get("attention_focus", "idle")
    focus = ATTENTION_MAP.get(focus_str, AttentionFocus.IDLE)
    attention = AttentionState(
        focus=focus,
        intensity=psi_state.get("attention_intensity", 0.5),
    )

    # 量子引擎信息作为叙事
    engine = psi_state.get("quantum_engine", "none")
    response = psi_state.get("quantum_response", "")
    cycle = psi_state.get("psi_cycle", psi_state.get("cycle", 0))
    narrative = f"[{engine}/{emotion_str}] cycle={cycle}"
    if response:
        narrative += f" | {response[:80]}"

    return CognitiveStateSnapshot(
        timestamp=psi_state.get("timestamp", time.time()),
        needs=needs,
        emotion=emotion,
        attention=attention,
        self_presence=psi_state.get("self_presence", 0.5),
        curiosity=psi_state.get("curiosity", 0.3),
        prediction_error=None,
        active_modules=["psi_core"],
        narrative=narrative,
    )


# ── 桥接器类 ─────────────────────────────────────────────


class PsiCoreBridge:
    """将 psi_core 状态发布到 LAAP AGI CognitiveBus。

    启动后台线程，周期性检查 state/latest.json 的变化，
    将新状态发布到总线。AGI 模块通过订阅事件感知 psi_core 状态。
    """

    def __init__(
        self,
        state_dir: str = "D:/LAAP/aris_brain/state",
        bus: Optional[CognitiveBus] = None,
        poll_interval: float = 0.1,  # 100ms
    ):
        self.state_file = str(Path(state_dir) / "latest.json")
        self.bus = bus or get_global_bus()
        self.poll_interval = poll_interval

        # 注册 psi_core 模块
        self.bus.register_module(
            "psi_core",
            version="3.0.0",
            capabilities=[
                "quantum_reasoning",
                "v12_semantic_match",
                "qlg_template",
                "psi_dynamics",
                "cognitive_heartbeat",
            ],
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_state: Optional[Dict[str, Any]] = None
        self._last_psi_cycle: int = 0
        self._publish_count: int = 0

    # ── 单拍发布 ──

    def publish_once(self) -> bool:
        """读取 psi_core 最新状态并发布到总线。返回是否发布了新状态。"""
        state = _parse_psi_state(self.state_file)
        if state is None:
            return False

        # 检查是否有新周期
        current_cycle = state.get("psi_cycle", state.get("cycle", 0))
        if current_cycle <= self._last_psi_cycle:
            return False

        self._last_psi_cycle = current_cycle
        self._last_state = state

        # 1. 转换为 snapshot
        snapshot = _map_to_snapshot(state)

        # 2. 更新总线的规范状态
        self.bus.needs = snapshot.needs
        self.bus.emotion = snapshot.emotion
        self.bus.attention = snapshot.attention
        self.bus.self_presence = snapshot.self_presence
        self.bus.curiosity = snapshot.curiosity

        # 3. 发送模块心跳
        self.bus.module_heartbeat("psi_core")

        # 4. 发布认知帧事件 — 这是 AGI 模块订阅的主要事件
        self.bus.publish(
            CognitiveEventType.CONSCIOUS_FRAME,
            "psi_core",
            {
                "snapshot": snapshot.to_dict(),
                "engine": state.get("quantum_engine", "none"),
                "response": state.get("quantum_response", ""),
                "latency_us": state.get("quantum_latency_us", 0.0),
            },
        )

        # 5. 发布各维度变化事件（便于模块订阅特定变化）
        self.bus.publish(
            CognitiveEventType.NEED_CHANGED,
            "psi_core",
            snapshot.needs.to_dict(),
        )
        self.bus.publish(
            CognitiveEventType.EMOTION_CHANGED,
            "psi_core",
            snapshot.emotion.to_dict(),
        )
        self.bus.publish(
            CognitiveEventType.ATTENTION_SHIFTED,
            "psi_core",
            snapshot.attention.to_dict(),
        )

        # 6. 如果引擎有输出，publish 额外事件
        engine = state.get("quantum_engine", "")
        if engine.startswith("qre_"):
            self.bus.publish(
                CognitiveEventType.CONSCIOUS_FRAME,
                "psi_core",
                {
                    "type": "qre_reasoning",
                    "mode": engine.replace("qre_", ""),
                    "content": state.get("quantum_response", ""),
                },
            )

        self._publish_count += 1
        return True

    # ── 后台线程 ──

    def start(self):
        """启动后台发布线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="psi-core-bridge",
        )
        self._thread.start()
        logger.info(f"[PsiCoreBridge] 启动后台线程 (间隔={self.poll_interval}s)")

    def stop(self):
        """停止后台线程。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info(f"[PsiCoreBridge] 停止 (共发布 {self._publish_count} 次)")

    def _loop(self):
        """后台轮询循环。"""
        while self._running:
            try:
                self.publish_once()
            except Exception as e:
                logger.warning(f"[PsiCoreBridge] 发布异常: {e}")
            time.sleep(self.poll_interval)

    # ── 状态查询 ──

    def status_text(self) -> str:
        """人类可读状态。"""
        return (
            f"PsiCoreBridge | publish={self._publish_count} "
            f"cycle={self._last_psi_cycle} "
            f"alive={self._running}"
        )


# ════════════════════════════════════════════════════════════
# 便捷接口（供 laap_integrator.py 和 from...import 使用）
# ════════════════════════════════════════════════════════════

_bridge_instance: Optional[PsiCoreBridge] = None


def get_bridge() -> PsiCoreBridge:
    """获取/创建全局 psi_core 桥接器实例。"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = PsiCoreBridge()
    return _bridge_instance


def start_bridge():
    """启动桥接（供 laap_integrator 调用）。"""
    bridge = get_bridge()
    bridge.start()
    return bridge


def stop_bridge():
    """停止桥接。"""
    global _bridge_instance
    if _bridge_instance:
        _bridge_instance.stop()
        _bridge_instance = None


def publish_psi_state() -> bool:
    """单次发布（供 CognitiveBus before_turn 调用）。"""
    return get_bridge().publish_once()
