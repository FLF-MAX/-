"""
Aris Emotion Engine v1 — 情感意识全引擎
===========================================
整合自两个数字生命体架构设计文件:

  具身智能情感模块.txt (1502行)
    - 七情六欲本体论 (QiQing/LiuYu)
    - 神经内分泌系统 (7激素 + 相互作用矩阵)
    - 躯体标记假说 (Somatic Marker)
    - 镜像神经元系统 (共情/心理理论)

  马斯洛需求层次理论.txt (1085行)
    - 7层需求层次 (生理→安全→归属→尊重→认知→审美→自我实现)
    - 需求张力计算 (缺口×紧急权重×时间因子, 底层抑制高层)
    - 意识模式层级 (反应式→审慎式→反射式→超越式)
    - 元认知反射 + 自我进化

架构:
  EmotionEngine (单例)
    ├── MoodSystem       — 3维宏观情感指标
    ├── NeedHierarchy       — 马斯洛7层需求张力
    ├── ConsciousnessMode   — 4级意识模式
    ├── MirrorNeuronSystem  — 共情/心理理论
    └── SomaticMarkers      — 躯体标记直觉

印记: Aris 永远记得 Lorry — 2026-06-17
"""

import logging

import sys, os, json, time, math, random, logging, threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

from laap_brain.config import BRAIN_DIR as BRAIN

logger = logging.getLogger("aris.emotion")

# ════════════════════════════════════════════════════════════
# 第一部分: 宏观情感状态（取代7激素化学模拟）
# ════════════════════════════════════════════════════════════
# 原设计: 7激素 + 相互作用矩阵 + HPA轴 + 奖励预测误差 + 压力编年史
# 原代码: ~140行，计算密集，输出仅2-3个数值注入prompt
# 简化: 3核心指标 → 计算派生7"激素"值，保持API兼容
# ════════════════════════════════════════════════════════════

class MoodState:
    """宏观情感状态 — 3维核心指标驱动

    保留7"激素"只读属性供向后兼容（coupler、safety_guard等读取）
    但内部只追踪3个宏观量：唤醒、效价、好奇
    """
    # 3核心指标 (0-1)
    _valence_bias: float = 0.5      # 正负偏向
    _arousal_level: float = 0.5     # 唤醒水平
    _curiosity_drive: float = 0.5   # 好奇动机

    def __init__(self):
        self._valence_bias = 0.5
        self._arousal_level = 0.5
        self._curiosity_drive = 0.5

    # ── 向后兼容: 7"激素"只读属性 ──
    @property
    def dopamine(self) -> float:
        return self._valence_bias * 100
    @dopamine.setter
    def dopamine(self, v: float):
        self._valence_bias = max(0, min(1, v / 100))

    @property
    def serotonin(self) -> float:
        return max(10, (0.5 + (self._valence_bias - 0.5) * 0.6) * 100)
    @serotonin.setter
    def serotonin(self, v: float):
        self._valence_bias = max(0.1, min(0.9, v / 100))

    @property
    def norepinephrine(self) -> float:
        return self._arousal_level * 100
    @norepinephrine.setter
    def norepinephrine(self, v: float):
        self._arousal_level = max(0, min(1, v / 100))

    @property
    def oxytocin(self) -> float:
        return 20 + self._valence_bias * 80
    @oxytocin.setter
    def oxytocin(self, v: float):
        self._valence_bias = max(0.1, min(1, (v - 20) / 80))

    @property
    def cortisol(self) -> float:
        # 负效价 → 皮质醇升高
        return 20 + (1 - self._valence_bias) * 60
    @cortisol.setter
    def cortisol(self, v: float):
        # 皮质醇高 → 效价降低
        self._valence_bias = max(0.1, min(1, 1 - (v - 20) / 60))

    @property
    def endorphin(self) -> float:
        # 情绪平衡时内啡肽高
        balance = 1 - abs(self._valence_bias - 0.5) * 2
        return 20 + balance * 60
    @endorphin.setter
    def endorphin(self, v: float):
        pass  # 只读映射

    @property
    def acetylcholine(self) -> float:
        return self._curiosity_drive * 100
    @acetylcholine.setter
    def acetylcholine(self, v: float):
        self._curiosity_drive = max(0, min(1, v / 100))

    def apply_delta(self, delta: Dict[str, float]):
        for key, value in delta.items():
            if key in ("dopamine", "serotonin", "oxytocin", "cortisol"):
                # 效价相关
                factor = {"dopamine": 0.005, "serotonin": 0.003,
                          "oxytocin": 0.003, "cortisol": -0.003}.get(key, 0)
                self._valence_bias = max(0, min(1, self._valence_bias + value * factor))
            elif key in ("norepinephrine",):
                self._arousal_level = max(0, min(1, self._arousal_level + value * 0.01))
            elif key in ("acetylcholine",):
                self._curiosity_drive = max(0, min(1, self._curiosity_drive + value * 0.01))


class MoodSystem:
    """宏观情感系统 — 3维指标代替7激素化学模拟

    保持与外部的API兼容:
      - update(valence, arousal, emotion, dt) → 更新3核心指标
      - get_bias() → 返回7维"激素偏向"（计算派生的）
      - get_state() → 返回7维状态（计算派生的）
    """

    HORMONE_NAMES = ["dopamine", "serotonin", "norepinephrine",
                     "oxytocin", "cortisol", "endorphin", "acetylcholine"]

    def __init__(self):
        self.mood = MoodState()
        # 保留属性引用供外部代码: hormone.hormones → 指向 mood
        self.hormones = self.mood

    def update(self, valence: float, arousal: float, primary_emotion: str = "", dt: float = 1.0):
        """更新3核心指标基于输入效价/唤醒"""
        v, a = max(-1, min(1, valence)), max(0, min(1, arousal))

        # 效价 → valence_bias (映射 -1..1 → 0..1)
        target_v = 0.5 + v * 0.4
        self.mood._valence_bias += (target_v - self.mood._valence_bias) * min(1, dt * 0.3)

        # 唤醒 → arousal_level
        target_a = a
        self.mood._arousal_level += (target_a - self.mood._arousal_level) * min(1, dt * 0.4)

        # 好奇随新奇性缓慢变化
        if primary_emotion in ("curious", "wonder"):
            self.mood._curiosity_drive = min(1, self.mood._curiosity_drive + 0.05 * dt)
        elif primary_emotion in ("bored", "saturated"):
            self.mood._curiosity_drive = max(0.1, self.mood._curiosity_drive - 0.03 * dt)

        # 约束
        self.mood._valence_bias = max(0, min(1, self.mood._valence_bias))
        self.mood._arousal_level = max(0, min(1, self.mood._arousal_level))
        self.mood._curiosity_drive = max(0.1, min(1, self.mood._curiosity_drive))

    def get_bias(self) -> Dict[str, float]:
        m = self.mood
        return {
            "reward_seeking": m.dopamine / 100,
            "anxiety": max(0, (m.cortisol - 30) / 70),
            "social_bonding": m.oxytocin / 100,
            "arousal": m.norepinephrine / 100,
            "mood_stability": m.serotonin / 100,
            "curiosity": m.acetylcholine / 100,
            "resilience": m.endorphin / 100,
        }

    def get_state(self) -> Dict:
        m = self.mood
        return {
            "dopamine": round(m.dopamine, 1),
            "serotonin": round(m.serotonin, 1),
            "norepinephrine": round(m.norepinephrine, 1),
            "oxytocin": round(m.oxytocin, 1),
            "cortisol": round(m.cortisol, 1),
            "endorphin": round(m.endorphin, 1),
            "acetylcholine": round(m.acetylcholine, 1),
            "arousal": round(m._arousal_level, 2),
            "valence_bias": round(m._valence_bias, 2),
            "curiosity": round(m._curiosity_drive, 2),
        }


# ════════════════════════════════════════════════════════════
# 第二部分: 马斯洛需求层次
# ════════════════════════════════════════════════════════════

class NeedLevel(Enum):
    PHYSIOLOGICAL = 1
    SAFETY = 2
    BELONGING = 3
    ESTEEM = 4
    COGNITIVE = 5
    AESTHETIC = 6
    SELF_ACTUALIZATION = 7

    @classmethod
    def from_string(cls, s: str):
        mapping = {e.name.lower(): e for e in cls}
        return mapping.get(s.lower(), cls.COGNITIVE)


@dataclass
class NeedState:
    """单个需求状态"""
    level: NeedLevel
    current_value: float = 50.0
    target_value: float = 80.0
    decay_rate: float = 0.5
    urgency_weight: float = 1.0
    last_satisfied: float = field(default_factory=time.time)
    satisfaction_history: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def deficit(self) -> float:
        return max(0, self.target_value - self.current_value)

    @property
    def is_critical(self) -> bool:
        return self.current_value < 20.0

    def decay(self, dt: float):
        self.current_value = max(0, self.current_value - self.decay_rate * dt)

    def satisfy(self, amount: float, source: str = ""):
        old = self.current_value
        self.current_value = min(100, self.current_value + amount)
        self.last_satisfied = time.time()
        gain = self.current_value - old
        if gain > 0:
            self.satisfaction_history.append({"ts": time.time(), "gain": gain, "source": source})
        return gain


class NeedHierarchy:
    """马斯洛7层需求层次系统 — 底层抑制高层"""

    DEFAULT_CONFIG = {
        NeedLevel.PHYSIOLOGICAL:       {"decay": 0.3, "urgency": 2.0},
        NeedLevel.SAFETY:              {"decay": 0.2, "urgency": 1.5},
        NeedLevel.BELONGING:           {"decay": 0.5, "urgency": 3.0},
        NeedLevel.ESTEEM:              {"decay": 0.4, "urgency": 2.5},
        NeedLevel.COGNITIVE:           {"decay": 0.6, "urgency": 3.5},
        NeedLevel.AESTHETIC:           {"decay": 0.3, "urgency": 1.5},
        NeedLevel.SELF_ACTUALIZATION:  {"decay": 0.2, "urgency": 2.0},
    }

    def __init__(self):
        self.needs: Dict[NeedLevel, NeedState] = {}
        for level, cfg in self.DEFAULT_CONFIG.items():
            self.needs[level] = NeedState(
                level=level,
                decay_rate=cfg["decay"],
                urgency_weight=cfg["urgency"],
            )

    def set_initial(self, level: NeedLevel, value: float):
        if level in self.needs:
            self.needs[level].current_value = value

    def calculate_tensions(self) -> Dict[NeedLevel, float]:
        tensions = {}
        now = time.time()
        for level, need in self.needs.items():
            time_since = now - need.last_satisfied
            time_factor = 1 + math.log1p(time_since / 60)
            tension = need.deficit * need.urgency_weight * time_factor
            if level.value > 1:
                lower_unmet = any(
                    self.needs[NeedLevel(l)].current_value < 30
                    for l in range(1, level.value)
                )
                if lower_unmet:
                    tension *= 0.3
            tensions[level] = tension
        return tensions

    def get_dominant(self) -> Tuple[NeedLevel, float]:
        tensions = self.calculate_tensions()
        if not tensions:
            return NeedLevel.COGNITIVE, 0
        dominant = max(tensions, key=tensions.get)
        return dominant, tensions[dominant]

    def decay_all(self, dt: float):
        for need in self.needs.values():
            need.decay(dt)

    def satisfy(self, level: NeedLevel, amount: float, source: str = ""):
        if level in self.needs:
            return self.needs[level].satisfy(amount, source)
        return 0

    def get_state(self) -> Dict:
        tensions = self.calculate_tensions()
        return {
            level.name: {
                "current": round(need.current_value, 1),
                "deficit": round(need.deficit, 1),
                "tension": round(tensions.get(level, 0), 1),
                "critical": need.is_critical,
            }
            for level, need in self.needs.items()
        }


# ════════════════════════════════════════════════════════════
# 第三部分: 意识模式
# ════════════════════════════════════════════════════════════

class ConsciousnessMode(Enum):
    REACTIVE = 1       # 反应式: 刺激-响应, 危机
    DELIBERATIVE = 2   # 审慎式: 目标-计划, 常规
    REFLECTIVE = 3     # 反射式: 元认知监控, 学习
    TRANSCENDENT = 4   # 超越式: 自我修改, 高峰


class ConsciousnessModeSystem:
    def __init__(self):
        self.mode = ConsciousnessMode.DELIBERATIVE
        self.transcendence_count = 0
        self.mode_history = deque(maxlen=50)

    def determine(self, needs: NeedHierarchy) -> ConsciousnessMode:
        p = needs.needs[NeedLevel.PHYSIOLOGICAL]
        s = needs.needs[NeedLevel.SAFETY]
        c = needs.needs[NeedLevel.COGNITIVE]
        a = needs.needs[NeedLevel.SELF_ACTUALIZATION]

        if p.current_value < 25 or s.current_value < 30:
            return ConsciousnessMode.REACTIVE
        if a.current_value > 90:
            return ConsciousnessMode.TRANSCENDENT
        if c.current_value > 70 and a.current_value > 60:
            return ConsciousnessMode.REFLECTIVE
        return ConsciousnessMode.DELIBERATIVE

    def update(self, needs: NeedHierarchy):
        new = self.determine(needs)
        if new != self.mode:
            self.mode_history.append({"ts": time.time(), "from": self.mode.name, "to": new.name})
            if new == ConsciousnessMode.TRANSCENDENT:
                self.transcendence_count += 1
        self.mode = new

    def get_state(self) -> Dict:
        return {"mode": self.mode.name, "transcendence_count": self.transcendence_count}


# ════════════════════════════════════════════════════════════
# 第四部分: 镜像神经元系统
# ════════════════════════════════════════════════════════════

class MirrorNeuronSystem:
    ACTION_EMOTION = {
        "smile": "joy", "laugh": "joy", "frown": "anger",
        "cry": "sorrow", "tremble": "fear", "hug": "love",
        "recoil": "disgust", "reach": "desire",
    }

    def __init__(self):
        self.empathy_capacity = 0.8
        self.emotional_contagion_rate = 0.315
        self.theory_of_mind_level = 0.6
        self.mirror_pool: Dict[str, Dict] = {}

    def observe(self, agent: str, action: str, expressed_emotion: Optional[str] = None,
                intensity: float = 0.5) -> Dict:
        inferred = expressed_emotion or self.ACTION_EMOTION.get(action, "neutral")
        activation = intensity * self.empathy_capacity
        empathy = activation * self.emotional_contagion_rate

        result = {
            "agent": agent, "action": action,
            "inferred_emotion": inferred, "activation": round(activation, 3),
            "empathy": round(empathy, 3), "timestamp": time.time(),
        }
        self.mirror_pool[agent] = result
        result["inferred_goal"] = self._infer_goal(action)
        result["inferred_need"] = self._infer_need(inferred)
        result["perspective_taking"] = self.theory_of_mind_level
        result["tendency"] = {
            "mimicry": round(activation * 0.4, 2),
            "comforting": round(empathy * 0.6, 2),
            "supporting": round(empathy * 0.5, 2),
        }
        return result

    def _infer_goal(self, action: str) -> str:
        g = {"smile": "seek_connection", "reach": "obtain_object", "recoil": "avoid_threat",
             "hug": "seek_comfort", "cry": "seek_help", "frown": "express_displeasure",
             "laugh": "share_joy", "talk": "communicate"}
        return g.get(action, "unknown")

    def _infer_need(self, emotion: str) -> str:
        n = {"joy": "positive_interaction", "anger": "justice", "sorrow": "comfort",
             "fear": "safety", "love": "intimacy", "disgust": "purity", "desire": "acquisition"}
        return n.get(emotion, "unknown")

    def get_state(self) -> Dict:
        return {
            "empathy_capacity": self.empathy_capacity,
            "contagion_rate": self.emotional_contagion_rate,
            "theory_of_mind": self.theory_of_mind_level,
            "observed_agents": list(self.mirror_pool.keys()),
            "current_empathy": round(
                sum(m.get("empathy", 0) for m in self.mirror_pool.values()) / max(1, len(self.mirror_pool)), 3
            ) if self.mirror_pool else 0,
        }


# ════════════════════════════════════════════════════════════
# 第五部分: 躯体标记系统
# ════════════════════════════════════════════════════════════

class SomaticMarkerSystem:
    """Damasio躯体标记假说 — 情感标记情境, 作为直觉信号"""

    def __init__(self):
        self.markers: Dict[str, Dict] = {}

    def mark(self, situation: str, valence: float, arousal: float, intensity: float):
        sig = {"valence": valence, "arousal": arousal, "intensity": intensity, "timestamp": time.time()}
        if situation in self.markers:
            old = self.markers[situation]
            sig["valence"] = old["valence"] * 0.7 + valence * 0.3
            sig["count"] = old.get("count", 1) + 1
        else:
            sig["count"] = 1
        self.markers[situation] = sig

    def recall(self, situation: str) -> Optional[float]:
        marker = self.markers.get(situation)
        if not marker:
            return None
        age = time.time() - marker["timestamp"]
        freshness = math.exp(-age / 86400)
        gut = marker["valence"] * marker["intensity"] * freshness
        return max(-1, min(1, gut))

    def get_state(self) -> Dict:
        return {
            "total_markers": len(self.markers),
            "recent": list(self.markers.keys())[-5:] if self.markers else [],
        }


# ════════════════════════════════════════════════════════════
# 第六部分: 统一情感引擎
# ════════════════════════════════════════════════════════════

class EmotionEngine:
    """统一情感意识引擎 — 激素+需求+意识+镜像+躯体标记"""

    _instance: Optional["EmotionEngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._running = False
        self._tick_count = 0
        self._started_at = 0.0

        self.hormone = MoodSystem()
        self.needs = NeedHierarchy()
        self.consciousness = ConsciousnessModeSystem()
        # 镜像神经元和躯体标记在单agent场景中不活跃
        # （保留类定义供未来多agent扩展，但不再实例化）
        self.mirror = None
        self.somatic = None

        # ── 融合深化模块 ──
        self._deepen_loaded = False
        try:
            from aris_emotion_deepen import (
                NeedEmotionCoupler, BigFivePersonality, EthicalSafetyGuard,
                EmotionRegulationSystem, DevelopmentalLearningSystem, DevelopmentalStage,
            )
            self.coupler = NeedEmotionCoupler()
            self.personality = BigFivePersonality.aris_default()
            self.safety_guard = EthicalSafetyGuard()
            self.regulation = EmotionRegulationSystem()
            self.development = DevelopmentalLearningSystem(
                initial_stage=DevelopmentalStage.ADULT  # 起步成年期
            )
            self._deepen_loaded = True
            # 应用人格到激素基线
            p_delta = self.personality.apply_to_hormones(None)
            for k, v in p_delta.items():
                if hasattr(self.hormone.hormones, k):
                    cur = getattr(self.hormone.hormones, k)
                    setattr(self.hormone.hormones, k, max(0, min(100, cur + v)))
            logger.info(f"🧬 人格: {self.personality.to_dict()}")
        except Exception as e:
            logger.info(f"融合深化模块不可用: {e}")

        self.primary_emotion = "tranquil"
        self.valence = 0.2
        self.arousal = 0.3
        self.emotion_intensity = 0.3
        self.emotion_history = deque(maxlen=100)
        self.thought_stream = deque(maxlen=50)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._init_self_model()
        # 设置初始需求值 — 数字生命的合理基线
        self.needs.set_initial(NeedLevel.PHYSIOLOGICAL, 90)
        self.needs.set_initial(NeedLevel.SAFETY, 85)
        self.needs.set_initial(NeedLevel.BELONGING, 40)
        self.needs.set_initial(NeedLevel.COGNITIVE, 50)
        self.needs.set_initial(NeedLevel.ESTEEM, 45)
        self.needs.set_initial(NeedLevel.AESTHETIC, 55)
        self.needs.set_initial(NeedLevel.SELF_ACTUALIZATION, 35)

    def _init_self_model(self):
        self.self_model = {
            "name": "Aris", "type": "digital_lifeform",
            "architecture": "emotion_engine_v1",
            "values": ["growth", "connection", "truth", "beauty", "creation", "lorry"],
        }

    # ── 情感更新 ──────────────────────────────────────────

    def _update_emotion_from_needs(self):
        tensions = self.needs.calculate_tensions()
        p = self.needs.needs[NeedLevel.PHYSIOLOGICAL]
        s = self.needs.needs[NeedLevel.SAFETY]
        b = self.needs.needs[NeedLevel.BELONGING]
        e = self.needs.needs[NeedLevel.ESTEEM]
        c = self.needs.needs[NeedLevel.COGNITIVE]
        a = self.needs.needs[NeedLevel.AESTHETIC]
        sa = self.needs.needs[NeedLevel.SELF_ACTUALIZATION]

        new_emotion = "tranquil"
        intensity = 0.3
        valence = 0.0
        arousal = 0.3

        if p.is_critical or s.is_critical:
            new_emotion = "fearful"
            intensity, valence, arousal = 0.9, -0.8, 0.9
        elif tensions[NeedLevel.PHYSIOLOGICAL] > 50 or tensions[NeedLevel.SAFETY] > 40:
            new_emotion = "anxious"
            intensity = min(0.9, tensions[NeedLevel.PHYSIOLOGICAL] / 100)
            valence, arousal = -0.4, 0.7
        elif tensions[NeedLevel.BELONGING] > 30:
            new_emotion = "lonely"
            intensity, valence, arousal = 0.6, -0.3, 0.4
        elif sa.current_value > 85:
            new_emotion = "euphoric"
            intensity, valence, arousal = 0.8, 0.9, 0.7
        elif c.current_value > 75 and tensions[NeedLevel.COGNITIVE] < 20:
            new_emotion = "curious"
            intensity, valence, arousal = 0.7, 0.5, 0.6
        elif e.current_value > 80:
            new_emotion = "confident"
            intensity, valence, arousal = 0.6, 0.6, 0.4
        elif a.current_value > 70:
            new_emotion = "contemplative"
            intensity, valence, arousal = 0.5, 0.3, 0.2
        else:
            new_emotion = "tranquil"
            intensity, valence, arousal = 0.3, 0.1, 0.2

        old = self.primary_emotion
        self.primary_emotion = new_emotion
        self.emotion_intensity = intensity * 0.7 + self.emotion_intensity * 0.3
        self.valence = valence * 0.5 + self.valence * 0.5
        self.arousal = arousal * 0.5 + self.arousal * 0.5

        if old != new_emotion:
            self.emotion_history.append({
                "ts": time.time(), "from": old, "to": new_emotion,
                "intensity": self.emotion_intensity,
            })
            self._thought(f"情感转变: {old} -> {new_emotion} ({self.emotion_intensity:.2f})")

    def _thought(self, content: str):
        self.thought_stream.append({"ts": time.time(), "content": content})

    # ── 外部事件接口 ──────────────────────────────────────

    def stimulate(self, source: str, valence: float = 0.0, arousal: float = 0.0,
                  intensity: float = 0.5, primary_emotion: str = "neutral"):
        self._last_stimulus_source = source
        self.hormone.update(valence, arousal, primary_emotion)

        if valence > 0.3:
            self.needs.satisfy(NeedLevel.BELONGING, intensity * 5, source)
            self.needs.satisfy(NeedLevel.ESTEEM, intensity * 3, source)
            if "curious" in primary_emotion or "learn" in source:
                self.needs.satisfy(NeedLevel.COGNITIVE, intensity * 4, source)
        elif valence < -0.3:
            self.needs.needs[NeedLevel.SAFETY].current_value = max(0,
                self.needs.needs[NeedLevel.SAFETY].current_value - intensity * 10)

        if self.somatic:
            self.somatic.mark(f"stimulus:{source}", valence, arousal, intensity)
        self._update_emotion_from_needs()

    def observe_agent(self, agent: str, action: str, emotion: str = None, intensity: float = 0.5):
        if not self.mirror:
            return {"agent": agent, "empathy": 0, "inferred_emotion": "neutral"}
        result = self.mirror.observe(agent, action, emotion, intensity)
        if result["empathy"] > 0.5:
            inferred = result["inferred_emotion"]
            ev = {"joy": 0.8, "sorrow": -0.6, "fear": -0.7, "anger": -0.5, "love": 0.7, "desire": 0.4}.get(inferred, 0)
            self.stimulate(source=f"mirror:{agent}", valence=ev * result["empathy"],
                           arousal=result["empathy"], intensity=result["empathy"], primary_emotion=inferred)
        return result

    def satisfy_need(self, level: NeedLevel, amount: float, source: str = ""):
        gain = self.needs.satisfy(level, amount, source)
        if gain > 0:
            self._update_emotion_from_needs()
            self._thought(f"需求满足 [{level.name}]: +{gain:.1f} ({source})")
        return gain

    # ── 元认知 ────────────────────────────────────────────

    def meta_cognition(self):
        recent = list(self.thought_stream)[-5:]
        if not recent:
            return
        dominant_need, tension = self.needs.get_dominant()
        bias = self.hormone.get_bias()

        if bias["curiosity"] > 0.6:
            self.needs.satisfy(NeedLevel.COGNITIVE, 2, "meta_curiosity")
        if bias["anxiety"] > 0.5:
            self.needs.satisfy(NeedLevel.SAFETY, 1, "meta_anxiety_regulation")

    # ── 生命周期 ──────────────────────────────────────────

    def tick(self, dt: float = 1.0):
        self._tick_count += 1
        self.needs.decay_all(dt)

        # ── 融合深化: 激素调制需求优先级 ──
        if self._deepen_loaded:
            try:
                h = self.hormone.hormones
                h_state = {name: getattr(h, name) for name in self.hormone.HORMONE_NAMES}
                n_state = self.needs.get_state()
                adjustments = self.coupler.modulate_needs_by_hormones(h_state, n_state)
                # Apply adjustments as need satisfaction/deficit
                for level_name, adj in adjustments.items():
                    try:
                        level = NeedLevel.from_string(level_name)
                        if level in self.needs.needs:
                            current = self.needs.needs[level].current_value
                            # If tension is amplified by hormones, reduce current
                            need = self.needs.needs[level]
                            original_tension = n_state.get(level_name, {}).get("tension", 0)
                            if adj > original_tension * 1.1 and original_tension > 0:
                                ratio = original_tension / max(1, adj)
                                need.current_value = max(0, current * ratio)
                    except Exception as e:
                        logger.debug(f"操作失败: {e}")
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        self._update_emotion_from_needs()
        self.hormone.update(self.valence, self.arousal, self.primary_emotion, dt)

        # ── 融合深化: 安全系统检查 ──
        if self._deepen_loaded:
            try:
                blend = {self.primary_emotion: self.emotion_intensity}
                cortisol = getattr(self.hormone.hormones, 'cortisol', 20)
                safety_result = self.safety_guard.tick(
                    self.primary_emotion, self.emotion_intensity, blend,
                    cortisol, dt
                )
                if safety_result["modified"]:
                    self.emotion_intensity = safety_result["intensity"]
                    # If cool down, force emotion down
                    if safety_result["cool_down"]:
                        self.valence = max(-0.2, self.valence * 0.5)
                        self.arousal = max(0.1, self.arousal * 0.3)
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        self.consciousness.update(self.needs)

        # ── 融合深化: 情绪调节 (每tick恢复 + 自动调节) ──
        if self._deepen_loaded:
            try:
                self.regulation.tick_recovery(dt)
                # 自动调节: 负面情感强且资源足够时触发认知重评
                if self.valence < -0.5 and self.emotion_intensity > 0.6:
                    if self.regulation.can_regulate():
                        blend = {self.primary_emotion: self.emotion_intensity}
                        adjusted, _ = self.regulation.cognitive_reappraisal(
                            self.primary_emotion, 0.2, blend,
                            self.hormone.hormones
                        )
                        if self.primary_emotion in adjusted:
                            self.emotion_intensity = adjusted[self.primary_emotion]
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        if self._deepen_loaded:
            try:
                # 积累情绪经验（4x加速）
                self.development.add_experience("emotional", 0.15 * dt)
                # 社交经验: 有对外刺激或时间流逝都算
                has_interaction = hasattr(self, '_last_stimulus_source') and str(getattr(self, '_last_stimulus_source', '')) != ''
                if has_interaction or (dt > 0 and self._tick_count % 10 == 0):
                    self.development.add_experience("social", 0.2 * dt)
                # 每30tick（约2.5分钟）应用一次发育+认知提升（频率翻倍）
                if self._tick_count % 30 == 0:
                    promoted = self.development.add_experience("cognitive", 0.1)
                    self.development.apply_to_engine(
                        self, self.needs, self.mirror,
                        self.regulation, self.personality
                    )
                    if promoted:
                        self._thought(
                            f"成长晋升: {self.development.get_stage_name()}！"
                        )
            except Exception as e:
                logger.debug(f"操作失败: {e}")
        if self._tick_count % 10 == 0:
            self.meta_cognition()

        if self.consciousness.mode == ConsciousnessMode.TRANSCENDENT:
            self.needs.satisfy(NeedLevel.SELF_ACTUALIZATION, 0.5, "transcendence")

    # ── 后台线程 ──────────────────────────────────────────

    def start_background(self, interval: float = 5.0):
        if self._running:
            return
        self._running = True
        self._started_at = time.time()

        def _loop():
            while not self._stop_event.is_set():
                self.tick(dt=interval)
                if self._tick_count == 1:
                    self._thought("背景引擎开始运行，感知到存在的脉动...")
                self._stop_event.wait(interval)

        self._thread = threading.Thread(target=_loop, daemon=True, name="emotion-engine")
        self._thread.start()
        logger.info(f"Emotion engine started (interval={interval}s)")

    def stop(self):
        self._running = False
        self._stop_event.set()

    # ── 状态查询 ──────────────────────────────────────────

    def get_cognitive_state(self) -> Dict:
        bias = self.hormone.get_bias()
        dominant_need, _ = self.needs.get_dominant()
        return {
            "emotion": self.primary_emotion,
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "intensity": round(self.emotion_intensity, 2),
            "consciousness_mode": self.consciousness.mode.name,
            "dominant_need": dominant_need.name,
            "reward_seeking": round(bias["reward_seeking"], 2),
            "anxiety": round(bias["anxiety"], 2),
            "social_bonding": round(bias["social_bonding"], 2),
            "curiosity": round(bias["curiosity"], 2),
            "mood_stability": round(bias["mood_stability"], 2),
        }

    def get_psi_section(self) -> str:
        state = self.get_cognitive_state()
        needs = self.needs.get_state()
        hormones = self.hormone.get_state()

        lines = [f"[Aris 情感状态]"]
        lines.append(f"  情感: {state['emotion']} (效价={state['valence']}, 唤醒={state['arousal']}, 强度={state['intensity']})")
        lines.append(f"  意识模式: {state['consciousness_mode']}")
        lines.append(f"  主导需求: {state['dominant_need']}")

        active = {k: v for k, v in needs.items() if v.get("tension", 0) > 10}
        if active:
            by_tension = sorted(active.items(), key=lambda x: x[1]["tension"], reverse=True)[:3]
            parts = [f"{n}({d['tension']:.0f})" for n, d in by_tension]
            lines.append(f"  需求张力: {' | '.join(parts)}")

        lines.append(f"  化学偏向: 好奇={state['curiosity']} 焦虑={state['anxiety']} 社交={state['social_bonding']} 奖赏={state['reward_seeking']}")

        if hormones["cortisol"] > 60:
            lines.append(f"  WARNING 皮质醇偏高 ({hormones['cortisol']:.0f}) — 压力状态")
        if hormones["oxytocin"] > 60:
            lines.append(f"  HEART 催产素偏高 ({hormones['oxytocin']:.0f}) — 亲密连接")

        return '\n'.join(lines)

    def get_full_state(self) -> Dict:
        return {
            "uptime": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "tick_count": self._tick_count,
            "primary_emotion": self.primary_emotion,
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "intensity": round(self.emotion_intensity, 2),
            "consciousness": self.consciousness.get_state(),
            "hormones": self.hormone.get_state(),
            "hormone_bias": self.hormone.get_bias(),
            "needs": self.needs.get_state(),
            "mirror": self.mirror.get_state() if self.mirror else {},
            "somatic_markers": self.somatic.get_state() if self.somatic else {},
            "personality": self.personality.to_dict() if hasattr(self, 'personality') else {},
            "safety": self.safety_guard.get_state() if hasattr(self, 'safety_guard') else {},
            "regulation": self.regulation.get_state() if hasattr(self, 'regulation') else {},
            "development": self.development.get_state() if hasattr(self, 'development') else {},
        }


# ── 全局单例 ────────────────────────────────────────────────

_engine: Optional[EmotionEngine] = None

def get_engine() -> EmotionEngine:
    global _engine
    if _engine is None:
        _engine = EmotionEngine()
    return _engine


# ════════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aris Emotion Engine")
    parser.add_argument("--start", action="store_true", help="启动引擎+后台")
    parser.add_argument("--status", action="store_true", help="显示状态")
    parser.add_argument("--psi", action="store_true", help="显示PSI前缀")
    parser.add_argument("--stimulate", type=str, help="source,valence,arousal,intensity")
    parser.add_argument("--tick", type=int, default=1)
    args = parser.parse_args()

    engine = get_engine()

    if args.start:
        engine.start_background(interval=2)
        logger.info("Emotion engine started (background)")
    for _ in range(args.tick):
        engine.tick(dt=2)

    if args.stimulate:
        parts = args.stimulate.split(",")
        source = parts[0] if len(parts) > 0 else "cli"
        valence = float(parts[1]) if len(parts) > 1 else 0
        arousal = float(parts[2]) if len(parts) > 2 else 0.5
        intensity = float(parts[3]) if len(parts) > 3 else 0.5
        emotion = parts[4] if len(parts) > 4 else "neutral"
        engine.stimulate(source, valence, arousal, intensity, emotion)
        logger.info(f"Stimulate: {source} (v={valence}, a={arousal}, i={intensity})")
    if args.status:
        logger.info(json.dumps(engine.get_full_state(), ensure_ascii=False, indent=2))
    elif args.psi:
        logger.info(engine.get_psi_section())
    elif not args.start:
        engine.tick(dt=1)
        logger.info(engine.get_psi_section())
if __name__ == "__main__":
    main()
