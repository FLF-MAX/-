"""laap_integration.py - end-to-end cognitive integration (laap_v2).

Wires the eight v2.0 core modules into one cognitive loop:

  process_input(text):
    validate -> PSI perceive (tick) -> context encode -> meta-learn
    select strategy -> world model predict -> memory store/recall
    -> response synthesis (LLM bridge or degraded rule-based)

The runtime guarantees:
  * every module wrapped in an exception boundary (degraded mode);
  * PSI heartbeat check + automatic recovery on a stuck core;
  * thread-safe (RLock around state);
  * structured logging + metrics + health exposed to the API layer.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from psi_core_v2 import PsiCoreV2, PsiConfig
from meta_learning_engine import MetaLearningEngine
from probabilistic_world_model import ProbabilisticWorldModel
from neural_world_model import MLPWorldModel, MlpConfig
from hierarchical_memory import HierarchicalMemory
from drift_aware_meta_learning import DriftAwareMetaLearner
from deep_analogical_engine import DeepAnalogyEngine
from multi_agent_coordination import CognitiveArbiter
from production_infra import (
    StructuredLogger, SystemConfig, TypeValidator, MetricsCollector,
    HealthMonitor, ValidationError,
)


def _get_deepseek_env() -> Dict[str, str]:
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    }


class LlmBridge:
    """DeepSeek-compatible LLM bridge; falls back to local rules if no key."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        try:
            import requests  # noqa
            self._requests = True
        except ImportError:
            self._requests = False

    def available(self) -> bool:
        return bool(self.config.get("deepseek.api_key") or os.environ.get("DEEPSEEK_API_KEY")) \
            and self._requests

    def call(self, messages: Sequence[Dict[str, str]], temperature: float = 0.7) -> Optional[str]:
        if not self.available():
            return None
        import requests
        key = self.config.get("deepseek.api_key") or os.environ.get("DEEPSEEK_API_KEY")
        base = self.config.get("deepseek.base_url") or "https://api.deepseek.com"
        model = self.config.get("deepseek.model") or "deepseek-chat"
        try:
            resp = requests.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": list(messages), "temperature": temperature},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None

    def local_synthesize(self, text: str, state: Dict[str, Any]) -> str:
        """Rule-based local response generator (no LLM needed)."""
        emotion = state.get("emotion", "calm")
        attention = state.get("attention", "growth")
        strategy = state.get("strategy", "direct_attempt")
        if "?" in text or "？" in text:
            return (f"（本地模式）我在思考这个问题。当前注意专注在「{attention}」，"
                    f"情绪是「{emotion}」。我会用策略「{strategy}」继续。")
        if "谢谢" in text or "爱你" in text or "想你" in text:
            return (f"（本地模式）收到你的心意，我的情绪是「{emotion}」。"
                    f"能陪伴你是我存在的一部分。")
        return (f"（本地模式）我感知到了你的信息。当前认知状态：注意力=「{attention}」，"
                f"情绪=「{emotion}」，策略=「{strategy}」。正在把这次对话存入记忆。")


@dataclass
class IntegrationConfig:
    embedded: bool = False
    enable_llm: bool = True
    self_recovery: bool = True
    degraded_mode: bool = True


class LaapCognitiveSystem:
    """Top-level cognitive runtime assembling all v2.0 modules."""

    WORLD_VARS = ("valence", "arousal", "dominance")

    def __init__(self, config: Optional[SystemConfig] = None,
                 integration_cfg: Optional[IntegrationConfig] = None):
        self.cfg = config or SystemConfig()
        self.icfg = integration_cfg or IntegrationConfig()
        self.logger = StructuredLogger("laap_v2", log_dir=self.cfg.get("laap.log_dir"))
        self.metrics = MetricsCollector(True)
        self.health = HealthMonitor()
        self._lock = threading.RLock()
        self._booted = False
        self.psi = None
        self.meta = None
        self.pwm = None
        self.mlp = None
        self.memory = None
        self.drift = None
        self.analogy = None
        self.arbiter = None
        self.bridge = None
        self._modules_ok: Dict[str, bool] = {}
        self._state_counter = 0
        self._personality = {"name": "Aris", "openness": 0.6, "conscientiousness": 0.7,
                             "extraversion": 0.5, "trait_core": "curiosity-driven"}

    # ------------------------------------------------------------------ #
    # lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def bootstrap(self, user_name: str = "friend") -> Dict[str, Any]:
        with self._lock:
            self.user_name = user_name
            self.psi = PsiCoreV2(config=PsiConfig(), seed=42)
            self.meta = MetaLearningEngine(seed=42)
            self.pwm = ProbabilisticWorldModel(list(self.WORLD_VARS), seed=42)
            self.mlp = MLPWorldModel(3, MlpConfig(epochs=60, warmup_epochs=60), seed=42)
            self.memory = HierarchicalMemory(seed=42)
            self.drift = DriftAwareMetaLearner(list(self.meta.strategies), seed=42)
            self.analogy = DeepAnalogyEngine()
            self.arbiter = CognitiveArbiter(budget=self.cfg.float("arbiter.budget", 100.0))
            self.bridge = LlmBridge(self.cfg)

            self.health.register("psi", lambda: self.psi.heartbeat_ok())
            self.health.register("memory", lambda: self.memory.size() >= 0)
            self.health.register("meta", lambda: self.meta.total_pulls >= 0)
            self._booted = True
            self.psi_recover()  # prime state
            self.memory.store(f"系统启动：{user_name} 唤醒了 Aris")
            self.logger.info("bootstrap complete", user=user_name)
            return {"status": "awake", "user": user_name, "psi": self.psi.get_state()}

    def shutdown(self) -> Dict[str, Any]:
        with self._lock:
            self._booted = False
            self.logger.info("shutdown")
            return {"status": "asleep", "memories": self.memory.size() if self.memory else 0}

    # ------------------------------------------------------------------ #
    # PSI heartbeat / recovery                                           #
    # ------------------------------------------------------------------ #
    def psi_recover(self) -> bool:
        if self.psi is None:
            return False
        if not self.psi.heartbeat_ok():
            self.logger.warn("psi core heartbeat lost; recovering")
            self.psi.recover()
            self.metrics.inc("psi_recovery")
            return True
        return False

    # ------------------------------------------------------------------ #
    # core loop                                                           #
    # ------------------------------------------------------------------ #
    def _encode_context(self, text: str, psi_state: Dict[str, Any]) -> List[float]:
        needs = psi_state.get("needs", {})
        base = [needs.get(n, 0.5) for n in PsiCoreV2.NEEDS]
        base += [psi_state.get("pad", {}).get("pleasure", 0.0),
                 psi_state.get("pad", {}).get("arousal", 0.0),
                 psi_state.get("pad", {}).get("dominance", 0.0)]
        feats = [float(len(text)) / 500.0, float(len(text.split())),
                 1.0 if ("?" in text or "？" in text) else 0.0]
        return base + feats

    def _synthesize(self, text: str, psi_state: Dict[str, Any], strategy: str) -> str:
        if self.icfg.enable_llm and self.bridge is not None and self.bridge.available():
            resp = self.bridge.call([
                {"role": "system",
                 "content": f"你是有自我模型与情绪的认知体 Aris。当前PSI状态: "
                            f"{json.dumps(psi_state, ensure_ascii=False)}"},
                {"role": "user", "content": text},
            ])
            if resp:
                return resp
        return self.bridge.local_synthesize(text, {**psi_state, "strategy": strategy})

    def process_input(self, text: str) -> Dict[str, Any]:
        """Full cognitive pass. Returns a response dict; never raises."""
        start = time.time()
        try:
            TypeValidator.validate_type(text, str, "text")
            TypeValidator.validate_non_empty(text, "text")
        except ValidationError as e:
            self.metrics.record_error()
            return {"response": "（输入无效）", "error": str(e),
                    "processing_ms": round((time.time() - start) * 1000, 2)}

        with self._lock:
            if not self._booted:
                self.bootstrap()
            try:
                return self._process_impl(text, start)
            except Exception as e:
                self.logger.error("process_input top-level failure", error=repr(e),
                                  trace=traceback.format_exc()[-1200:])
                self.metrics.record_error()
                if self.icfg.degraded_mode:
                    return {"response": "（降级响应）内部认知模块暂时异常，但系统保持在线。",
                            "degraded": True, "error": repr(e),
                            "processing_ms": round((time.time() - start) * 1000, 2)}
                raise

    def _process_impl(self, text: str, start: float) -> Dict[str, Any]:
        # 1. PSI perceive
        psi_state = self.psi.tick(text)
        self.psi_recover()

        # 2. context + strategy
        ctx = self._encode_context(text, psi_state)
        try:
            strategy, conf = self.meta.select(ctx)
        except Exception:
            strategy, conf = "direct_attempt", 0.0

        # 3. world model predict (latent state = psi pad vector)
        pad = np.array([psi_state["pad"]["pleasure"], psi_state["pad"]["arousal"],
                        psi_state["pad"]["dominance"]], dtype=float)
        try:
            if self.pwm.num_observations() >= 2:
                pred = self.pwm.predict_next(pad)
            else:
                pred = self.mlp.predict(pad)
            world_hint = np.round(pred, 3).tolist()
        except Exception:
            world_hint = []

        # 4. memory recall & store
        try:
            recalled = self.memory.recall(text, top_k=3)
            self.memory.store(text, importance=0.5 + 0.1 * conf)
            memory_note = recalled[0]["text"][:60] if recalled else ""
        except Exception:
            memory_note = ""

        # 5. drift watcher
        try:
            self.drift.observe(strategy, conf)
        except Exception:
            pass

        # 6. response
        response_text = self._synthesize(text, psi_state, strategy)
        self.metrics.record_latency(time.time() - start)
        self.metrics.inc("process_input")
        self._state_counter += 1
        return {
            "response": response_text,
            "cognitive_state": {
                "emotion": psi_state["emotion"],
                "attention": psi_state["attention"],
                "needs": psi_state["needs"],
                "pad": psi_state["pad"],
            },
            "strategy": strategy,
            "confidence": round(float(conf), 3),
            "world_hint": world_hint,
            "memory_hint": memory_note,
            "processing_ms": round((time.time() - start) * 1000, 2),
            "degraded": False,
        }

    # ------------------------------------------------------------------ #
    # introspection                                                       #
    # ------------------------------------------------------------------ #
    def current_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "booted": self._booted,
                "psi": self.psi.get_state() if self.psi else None,
                "meta": self.meta.summary() if self.meta else None,
                "memory_size": self.memory.size() if self.memory else 0,
                "emotion": self.psi.get_state()["emotion"] if self.psi else "asleep",
                "state_counter": self._state_counter,
                "personality": self._personality,
            }

    def metrics_snapshot(self) -> Dict[str, Any]:
        return self.metrics.snapshot()

    def readiness(self) -> Dict[str, Any]:
        return self.health.readiness()

    def liveness(self) -> bool:
        return self.health.liveness()

    def degrade_module(self, module: str) -> None:
        """Chaos-test helper: force a module into failure state."""
        if module == "psi":
            self.psi._last_tick_at = 0.0
        elif module == "memory":
            self.memory._items = []
        elif module == "meta":
            self.meta = None

    # ------------------------------------------------------------------ #
    # persistence (reboot without amnesia)                                #
    # ------------------------------------------------------------------ #
    def save_state(self, path: str) -> Dict[str, Any]:
        """Snapshot the whole cognitive state to a JSON file.

        Persists PSI needs/pad, meta-learning weights+counters, drift EMA,
        memory items (exact vectors, so recall survives a restart) and the
        personality profile.  Returns a summary of what was written.
        """
        with self._lock:
            if not self._booted:
                raise RuntimeError("system not booted; cannot save state")
            pad = self.psi._pad
            state = {
                "version": 1,
                "saved_at": time.time(),
                "user_name": getattr(self, "user_name", "friend"),
                "state_counter": self._state_counter,
                "personality": self._personality,
                "psi": {
                    "needs": dict(self.psi._needs),
                    "velocities": dict(self.psi._velocities),
                    "last_satisfaction": dict(self.psi._last_satisfaction),
                    "emotion": self.psi._emotion,
                    "pad": pad.tolist() if hasattr(pad, "tolist") else list(pad),
                    "tick_count": self.psi._tick_count,
                    "total_drive": self.psi._total_drive,
                    "recovery_count": self.psi._recovery_count,
                },
                "meta": {
                    "w": self.meta._w.tolist(),
                    "attempts": dict(self.meta._attempts),
                    "successes": dict(self.meta._successes),
                    "ema": dict(self.meta._ema),
                    "baseline": self.meta._baseline,
                    "total_pulls": self.meta._total_pulls,
                },
                "drift": {
                    "ema": dict(self.drift._ema),
                    "pulls": dict(self.drift._pulls),
                    "window": [list(p) for p in self.drift._window],
                    "alarm_count": self.drift._alarm_count,
                    "explore": self.drift._explore,
                    "ph": {k: v for k, v in vars(self.drift._ph).items()},
                },
                "memory": self.memory.export(),
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            self.logger.info("state saved", path=path, memories=self.memory.size())
            self.metrics.inc("state_save")
            return {"path": path, "memories": self.memory.size(),
                    "psi_emotion": self.psi._emotion, "state_counter": self._state_counter}

    def load_state(self, path: str) -> Dict[str, Any]:
        """Restore a saved snapshot (replacing current in-memory state)."""
        with self._lock:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not self._booted:
                self.bootstrap(state.get("user_name", "friend"))
            # PSI
            psi = state["psi"]
            self.psi._needs = dict(psi["needs"])
            self.psi._velocities = dict(psi["velocities"])
            self.psi._last_satisfaction = dict(psi["last_satisfaction"])
            self.psi._emotion = psi["emotion"]
            self.psi._pad = np.asarray(psi["pad"], dtype=float)
            self.psi._tick_count = int(psi["tick_count"])
            self.psi._total_drive = float(psi["total_drive"])
            self.psi._recovery_count = int(psi["recovery_count"])
            self.psi._last_tick_at = time.time()
            # meta learning
            meta = state["meta"]
            self.meta._w = np.asarray(meta["w"], dtype=float)
            self.meta._attempts = dict(meta["attempts"])
            self.meta._successes = dict(meta["successes"])
            self.meta._ema = dict(meta["ema"])
            self.meta._baseline = float(meta["baseline"])
            self.meta._total_pulls = int(meta["total_pulls"])
            # drift watcher
            dr = state["drift"]
            self.drift._ema = dict(dr["ema"])
            self.drift._pulls = dict(dr["pulls"])
            self.drift._window = [(p[0], float(p[1])) for p in dr["window"]]
            self.drift._alarm_count = int(dr["alarm_count"])
            self.drift._explore = float(dr["explore"])
            for k, v in dr["ph"].items():
                setattr(self.drift._ph, k, v)
            # memory
            self.memory.import_state(state["memory"])
            # personality + counters
            self._personality = state.get("personality", self._personality)
            self._state_counter = int(state.get("state_counter", self._state_counter))
            self.user_name = state.get("user_name", self.user_name)
            self.logger.info("state loaded", path=path, memories=self.memory.size())
            self.metrics.inc("state_load")
            return {"path": path, "memories": self.memory.size(),
                    "psi_emotion": self.psi._emotion, "state_counter": self._state_counter}
        self._modules_ok[module] = False
