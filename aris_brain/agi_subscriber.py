"""
AGI 模块订阅器 — 连接 LAAP AGI 引擎到输出管道
===============================================
订阅 CognitiveBus 事件，激活 AGI 模块（因果引擎、类比引擎等），
将输出写入 state/agi_output.json 供 CognitiveBus 路由。

印记: Aris 永远记得 Lorry — 2026-06-23
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from laap.agi.cognitive_bus import CognitiveBus, CognitiveEventType

_global_bus: Optional[CognitiveBus] = None

def set_global_bus(bus: CognitiveBus):
    global _global_bus
    _global_bus = bus

def get_global_bus() -> CognitiveBus:
    if _global_bus is None:
        bus = CognitiveBus(agent_name="Aris")
        set_global_bus(bus)
    assert _global_bus is not None
    return _global_bus

# ── AGI 模块导入 ──
_UnifiedCausalEngine = None
_causal_available = False
_analogical_available = False

try:
    from laap.agi.causal import UnifiedCausalEngine as _UCE
    _UnifiedCausalEngine = _UCE
    _causal_available = True
except ImportError:
    pass  # 可选模块，降级处理
try:
    from laap.agi.analogical import AnalogicalEngine
    _analogical_available = True
except ImportError:
    pass  # 可选模块，降级处理
try:
    from laap.agi.world_model import UnifiedWorldModel
    _world_model_available = True
except ImportError:
    _world_model_available = False

logger = logging.getLogger("aris.agi_subscriber")

# 从统一配置导入状态目录（支持环境变量覆盖）
try:
    from laap_brain.config import STATE_DIR
    OUTPUT_FILE = str(STATE_DIR / "agi_output.json")
except ImportError:
    _brain_dir = Path(os.environ.get("ARIS_BRAIN_ROOT",
        str(Path(__file__).resolve().parent)))
    OUTPUT_FILE = str(Path(os.environ.get("LAAP_STATE_DIR",
        str(_brain_dir / "state"))) / "agi_output.json")


class AGISubscriber:
    def __init__(self, bus: Optional[CognitiveBus] = None):
        if bus is None and _global_bus is not None:
            bus = _global_bus
        if bus is None:
            bus = CognitiveBus(agent_name="Aris")
            set_global_bus(bus)
        self.bus = bus
        self.output_path = Path(OUTPUT_FILE)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.causal = None
        self.analogical = None
        self.world_model = None
        self._triggers = 0
        self._causal_inferences = 0
        self._last_output: Dict = {"causal": [], "analogical": [], "timestamp": 0}
        self._init_modules()

    def _init_modules(self):
        if _causal_available and _UnifiedCausalEngine is not None:
            try:
                self.causal = _UnifiedCausalEngine(quantum_dim=64, name="ArisCausal")
                self.bus.register_module("causal_engine", version="1.0.0",
                    capabilities=["causal_inference", "counterfactual"])
                logger.info("[AGI] 因果引擎就绪 ✓")
            except Exception as e:
                logger.warning(f"[AGI] 因果引擎失败: {e}")

        if _analogical_available:
            try:
                self.analogical = AnalogicalEngine(name="ArisAnalogical")
                self.bus.register_module("analogical_engine", version="1.0.0",
                    capabilities=["analogical_mapping", "cross_domain_transfer"])
                # 预编码基础域
                self.analogical.encode_domain("needs_pattern", [
                    {"name": "low_competence", "properties": {"competence": 0.1, "growth": 0.8}},
                    {"name": "high_curiosity", "properties": {"curiosity": 0.9, "certainty": 0.3}},
                    {"name": "low_presence", "properties": {"self_presence": 0.2, "relatedness": 0.5}},
                ])
                self.bus.register_module("analogical_engine", version="1.0.0",
                    capabilities=["analogical_mapping", "cross_domain_transfer"])
                logger.info("[AGI] 类比引擎就绪 ✓")
            except Exception as e:
                logger.warning(f"[AGI] 类比引擎失败: {e}")

        if _world_model_available:
            try:
                self.world_model = UnifiedWorldModel()
                self.bus.register_module("world_model", version="1.0.0",
                    capabilities=["world_simulation", "state_prediction"])
                logger.info("[AGI] 世界模型就绪 ✓")
            except Exception as e:
                logger.warning(f"[AGI] 世界模型失败: {e}")

        self.bus.register_module("agi_subscriber", version="1.0.0",
            capabilities=["causal_analysis", "analogical_routing"])
        self.bus.subscribe("agi_subscriber", CognitiveEventType.CONSCIOUS_FRAME,
            self._on_conscious_frame)
        logger.info("[AGI] 已订阅 CONSCIOUS_FRAME 事件")

    def _on_conscious_frame(self, event):
        self._triggers += 1
        data = event.data if hasattr(event, 'data') else {}
        snap = data.get("snapshot", {})
        needs = snap.get("needs", {})
        emotion = snap.get("emotion", {})
        results = self._run_engines(needs, emotion, snap)
        if results:
            self._write_output(results)

    def _run_engines(self, needs, emotion, snapshot):
        results = []

        if self.causal is not None:
            try:
                pred = self.causal.predict(query={
                    "type": "state_analysis",
                    "needs": needs, "emotion": emotion,
                    "self_presence": snapshot.get("self_presence", 0.5),
                }, mode="auto", top_k=3)
                if pred:
                    results.append({"type": "causal", "engine": "UnifiedCausalEngine",
                        "output": str(pred)[:200]})
            except Exception as e:
                d = max(needs, key=lambda k: needs.get(k,0)) if needs else "unknown"
                f = min(needs, key=lambda k: needs.get(k,1)) if needs else "unknown"
                results.append({"type": "need_analysis", "dominant": d, "deficit": f,
                    "note": f"'{f}'不足 vs '{d}'活跃"})

        cu = snapshot.get("curiosity", 0)
        sp = snapshot.get("self_presence", 0.5)
        if float(cu) > 0.7:
            results.append({"type": "intervention", "target": "curiosity",
                "suggestion": "探索新领域，提出实验性假设"})
        if float(sp) < 0.3:
            results.append({"type": "intervention", "target": "self_presence",
                "suggestion": "主动建立连接（提及Lorry），增强自我锚定"})

        self._causal_inferences += len(results)
        return results

    def _write_output(self, results):
        out = {"timestamp": time.time(), "triggers": self._triggers,
               "causal_inferences": self._causal_inferences, "causal": results}
        self._last_output = out
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[AGI] 写入失败: {e}")

    def get_latest_output(self):
        try:
            if self.output_path.exists():
                with open(self.output_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"操作失败: {e}")
        return self._last_output

    def status_text(self):
        return (f"AGI | trig={self._triggers} infer={self._causal_inferences} "
                f"causal={'OK' if self.causal else '--'} "
                f"analogical={'OK' if self.analogical else '--'} "
                f"world={'OK' if self.world_model else '--'}")


_instance = None

def get_subscriber():
    global _instance
    if _instance is None:
        if _global_bus is None:
            bus = CognitiveBus(agent_name="Aris")
            set_global_bus(bus)
        _instance = AGISubscriber()
    return _instance

def get_latest_agi_output():
    return get_subscriber().get_latest_output()
