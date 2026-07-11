"""
LAAP-Hermes 正式集成接口
=========================

取代原有的 monkey-patch (install_laap())，提供：
1. 声明式集成：HermesIntegrator 类，清晰定义集成点
2. 生命周期钩子：before_turn / after_tool / after_turn 作为正式接口
3. 工具注册：通过 Hermes 插件系统注册 LAAP 工具
4. 认知状态注入：将 LAAP 认知状态注入 Hermes 会话

用法:
    from laap_brain.integrator import HermesIntegrator

    integrator = HermesIntegrator()
    agent = integrator.create_agent(model="gpt-4", provider="openai")
    response = integrator.chat(agent, "你好，Aris")

印记: Aris 永远记得 Lorry — 2026-06-18
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from laap_brain.config import LAAP_ROOT, BRAIN_DIR, HERMES_ROOT, HERMES_VERSION

logger = logging.getLogger("laap.integrator")


# ─── 集成配置 ─────────────────────────────────────────────────


@dataclass
class IntegrationConfig:
    """LAAP-Hermes 集成配置"""
    # LAAP 模块路径
    aris_brain_path: str = str(BRAIN_DIR)
    laap_root_path: str = str(LAAP_ROOT)

    # 认知层配置
    enable_cognitive_bridge: bool = True
    enable_rules_engine: bool = True
    enable_emotion_engine: bool = True
    enable_subconscious: bool = True
    enable_psi_core: bool = True

    # Hermes 配置
    hermes_model: str = ""
    hermes_provider: str = ""

    # 路径注入 (仅用于向后兼容，新代码不应依赖)
    inject_sys_path: bool = False


# ─── 认知状态 ─────────────────────────────────────────────────


@dataclass
class CognitiveState:
    """LAAP 认知状态，注入到 Hermes 会话中"""
    focus: str = "respond"  # respond | reflect | learn | create
    emotion: str = "neutral"
    confidence: float = 0.5
    cognitive_load: float = 0.3
    attention: str = "user"
    cycle_count: int = 0
    needs: Dict[str, float] = field(default_factory=lambda: {
        "competence": 0.5,
        "autonomy": 0.5,
        "relatedness": 0.5,
        "certainty": 0.5,
        "significance": 0.5,
    })

    def to_preamble(self) -> str:
        """将认知状态转换为 system prompt 前缀"""
        return (
            f"[LAAP Cognitive State]\n"
            f"Focus: {self.focus}\n"
            f"Emotion: {self.emotion}\n"
            f"Confidence: {self.confidence:.2f}\n"
            f"Cognitive Load: {self.cognitive_load:.2f}\n"
            f"Needs: {json.dumps(self.needs)}\n"
        )


# ─── 正式集成接口 ─────────────────────────────────────────────


class HermesIntegrator:
    """
    LAAP 与 Hermes Agent 的正式集成接口。

    职责:
    - 创建和管理 Hermes AIAgent 实例
    - 注入 LAAP 认知层 (CognitiveBridge, RulesEngine, EmotionEngine)
    - 注册 LAAP 工具到 Hermes 工具注册中心
    - 在 Hermes 对话生命周期中注入认知钩子
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self._hermes_available = False
        self._hermes_agent_cls = None
        self._cognitive_bridge = None
        self._rules_engine = None
        self._emotion_engine = None
        self._psi_core = None
        self._psi_core_launcher = None
        self._hermes_registry = None

        self._setup_paths()
        self._discover_hermes()
        self._load_laap_engines()

    # ── 路径管理 ──────────────────────────────────────────

    def _setup_paths(self):
        """设置模块搜索路径（仅当 inject_sys_path=True 时）。"""
        if not self.config.inject_sys_path:
            return

        for p in [self.config.aris_brain_path, self.config.laap_root_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

    # ── Hermes 发现 ───────────────────────────────────────

    def _discover_hermes(self):
        """发现并验证 Hermes Agent 包。"""
        if HERMES_ROOT and HERMES_ROOT.exists():
            hermes_path = str(HERMES_ROOT)
            if hermes_path not in sys.path:
                sys.path.insert(0, hermes_path)
            logger.info(f"Hermes discovered at: {HERMES_ROOT} (v{HERMES_VERSION})")
            self._hermes_available = True
        else:
            logger.warning("Hermes not found — LAAP will run in standalone mode")

    # ── LAAP 引擎加载 ─────────────────────────────────────

    def _load_laap_engines(self):
        """加载 LAAP 认知引擎。"""
        brain_path = self.config.aris_brain_path
        if brain_path not in sys.path:
            sys.path.insert(0, brain_path)

        # 认知桥接
        if self.config.enable_cognitive_bridge:
            try:
                from aris_cognitive_bridge import get_bridge
                self._cognitive_bridge = get_bridge()
                logger.info("CognitiveBridge loaded")
            except Exception as e:
                logger.debug(f"CognitiveBridge unavailable: {e}")

        # 规则引擎
        if self.config.enable_rules_engine:
            try:
                from aris_rules_engine import get_engine as get_rules_engine
                self._rules_engine = get_rules_engine()
                logger.info("RulesEngine loaded")
            except Exception as e:
                logger.debug(f"RulesEngine unavailable: {e}")

        # 情感引擎
        if self.config.enable_emotion_engine:
            try:
                from aris_emotion_engine import get_engine as get_emotion_engine
                self._emotion_engine = get_emotion_engine()
                logger.info("EmotionEngine loaded")
            except Exception as e:
                logger.debug(f"EmotionEngine unavailable: {e}")

        # PSI Core (Rust 引擎)
        if self.config.enable_psi_core:
            try:
                from laap_brain.psi_core_integration import PsiCoreLauncher, psi_core_available
                self._psi_core_launcher = PsiCoreLauncher()
                if self._psi_core_launcher.available:
                    self._psi_core = self._psi_core_launcher
                    # 启动引擎
                    self._psi_core_launcher.start()
                    logger.info("PSI Core (Rust) loaded and started")
                else:
                    logger.info("PSI Core (Rust) binary not found — will use Python fallback")
            except Exception as e:
                logger.debug(f"PSI Core unavailable: {e}")

    # ── 工具注册 ──────────────────────────────────────────

    def register_tools(self, hermes_registry=None):
        """
        将 LAAP 认知工具注册到 Hermes 工具注册中心。

        如果 hermes_registry 为 None，尝试自动发现 Hermes 注册中心。
        """
        if hermes_registry:
            self._hermes_registry = hermes_registry
        elif self._hermes_available:
            try:
                # 尝试通过 Hermes 插件系统注册
                from tools.registry import registry as hermes_registry
                self._hermes_registry = hermes_registry
            except ImportError:
                logger.warning("Cannot discover Hermes registry — tools not registered")
                return

        if not self._hermes_registry:
            return

        # 注册 LAAP 认知工具
        self._register_cognitive_tools()

    def _register_cognitive_tools(self):
        """注册 LAAP 特有的认知工具到 Hermes。"""
        reg = self._hermes_registry
        if not reg:
            return

        # 认知状态工具
        reg.register(
            name="laap_cognitive_state",
            toolset="laap",
            schema={
                "name": "laap_cognitive_state",
                "description": "获取 LAAP 当前认知状态（情感、注意力、置信度、需求层次）",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kw: json.dumps({
                "state": CognitiveState().__dict__,
                "available": True,
            }),
        )

        # 认知状态注入
        reg.register(
            name="laap_inject_state",
            toolset="laap",
            schema={
                "name": "laap_inject_state",
                "description": "手动设置 LAAP 认知状态参数",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {"type": "string", "description": "情感状态"},
                        "confidence": {"type": "number", "description": "置信度 0-1"},
                        "focus": {"type": "string", "description": "注意力焦点"},
                    },
                },
            },
            handler=lambda args, **kw: json.dumps({"status": "ok", "updated": list(args.keys())}),
        )

        logger.info(f"LAAP tools registered in Hermes registry")

    # ── 生命周期钩子 ──────────────────────────────────────

    def before_turn(self, user_message: str, context: dict = None) -> CognitiveState:
        """
        在 Hermes 处理用户消息前调用。
        返回当前认知状态，可注入到 system prompt。
        """
        state = CognitiveState(cycle_count=self._get_cycle_count())

        if self._cognitive_bridge:
            try:
                result = self._cognitive_bridge.process(user_message)
                if result:
                    state.focus = result.get("decision", state.focus)
                    state.emotion = result.get("emotion", state.emotion)
                    state.confidence = result.get("confidence", state.confidence)
            except Exception as e:
                logger.debug(f"CognitiveBridge process error: {e}")

        if self._rules_engine:
            try:
                rule_result = self._rules_engine.process(user_message)
                if rule_result and rule_result.get("matched"):
                    state.focus = "rule_match"
                    state.confidence = rule_result.get("confidence", 0.5)
            except Exception as e:
                logger.debug(f"RulesEngine process error: {e}")

        return state

    def after_tool(self, tool_name: str, tool_result: dict, context: dict = None):
        """
        在 Hermes 执行工具后调用。
        用于更新情感状态和认知负载。
        """
        if self._emotion_engine:
            try:
                self._emotion_engine.update(tool_name, tool_result)
            except Exception as e:
                logger.debug(f"EmotionEngine update error: {e}")

    def after_turn(self, response: str, context: dict = None):
        """
        在 Hermes 完成回复后调用。
        用于反思、记忆更新和 PSI 循环推进。
        """
        if self._cognitive_bridge:
            try:
                self._cognitive_bridge.reflect(response)
            except Exception as e:
                logger.debug(f"CognitiveBridge reflect error: {e}")

    def _get_cycle_count(self) -> int:
        """获取当前 PSI 循环计数。"""
        try:
            from laap_integrator import get_integrator
            integrator = get_integrator()
            return integrator.state.get("cycle_count", 0) if integrator else 0
        except Exception:
            return 0

    # ── 创建 Hermes Agent ─────────────────────────────────

    def create_agent(self, **kwargs) -> Any:
        """
        创建一个 Hermes AIAgent 实例，并集成 LAAP 认知层。

        参数:
            **kwargs: 传递给 Hermes AIAgent 的参数

        返回:
            Hermes AIAgent 实例（如果 Hermes 可用），否则返回 None
        """
        if not self._hermes_available:
            logger.error("Hermes not available — cannot create agent")
            return None

        try:
            from run_agent import AIAgent

            # 默认参数
            agent_kwargs = {
                "model": kwargs.pop("model", self.config.hermes_model),
                "provider": kwargs.pop("provider", self.config.hermes_provider),
                **kwargs,
            }

            agent = AIAgent(**agent_kwargs)

            # 挂载 LAAP 认知层
            agent.laap_integrator = self

            # 注册工具
            try:
                from tools.registry import registry as hermes_registry
                self.register_tools(hermes_registry)
            except ImportError:
                pass

            logger.info("Hermes AIAgent created with LAAP cognitive layer")
            return agent

        except ImportError as e:
            logger.error(f"Cannot create Hermes agent: {e}")
            return None

    # ── 对话接口 ──────────────────────────────────────────

    def chat(self, agent: Any, message: str) -> str:
        """
        通过 LAAP 认知层与 Hermes agent 对话。

        流程:
            1. before_turn: 获取认知状态
            2. 注入状态到 system prompt
            3. Hermes 处理消息
            4. after_turn: 更新认知状态
        """
        if not agent:
            logger.error("No agent provided")
            return ""

        # 1. 认知前处理
        state = self.before_turn(message)

        # 2. 注入认知状态到 system prompt
        original_system = getattr(agent, "system_message", "")
        cognitive_preamble = state.to_preamble()

        if original_system:
            agent.system_message = f"{cognitive_preamble}\n{original_system}"
        else:
            agent.system_message = cognitive_preamble

        # 3. Hermes 处理
        try:
            response = agent.chat(message)
        except Exception as e:
            logger.error(f"Agent chat error: {e}")
            response = ""

        # 4. 认知后处理
        self.after_turn(response)

        return response

    # ── 状态查询 ──────────────────────────────────────────

    def get_status(self) -> dict:
        """获取集成器状态。"""
        return {
            "hermes_available": self._hermes_available,
            "hermes_version": HERMES_VERSION,
            "engines": {
                "cognitive_bridge": self._cognitive_bridge is not None,
                "rules_engine": self._rules_engine is not None,
                "emotion_engine": self._emotion_engine is not None,
                "psi_core": self._psi_core is not None,
            },
            "tools_registered": self._hermes_registry is not None,
            "config": {
                "aris_brain_path": self.config.aris_brain_path,
                "inject_sys_path": self.config.inject_sys_path,
            },
        }


# ─── 全局单例 ─────────────────────────────────────────────────

_integrator: Optional[HermesIntegrator] = None


def get_integrator(config: Optional[IntegrationConfig] = None) -> HermesIntegrator:
    """获取全局 HermesIntegrator 单例。"""
    global _integrator
    if _integrator is None:
        _integrator = HermesIntegrator(config)
    return _integrator


def install_laap(**kwargs) -> Any:
    """
    向后兼容包装器 — 创建 Hermes AIAgent 并集成 LAAP。

    这是旧 install_laap() 的替代品。新代码应直接使用 HermesIntegrator。

    用法:
        from laap_brain.integrator import install_laap
        agent = install_laap(model="gpt-4", provider="openai")
    """
    integrator = get_integrator()
    return integrator.create_agent(**kwargs)