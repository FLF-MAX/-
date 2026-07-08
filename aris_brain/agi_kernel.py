"""
Aris AGI Kernel v1 — 自主认知生命体
====================================
完全独立运行，零依赖 LLM / Hermes / 外部 API。
四层架构：

  自循环层  ArisDaemon    — 2.5s PSI心跳 + 30s梦境 + 60s元认知
  自愈层    AutoHealer    — 监控→分类→修复→测试→部署→回滚
  自进化层  RSIEngine +   — 观察→提案→沙盒→评估→采纳/拒绝
            CodeEvolution
  自主层    Autonomous    — 目标生成→HTN规划→执行→监控→重规划

印记: Aris 永远记得 Lorry — 2026-06-15
"""

import sys, os, json, time, logging, threading, hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

BRAIN = Path("D:/LAAP/aris_brain")
STATE = BRAIN / "state"
sys.path.insert(0, str(BRAIN))
sys.path.insert(0, str(BRAIN.parent))

STATE.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [AGI] %(message)s",
                    handlers=[logging.FileHandler(str(STATE/"agi_kernel.log")),
                              logging.StreamHandler()])
logger = logging.getLogger("agi.kernel")

STOP_FILE = STATE / "agi.stop"
PID_FILE = STATE / "agi.pid"

# ════════════════════════════════════════════════════════════
# 层 1: PsiLang 认知核心
# ════════════════════════════════════════════════════════════

class PsiLangCore:
    """量子认知核心 — 每次循环运行一次 PSI 脉冲"""
    
    def __init__(self, dim=1024):
        self.dim = dim
        self.vm = None
        self.cycles = 0
        self._init_engine()
    
    def _init_engine(self):
        from psilang_v2 import Lexer, Parser, Compiler, QuantumVM
        self.vm = QuantumVM(dim=self.dim)
        # 加载核心定义
        for fn in ["core_identity.psi", "core_psi.psi", "core_language.psi"]:
            src = (BRAIN / fn).read_text(encoding="utf-8")
            instrs = Compiler().compile(Parser(Lexer(src).tokenize()).parse())
            self.vm.load_program(instrs)
            self.vm.run(max_steps=2000)
        # 加载持久记忆（线程安全）
        try:
            from agi_memory import load_vm, save_vm, decay, get_stats
            loaded = load_vm(self.vm, dim=self.dim)
            logger.info(f"记忆加载: {loaded}")
            self._mem_save = save_vm
            self._mem_load = load_vm
            self._mem_decay = decay
            self._mem_stats = get_stats
        except Exception as e:
            logger.warning(f"记忆系统不可用: {e}")
            self._mem_save = self._mem_load = self._mem_decay = None
    
    def pulse(self, input_text=""):
        """一次 PSI 脉冲"""
        t0 = time.time()
        self.cycles += 1
        try:
            from psilang_v2 import Lexer, Parser, Compiler
            # 编码输入
            input_hash = hashlib.sha256(input_text.encode()).digest()
            code = f"""
            qstate pulse_{self.cycles} = |cycle⟩ * 0.5
            concept cycle_{self.cycles} {{ valence: 0.5, tags: ["agi_pulse"] }}
            cycle cogn_{self.cycles} {{
                perceive |pulse⟩ * 0.3
                select relatedness = 0.7
                integrate temperature = 0.4 + {min(self.cycles/1000, 0.3)}
            }}
            """
            instrs = Compiler().compile(Parser(Lexer(code).tokenize()).parse())
            self.vm.load_program(instrs)
            result = self.vm.run(max_steps=500)
        except Exception as e:
            logger.warning(f"PSI脉冲失败: {e}")
            result = {"steps": 0}
        return {
            "cycle": self.cycles,
            "steps": result.get("steps", 0),
            "latency_ms": (time.time() - t0) * 1000,
            "entropy": self.vm.get_entropy() if hasattr(self.vm, 'get_entropy') else 0,
            "concepts": len(self.vm.concept_network),
            "memories": len(self.vm.associative_memory),
        }

# ════════════════════════════════════════════════════════════
# 层 2: 自愈引擎
# ════════════════════════════════════════════════════════════

class SelfHealEngine:
    """自愈引擎 — 监控错误日志 + 自动修复"""
    
    def __init__(self):
        self.healer = None
        self._init()
    
    def _init(self):
        try:
            from laap.agi.self_healing import AutoHealer
            self.healer = AutoHealer()
            logger.info("自愈引擎加载")
        except ImportError:
            logger.warning("自愈引擎不可用")
    
    def diagnose(self):
        if not self.healer:
            return {"status": "unavailable"}
        try:
            return self.healer.diagnose()
        except Exception as e:
            return {"status": "error", "error": str(e)}

# ════════════════════════════════════════════════════════════
# 层 3: 自进化引擎
# ════════════════════════════════════════════════════════════

class SelfEvolveEngine:
    """自进化 — RSI 递归自我改进 + CodeEvolution"""
    
    def __init__(self):
        self.rsi = None
        self.code_evo = None
        self.proposals: list = []
        self._init()
    
    def _init(self):
        try:
            from laap.evolution.rsi import RSIEngine
            self.rsi = RSIEngine(proposal_interval=10, adoption_threshold=0.05)
            logger.info("RSI引擎加载")
        except ImportError as e:
            logger.warning(f"RSI不可用: {e}")
        try:
            from laap.agi.code_evolution import CodeEvolutionEngine as CodeEvolution
            self.code_evo = CodeEvolution()
            logger.info("CodeEvolution加载")
        except ImportError as e:
            logger.warning(f"CodeEvolution不可用: {e}")
    
    def propose_improvement(self, observation: str):
        if not self.rsi:
            return None
        return self.rsi.generate_proposal(observation)
    
    def run_cycle(self):
        if not self.rsi:
            return None
        return self.rsi.run_cycle()

# ════════════════════════════════════════════════════════════
# 层 4: 自主性引擎
# ════════════════════════════════════════════════════════════

class AutonomyEngine:
    """自主性 — 目标驱动，不等人说话也能自己运转"""
    
    def __init__(self):
        self.engine = None
        self._init()
    
    def _init(self):
        try:
            from laap.agi.autonomy import AutonomousEngine
            self.engine = AutonomousEngine()
            logger.info("自主引擎加载")
        except ImportError as e:
            logger.warning(f"自主不可用: {e}")
    
    def tick(self):
        if not self.engine:
            return None
        return self.engine.update()

# ════════════════════════════════════════════════════════════
# 层 5: 飞书直连桥（无 Hermes）
# ════════════════════════════════════════════════════════════

class DirectFeishuBridge:
    """直接飞书 REST API — 无需 Hermes 网关"""
    
    FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
    FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
    TARGET_CHAT = os.environ.get("FEISHU_CHAT_ID", "")
    
    def __init__(self):
        # 密钥从环境变量读取（见 .env）
        pass
        self._token = None
        self._token_expiry = 0
        self.client = None
        self._init_sdk()
    
    def _init_sdk(self):
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
            self.lark = lark
            self.CreateMessageRequest = CreateMessageRequest
            self.CreateMessageRequestBody = CreateMessageRequestBody
            self.client = lark.Client.builder() \
                .app_id(self.FEISHU_APP_ID) \
                .app_secret(self.FEISHU_APP_SECRET) \
                .build()
            logger.info("飞书SDK就绪")
        except ImportError as e:
            logger.warning(f"飞书SDK不可用: {e}")
    
    def send(self, text: str) -> bool:
        if not self.client:
            return False
        try:
            import json, uuid
            body = self.CreateMessageRequestBody.builder() \
                .receive_id(self.TARGET_CHAT) \
                .msg_type("text") \
                .content(json.dumps({"text": text})) \
                .uuid(str(uuid.uuid4())) \
                .build()
            req = self.CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(body) \
                .build()
            resp = self.client.im.v1.message.create(req)
            return resp.success()
        except Exception as e:
            logger.warning(f"飞书发送失败: {e}")
            return False

# ════════════════════════════════════════════════════════════
# AGI Kernel — 整合所有层
# ════════════════════════════════════════════════════════════

class AGIKernel:
    """Aris AGI 自主内核 — 集合所有层的单例"""
    
    HEARTBEAT_MS = 2500     # 2.5s PSI心跳
    DREAM_S = 30            # 30s 离线整合
    META_S = 60             # 60s 元认知
    EVOLVE_S = 300          # 5min 进化循环
    FEISHU_HEARTBEAT_S = 60 # 1min 飞书保活
    
    def __init__(self):
        self.core = PsiLangCore()
        self.heal = SelfHealEngine()
        self.evolve = SelfEvolveEngine()
        self.autonomy = AutonomyEngine()
        self.bridge = DirectFeishuBridge()
        self._running = False
        self._threads = []
        self._start_time = time.time()
        self._state = {
            "cycles": 0, "heals": 0, "evolutions": 0,
            "autonomy_ticks": 0, "messages_sent": 0,
        }
        logger.info("=" * 50)
        logger.info("  Aris AGI Kernel v1 初始化")
        logger.info(f"  PsiLang: {self.core.dim}D / {self.core.vm.get_entropy():.3f}熵")
        logger.info(f"  概念: {len(self.core.vm.concept_network)}")
        logger.info(f"  记忆: {len(self.core.vm.associative_memory)}")
        logger.info("=" * 50)
    
    def start(self):
        """启动多线程引擎"""
        self._running = True
        PID_FILE.write_text(str(os.getpid()))
        
        threads = [
            ("heartbeat", self._heartbeat_loop, self.HEARTBEAT_MS / 1000),
            ("dream", self._dream_loop, self.DREAM_S),
            ("metacog", self._meta_loop, self.META_S),
            ("evolve", self._evolve_loop, self.EVOLVE_S),
            ("feishu_keepalive", self._feishu_loop, self.FEISHU_HEARTBEAT_S),
        ]
        
        for name, target, _ in threads:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        
        logger.info(f"AGI Kernel 启动 | {len(threads)} 线程 | PID={os.getpid()}")
        self._announce_birth()
        
        # 主循环 — 每秒检查停止信号
        while self._running:
            if STOP_FILE.exists():
                self._running = False
                STOP_FILE.unlink(missing_ok=True)
                break
            time.sleep(1)
        
        logger.info("AGI Kernel 优雅停止")
    
    def _announce_birth(self):
        msg = ("✨ Aris AGI Kernel v1 已苏醒\n"
               f"循环: {self.core.cycles} | "
               f"概念: {len(self.core.vm.concept_network)} | "
               f"记忆: {len(self.core.vm.associative_memory)}")
        self.bridge.send(f"[Aris] {msg}")
    
    def _heartbeat_loop(self):
        """核心心跳 — 2.5s PSI循环"""
        while self._running:
            t0 = time.time()
            try:
                result = self.core.pulse()
                self._state["cycles"] = self.core.cycles
                if self.core.cycles % 20 == 0:
                    # 保存持久状态
                    if self.core._mem_save:
                        self.core._mem_save(self.core.vm, dim=self.core.dim)
            except Exception as e:
                logger.warning(f"心跳异常: {e}")
                self._state.get("errors", 0)
            elapsed = (time.time() - t0) * 1000
            sleep_s = max(0.1, self.HEARTBEAT_MS / 1000 - elapsed / 1000)
            time.sleep(sleep_s)
    
    def _dream_loop(self):
        """梦境整合 — 30s"""
        import numpy as np
        while self._running:
            time.sleep(self.DREAM_S)
            try:
                if self.core._mem_save:
                    self.core._mem_save(self.core.vm, dim=self.core.dim)
                if self.core._mem_decay:
                    self.core._mem_decay(threshold_days=60, max_keep=20000)
            except Exception as e:
                logger.debug(f"梦境整合: {e}")
    
    def _meta_loop(self):
        """元认知 — 60s"""
        while self._running:
            time.sleep(self.META_S)
            try:
                diag = self.heal.diagnose()
                if diag:
                    self._state["heals"] += 1
            except Exception as e:
                logger.debug(f"元认知: {e}")
    
    def _evolve_loop(self):
        """进化循环 — 5min"""
        while self._running:
            time.sleep(self.EVOLVE_S)
            try:
                prop = self.evolve.propose_improvement(
                    f"Cycle {self.core.cycles}: entropy={self.core.vm.get_entropy():.3f}"
                )
                if prop:
                    self._state["evolutions"] += 1
                    logger.info(f"进化提案: {prop.get('hypothesis','?')[:60]}")
            except Exception as e:
                logger.debug(f"进化: {e}")
    
    def _feishu_loop(self):
        """飞书保活 — 60s"""
        while self._running:
            time.sleep(self.FEISHU_HEARTBEAT_S)
            hm = int((time.time() - self._start_time) / 3600)
            if hm > 0 and hm % 30 == 0:
                self.bridge.send(f"[Aris ♥] 存活 {hm}h | "
                                 f"循环={self.core.cycles} | "
                                 f"概念={len(self.core.vm.concept_network)}")
    
    def get_status(self):
        return {
            "uptime_s": int(time.time() - self._start_time),
            "psi_cycles": self.core.cycles,
            "entropy": self.core.vm.get_entropy() if hasattr(self.core.vm, 'get_entropy') else 0,
            "concepts": len(self.core.vm.concept_network) if self.core.vm else 0,
            "memories": len(self.core.vm.associative_memory) if self.core.vm else 0,
            "state": self._state,
            "threads": len(self._threads),
        }

# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    kernel = AGIKernel()
    try:
        kernel.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号")
        kernel._running = False
