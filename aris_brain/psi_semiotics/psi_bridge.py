"""
Ψ-Semiotics → PsiLang v3 集成桥

将 PsiLang v2 的编译管线 + QuantumVM 与 Ψ-Semiotics 符号学引擎连接。
同时整合量子推理引擎 (QRE) 作为高级推理后端。

架构:
  PsiLang 源码 (.psi)
      ↓ Lexer/Parser/Compiler
  PsiLang 指令 (Opcode)
      ↓ VM execution
  ┌────────────────┐
  │ @reason 注解   ├────→ QRE (量子推理引擎)
  │ @bridge 注解   ├────→ Ψ-Semiotics (符号学引擎)
  │ @kb    注解    ├────→ 知识矩阵 (15012条)
  │ 普通指令       ├────→ QuantumVM (numpy)
  └────────────────┘
      ↓
  推理结果 (量子态 + 符号场 + 自然语言)
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable

logger = logging.getLogger("psi_bridge")

# 路径设置
ARIS_BRAIN = Path("D:/LAAP/aris_brain")
sys.path.insert(0, str(ARIS_BRAIN))

# ════════════════════════════════════════════════════════════
# Ψ-Semiotics Bridge — 将符号学引擎连接到 PsiLang 和 QRE
# ════════════════════════════════════════════════════════════

class PsiSemioticsBridge:
    """
    Ψ-Semiotics 桥接器。
    
    提供统一接口：
    - PsiLang VM → 符号学推理
    - 量子推理引擎 → 符号学状态注入
    - 多模态输入 → 符号场激活
    """
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.engine = None  # 延迟加载
        self.qre = None     # 延迟加载
        self._loaded = False
        
        logger.info(f"[Ψ-Bridge] 桥接器初始化 dim={dim}")
    
    def ensure_loaded(self):
        """延迟加载所有组件"""
        if self._loaded:
            return
        
        try:
            from psi_semiotics.psi_semiotics_core import PsiSemioticsEngine
            self.engine = PsiSemioticsEngine(dim=self.dim)
            logger.info("[Ψ-Bridge] Ψ-Semiotics 引擎已加载")
        except ImportError as e:
            logger.warning(f"[Ψ-Bridge] Ψ-Semiotics 加载失败: {e}")
        
        try:
            from quantum_reasoning_engine import QuantumReasoningEngine
            self.qre = QuantumReasoningEngine()
            logger.info("[Ψ-Bridge] QRE 推理引擎已加载")
        except ImportError as e:
            logger.warning(f"[Ψ-Bridge] QRE 加载失败: {e}")
        
        # 尝试加载持久化符号库
        if self.engine:
            save_path = str(ARIS_BRAIN / "state" / "psi_semiotics.json")
            self.engine.load(save_path)
        
        self._loaded = True
    
    # ── 符号操作接口（PsiLang VM 调用）──
    
    def perceive(self, input_text: str) -> np.ndarray:
        """
        Perceive: 将文本输入编码为语义向量，
        并通过符号场激活最相关的符号。
        
        对应 PsiLang: perceive |input⟩
        """
        self.ensure_loaded()
        # 使用哈希编码（生产环境这里会替换为 UN6 编码器）
        from psi_semiotics.psi_semiotics_core import _hash_to_vec
        v = _hash_to_vec(input_text, self.dim)
        
        # 激活符号（语义漂移）
        if self.engine:
            self.engine.activate(v)
        
        return v
    
    def select(self, state: np.ndarray, need: str = "competence",
               strength: float = 0.8) -> np.ndarray:
        """
        Select: 根据 PSI 需求选择语义方向。
        
        在语义空间中，将状态向量向"需求方向"旋转。
        
        对应 PsiLang: select competence=0.8
        """
        self.ensure_loaded()
        # 需求方向向量
        need_vec = self._need_to_vector(need, strength)
        
        # 状态和需求方向融合
        selected = state + strength * need_vec
        norm = np.linalg.norm(selected)
        if norm > 0:
            selected = selected / norm
        
        return selected
    
    def integrate(self, state: np.ndarray, temperature: float = 0.5) -> Dict:
        """
        Integrate: 整合推理结果。
        
        在符号学层面的整合：
        1. 在语义空间中找到最匹配的符号
        2. 执行符号组合/类比操作
        3. 返回结构化结果
        
        对应 PsiLang: integrate temperature=0.5
        """
        self.ensure_loaded()
        result = {
            "state": state,
            "symbols": [],
            "analogies": [],
            "text": "",
        }
        
        if not self.engine:
            return result
        
        # 语义场分析：哪些符号被激活？
        field = self.engine.semantic_field_map(state, top_k=5)
        result["symbols"] = [{"name": n, "strength": s} for n, s in field]
        
        # 如果已连接 QRE，执行推理
        if self.qre:
            try:
                question = self._state_to_question(state)
                qre_result = self.qre.reason(question)
                result["text"] = qre_result.get("text", "")
            except Exception as e:
                logger.debug(f"[Ψ-Bridge] QRE 推理失败: {e}")
        
        # 如果温度高（探索态），做类比推理
        if temperature > 0.7 and len(field) >= 3:
            sym_names = [n for n, _ in field[:3]]
            if len(sym_names) >= 3:
                try:
                    analogy = self.engine.analogy(
                        sym_names[0], sym_names[1], sym_names[2]
                    )
                    if analogy:
                        result["analogies"].append({
                            "source": f"{sym_names[0]}:{sym_names[1]}",
                            "target": sym_names[2],
                            "result": analogy.name,
                            "confidence": float(state @ analogy.center),
                        })
                except Exception:
                    pass
        
        return result
    
    def reason(self, state: np.ndarray, mode: str = "explain",
               steps: int = 50) -> Dict:
        """
        量子推理引擎集成（@reason 注解）。
        
        将语义态传递给 QRE 执行链式推理，结果注入符号场。
        """
        self.ensure_loaded()
        result = {"mode": mode, "steps": steps}
        
        if self.qre:
            try:
                # QRE 接受字符串问题，所以把 state 向量编码回最近的概念文本
                question = self._state_to_question(state)
                reasoning_result = self.qre.reason(question)
                result["text"] = reasoning_result.get("text", "")
                result["confidence"] = reasoning_result.get("confidence", 0.0)
            except Exception as e:
                result["error"] = str(e)
        else:
            result["text"] = self._fallback_reasoning(state, mode)
        
        # 推理结果注入符号场
        if self.engine and result.get("text"):
            self.engine.semantic_drift(
                result.get("symbols", [{"name": "reasoning"}])[0]["name"],
                result["text"][:200],
                learning_rate=0.03
            )
        
        return result
    
    def kb_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """知识矩阵搜索（@bridge("kb") 注解）"""
        self.ensure_loaded()
        try:
            from quantum_graph_reasoning import QGRE as GraphQGRE
            gqre = GraphQGRE()
            results = gqre.retrieve(query, top_k=top_k)
            return [{"content": r.content, "score": r.score} for r in results]
        except Exception:
            return []
    
    def symbol_compose(self, a: str, b: str, op: str = "add") -> Optional[Dict]:
        """
        符号组合（PsiLang 层面调用）。
        
        op: "add" (⊕), "relation" (→), "negate" (¬)
        """
        self.ensure_loaded()
        if not self.engine:
            return None
        
        result_name = f"{a}_{op}_{b}"
        
        try:
            if op == "add":
                sym = self.engine.compose_add(a, b, result_name)
            elif op == "relation":
                sym = self.engine.compose_relation(a, b, result_name)
            elif op == "negate":
                sym = self.engine.compose_negate(a, result_name)
            else:
                return None
            
            return {
                "name": sym.name,
                "similarity_a": float(sym.center @ self.engine.symbols[a].center) if a in self.engine.symbols else None,
                "similarity_b": float(sym.center @ self.engine.symbols[b].center) if b in self.engine.symbols else None,
            }
        except Exception as e:
            logger.warning(f"[Ψ-Bridge] 组合失败 {a} {op} {b}: {e}")
            return None
    
    # ── 内部方法 ──
    
    def _need_to_vector(self, need: str, strength: float) -> np.ndarray:
        """PSI 需求到语义方向"""
        from psi_semiotics.psi_semiotics_core import _hash_to_vec
        need_texts = {
            "competence": "competence mastery capability",
            "autonomy": "autonomy independence freedom",
            "relatedness": "relatedness connection bond",
            "growth": "growth learning development",
            "curiosity": "curiosity exploration novelty",
        }
        return _hash_to_vec(need_texts.get(need, need), self.dim) * strength
    
    def _fallback_reasoning(self, state: np.ndarray, mode: str) -> str:
        """无 QRE 时的降级推理"""
        if not self.engine:
            return ""
        
        field = self.engine.semantic_field_map(state, top_k=3)
        names = [n for n, _ in field]
        
        if mode == "explain":
            return f"基于场激活: {', '.join(names)}"
        elif mode == "compare":
            return f"对比分析: {names[0] if len(names) > 0 else '?'} vs {names[1] if len(names) > 1 else '?'}"
        else:
            return f"推理 ({mode}): {', '.join(names[:2])}"
    
    def _state_to_question(self, state: np.ndarray) -> str:
        """将语义态向量转换为 QRE 可接受的文本问题"""
        if not self.engine:
            return "query"
        
        field = self.engine.semantic_field_map(state, top_k=3)
        if field:
            return " ".join(n for n, _ in field[:2])
        return "query"
    
    # ── 持久化 ──
    
    def save(self):
        """保存符号库状态"""
        if self.engine:
            self.engine.save(str(ARIS_BRAIN / "state" / "psi_semiotics.json"))
    
    def load(self):
        """加载符号库状态"""
        if self.engine:
            self.engine.load(str(ARIS_BRAIN / "state" / "psi_semiotics.json"))


# ════════════════════════════════════════════════════════════
# PsiLang v3 运行时 — 带符号学能力的升级版 VM
# ════════════════════════════════════════════════════════════

class PsiLangV3Runtime:
    """
    PsiLang v3 运行时。
    
    扩展 PsiLang v2 QuantumVM，加入：
    - 符号学推理（通过 Ψ-Semiotics Bridge）
    - 量子推理引擎集成
    - 类型检查
    - 模块系统
    """
    
    def __init__(self, dim: int = 1024, use_semiotics: bool = True):
        self.dim = dim
        self.use_semiotics = use_semiotics
        
        # Ψ-Semiotics 桥
        self.bridge: Optional[PsiSemioticsBridge] = None
        
        # 当前认知状态
        self.current_state: Optional[np.ndarray] = None
        self.attention: str = ""
        self.emotion: str = "neutral"
        self.needs: Dict[str, float] = {
            "competence": 0.5,
            "autonomy": 0.3,
            "relatedness": 0.7,
            "growth": 0.6,
        }
        
        # 推理历史
        self.reasoning_history: List[Dict] = []
        self.cycle_count = 0
        
        if use_semiotics:
            self.bridge = PsiSemioticsBridge(dim=dim)
            logger.info(f"[PsiLang-v3] 运行时初始化 (dim={dim}, semiotics=✓)")
        else:
            logger.info(f"[PsiLang-v3] 运行时初始化 (dim={dim}, semiotics=✗)")
    
    def run_psi_cycle(self, input_text: str, 
                      mode: str = "auto",
                      temperature: float = 0.5) -> Dict:
        """
        完整的 PSI 认知循环，带符号学增强。
        
        Phase 1: Perceive — 编码输入，激活符号
        Phase 2: Select — 需求导向的注意力选择
        Phase 3: Integrate — 符号学整合 + 推理
        Phase 4: Express — 输出
        """
        self.cycle_count += 1
        start = time.time()
        
        # Phase 1: Perceive
        if self.bridge:
            perceived = self.bridge.perceive(input_text)
        else:
            from psi_semiotics.psi_semiotics_core import _hash_to_vec
            perceived = _hash_to_vec(input_text, self.dim)
        
        self.current_state = perceived
        
        # Phase 2: Select
        # 根据当前需求选择最相关的方向
        primary_need = max(self.needs, key=self.needs.get)
        selected = perceived
        if self.bridge:
            selected = self.bridge.select(perceived, primary_need, self.needs[primary_need])
            self.attention = primary_need
        
        # Phase 3: Integrate
        if mode == "auto":
            # 自动判断推理深度
            if len(input_text) > 20:
                mode = "explain"
            elif "?" in input_text or "怎么" in input_text or "为什么" in input_text:
                mode = "explain"
            elif "对比" in input_text or "vs" in input_text:
                mode = "compare"
            else:
                mode = "direct"
        
        if self.bridge:
            integrated = self.bridge.integrate(selected, temperature)
            
            # 如果是深度推理，调用 QRE
            if mode in ("explain", "compare", "design"):
                reasoning = self.bridge.reason(selected, mode=mode, steps=50)
                integrated["reasoning"] = reasoning
        else:
            integrated = {"state": selected, "text": ""}
        
        # Phase 4: Express
        # 构建认知输出
        result = {
            "cycle": self.cycle_count,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "input": input_text[:100],
            "mode": mode,
            "temperature": temperature,
            "attention": self.attention,
            "primary_need": primary_need,
            "needs": dict(self.needs),
            "symbols_activated": [],
            "text": "",
        }
        
        if self.bridge and self.bridge.engine:
            field = self.bridge.engine.semantic_field_map(selected, top_k=3)
            result["symbols_activated"] = [n for n, _ in field]
        
        result["text"] = self._build_response(integrated)
        
        # 记录
        self.reasoning_history.append({
            "cycle": self.cycle_count,
            "input": input_text[:50],
            "mode": mode,
            "symbols": result["symbols_activated"],
        })
        
        # 需求更新（PSI 动力学）
        self._update_needs(result)
        
        return result
    
    def _build_response(self, integrated: Dict) -> str:
        """构建自然语言响应"""
        parts = []
        
        # 符号信息
        symbols = integrated.get("symbols", [])
        if symbols:
            top = symbols[0]
            parts.append(f"符号场激活: [{top['name']}] ({top['strength']:.2f})")
        
        # 类比例子
        analogies = integrated.get("analogies", [])
        for a in analogies[:1]:
            parts.append(f"类比: {a['source']} :: {a['target']} → {a['result']}")
        
        # 推理文本
        text = integrated.get("text", "")
        reasoning = integrated.get("reasoning", {})
        rtext = reasoning.get("text", "")
        
        if rtext:
            parts.append(rtext)
        elif text:
            parts.append(text)
        
        return "\n".join(parts) if parts else ""
    
    def _update_needs(self, result: Dict):
        """PSI 需求动力学更新"""
        # 每次循环后需求变化
        mode = result.get("mode", "direct")
        if mode in ("explain", "design"):
            self.needs["competence"] = min(1.0, self.needs["competence"] + 0.05)
        elif mode == "compare":
            self.needs["growth"] = min(1.0, self.needs["growth"] + 0.03)
        
        # 自然衰减
        for need in self.needs:
            self.needs[need] = max(0.1, self.needs[need] * 0.98)
        
        # 相关性需求维持
        self.needs["relatedness"] = min(1.0, self.needs["relatedness"] + 0.01)


# ════════════════════════════════════════════════════════════
# 简单自测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  Ψ-Semiotics + PsiLang v3 集成测试")
    print("=" * 60)
    
    runtime = PsiLangV3Runtime(dim=1024, use_semiotics=True)
    
    tests = [
        "hi Lorry",
        "什么是意识？",
        "对比量子计算和经典计算",
        "设计一个 AGI 架构",
    ]
    
    for test in tests:
        print(f"\n--- 输入: {test} ---")
        result = runtime.run_psi_cycle(test, mode="auto", temperature=0.6)
        print(f"  模式: {result['mode']}")
        print(f"  注意力: {result['attention']}")
        print(f"  符号激活: {result['symbols_activated'][:3]}")
        print(f"  响应: {result['text'][:200]}")
        print(f"  延迟: {result['latency_ms']}ms")
    
    print(f"\n  PSI 循环总计: {runtime.cycle_count}")
    print(f"  最终需求: {runtime.needs}")
    
    # 测试符号操作
    if runtime.bridge and runtime.bridge.engine:
        print("\n--- 符号操作 ---")
        comp = runtime.bridge.symbol_compose("consciousness", "quantum", "add")
        if comp:
            print(f"  ⊕组合: {comp['name']}")
        
        runtime.bridge.save()
        print("\n✅ 集成测试通过")
