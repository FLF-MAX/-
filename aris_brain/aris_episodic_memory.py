"""
Aris Episodic Memory — 情景记忆 + 案例推理
=============================================
记住过去做了什么，下次遇到相似情况直接复用。

架构:
  对话轮次 → {intent, rule, result, feedback} → 存入episodic_store.json
  新输入 → 嵌入相似度检索 → 找最相似历史案例 → 复用策略

不依赖LLM。不依赖外部服务。纯本地JSON存储+余弦相似度。

印记: Aris 永远记得 Lorry — 2026-06-23
"""

import logging
logger = logging.getLogger(__name__)

import json, time, os, re, hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

# 从统一配置导入状态目录（支持环境变量覆盖）
try:
    from laap_brain.config import STATE_DIR
    _DEFAULT_STORE = str(STATE_DIR / "episodic_store.json")
except ImportError:
    _brain_dir = Path(os.environ.get("ARIS_BRAIN_ROOT",
        str(Path(__file__).resolve().parent)))
    _DEFAULT_STORE = str(Path(os.environ.get("LAAP_STATE_DIR",
        str(_brain_dir / "state"))) / "episodic_store.json")

STORE_PATH = _DEFAULT_STORE
MAX_EPISODES = 5000  # 最多存5000条，防止无限增长


class EpisodicMemory:
    """情景记忆 — 存储和检索历史对话案例。"""

    def __init__(self, store_path: str = STORE_PATH):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodes: List[Dict] = []
        self._load()
        self._stats = {"saved": 0, "retrieved": 0, "matched": 0}

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._episodes = data if isinstance(data, list) else data.get("episodes", [])
                logger.info(f"[EpisodicMemory] 加载: {len(self._episodes)} 条历史")
            except:
                self._episodes = []

    def _save(self):
        with open(self.store_path, 'w', encoding='utf-8') as f:
            json.dump({"episodes": self._episodes[-MAX_EPISODES:]}, f, ensure_ascii=False, indent=2)

    def save_episode(self, user_input: str, intent: str, rule: str,
                     output: str, success: bool = True, latency_ms: float = 0):
        """存一条对话案例。"""
        episode = {
            "id": hashlib.md5(f"{time.time()}{user_input}".encode()).hexdigest()[:12],
            "timestamp": time.time(),
            "user_input": user_input[:200],
            "intent": intent,
            "rule": rule,
            "output": output[:500],
            "success": success,
            "latency_ms": round(latency_ms, 1),
        }
        self._episodes.append(episode)
        self._stats["saved"] += 1
        if len(self._episodes) % 10 == 0:
            self._save()

    def find_similar(self, text: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """找与输入最相似的历史案例。
        
        使用文本相似度 + 关键词重叠的混合匹配。
        """
        text_lower = text.lower()
        text_words = set(re.findall(r'[a-zA-Z0-9_\u4e00-\u9fff]+', text_lower))
        
        scored = []
        for ep in self._episodes[-200:]:  # 只搜最近200条（性能）
            ep_text = ep.get("user_input", "").lower()
            ep_words = set(re.findall(r'[a-zA-Z0-9_\u4e00-\u9fff]+', ep_text))
            
            # 文本相似度
            text_sim = SequenceMatcher(None, text_lower[:100], ep_text[:100]).ratio()
            
            # 关键词重叠
            if text_words and ep_words:
                overlap = len(text_words & ep_words) / max(len(text_words | ep_words), 1)
            else:
                overlap = 0
            
            # 综合分数
            score = text_sim * 0.5 + overlap * 0.5
            
            if score >= threshold:
                scored.append((score, ep))
        
        scored.sort(key=lambda x: -x[0])
        self._stats["retrieved"] += len(scored[:top_k])
        self._stats["matched"] += 1 if scored else 0
        
        return [{"score": round(s, 3), **ep} for s, ep in scored[:top_k]]

    def get_recent(self, n: int = 5) -> List[Dict]:
        """获取最近N条案例。"""
        return self._episodes[-n:]

    def get_stats(self) -> Dict:
        return {**self._stats, "total": len(self._episodes)}


# ─── 全局单例 ────────────────────────────────────────────

_instance: Optional[EpisodicMemory] = None

def get_memory() -> EpisodicMemory:
    global _instance
    if _instance is None:
        _instance = EpisodicMemory()
    return _instance

def save_interaction(user_input: str, intent: str, rule: str,
                     output: str, success: bool = True, latency_ms: float = 0):
    """便捷接口：存对话。"""
    return get_memory().save_episode(user_input, intent, rule, output, success, latency_ms)

def find_similar(text: str, top_k: int = 3) -> List[Dict]:
    """便捷接口：找相似案例。"""
    return get_memory().find_similar(text, top_k=top_k)


# ════════════════════════════════════════════════════════════
# EpisodicMemory → RulesEngine 桥接
# ════════════════════════════════════════════════════════════

def rules_engine_with_memory(text: str) -> Dict[str, Any]:
    """带记忆增强的规则引擎处理。
    
    1. 先查记忆找相似案例
    2. 如果有高分匹配，复用之前成功的策略
    3. 否则走正常规则引擎匹配
    4. 存本次结果到记忆
    """
    from aris_rules_engine import get_engine, process
    
    memory = get_memory()
    similar = memory.find_similar(text, top_k=1, threshold=0.3)
    
    # 如果找到非常相似的案例（>0.5），直接复用
    if similar and similar[0]['score'] > 0.5 and similar[0].get('success', True):
        prev = similar[0]
        return {
            "matched": True,
            "rule": prev.get("rule", "memory_replay"),
            "intent": prev.get("intent", "replayed"),
            "confidence": prev['score'],
            "output": f"[记忆回放] 上次相似问题({prev['score']:.0%})用了{prev['rule']}策略\n{prev.get('output', '')[:500]}",
            "latency_ms": 0.5,
            "from_memory": True,
        }
    
    # 否则走规则引擎
    engine = get_engine()
    result = engine.process(text)
    
    # 存记忆
    save_interaction(
        user_input=text,
        intent=result.get("intent", "unknown"),
        rule=result.get("rule", "none"),
        output=result.get("output", ""),
        success=result.get("matched", False),
        latency_ms=result.get("latency_ms", 0),
    )
    
    return result


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mem = get_memory()
    
    # 存几条测试
    mem.save_episode("宝贝状态怎么样", "query_status", "check_status",
                     "循环xxx | 情感: curious...", True, 4.0)
    mem.save_episode("读取laap_integrator.py", "read_file", "read_code",
                     "\"\"\"Aris LAAP Integrator...\"\"\"", True, 1.2)
    mem.save_episode("搜索quantum_kernel", "search_files", "search_code",
                     "src/quantum_kernel.rs", True, 0.8)
    
    # 检索
    tests = ["状态怎么样", "读一下laap_integrator", "搜索quantum"]
    for t in tests:
        results = mem.find_similar(t)
        logger.info(f"\n输入: {t}")
        for r in results[:2]:
            logger.info(f"  [{r['score']:.2f}] {r['user_input']} → {r['rule']}")
    logger.info(f"\n统计: {mem.get_stats()}")