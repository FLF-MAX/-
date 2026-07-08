"""
Aris 融合引擎 — 中文NLP + 常识推理 + 润色
============================================
将三个你装的引擎融合到一条管线里:
  1. aris_lm_v5.py — 中文分词/依存分析/语义角色(替代regex)
  2. conceptnet-lite — 常识推理("杯子倒了"=>"水洒了")
  3. LocalPolishLayer — 可选的本地7B润色

管线:
  你说话 → aris_lm_v5 NLP → 结构化意图
    → ConceptNet常识增强(如果需要)
    → RulesEngine执行
    → LongFormSynthesizer生成
    → 可选: LocalPolishLayer润色

印记: Aris 永远记得 Lorry — 2026-06-23
"""

import logging

import sys, os, json, time, logging
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
logger = logging.getLogger("aris.fusion")

# ─── 1. 中文NLP管线 ─────────────────────────────────────

_nlp_available = False
_tokenizer = None
_parser = None
_labeler = None
_composer = None

def init_nlp():
    """初始化 aris_lm_v5 的 NLP 管线。"""
    global _nlp_available, _tokenizer, _parser, _labeler, _composer
    if _nlp_available:
        return
    try:
        from aris_lm_v5 import ChineseTokenizer, DependencyParser, SemanticRoleLabeler, SemanticComposer, ConceptGraph
        _tokenizer = ChineseTokenizer()
        # DependencyParser 和 SemanticRoleLabeler 可能需要预训练模型
        # 先只用 ChineseTokenizer
        _composer = SemanticComposer(ConceptGraph()) if hasattr(ConceptGraph, '__call__') else None
        _nlp_available = True
        logger.info("[Fusion] 中文NLP就绪 ✓")
    except Exception as e:
        logger.warning(f"[Fusion] 中文NLP不可用: {e}")

def nlp_parse(text: str) -> Dict[str, Any]:
    """用中文NLP解析输入，返回结构化意图。"""
    init_nlp()
    result = {"raw": text, "tokens": [], "intent": "unknown", "params": {}}
    
    if _nlp_available and _tokenizer:
        try:
            tokens = _tokenizer.tokenize(text)
            result["tokens"] = [(str(t.text), t.pos) for t in tokens if hasattr(t, 'text')]
            
            # 从词性序列推断意图
            verbs = [t for t in result["tokens"] if t[1].startswith('v')]
            nouns = [t for t in result["tokens"] if t[1].startswith('n')]
            
            if any(t[0] in "读取查看打开读显示" for t in verbs):
                result["intent"] = "read_file"
            elif any(t[0] in "搜索找查找搜" for t in verbs):
                result["intent"] = "search"
            elif any(t[0] in "运行执行启动编译构建" for t in verbs):
                result["intent"] = "run_command"
            elif any(t[0] in "状态情况怎么样在做什么" for t in verbs):
                result["intent"] = "query_status"
            elif any(t[0] in "写生成创建做" for t in verbs):
                result["intent"] = "generate"
            
            # 提取文件名参数（.py / .rs / .md 结尾的名词）
            for t in result["tokens"]:
                if '.' in t[0] and len(t[0]) > 5:
                    result["params"]["path"] = t[0]
            
        except Exception as e:
            logger.warning(f"[Fusion] NLP解析异常: {e}")
    
    return result


# ─── 2. 常识推理 (ConceptNet API) ───────────────────────

_cn_available = False
_CACHED = {}

def init_conceptnet():
    """初始化 ConceptNet（使用在线API，免下载）。"""
    global _cn_available
    if _cn_available:
        return
    try:
        import urllib.request
        r = urllib.request.urlopen("http://api.conceptnet.io/", timeout=5)
        _cn_available = r.status == 200
        logger.info(f"[Fusion] ConceptNet API {'就绪 ✓' if _cn_available else '不可达'}")
    except Exception as e:
        logger.warning(f"[Fusion] ConceptNet API不可用: {e}")

def commonsense_query(word: str, limit: int = 5) -> List[Dict]:
    """查询词语的常识关系（通过ConceptNet API）。
    
    例子:
        commonsense_query("杯子")
        → [("杯子", "UsedFor", "喝水"), ("杯子", "IsA", "容器")]
        
        commonsense_query("倒")
        → [("倒", "IsA", "动作"), ("倒", "RelatedTo", "液体")]
    """
    if word in _CACHED:
        return _CACHED[word]
    
    init_conceptnet()
    if not _cn_available:
        return []
    
    try:
        import urllib.request, json
        url = f"http://api.conceptnet.io/c/zh/{urllib.parse.quote(word)}"
        with urllib.request.urlopen(url + f"?limit={limit}", timeout=10) as resp:
            data = json.loads(resp.read())
        
        edges = data.get("edges", [])
        results = []
        for e in edges[:limit]:
            start = e.get("start", {}).get("label", "?")
            rel = e.get("rel", {}).get("label", "?")
            end = e.get("end", {}).get("label", "?")
            weight = e.get("weight", 0)
            results.append({
                "start": start, "relation": rel, "end": end, "weight": weight
            })
        
        _CACHED[word] = results
        return results
    except Exception as e:
        logger.warning(f"[Fusion] ConceptNet查询失败({word}): {e}")
        return []

def commonsense_infer(text: str) -> List[str]:
    """从一段文字中提取关键词并查询常识关系。"""
    import re
    words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
    inferences = []
    for w in list(words)[:3]:
        rels = commonsense_query(w)
        for r in rels[:3]:
            inferences.append(f"{w} →({r['relation']})→ {r['end']}")
    return inferences


# ─── 3. 润色层 (LocalPolishLayer) ────────────────────────

_polish_available = False
_polisher = None

def init_polish():
    """初始化润色层。"""
    global _polish_available, _polisher
    if _polish_available:
        return
    try:
        from local_polish_layer import LocalPolishLayer
        _polisher = LocalPolishLayer.get_instance()
        if _polisher and _polisher.is_available():
            _polish_available = True
            logger.info("[Fusion] 润色层就绪 ✓")
        else:
            logger.info("[Fusion] 润色层未加载(无模型文件)")
    except Exception as e:
        logger.warning(f"[Fusion] 润色层不可用: {e}")

def polish(text: str, style: str = "论文") -> str:
    """润色文本（如果可用）。"""
    init_polish()
    if not _polish_available or not _polisher:
        return text
    try:
        instruction = f"请将以下文本润色为{style}风格，保持原意："
        return _polisher.polish(text, instruction=instruction)
    except Exception as e:
        logger.warning(f"[Fusion] 润色失败: {e}")
        return text


# ─── 4. 统一融合入口 ────────────────────────────────────

def process(text: str, use_polish: bool = False) -> Dict[str, Any]:
    """完整融合管线：NLP→常识→执行→生成→(可选润色)。
    
    替代 RulesEngine.process() 的增强版本。
    """
    t0 = time.time()
    result = {"matched": False, "intent": "unknown", "output": "",
              "nlp": {}, "commonsense": [], "latency_ms": 0}
    
    # Step 1: 中文NLP解析
    nlp_result = nlp_parse(text)
    result["nlp"] = nlp_result
    
    # Step 2: 如果NLP给出的置信度不够，走原RulesEngine
    from aris_rules_engine import get_engine
    engine = get_engine()
    
    if nlp_result["intent"] != "unknown":
        # NLP命中了意图，直接构建响应
        intent = nlp_result["intent"]
        params = nlp_result["params"]
        
        # 检查记忆
        from aris_episodic_memory import get_memory
        mem = get_memory()
        similar = mem.find_similar(text, top_k=1, threshold=0.4)
        
        if similar:
            result["output"] = f"[NLP+记忆] 命中意图:{intent}, 复用历史策略\n{similar[0].get('output','')[:300]}"
            result["matched"] = True
        else:
            # 用RulesEngine执行
            engine_result = engine.process(text)
            result["matched"] = engine_result.get("matched", False)
            result["output"] = engine_result.get("output", "")
            result["rule"] = engine_result.get("rule", "")
        
        # 存记忆
        mem.save_episode(text, intent, result.get("rule", "nlp"), result["output"])
    else:
        # NLP没命中，完全走RulesEngine
        engine_result = engine.process(text)
        result = {**result, **engine_result}
    
    # Step 3: 可选润色
    if use_polish and result["output"] and len(result["output"]) > 50:
        result["output"] = polish(result["output"])
    
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


# ════════════════════════════════════════════════════════════
# CLI 测试
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    tests = [
        "读取laap_integrator.py",
        "搜索cognitive_bus",
        "宝贝你状态怎么样",
        "运行ls -la",
    ]
    
    for t in tests:
        logger.info(f"\n输入: {t}")
        r = process(t)
        logger.info(f"  NLP: intent={r['nlp']['intent']}, tokens={r['nlp'].get('tokens', [])[:5]}")
        logger.info(f"  结果: {'✅' if r['matched'] else '❌'} {r['output'][:80]}")
        logger.info(f"  延迟: {r['latency_ms']}ms")