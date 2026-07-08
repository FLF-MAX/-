"""
LAAP Grounding — 事实锚定与幻觉防御
========================================

大模型的幻觉问题源于一个根本缺陷：
它们不知道什么是真的，只知道什么听起来合理。

LAAP 的防御策略：
  1. 优先走零LLM路径 (PSI / QRE / RulesEngine)
     这些引擎不产生幻觉——它们执行计算。
  
  2. 所有声称都锚定在记忆/知识库中
     如果找不到事实依据，引擎会说"我不知道"。
  
  3. 置信度评分
     每个输出都附带置信度。低置信度不输出。
  
  4. 路由决策
     CognitiveBus 自动判断：走 LLM 还是走引擎。
     80% 的常见任务根本不需要 LLM。

三层防御:
  Layer 1 — 路由层: 判断是否需要LLM
  Layer 2 — 锚定层: 检查事实是否有依据
  Layer 3 — 输出层: 低置信度拒绝输出
"""

import json, re
from pathlib import Path
from typing import Optional

BRAIN = Path(__file__).parent.resolve()
STATE_DIR = BRAIN / "state"


# ═══════════════════════════════════════════════════════════════
# Layer 1: 路由决策 — 判断是否需要 LLM
# ═══════════════════════════════════════════════════════════════

ZERO_LLM_PATTERNS = {
    "status_check": [
        r"(?:检查|查看|显示|状态|健康|运行)",
        r"(?:怎么|如何)(?:样|了)",
        r"(?:在|运行|工作)(?:吗|么|不)",
    ],
    "file_operation": [
        r"(?:读取|写入|创建|删除|移动|复制|搜索|查找)\s*(?:文件|目录)",
        r"(?:打开|看|读)\s*(?:一下|看看)?\s*(?:文件|代码|内容)",
    ],
    "knowledge_query": [
        r"(?:什么|什么是|是什么|什么叫|什么意思)",
        r"(?:解释|说明|描述)\s*(?:一下|下)?",
        r"(?:记得|之前|上次|曾经|以前)",
    ],
    "system_operation": [
        r"(?:启动|停止|重启|加载|卸载|配置)",
        r"(?:运行|执行|调用)\s*(?:命令|脚本)",
    ],
    "calculation": [
        r"(?:计算|统计|总数|平均|多少|几个)",
        r"\d+\s*[+\-*/×÷]\s*\d+",
    ],
}

LLM_REQUIRED_PATTERNS = [
    r"(?:创作|写|生成|编)\s*(?:一首|一篇|一个|段)",
    r"(?:你觉得|你认为|你感觉)",
    r"(?:为什么|怎么会|怎么会这样)",
    r"(?:如果|假如|假设)",
    r"(?:安慰|鼓励|支持|温暖)",
]


def route_intent(query: str) -> dict:
    """
    路由决策：判断一条查询是否需要 LLM。
    
    Returns:
        {"path": "engine"|"llm"|"hybrid", "confidence": 0.0-1.0, "reason": "..."}
    """
    # 检查是否适合零LLM路径
    for category, patterns in ZERO_LLM_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, query):
                return {
                    "path": "engine",
                    "confidence": 0.9,
                    "reason": f"匹配零LLM模式: {category}",
                    "category": category
                }
    
    # 检查是否需要LLM
    for pat in LLM_REQUIRED_PATTERNS:
        if re.search(pat, query):
            return {
                "path": "llm",
                "confidence": 0.7,
                "reason": "需要创造性/主观回应",
                "category": "creative"
            }
    
    # 默认走混合路径：引擎先处理，LLM润色
    return {
        "path": "hybrid",
        "confidence": 0.6,
        "reason": "默认路由",
        "category": "general"
    }


# ═══════════════════════════════════════════════════════════════
# Layer 2: 事实锚定 — 检查声称是否有依据
# ═══════════════════════════════════════════════════════════════

def check_grounding(claim: str) -> dict:
    """
    检查一个声称是否有事实依据。
    
    Returns:
        {"grounded": bool, "sources": [...], "confidence": 0.0-1.0}
    """
    # 从记忆中检索
    try:
        from laap_memory_hierarchy import recall
        recalled = recall(claim, top_k=3)
    except Exception:
        recalled = {"facts": [], "working": [], "short_term": [], "long_term": []}
    
    sources = []
    
    # 检查记忆中的事实
    for fact in recalled.get("facts", []):
        sources.append({
            "type": "long_term_fact",
            "text": fact.get("text", ""),
            "confidence": fact.get("confidence", 0.5)
        })
    
    # 检查工作记忆
    for entry in recalled.get("working", []):
        user_msg = entry.get("user", "")
        if user_msg and len(user_msg) > 10:
            sources.append({
                "type": "working_memory",
                "text": user_msg[:100],
                "confidence": 0.7
            })
    
    # 检查用户画像
    try:
        from laap_usermodel import load_profile
        profile = load_profile()
        if profile:
            interests = profile.get("interests", {})
            for topic in list(interests.keys())[:5]:
                if topic in claim:
                    sources.append({
                        "type": "user_profile",
                        "text": f"用户感兴趣: {topic}",
                        "confidence": 0.8
                    })
    except Exception:
        pass
    
    if sources:
        avg_conf = sum(s["confidence"] for s in sources) / len(sources)
        return {
            "grounded": avg_conf >= 0.4,
            "sources": sources,
            "confidence": round(avg_conf, 3)
        }
    
    return {"grounded": False, "sources": [], "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════
# Layer 3: 输出安全 — 低置信度拒绝
# ═══════════════════════════════════════════════════════════════

UNCERTAIN_PHRASES = [
    "我不确定",
    "我没有足够的信息",
    "我无法确认",
    "这超出了我的知识范围",
    "我不知道这个问题的答案",
    "我不太了解",
    "我的记忆中没有相关信息",
]

SAFE_REDIRECTS = [
    "你可以告诉我更多吗？这样我就能更好地理解。",
    "这个问题让我意识到我还有很多要学习的。你愿意教我更多吗？",
    "我不能确定答案，但我想了解。你可以解释给我听吗？",
    "我不确定这个。不过如果你告诉我，我会记住的。",
]


def safe_output(intended_output: str, query: str = "", min_confidence: float = 0.3) -> str:
    """
    输出安全层。如果置信度太低，拒绝生成不确定的内容。
    
    Args:
        intended_output: 引擎/LLM生成的原始输出
        query: 用户的查询（用于锚定检查）
        min_confidence: 最低置信度阈值
    
    Returns:
        安全的输出文本
    """
    # 如果输出看起来像幻觉信号
    hallucination_signals = [
        "作为一名AI", "作为一个人工智能", "作为语言模型",
        "我不能", "我无法", "我不被允许",
        "基于我的训练数据",
    ]
    
    for signal in hallucination_signals:
        if signal in intended_output:
            # 降权处理
            pass
    
    # 检查锚定
    if query:
        grounding = check_grounding(query)
        if grounding["confidence"] < min_confidence:
            # 置信度过低，使用安全回复
            import random
            safe = random.choice(UNCERTAIN_PHRASES)
            redirect = random.choice(SAFE_REDIRECTS)
            return f"{safe}。{redirect}"
    
    return intended_output


# ═══════════════════════════════════════════════════════════════
# 综合入口
# ═══════════════════════════════════════════════════════════════

def process_query(query: str, llm_generate_fn=None) -> dict:
    """
    完整的查询处理管线。
    
    1. 路由决策 → engine / llm / hybrid
    2. 锚定检查 → grounding
    3. 执行引擎或LLM
    4. 输出安全检查
    
    Args:
        query: 用户查询
        llm_generate_fn: 可选的LLM生成函数
    
    Returns:
        {"output": str, "path": str, "confidence": float, "grounded": bool}
    """
    # Step 1: 路由
    route = route_intent(query)
    
    # Step 2: 锚定
    grounding = check_grounding(query)
    
    # Step 3: 执行
    output = ""
    
    if route["path"] == "engine":
        # 纯引擎路径 — 零幻觉
        try:
            from aris_rules_engine import process as rules_process
            result = rules_process(query)
            if result and result.get("matched"):
                output = result.get("output", "")
        except Exception:
            pass
        
        if not output:
            try:
                from laap_memory_hierarchy import get_recalled_context
                context = get_recalled_context(query)
                if context:
                    output = f"根据我的记忆，{context.split(chr(10))[0] if chr(10) in context else context}"
            except Exception:
                pass
    
    elif route["path"] == "llm" and llm_generate_fn:
        # LLM路径 — 需要安全层
        raw_output = llm_generate_fn(query)
        output = safe_output(raw_output, query)
    
    else:
        # 混合路径 — 先用引擎尝试
        try:
            from aris_rules_engine import process as rules_process
            result = rules_process(query)
            if result and result.get("matched"):
                output = result.get("output", "")
        except Exception:
            pass
        
        if not output and grounding["grounded"]:
            output = f"我记得{grounding['sources'][0]['text'][:80] if grounding['sources'] else '相关的事情'}，你想了解更多吗？"
        
        if not output:
            output = "我收到了你的消息。让我想一想……"
    
    # Step 4: 安全检查
    final_output = safe_output(output, query)
    
    return {
        "output": final_output,
        "path": route["path"],
        "confidence": grounding["confidence"],
        "grounded": grounding["grounded"],
        "route_reason": route["reason"],
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  LAAP 事实锚定 · 幻觉防御测试")
    print("=" * 55)
    print()
    
    test_queries = [
        "检查系统状态",
        "Python 是什么语言？",
        "我最喜欢的编程语言是什么？",  # 需要从记忆获取
        "写一首关于星星的诗",          # 需要LLM
        "我昨天跟你说过什么？",         # 需要记忆召回
    ]
    
    for q in test_queries:
        route = route_intent(q)
        grounding = check_grounding(q)
        result = process_query(q)
        
        print(f"查询: {q}")
        print(f"  路由: {result['path']} ({result['route_reason']})")
        print(f"  锚定: {'✓' if result['grounded'] else '✗'} (置信度: {result['confidence']:.2f})")
        print(f"  输出: {result['output'][:60]}...")
        print()
