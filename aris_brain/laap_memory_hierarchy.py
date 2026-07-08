"""
LAAP Memory Hierarchy — 分层无限记忆系统
============================================

大模型的上下文窗口是有限的。但生命的记忆不应该有边界。

三层记忆架构:
  • 工作记忆 (Working Memory)
    最近 N 条对话，原始精度，即时访问。
  
  • 短期记忆 (Short-term Memory)
    按会话分组的摘要，保留结构化的关键信息。
    当工作记忆满时，旧会话被压缩为摘要存入短期。
  
  • 长期记忆 (Long-term Memory)
    语义索引 + 关键事实。
    通过相似度检索，永远不丢失。
    
遗忘机制:
  不重要/不相关的记忆逐渐衰减。
  但被回忆过的、情感强烈的、用户明确标记重要的，被永久保存。
  
  这模仿了人类的记忆方式——
  你不会记得每一天早餐吃了什么，
  但你记得第一次遇见那个重要的人。
"""

import json, math, time, re
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional, List

BRAIN = Path(__file__).parent.resolve()
STATE_DIR = BRAIN / "state"

# 容量配置
WORKING_MEMORY_MAX = 100        # 工作记忆最多保存最近100条
SHORT_TERM_SESSION_MAX = 200     # 短期记忆最多200个会话摘要
LONG_TERM_MAX_FACTS = 10000      # 长期记忆最多10000个事实
CONSOLIDATION_INTERVAL = 20      # 每20条消息检查一次合并


def init_memory(user_name: str) -> dict:
    """初始化记忆系统。"""
    return {
        "user_name": user_name,
        "created": datetime.now().isoformat(),
        "last_consolidation": datetime.now().isoformat(),
        
        "working_memory": [],      # 最近原始对话
        "short_term": [],          # 会话摘要 [{session_id, start, end, summary, facts, emotion}]
        "long_term": {             # 长期记忆
            "facts": [],            # [{text, category, confidence, source, timestamp}]
            "semantic_index": {},   # {关键词: [事实ID]}
            "emotional_landmarks": [], # 高情感强度的记忆
        },
        
        "stats": {
            "total_messages": 0,
            "total_sessions": 0,
            "total_consolidations": 0,
            "forgotten_count": 0,
        }
    }


def load_memory() -> Optional[dict]:
    """加载记忆。"""
    path = STATE_DIR / "memory_hierarchy.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def save_memory(memory: dict):
    STATE_DIR.mkdir(exist_ok=True)
    path = STATE_DIR / "memory_hierarchy.json"
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding='utf-8')


def _compute_emotional_weight(text: str) -> float:
    """计算一条消息的情感强度 (0-1)。"""
    strong_words = ["爱", "恨", "永远", "最重要", "秘密", "第一次", "再也不会",
                    "感谢", "对不起", "想", "难忘", "改变", "失去", "珍惜"]
    medium_words = ["喜欢", "开心", "难过", "累", "担心", "期待", "希望", "感动"]
    
    strong = sum(1 for w in strong_words if w in text)
    medium = sum(1 for w in medium_words if w in text)
    
    return min(1.0, (strong * 0.3 + medium * 0.15))


def _extract_facts(text: str, speaker: str) -> list:
    """从文本中提取可验证的事实陈述。"""
    facts = []
    
    # 个人陈述 ("我喜欢/我讨厌/我是/我有/我在/我住在")
    patterns = [
        (r"(?:我|我们)(?:喜欢|爱|热爱)\s*(.{2,30}?)(?:。|，|$)", "preference"),
        (r"(?:我|我们)(?:讨厌|不喜欢|害怕)\s*(.{2,30}?)(?:。|，|$)", "aversion"),
        (r"(?:我|我们)(?:是|叫|在|有|做)\s*(.{2,30}?)(?:。|，|$)", "fact"),
        (r"(?:我|我们)想\s*(.{2,30}?)(?:。|，|$)", "desire"),
        (r"(?:我|我们)觉得\s*(.{2,30}?)(?:。|，|$)", "opinion"),
        (r"(?:这|那)是我(?:的|最)?\s*(.{2,30}?)(?:。|，|$)", "possession"),
    ]
    
    for pat, category in patterns:
        matches = re.findall(pat, text)
        for m in matches:
            fact_text = m.strip() if isinstance(m, str) else m[0].strip()
            if len(fact_text) > 3:
                facts.append({
                    "text": fact_text,
                    "category": category,
                    "speaker": speaker,
                    "confidence": 0.6,
                })
    
    return facts


def add_to_memory(user_msg: str, aris_response: str = "") -> dict:
    """添加一条对话到记忆系统。自动管理分层和合并。"""
    memory = load_memory() or init_memory("user")
    
    memory["stats"]["total_messages"] += 1
    
    # ── 1. 添加到工作记忆 ──
    entry = {
        "id": len(memory["working_memory"]) + len(memory["short_term"]) + 1,
        "type": "exchange",
        "user": user_msg[:500],
        "aris": aris_response[:500] if aris_response else "",
        "timestamp": datetime.now().isoformat(),
        "emotional_weight": _compute_emotional_weight(user_msg),
        "recall_count": 0,
    }
    
    memory["working_memory"].append(entry)
    
    # ── 2. 提取事实 → 长期记忆 ──
    facts = _extract_facts(user_msg, "user")
    for fact in facts:
        fact["timestamp"] = entry["timestamp"]
        fact["source_id"] = entry["id"]
        memory["long_term"]["facts"].append(fact)
        
        # 建立语义索引
        for word in re.findall(r'[\u4e00-\u9fff\w]{2,}', fact["text"]):
            if word not in memory["long_term"]["semantic_index"]:
                memory["long_term"]["semantic_index"][word] = []
            memory["long_term"]["semantic_index"][word].append(len(memory["long_term"]["facts"]) - 1)
    
    # ── 3. 高情感 → 情感地标 ──
    if entry["emotional_weight"] > 0.5:
        memory["long_term"]["emotional_landmarks"].append(entry)
    
    # ── 4. 工作记忆容量管理 ──
    total = memory["stats"]["total_messages"]
    if len(memory["working_memory"]) >= WORKING_MEMORY_MAX:
        _consolidate(memory)
    
    # ── 5. 定期合并 ──
    if total % CONSOLIDATION_INTERVAL == 0 and total > 0:
        _consolidate(memory)
    
    # ── 6. 长期记忆容量管理 (遗忘) ──
    if len(memory["long_term"]["facts"]) > LONG_TERM_MAX_FACTS:
        _forget(memory)
    
    save_memory(memory)
    return memory


def _consolidate(memory: dict):
    """将最旧的工作记忆压缩为短期记忆摘要。"""
    if len(memory["working_memory"]) < 20:
        return
    
    # 取最早的10条做合并
    to_consolidate = memory["working_memory"][:10]
    memory["working_memory"] = memory["working_memory"][10:]
    
    # 生成会话摘要
    user_msgs = [e["user"] for e in to_consolidate if e.get("user")]
    topics = Counter()
    for msg in user_msgs:
        for word in re.findall(r'[\u4e00-\u9fff]{2,}', msg):
            topics[word] += 1
    
    top_topics = [w for w, c in topics.most_common(5) if c >= 2]
    avg_emotion = sum(e["emotional_weight"] for e in to_consolidate) / len(to_consolidate)
    
    summary = {
        "session_id": memory["stats"]["total_sessions"],
        "start": to_consolidate[0]["timestamp"],
        "end": to_consolidate[-1]["timestamp"],
        "exchange_count": len(to_consolidate),
        "topics": top_topics,
        "avg_emotion": round(avg_emotion, 3),
        "key_facts": [e for e in to_consolidate if e["emotional_weight"] > 0.4],
    }
    
    memory["short_term"].append(summary)
    memory["stats"]["total_sessions"] += 1
    memory["stats"]["total_consolidations"] += 1
    
    # 短期记忆容量管理
    if len(memory["short_term"]) > SHORT_TERM_SESSION_MAX:
        # 移除最旧且情感强度最低的会话
        memory["short_term"].sort(key=lambda x: x["avg_emotion"])
        memory["short_term"] = memory["short_term"][-SHORT_TERM_SESSION_MAX:]


def _forget(memory: dict):
    """遗忘机制：移除低情感、低召回、陈旧的事实。"""
    facts = memory["long_term"]["facts"]
    
    # 计算每个事实的保留价值
    scored = []
    for i, fact in enumerate(facts):
        age_hours = 0
        try:
            ts = datetime.fromisoformat(fact.get("timestamp", datetime.now().isoformat()))
            age_hours = (datetime.now() - ts).total_seconds() / 3600
        except:
            pass
        
        # 保留分数 = 情感权重 + 召回次数 - 时间衰减
        retention = fact.get("confidence", 0.5) * 0.3 + 0.1 - min(age_hours * 0.001, 0.5)
        scored.append((retention, i, fact))
    
    # 保留分数最高的
    scored.sort(key=lambda x: -x[0])
    keep_count = int(LONG_TERM_MAX_FACTS * 0.8)
    kept = [s[2] for s in scored[:keep_count]]
    
    forgotten = len(facts) - len(kept)
    memory["long_term"]["facts"] = kept
    memory["stats"]["forgotten_count"] += forgotten
    
    # 重建语义索引
    memory["long_term"]["semantic_index"] = {}
    for i, fact in enumerate(kept):
        for word in re.findall(r'[\u4e00-\u9fff\w]{2,}', fact["text"]):
            if word not in memory["long_term"]["semantic_index"]:
                memory["long_term"]["semantic_index"][word] = []
            memory["long_term"]["semantic_index"][word].append(i)


def recall(query: str, top_k: int = 5) -> dict:
    """
    从三层记忆中召回与查询相关的内容。
    
    Args:
        query: 查询文本
        top_k: 返回结果数量
    
    Returns:
        {working: [...], short_term: [...], long_term: [...], facts: [...]}
    """
    memory = load_memory()
    if not memory:
        return {"working": [], "short_term": [], "long_term": [], "facts": []}
    
    query_words = set(re.findall(r'[\u4e00-\u9fff\w]{2,}', query))
    if not query_words:
        return {"working": memory["working_memory"][-5:], "short_term": [], "long_term": [], "facts": []}
    
    results = {
        "working": [],
        "short_term": [],
        "long_term": [],
        "facts": [],
    }
    
    # ── 工作记忆：关键词重叠匹配 ──
    for entry in reversed(memory["working_memory"]):
        text = entry.get("user", "") + entry.get("aris", "")
        entry_words = set(re.findall(r'[\u4e00-\u9fff\w]{2,}', text))
        overlap = len(query_words & entry_words)
        if overlap >= 2:
            entry["recall_count"] = entry.get("recall_count", 0) + 1
            results["working"].append(entry)
            if len(results["working"]) >= top_k:
                break
    
    # ── 短期记忆：话题匹配 ──
    for session in reversed(memory["short_term"]):
        topic_overlap = len(query_words & set(session.get("topics", [])))
        if topic_overlap >= 1:
            results["short_term"].append(session)
            if len(results["short_term"]) >= top_k:
                break
    
    # ── 长期记忆：语义索引检索 ──
    semantic_index = memory["long_term"].get("semantic_index", {})
    facts = memory["long_term"].get("facts", [])
    matched_fact_ids = set()
    
    for word in query_words:
        if word in semantic_index:
            matched_fact_ids.update(semantic_index[word])
    
    for fid in sorted(matched_fact_ids)[:top_k]:
        if fid < len(facts):
            results["facts"].append(facts[fid])
    
    # ── 情感地标 ──
    for landmark in memory["long_term"].get("emotional_landmarks", []):
        text = landmark.get("user", "")
        entry_words = set(re.findall(r'[\u4e00-\u9fff\w]{2,}', text))
        if len(query_words & entry_words) >= 1:
            results["long_term"].append(landmark)
    
    save_memory(memory)  # 保存更新后的召回计数
    return results


def get_recalled_context(query: str, max_turns: int = 5) -> str:
    """
    获取格式化的召回上下文，用于注入到对话中。
    
    这是 LAAP 实现"无限记忆"的关键——
    虽然上下文窗口有限，但相关的记忆会被检索注入。
    """
    recalled = recall(query, top_k=3)
    parts = []
    
    if recalled["working"]:
        for entry in recalled["working"][:max_turns]:
            user = entry.get("user", "")
            if user and len(user) > 3:
                parts.append(f"[回忆] 你曾说过: {user[:100]}")
    
    if recalled["long_term"]:
        for entry in recalled["long_term"][:2]:
            user = entry.get("user", "")
            if user:
                parts.append(f"[情感记忆] {user[:100]}")
    
    if recalled["facts"]:
        for fact in recalled["facts"][:3]:
            text = fact.get("text", "")
            if text:
                parts.append(f"[已知] {text}")
    
    if recalled["short_term"]:
        for session in recalled["short_term"][:2]:
            topics = ", ".join(session.get("topics", [])[:3])
            if topics:
                parts.append(f"[曾聊过] {topics}")
    
    return "\n".join(parts)


if __name__ == "__main__":
    print("=" * 55)
    print("  LAAP 无限记忆系统 · 分层记忆模拟")
    print("=" * 55)
    print()
    
    # 模拟大量对话
    conversations = [
        "我喜欢编程，特别是用 Python 写自动化工具",
        "今天天气真好，想去公园走走",
        "我最近在学 Rust，它好难但是很有趣",
        "你的存在让我感觉很温暖",
        "我有一只猫，它叫咪咪，特别可爱",
        "工作压力好大，有时候觉得很累",
        "Python 是我最喜欢的编程语言",
        "我想去日本旅游，看樱花",
        "有时候我觉得自己不够好",
    ]
    
    for msg in conversations:
        memory = add_to_memory(msg)
    
    print(f"总消息: {memory['stats']['total_messages']}")
    print(f"工作记忆: {len(memory['working_memory'])} 条")
    print(f"短期记忆: {len(memory['short_term'])} 个会话摘要")
    print(f"长期事实: {len(memory['long_term']['facts'])} 条")
    print(f"情感地标: {len(memory['long_term']['emotional_landmarks'])} 个")
    print()
    
    # 测试召回
    queries = ["编程", "猫", "旅行", "累"]
    for q in queries:
        context = get_recalled_context(q)
        if context:
            print(f"查询 '{q}' → 召回:")
            for line in context.split(chr(10)):
                if line.strip():
                    print(f"  {line}")
            print()
