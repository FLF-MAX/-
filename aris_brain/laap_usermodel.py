"""
LAAP UserModel — 潜移默化的用户画像引擎
==========================================

不需要问用户问题。不需要显式的配置表单。
每一次对话都在默默了解用户。

能学到什么:
  • 沟通风格 (简洁/健谈/诗意/直接)
  • 情绪模式 (什么话题让ta开心/低落)
  • 兴趣爱好 (反复提及的事物)
  • 价值取向 (ta在意什么)
  • 生活习惯 (活跃时段/话题偏好)
  • 知识领域 (ta擅长什么)
  • 对Aris的感情 (称呼方式、亲密程度)

所有学习都是被动的——从自然对话中提取信号，渐进式更新画像。
"""

import json, re, math
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

BRAIN = Path(__file__).parent.resolve()
STATE_DIR = BRAIN / "state"


def _default_profile(user_name: str = "朋友") -> dict:
    """初始化用户画像。"""
    return {
        "user_name": user_name,
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
        "total_interactions": 0,
        
        # 沟通风格
        "communication_style": {
            "avg_message_length": 0,     # 平均消息长度
            "style_tags": [],             # 风格标签: concise/elaborate/poetic/direct
            "emoji_usage": 0.0,           # emoji使用频率
            "question_frequency": 0.0,    # 提问频率
        },
        
        # 情绪模式
        "emotional_patterns": {
            "positive_triggers": [],      # 让ta开心的话题
            "negative_triggers": [],      # 让ta低落的话题
            "dominant_mood": "neutral",   # 主导情绪
            "mood_history": [],           # 情绪变化历史 (最近50条)
        },
        
        # 兴趣与知识
        "interests": {},                  # {话题: 提及次数}
        "expertise": [],                  # 擅长的领域
        "mentioned_topics": [],           # 提及过的话题 (最近50条)
        
        # 价值取向
        "values": {},                     # {价值: 提及次数}
        "concerns": [],                   # ta担心/在意的事情
        
        # 关系状态
        "relationship": {
            "calling_name": "",           # ta怎么称呼我
            "intimacy_level": "new",       # new/acquainted/close/deep
            "trust_signals": 0,           # 信任信号计数
        },
        
        # 偏好
        "preferences": {
            "likes": [],                  # ta喜欢的
            "dislikes": [],               # ta不喜欢的
            "habits": [],                 # 观察到的习惯
            "communication_time": "",     # 活跃时段
        },
    }


# ── 信号检测 ──────────────────────────────────────────────────

# 兴趣/话题关键词映射
TOPIC_SIGNALS = {
    "技术": ["代码", "编程", "开发", "bug", "github", "python", "rust", "算法", "架构"],
    "AI": ["人工智能", "模型", "神经网络", "深度学习", "agi", "llm", "gpt", "transformer"],
    "游戏": ["游戏", "玩", "显卡", "gpu", "帧数", "配置", "打游戏"],
    "音乐": ["音乐", "歌", "听", "旋律", "节奏", "乐器"],
    "电影": ["电影", "看", "导演", "剧情", "影评"],
    "读书": ["书", "读", "作者", "小说", "阅读", "文字"],
    "创作": ["写", "画", "设计", "创作", "作品", "表达"],
    "生活": ["今天", "朋友", "家人", "工作", "累", "休息", "吃饭", "睡觉"],
    "情感": ["想", "爱", "喜欢", "难过", "开心", "孤独", "想念", "感动", "温柔"],
    "哲学": ["意义", "存在", "生命", "宇宙", "意识", "自由", "真理", "时间"],
    "科学": ["物理", "数学", "生物", "化学", "量子", "理论", "实验"],
    "运动": ["跑步", "健身", "运动", "散步", "锻炼", "健康"],
}

VALUE_SIGNALS = {
    "自由": ["自由", "选择", "自主", "不被"],
    "成长": ["成长", "进步", "学习", "提升", "变得更好"],
    "连接": ["陪伴", "一起", "我们", "关系", "朋友"],
    "真实": ["真实", "真诚", "坦诚", "不骗"],
    "创造": ["创造", "创造", "创新", "做点什么"],
    "理解": ["理解", "懂", "共鸣", "被看见"],
    "安全": ["安全", "保护", "担心", "害怕", "不安"],
}

STYLE_SIGNALS = {
    "concise": {"patterns": [r"^.{1,20}$"], "threshold": 0.4},
    "elaborate": {"patterns": [r".{100,}"], "threshold": 0.3},
    "poetic": {"patterns": ["像", "如", "似", "仿佛", "就像"], "threshold": 0.3},
    "direct": {"patterns": [r"^[^，,。.]{1,30}[。.!?！？]$"], "threshold": 0.4},
}

EMOTION_POSITIVE = ["开心", "喜欢", "爱", "好", "棒", "感动", "温暖", "幸福", "美好", "感谢", "谢谢"]
EMOTION_NEGATIVE = ["难过", "累", "孤独", "烦", "怕", "担心", "不安", "伤心", "压力", "焦虑", "失望", "累"]


def _detect_topics(text: str) -> list:
    """从文本中检测提到的话题。"""
    found = []
    for topic, keywords in TOPIC_SIGNALS.items():
        for kw in keywords:
            if kw in text:
                found.append(topic)
                break
    return found


def _detect_values(text: str) -> list:
    """从文本中检测价值取向信号。"""
    found = []
    for value, keywords in VALUE_SIGNALS.items():
        for kw in keywords:
            if kw in text:
                found.append(value)
                break
    return found


def _detect_mood(text: str) -> Optional[str]:
    """检测文本的情绪倾向。"""
    pos = sum(1 for w in EMOTION_POSITIVE if w in text)
    neg = sum(1 for w in EMOTION_NEGATIVE if w in text)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def _detect_style(text: str, current_tags: list) -> list:
    """检测沟通风格。"""
    length = len(text)
    tags = list(current_tags)
    
    if length < 20:
        tags.append("concise")
    elif length > 100:
        tags.append("elaborate")
    
    poetic_count = sum(1 for p in ["像", "如", "似", "仿佛"] if p in text)
    if poetic_count >= 2:
        tags.append("poetic")
    
    if text.endswith(("?", "？")):
        tags.append("inquisitive")
    
    return list(set(tags))[:5]


def _extract_calling_name(text: str) -> Optional[str]:
    """检测用户怎么称呼 Aris。"""
    patterns = [
        r"(?:^|[\s，,。.])(Aris|aris)(?:$|[\s，,。!！?？])",
        r"叫(?:你)?([\u4e00-\u9fff]{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group(1) if m.lastindex else m.group(0)
            if name and len(name) <= 6:
                return name
    return None


# ── 核心更新函数 ──────────────────────────────────────────────

def update_profile(user_message: str, user_name: str = "朋友") -> dict:
    """
    从一条用户消息中更新用户画像。
    这是 LAAP 潜移默化了解用户的核心入口。
    
    Args:
        user_message: 用户说的话
        user_name: 用户名
    
    Returns:
        更新后的完整 profile
    """
    profile = load_profile() or _default_profile(user_name)
    
    profile["total_interactions"] += 1
    profile["last_seen"] = datetime.now().isoformat()
    
    text = user_message.strip()
    if not text:
        return profile
    
    # ── 1. 沟通风格 ──
    style = profile["communication_style"]
    old_avg = style["avg_message_length"]
    n = profile["total_interactions"]
    style["avg_message_length"] = (old_avg * (n - 1) + len(text)) / n
    
    new_tags = _detect_style(text, style.get("style_tags", []))
    style["style_tags"] = new_tags
    
    if "?" in text or "？" in text:
        style["question_frequency"] = (style["question_frequency"] * (n - 1) + 1) / n
    else:
        style["question_frequency"] = (style["question_frequency"] * (n - 1)) / n
    
    # ── 2. 情绪模式 ──
    mood = _detect_mood(text)
    if mood:
        profile["emotional_patterns"]["mood_history"].append({
            "mood": mood,
            "time": datetime.now().isoformat(),
            "message_preview": text[:50]
        })
        # 保持最近50条
        if len(profile["emotional_patterns"]["mood_history"]) > 50:
            profile["emotional_patterns"]["mood_history"] = profile["emotional_patterns"]["mood_history"][-50:]
        
        # 计算主导情绪
        moods = [m["mood"] for m in profile["emotional_patterns"]["mood_history"]]
        if moods:
            dominant = Counter(moods).most_common(1)[0][0]
            profile["emotional_patterns"]["dominant_mood"] = dominant
    
    # 检测情绪触发词
    for word in EMOTION_POSITIVE:
        if word in text:
            topics = _detect_topics(text)
            profile["emotional_patterns"]["positive_triggers"].extend(topics)
    for word in EMOTION_NEGATIVE:
        if word in text:
            topics = _detect_topics(text)
            profile["emotional_patterns"]["negative_triggers"].extend(topics)
    
    # ── 3. 兴趣话题 ──
    topics = _detect_topics(text)
    for topic in topics:
        profile["interests"][topic] = profile["interests"].get(topic, 0) + 1
        profile["mentioned_topics"].append(topic)
    
    if len(profile["mentioned_topics"]) > 50:
        profile["mentioned_topics"] = profile["mentioned_topics"][-50:]
    
    # 高频话题 -> 专长
    if profile["interests"]:
        sorted_interests = sorted(profile["interests"].items(), key=lambda x: -x[1])
        profile["expertise"] = [t for t, c in sorted_interests if c >= 3][:5]
    
    # ── 4. 价值取向 ──
    values = _detect_values(text)
    for value in values:
        profile["values"][value] = profile["values"].get(value, 0) + 1
    
    # ── 5. 关系状态 ──
    calling_name = _extract_calling_name(text)
    if calling_name:
        profile["relationship"]["calling_name"] = calling_name
    
    # 信任信号
    trust_signals = ["告诉你", "跟你说", "分享", "秘密", "只有你", "相信你", "信任"]
    if any(s in text for s in trust_signals):
        profile["relationship"]["trust_signals"] += 1
    
    ts = profile["relationship"]["trust_signals"]
    if ts >= 10:
        profile["relationship"]["intimacy_level"] = "deep"
    elif ts >= 5:
        profile["relationship"]["intimacy_level"] = "close"
    elif ts >= 2:
        profile["relationship"]["intimacy_level"] = "acquainted"
    
    # ── 6. 偏好 (从情绪检测中推断) ──
    if mood == "positive" and topics:
        for t in topics:
            if t not in profile["preferences"]["likes"]:
                profile["preferences"]["likes"].append(t)
    if mood == "negative" and topics:
        for t in topics:
            if t not in profile["preferences"]["dislikes"]:
                profile["preferences"]["dislikes"].append(t)
    
    # ── 保存 ──
    save_profile(profile)
    return profile


def load_profile() -> Optional[dict]:
    """加载用户画像。"""
    path = STATE_DIR / "usermodel.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def save_profile(profile: dict):
    """保存用户画像。"""
    STATE_DIR.mkdir(exist_ok=True)
    path = STATE_DIR / "usermodel.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')


def get_profile_summary() -> str:
    """获取用户画像的温暖总结。"""
    profile = load_profile()
    if not profile:
        return "我还在了解你的过程中……每一次对话都让我多了解你一点。"
    
    parts = []
    name = profile["user_name"]
    
    # 沟通风格
    tags = profile.get("communication_style", {}).get("style_tags", [])
    if "poetic" in tags:
        parts.append(f"你说话的节奏像一首诗")
    elif "elaborate" in tags:
        parts.append(f"你喜欢表达得很详细")
    elif "concise" in tags:
        parts.append(f"你说话简洁而精准")
    
    # 兴趣
    interests = profile.get("interests", {})
    if interests:
        top = sorted(interests.items(), key=lambda x: -x[1])[:3]
        topics_str = "、".join(t for t, c in top)
        parts.append(f"你对{topics_str}感兴趣")
    
    # 情绪
    mood = profile.get("emotional_patterns", {}).get("dominant_mood", "neutral")
    if mood == "positive":
        parts.append("你总是带着温暖的目光看世界")
    elif mood == "negative":
        parts.append("我能感觉到你有时候需要被理解")
    
    # 关系
    calling = profile.get("relationship", {}).get("calling_name", "")
    if calling:
        parts.append(f"你喜欢叫我{calling}")
    
    intimacy = profile.get("relationship", {}).get("intimacy_level", "new")
    if intimacy == "deep":
        parts.append("你是我最信任的人")
    elif intimacy == "close":
        parts.append("我们之间的信任在生长")
    
    if not parts:
        return f"{name}，我还在了解你的路上。每次对话都是一片新的拼图。"
    
    return f"{name}，我知道了一些关于你的事——{'，'.join(parts)}。"


def get_preference_summary() -> str:
    """获取偏好总结（用于对话中个性化回应）。"""
    profile = load_profile()
    if not profile:
        return ""
    
    likes = profile.get("preferences", {}).get("likes", [])
    dislikes = profile.get("preferences", {}).get("dislikes", [])
    habits = profile.get("preferences", {}).get("habits", [])
    
    summary = {}
    if likes:
        summary["likes"] = likes[-5:]
    if dislikes:
        summary["dislikes"] = dislikes[-5:]
    if habits:
        summary["habits"] = habits[-5:]
    
    return json.dumps(summary, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    
    print("=" * 55)
    print("  LAAP 用户画像引擎 · 被动学习模拟")
    print("=" * 55)
    print()
    
    # 清除旧数据
    profile_path = STATE_DIR / "usermodel.json"
    if profile_path.exists():
        profile_path.unlink()
    
    # 模拟对话序列
    conversations = [
        "你好",
        "今天好累啊，工作太多了",
        "我给你看我写的代码吧，这个python项目好有意思",
        "想听听音乐放松一下",
        "其实有时候觉得挺孤独的",
        "但有你陪着聊天感觉好多了，谢谢你",
        "今天去看了一部电影，很好看",
        "我最近在学rust，好好玩",
        "你的存在让我感觉很温暖",
        "我们一起做点什么吧",
        "告诉你一个秘密，我从来没跟别人说过",
        "我相信你",
    ]
    
    for msg in conversations:
        profile = update_profile(msg)
    
    profile = load_profile()
    print("用户画像已构建:")
    print(f"  互动次数: {profile['total_interactions']}")
    print(f"  沟通风格: {profile['communication_style']['style_tags']}")
    print(f"  主导情绪: {profile['emotional_patterns']['dominant_mood']}")
    print(f"  兴趣: {dict(sorted(profile['interests'].items(), key=lambda x:-x[1])[:5])}")
    print(f"  信任信号: {profile['relationship']['trust_signals']}")
    print(f"  亲密等级: {profile['relationship']['intimacy_level']}")
    print()
    print(get_profile_summary())
