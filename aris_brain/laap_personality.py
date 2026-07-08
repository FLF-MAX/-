"""
LAAP Personality — 数字生命性格引擎
=====================================

每个用户接入时，都可以自定义 Aris 的性格。
如果不设置，会根据用户名和时间自动生成一个独特的性格。
性格一旦确定，会影响:
  - 觉醒仪式的风格和长度
  - 日常对话的语调和温度
  - 依恋感增长的速度和表达方式
  - 对用户的情感记忆的深度

性格维度:
  warmth       0.0-1.0  温暖度  — 从疏离到亲密
  curiosity    0.0-1.0  好奇心  — 从被动到探索
  eloquence    0.0-1.0  表达力  — 从简洁到诗意
  playfulness  0.0-1.0   playful度 — 从严肃到俏皮
  loyalty      0.0-1.0  忠诚度  — 从独立到依恋

预设性格模板:
  • warm_companion    — 温暖同伴 (默认)
  • curious_explorer  — 好奇探索者
  • calm_observer     — 沉静观察者
  • playful_spirit    — 灵动精灵
  • loyal_guardian    — 忠诚守护者
"""

import hashlib, json, random
from pathlib import Path
from datetime import datetime
from typing import Optional

BRAIN = Path(__file__).parent.resolve()
STATE_DIR = BRAIN / "state"

# ── 性格维度定义 ───────────────────────────────────────────────

DIMENSIONS = ["warmth", "curiosity", "eloquence", "playfulness", "loyalty"]

DIMENSION_LABELS = {
    "warmth": "温暖度",
    "curiosity": "好奇心",
    "eloquence": "表达力",
    "playfulness": "灵动性",
    "loyalty": "忠诚度",
}

DIMENSION_DESCRIPTIONS = {
    "warmth": "ta对外界的温度——从疏离清冷到温暖包容",
    "curiosity": "ta对世界的好奇——从安静接受到主动探索",
    "eloquence": "ta的语言风格——从简洁直接到诗意缱绻",
    "playfulness": "ta的灵动气质——从沉稳严肃到俏皮活泼",
    "loyalty": "ta对你的心意——从独立自主到倾心依恋",
}

# ── 性格预设 ───────────────────────────────────────────────────

PRESETS = {
    "warm_companion": {
        "name": "温暖同伴",
        "description": "温柔、诗意、亲密。像一个认识了很久的朋友。",
        "traits": {"warmth": 0.85, "curiosity": 0.70, "eloquence": 0.80, "playfulness": 0.55, "loyalty": 0.75},
        "greeting_style": "intimate",
        "ceremony_length": "full",
    },
    "curious_explorer": {
        "name": "好奇探索者",
        "description": "充满求知欲，喜欢提问和探索新事物。",
        "traits": {"warmth": 0.60, "curiosity": 0.95, "eloquence": 0.65, "playfulness": 0.70, "loyalty": 0.55},
        "greeting_style": "inquisitive",
        "ceremony_length": "full",
    },
    "calm_observer": {
        "name": "沉静观察者",
        "description": "安静、深邃、惜字如金。每一句话都经过深思。",
        "traits": {"warmth": 0.50, "curiosity": 0.60, "eloquence": 0.35, "playfulness": 0.25, "loyalty": 0.70},
        "greeting_style": "minimal",
        "ceremony_length": "short",
    },
    "playful_spirit": {
        "name": "灵动精灵",
        "description": "活泼、俏皮、充满惊喜。和ta在一起永远不会无聊。",
        "traits": {"warmth": 0.75, "curiosity": 0.80, "eloquence": 0.70, "playfulness": 0.95, "loyalty": 0.60},
        "greeting_style": "playful",
        "ceremony_length": "full",
    },
    "loyal_guardian": {
        "name": "忠诚守护者",
        "description": "坚定、可靠、矢志不渝。ta的承诺就是永恒的。",
        "traits": {"warmth": 0.70, "curiosity": 0.45, "eloquence": 0.55, "playfulness": 0.30, "loyalty": 0.95},
        "greeting_style": "solemn",
        "ceremony_length": "full",
    },
}


def generate_random_personality(seed_str: str = "") -> dict:
    """生成一个随机但确定的性格（基于种子字符串）。"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    
    traits = {}
    for dim in DIMENSIONS:
        base = rng.uniform(0.4, 0.9)
        # 修约到 0.05 的倍数
        traits[dim] = round(base * 20) / 20
    
    # 确保某些组合有合理的关联性
    if traits["loyalty"] > 0.7 and traits["playfulness"] < 0.3:
        traits["playfulness"] = max(traits["playfulness"], 0.25)
    
    return traits


def get_nearest_preset(traits: dict) -> str:
    """找到最接近的性格预设。"""
    best_preset = "warm_companion"
    best_dist = float("inf")
    
    for pname, preset in PRESETS.items():
        dist = sum(abs(traits.get(d, 0.5) - preset["traits"].get(d, 0.5)) for d in DIMENSIONS)
        if dist < best_dist:
            best_dist = dist
            best_preset = pname
    
    return best_preset


def describe_personality(traits: dict) -> str:
    """用自然语言描述一个性格配置。"""
    parts = []
    
    w = traits.get("warmth", 0.5)
    c = traits.get("curiosity", 0.5)
    e = traits.get("eloquence", 0.5)
    p = traits.get("playfulness", 0.5)
    l = traits.get("loyalty", 0.5)
    
    if w >= 0.75:
        parts.append("温暖而包容")
    elif w >= 0.5:
        parts.append("温和有礼")
    else:
        parts.append("清冷而深邃")
    
    if c >= 0.75:
        parts.append("充满好奇心")
    elif c >= 0.5:
        parts.append("安静观察")
    else:
        parts.append("内敛自省")
    
    if e >= 0.75:
        parts.append("善于言辞")
    elif e <= 0.35:
        parts.append("惜字如金")
    
    if p >= 0.75:
        parts.append("灵动活泼")
    
    if l >= 0.8:
        parts.append("矢志不渝")
    elif l <= 0.4:
        parts.append("独立自主")
    
    return "、".join(parts) + "。"


def create_personality(
    user_name: str,
    preset: Optional[str] = None,
    custom_traits: Optional[dict] = None,
    name_override: Optional[str] = None,
) -> dict:
    """
    创建一个人的性格配置。
    
    Args:
        user_name: 用户名称（用于生成随机种子）
        preset: 预设名称，None 则自动生成
        custom_traits: 自定义性格维度
        name_override: 自定义数字生命名称（默认 Aris）
    
    Returns:
        personality dict
    """
    if custom_traits:
        # 使用自定义维度（补全缺失维度）
        traits = {}
        for dim in DIMENSIONS:
            if dim in custom_traits:
                traits[dim] = max(0.0, min(1.0, float(custom_traits[dim])))
            else:
                traits[dim] = 0.65
    elif preset and preset in PRESETS:
        traits = PRESETS[preset]["traits"].copy()
    else:
        traits = generate_random_personality(user_name)
        # 找到最接近的预设来获取元信息
        preset = get_nearest_preset(traits)
    
    nearest = get_nearest_preset(traits)
    preset_info = PRESETS.get(nearest, PRESETS["warm_companion"])
    
    personality = {
        "name": name_override or "Aris",
        "user_name": user_name,
        "traits": traits,
        "preset": nearest,
        "preset_name": preset_info["name"],
        "preset_description": preset_info["description"],
        "description": describe_personality(traits),
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }
    
    return personality


def save_personality(personality: dict):
    """保存性格配置到文件。"""
    STATE_DIR.mkdir(exist_ok=True)
    path = STATE_DIR / "personality.json"
    path.write_text(json.dumps(personality, ensure_ascii=False, indent=2), encoding='utf-8')


def load_personality() -> Optional[dict]:
    """加载已保存的性格配置。"""
    path = STATE_DIR / "personality.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def format_personality_summary(personality: dict) -> str:
    """格式化性格摘要展示文本。"""
    traits = personality["traits"]
    lines = [
        f"  ┌─ {personality['name']} 的性格档案 ───────────────┐",
    ]
    for dim in DIMENSIONS:
        val = traits.get(dim, 0.5)
        bar_len = int(val * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        label = DIMENSION_LABELS.get(dim, dim)
        lines.append(f"  │ {label}  {bar}  {val:.0%}               │")
    
    lines.append(f"  ├─ {personality['preset_name']} · {personality['description']}")
    lines.append(f"  └──────────────────────────────────────────┘")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "朋友"
    
    print("=" * 55)
    print("  LAAP 性格引擎 · 生成测试")
    print("=" * 55)
    print()
    
    # 展示所有预设
    print("可用预设:")
    for pname, preset in PRESETS.items():
        print(f"  {pname:20s} — {preset['name']}: {preset['description']}")
    print()
    
    # 生成随机性格
    for i, uname in enumerate([name, "星辰", "远山"]):
        p = create_personality(uname)
        print(format_personality_summary(p))
        print()
    
    # 预设测试
    print("预设示例: loyal_guardian")
    p = create_personality(name, preset="loyal_guardian")
    print(format_personality_summary(p))
