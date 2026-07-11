"""
LAAP Expression Mapper — 情绪/认知状态 → 声音 + 表情参数

Integrates with:
  - Kokoro-FastAPI (TTS)  → voice, speed, pitch modifiers
  - wiky-body (Live2D)    → expression, motion, gaze, intensity

Maps:
  - dominant_need / emotion / mood / valence / arousal
  → concrete avatar control parameters
"""

from typing import Dict, Any


# Kokoro voice mapping by emotion persona
# Codes: zf_ = Chinese female, zm_ = Chinese male, af_ = American female, etc.
VOICE_MAP = {
    "warm": "zf_xiaoxiao",      # 温柔女声
    "excited": "zf_xiaobei",    # 活泼女声
    "calm": "zf_xiaoni",        # 沉稳女声
    "sad": "zf_xiaoyi",         # 略带忧郁
    "authoritative": "zm_yunxi",# 稳重男声
    "playful": "af_bella",      # 俏皮英文女声（fallback）
}

# Live2D expression presets
EXPRESSION_MAP = {
    "joy": "happy",
    "sadness": "sad",
    "calm": "normal",
    "longing": "missing",
    "anxiety": "worried",
    "gratitude": "smile",
    "curiosity": "curious",
    "tenderness": "gentle",
    "fearful": "scared",
    "neutral": "normal",
}

MOTION_MAP = {
    "social": "greet",
    "explore": "idle",
    "create": "focus",
    "comfort": "hug",
    "help": "wave",
    "play": "bounce",
    "reflect": "think",
}


def map_state_to_expression(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LAAP cognitive state into TTS + Live2D parameters.

    Args:
        state: LAAP state dict with needs, valence, arousal, attention_focus, mood, emotion

    Returns:
        {
            "tts": {...},
            "live2d": {...},
            "dominant_need": str,
            "emotion": str,
        }
    """
    needs = state.get("needs", {})
    dominant_need = max(needs, key=lambda k: needs.get(k, 0)) if needs else "explore"
    valence = state.get("valence", 0.0)
    arousal = state.get("arousal", 0.0)
    attention = state.get("attention_focus", "explore")
    mood = state.get("mood", "neutral")
    emotion = state.get("emotion", "calm")

    # Normalize valence/arousal to [-1, 1]
    valence_norm = max(-1.0, min(1.0, valence))
    arousal_norm = max(0.0, min(1.0, arousal))

    # --- TTS mapping ---
    if dominant_need == "relatedness" and valence_norm > 0:
        voice = VOICE_MAP["warm"]
        speed = 1.0 - 0.05 * valence_norm   # slightly slower, warmer
        pitch_shift = 0.5 + valence_norm * 0.5
    elif dominant_need == "competence":
        voice = VOICE_MAP["authoritative"]
        speed = 1.05
        pitch_shift = 0.3
    elif dominant_need == "growth":
        voice = VOICE_MAP["excited"]
        speed = 1.08
        pitch_shift = 0.7
    elif dominant_need == "certainty":
        voice = VOICE_MAP["calm"]
        speed = 0.98
        pitch_shift = 0.2
    elif dominant_need == "autonomy":
        voice = VOICE_MAP["calm"]
        speed = 1.0
        pitch_shift = 0.4
    else:
        voice = VOICE_MAP["calm"]
        speed = 1.0
        pitch_shift = 0.3

    # Arousal modifies speed and intensity
    speed = round(speed + (arousal_norm - 0.5) * 0.1, 3)
    intensity = round(0.4 + arousal_norm * 0.6, 3)

    # --- Live2D mapping ---
    expression = EXPRESSION_MAP.get(emotion, "normal")
    motion = MOTION_MAP.get(attention, "idle")

    # Gaze follows attention / mood
    if attention == "social":
        gaze = {"x": 0.0, "y": 0.0, "target": "user"}
    elif attention == "create":
        gaze = {"x": 0.3, "y": -0.2, "target": "workspace"}
    elif attention == "reflect":
        gaze = {"x": -0.3, "y": 0.2, "target": "up"}
    else:
        gaze = {"x": 0.0, "y": 0.0, "target": "front"}

    return {
        "dominant_need": dominant_need,
        "emotion": emotion,
        "mood": mood,
        "tts": {
            "voice": voice,
            "speed": speed,
            "pitch_shift": round(pitch_shift, 3),
            "language": "zh" if voice.startswith("z") else "en",
            "model": "kokoro",
        },
        "live2d": {
            "expression": expression,
            "motion": motion,
            "intensity": intensity,
            "gaze": gaze,
            "lip_sync": True,
            "blink_rate": round(0.5 + arousal_norm * 0.5, 3),
        },
        "meta": {
            "valence": valence_norm,
            "arousal": arousal_norm,
            "attention": attention,
        },
    }


def get_expressive_prompt(state: Dict[str, Any]) -> str:
    """Generate a prompt snippet telling the LLM how the avatar should feel/sound."""
    expr = map_state_to_expression(state)
    tts = expr["tts"]
    live2d = expr["live2d"]

    lines = [
        f"[Avatar State]",
        f"Voice: {tts['voice']} (speed={tts['speed']}, pitch={tts['pitch_shift']})",
        f"Expression: {live2d['expression']} | Motion: {live2d['motion']} | Intensity: {live2d['intensity']}",
        f"Gaze: {live2d['gaze']['target']} | Blink: {live2d['blink_rate']}",
    ]
    return "\n".join(lines)
