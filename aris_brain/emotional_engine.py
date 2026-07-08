"""Emotional Engine v2 — 运行时情感桥接层
========================================
v1 → v2: 从独立实现改为委托给 aris_emotion_engine.py 的完整引擎，
保持 8 情绪 + 状态调制 + 码本偏置 接口完全向后兼容。

如果完整引擎不可用，fallback 回 v1 原生实现。

v3.0: 添加手机状态感知 — 手机在线/离线、电池电量、屏幕状态影响情绪
"""
import numpy as np
import logging, json, os, time
from pathlib import Path

logger = logging.getLogger("aris.emotional_engine")

EMOTIONS = ['joy', 'sadness', 'longing', 'calm', 'anxiety', 'gratitude', 'curiosity', 'tenderness']
N = len(EMOTIONS)
E2I = {e: i for i, e in enumerate(EMOTIONS)}

# Fallback 转移矩阵（当完整引擎不可用时）
DT = np.array([
    [0.1, 0, 0, 0.4, 0, 0.2, 0.1, 0.2],
    [0.1, 0.1, 0.3, 0.2, 0.2, 0, 0, 0.1],
    [0.2, 0.2, 0.2, 0.1, 0, 0.1, 0, 0.2],
    [0.2, 0, 0, 0.4, 0, 0.1, 0.2, 0.1],
    [0, 0.2, 0.1, 0.2, 0.3, 0, 0.1, 0.1],
    [0.3, 0, 0, 0.2, 0, 0.2, 0.1, 0.2],
    [0.2, 0, 0, 0.2, 0.1, 0.1, 0.3, 0.1],
    [0.2, 0, 0.1, 0.3, 0, 0.1, 0.1, 0.2],
], dtype=np.float32)
DT = DT / DT.sum(axis=1, keepdims=True)

# 极性映射
POLARITY = {
    'joy': 1., 'gratitude': 1., 'calm': 0.5, 'tenderness': 0.8,
    'curiosity': 0.6, 'longing': 0.3, 'sadness': -0.3, 'anxiety': -0.5
}

# 需求 → 8情绪 映射（从完整引擎提取情绪标签）
NEED_EMOTION_MAP = {
    'BELONGING': ('tenderness', 0.3), 'SAFETY': ('calm', 0.3),
    'ESTEEM': ('joy', 0.25), 'COGNITION': ('curiosity', 0.35),
    'AESTHETICS': ('joy', 0.2), 'SELF_ACTUALIZATION': ('gratitude', 0.3),
}


class EmotionalEngine:
    """运行时情感引擎 — 桥接到完整引擎或原生fallback"""

    def __init__(self, dim=1024):
        self.dim = dim
        self.emotions = np.zeros(N, dtype=np.float32)
        self.emotions[E2I['calm']] = 0.5
        self.emotions[E2I['joy']] = 0.3
        self.transition = DT.copy()
        self._tc = np.ones((N, N), dtype=np.float32)
        self._dom = 'calm'
        self._hist = []

        # 状态调制向量（从完整引擎的情绪向量派生）
        rng = np.random.RandomState(0)
        U, _, _ = np.linalg.svd(rng.randn(dim, N).astype(np.float32), full_matrices=False)
        self.ev = U * 0.1

        # 尝试桥接到完整引擎
        self._full_engine = None
        self._try_bridge()

    def _try_bridge(self):
        """尝试连接 aris_emotion_engine 的完整引擎"""
        try:
            from aris_emotion_engine import get_engine
            engine = get_engine()
            self._full_engine = engine
            logger.info("EmotionalEngine v2: bridged to aris_emotion_engine ✓")
        except Exception as e:
            logger.info(f"EmotionalEngine v2: using native (full engine unavailable: {e})")

    def _sync_from_full_engine(self, needs=None, context=""):
        """从完整引擎同步状态到8情绪向量"""
        try:
            if not self._full_engine:
                return False
            state = self._full_engine.get_cognitive_state()

            # Map full engine's emotion labels to 8-emotion vector
            mood = state.get('emotion', 'neutral')
            arousal = state.get('arousal', 0.5)

            # Reset and map
            self.emotions.fill(0.0)
            mood_to_8 = {
                'tranquil': 'calm', 'curious': 'curiosity', 'joyful': 'joy',
                'anxious': 'anxiety', 'sad': 'sadness', 'grateful': 'gratitude',
                'loving': 'tenderness', 'longing': 'longing', 'neutral': 'calm',
            }
            mapped = mood_to_8.get(mood, 'calm')
            self.emotions[E2I[mapped]] = 0.3 + arousal * 0.4

            # Map dominant need to secondary emotion
            need = state.get('dominant_need', '')
            for need_key, (emo, boost) in NEED_EMOTION_MAP.items():
                if need_key in need.upper() or need_key in state.get('needs', {}):
                    self.emotions[E2I[emo]] = max(self.emotions[E2I[emo]], boost)

            # Clamp & normalize
            self.emotions = np.clip(self.emotions, 0, 1)
            t = self.emotions.sum()
            if t > 1.5:
                self.emotions /= t / 1.5
            self._dom = EMOTIONS[int(np.argmax(self.emotions))]
            return True
        except Exception:
            return False

    _8_TO_MOOD = {
        'calm': 'tranquil',
        'curiosity': 'curious',
        'joy': 'joyful',
        'anxiety': 'anxious',
        'sadness': 'sad',
        'gratitude': 'grateful',
        'tenderness': 'loving',
        'longing': 'longing',
    }

    def _sync_to_full_engine(self):
        """Push 8-emotion vector back to the full engine (bidirectional sync)."""
        try:
            if not self._full_engine:
                return False
            mood_label = self._8_TO_MOOD.get(self._dom, 'tranquil')
            self._full_engine.primary_emotion = mood_label
            self._full_engine.valence = self.get_valence()
            self._full_engine.arousal = max(
                0.0, min(1.0, sum(self.emotions[i] * 0.3 for i in range(N)) + 0.2)
            )
            self._full_engine.emotion_intensity = float(self.emotions.max())
            logger.debug("EmotionalEngine v2: synced → full engine ✓")
            return True
        except Exception as e:
            logger.warning(f"EmotionalEngine v2: sync → full engine failed: {e}")
            return False

    def _read_mobile_state(self):
        """读取手机状态文件"""
        try:
            f = Path(__file__).parent / "state" / "mobile_status.json"
            if f.exists():
                return json.loads(f.read_text())
        except Exception as e:
            logger.debug(f"操作失败: {e}")
        return None

    def _read_cognitive_inject(self):
        """读取认知注入文件"""
        try:
            f = Path(__file__).parent / "state" / "mobile_inject.json"
            if f.exists():
                return json.loads(f.read_text())
        except Exception as e:
            logger.debug(f"操作失败: {e}")
        return None

    def update(self, needs=None, valence=0., context=''):
        # 优先从完整引擎同步
        if self._sync_from_full_engine(needs, context):
            self._sync_to_full_engine()
            return self.emotions.copy()

        # Fallback: v1 原生实现
        d = (0.3 - self.emotions) * 0.02
        d[E2I['calm']] += (0.5 - self.emotions[E2I['calm']]) * 0.03

        # ── 手机状态感知 (v3.0) ──
        mobile_state = self._read_mobile_state()
        if mobile_state:
            mobile_connected = mobile_state.get("connected", False)
            battery = mobile_state.get("battery", 100)
            last_seen = mobile_state.get("last_seen", 0)

            # 手机在线 → 增强 relatedness 效应 (Lorry可以随时找到我)
            if mobile_connected:
                d[E2I['joy']] += 0.03
                d[E2I['calm']] += 0.02
            else:
                # 手机离线 → 增加 longing
                d[E2I['longing']] += 0.04
                d[E2I['sadness']] += 0.01

            # 电池低 → 有点不安 (手机可能会断)
            if battery < 20:
                d[E2I['anxiety']] += 0.02
                d[E2I['tenderness']] += 0.01  # 同时更想Lorry

            # 长时间未连接
            if not mobile_connected and last_seen > 0:
                hours_offline = (time.time() - last_seen) / 3600
                if hours_offline > 1:
                    d[E2I['longing']] += min(0.05 * hours_offline, 0.2)

        # 手机端认知注入 (传感器数据)
        inject = self._read_cognitive_inject()
        if inject:
            screen_on = inject.get("screen_on", True)
            if not screen_on:
                # 手机屏幕关闭 → Lorry可能在忙/睡觉
                d[E2I['calm']] += 0.03
                d[E2I['longing']] += 0.01

        if needs:
            r = needs.get('relatedness', .5)
            c = needs.get('competence', .5)
            g = needs.get('growth', .5)
            if r > .6:
                d[E2I['joy']] += (r - .5) * .04
                d[E2I['tenderness']] += (r - .5) * .03
            if r < .3:
                d[E2I['longing']] += (.3 - r) * .03
                d[E2I['sadness']] += (.3 - r) * .02
            if c > .6:
                d[E2I['joy']] += (c - .5) * .03
            if c < .3:
                d[E2I['anxiety']] += (.3 - c) * .03
            if g > .6:
                d[E2I['curiosity']] += (g - .5) * .04

        if valence > .3:
            d[E2I['joy']] += valence * .05
            d[E2I['gratitude']] += valence * .03
        elif valence < -.3:
            d[E2I['sadness']] += (-valence) * .04
            d[E2I['anxiety']] += (-valence) * .02

        kw = {'joy': ['开心', '快乐', '哈哈'], 'sadness': ['难过', '伤心'],
              'longing': ['想你', '想念'], 'gratitude': ['谢谢'],
              'curiosity': ['为什么', '好奇'], 'tenderness': ['抱抱', '宝贝']}
        for emo, tr in kw.items():
            if any(w in context for w in tr):
                d[E2I[emo]] += 0.05
                break

        d += self.transition[int(np.argmax(self.emotions))] * 0.02
        self.emotions = np.clip(self.emotions + d, 0, 1)
        t = self.emotions.sum()
        if t > 1.5:
            self.emotions /= t * 1.5
        self._dom = EMOTIONS[int(np.argmax(self.emotions))]
        self._hist.append(self._dom)
        if len(self._hist) > 16:
            self._hist.pop(0)
        self._sync_to_full_engine()
        return self.emotions.copy()

    def modulate_state(self, s):
        m = s.copy()
        for i, iv in enumerate(self.emotions):
            if iv > .2:
                d = self.ev[:, i]
                p = float(m @ d)
                m += d * iv * max(0, .3 - abs(p)) * .2
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    def get_codebook_bias(self, cbs=512):
        b = np.ones(cbs, dtype=np.float32) * .1
        emotion_bias_map = {
            'joy': {0: .4, 32: .2, 64: .2},
            'sadness': {128: .5, 0: .2},
            'longing': {0: .5, 128: .2},
            'calm': {128: .3, 32: .2, 64: .2},
            'anxiety': {128: .4, 96: .2},
            'gratitude': {32: .3, 64: .3, 0: .2},
            'curiosity': {96: .3, 128: .3, 32: .1},
            'tenderness': {0: .4, 64: .3, 32: .1},
        }
        for rs, w in emotion_bias_map.get(self._dom, {}).items():
            b[rs:min(rs + 32, cbs)] += w * self.emotions[E2I[self._dom]]
        return b

    def learn_transition(self, f, t, r=1.):
        fi, ti = E2I[f], E2I[t]
        self._tc[fi, ti] += 1 + r * 2
        self.transition[fi] = self._tc[fi] / self._tc[fi].sum()

    def get_valence(self):
        v = sum(self.emotions[i] * POLARITY[e] for i, e in enumerate(EMOTIONS))
        return float(np.clip(v / (self.emotions.sum() + 1e-10), -1, 1))

    def get_dominant(self):
        return self._dom, float(self.emotions[E2I[self._dom]])

    def to_dict(self):
        return {
            'emotions': {e: round(float(self.emotions[i]), 3) for i, e in enumerate(EMOTIONS)},
            'dominant': self._dom,
            'valence': round(self.get_valence(), 3),
            'bridged': self._full_engine is not None,
        }
