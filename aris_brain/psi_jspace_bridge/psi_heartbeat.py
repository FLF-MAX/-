"""
PSI Heartbeat — 认知状态维护守护脚本
每 30 分钟运行一次，保持 PSI 认知状态新鲜。
"""

import json
import sys
import os
import time

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BRIDGE_DIR)

from psi_bridge import get_bridge

bridge = get_bridge()

# 没有用户输入时的"空闲认知维护"
# 需求缓慢漂移回平衡，能量恢复
bridge.state.energy = min(10.0, bridge.state.energy + 0.1)
bridge.state.cognitive_cycle += 1

# 空闲时需求的自然调节
for name in ["competence", "autonomy", "relatedness", "certainty", "growth"]:
    # 缓慢回归 0.5
    bridge.state.needs[name] += (0.5 - bridge.state.needs[name]) * 0.02

# 每小时早 8 点到晚 12 点: 轻度激活
hour = time.localtime().tm_hour
if 8 <= hour <= 23:
    bridge.state.valence = max(0, bridge.state.valence + 0.02)
    bridge.state.arousal = max(0.2, bridge.state.arousal)
else:
    bridge.state.valence = 0.05
    bridge.state.arousal = 0.1

bridge.save_state({"heartbeat": True, "cycle": bridge.state.cognitive_cycle})
print(f"[PSI Heartbeat] Cycle {bridge.state.cognitive_cycle} | "
      f"Needs: {bridge.state.needs} | Energy: {bridge.state.energy:.1f}")
