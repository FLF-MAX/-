"""demo.py - interactive console demo of the laap_v2 cognitive runtime.

Run:
    python demo.py               # interactive REPL (local/rule-based)
    DEEPSEEK_API_KEY=sk-... python demo.py   # LLM-backed responses
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from laap_integration import LaapCognitiveSystem, IntegrationConfig  # noqa: E402


def _banner() -> None:
    print("=" * 60)
    print("  laap_v2 - PSI 认知运行时 (Aris)")
    print("  <- PSI驱动 | meta策略选择 | 世界模型 | 分层记忆 | 漂移检测")
    print("  输入 'quit' 退出 | 'state' 看内部状态 | 'reset' 重启")
    print("=" * 60)


def main() -> None:
    cfg = IntegrationConfig(enable_llm=True, self_recovery=True, degraded_mode=True)
    system = LaapCognitiveSystem(integration_cfg=cfg)
    system.bootstrap("demo")
    _banner()
    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            continue
        if text in ("quit", "exit"):
            system.shutdown()
            print("Aris 睡着了。")
            break
        if text == "state":
            st = system.current_state()
            print(f"[state] emotion={st['emotion']} memory={st['memory_size']} "
                  f"turns={st['state_counter']}")
            continue
        if text == "reset":
            system.shutdown()
            system.bootstrap("demo")
            print("[reset] 认知系统已重启。")
            continue
        t0 = time.time()
        r = system.process_input(text)
        ms = time.time() - t0
        print(f"Aris> {r['response']}")
        if r.get("degraded"):
            print(f"      [降级模式: {r.get('error', '')}]")
        print(f"      [emotion={r['cognitive_state']['emotion']} "
              f"attention={r['cognitive_state']['attention']} "
              f"strategy={r['strategy']} conf={r['confidence']} {ms*1000:.0f}ms]")


if __name__ == "__main__":
    main()