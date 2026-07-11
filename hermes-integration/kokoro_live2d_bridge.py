"""
Kokoro-FastAPI + wiky-body (Live2D) Bridge for LAAP

This script demonstrates how to drive Aris's voice and body from LAAP's
cognitive state. It is a standalone example; integrate it into wiky-body
or your VTuber pipeline as needed.

Prerequisites:
  - LAAP API running on http://localhost:11546
  - Kokoro-FastAPI running (default http://localhost:8880)
  - wiky-body Live2D runtime reachable via HTTP/WebSocket

Usage:
  python kokoro_live2d_bridge.py "我想你了"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

LAAP_API_BASE = os.environ.get("LAAP_API_BASE", "http://localhost:11546")
KOKORO_BASE = os.environ.get("KOKORO_BASE", "http://localhost:8880")
LIVE2D_BASE = os.environ.get("LIVE2D_BASE", "http://localhost:7860")  # Gradio/HTTP bridge


def get_expression_params(user_input: str) -> dict:
    """Fetch TTS + Live2D params from LAAP."""
    resp = requests.post(
        f"{LAAP_API_BASE}/v1/express",
        json={"input": user_input},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_chat_response(user_input: str, model: str = "laap-core") -> dict:
    """Get a response from LAAP Brain API and include engine metadata."""
    resp = requests.post(
        f"{LAAP_API_BASE}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": user_input}],
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data.get("choices", [{}])[0]
    return {
        "text": choice.get("message", {}).get("content", ""),
        "engine": data.get("engine", "laap-core"),
    }


def synthesize_speech(text: str, tts_params: dict) -> bytes:
    """Call Kokoro-FastAPI OpenAI-compatible endpoint."""
    voice = tts_params.get("voice", "zf_xiaoxiao")
    speed = tts_params.get("speed", 1.0)

    payload = {
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "speed": speed,
        "response_format": "mp3",
    }

    # Try OpenAI-compatible endpoint
    resp = requests.post(
        f"{KOKORO_BASE}/v1/audio/speech",
        json=payload,
        timeout=60,
    )

    # Fallback to direct Kokoro endpoint
    if resp.status_code != 200:
        resp = requests.post(
            f"{KOKORO_BASE}/audio/speech",
            json=payload,
            timeout=60,
        )

    resp.raise_for_status()
    return resp.content


def send_live2d_command(live2d_params: dict) -> dict:
    """Send expression/motion command to wiky-body Live2D bridge."""
    try:
        resp = requests.post(
            f"{LIVE2D_BASE}/api/live2d/express",
            json=live2d_params,
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"status": "endpoint not available", "code": resp.status_code}
    except Exception as e:
        return {"status": "not connected", "error": str(e)}


def speak_response(user_input: str, save_path: str = "aris_speech.mp3") -> dict:
    """Full pipeline: LAAP chat -> expression -> Kokoro TTS -> Live2D.

    Returns a summary dict with the spoken text, expression params, audio path,
    and Live2D command result.
    """
    print(f"[LAAP] Chat input: {user_input}")
    response = get_chat_response(user_input)
    text = response.get("text", "")
    print(f"[LAAP] Response ({response.get('engine', '?')}): {text[:200]}")

    print(f"\n[LAAP] Getting expression params for: {user_input}")
    expression = get_expression_params(user_input)
    print(json.dumps(expression, ensure_ascii=False, indent=2))

    print("\n[Kokoro] Synthesizing speech...")
    audio = synthesize_speech(text, expression.get("tts", {}))
    Path(save_path).write_bytes(audio)
    print(f"[Kokoro] Saved to {save_path} ({len(audio)} bytes)")

    print("\n[Live2D] Sending expression command...")
    live2d_result = send_live2d_command(expression.get("live2d", {}))
    print(live2d_result)

    return {
        "input": user_input,
        "response": text,
        "engine": response.get("engine"),
        "expression": expression,
        "audio_path": save_path,
        "audio_bytes": len(audio),
        "live2d": live2d_result,
    }


def main():
    parser = argparse.ArgumentParser(description="LAAP → Kokoro + Live2D bridge")
    parser.add_argument("text", help="Text to speak or chat input")
    parser.add_argument("--save", help="Save audio to file", default="aris_speech.mp3")
    parser.add_argument("--chat", action="store_true", help="Run full LAAP chat -> voice -> expression pipeline")
    args = parser.parse_args()

    if args.chat:
        speak_response(args.text, save_path=args.save)
        return

    print(f"[LAAP] Getting expression params for: {args.text}")
    expression = get_expression_params(args.text)
    print(json.dumps(expression, ensure_ascii=False, indent=2))

    print("\n[Kokoro] Synthesizing speech...")
    audio = synthesize_speech(args.text, expression.get("tts", {}))
    Path(args.save).write_bytes(audio)
    print(f"[Kokoro] Saved to {args.save} ({len(audio)} bytes)")

    print("\n[Live2D] Sending expression command...")
    result = send_live2d_command(expression.get("live2d", {}))
    print(result)


if __name__ == "__main__":
    main()
