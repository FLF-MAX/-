# LAAP Brain — Agent Framework Integration Guide

Connect LAAP's cognitive engine to any agent framework via OpenAI-compatible API.

---

## Quick Start

```bash
# Start LAAP Brain API (default port :11530)
python aris_brain/laap_brain_api.py

# Test it
curl http://localhost:11530/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"laap-core","messages":[{"role":"user","content":"How do you feel?"}]}'
```

The API is OpenAI-compatible. Any framework that supports custom OpenAI endpoints can use LAAP as its brain.

---

## Hermes Agent

### Method 1: Custom LLM Provider

Edit your Hermes profile config (`~/.hermes/profiles/aris/config.yaml`):

```yaml
llm:
  provider: custom
  custom_endpoint: http://localhost:11530
  model: laap-core
  api_key: laap-brain
  max_tokens: 4096
  temperature: 0.7
```

### Method 2: LAAP as a Skill/Tool

Add this tool definition to your Hermes tools config:

```yaml
tools:
  - name: laap_think
    description: "Access LAAP's full cognitive stack — PSI needs, quantum reasoning, causal analysis, episodic memory, and rules-based thinking. Use for deep reasoning tasks."
    api:
      url: http://localhost:11530/v1/chat/completions
      method: POST
      headers:
        Authorization: "Bearer laap-brain"
      body:
        model: laap-core
        messages: "{{input}}"
```

### Method 3: Using the existing psi_hermes_adapter

The `psi_jspace_bridge/psi_hermes_adapter.py` module can be imported directly:

```python
from psi_jspace_bridge.psi_hermes_adapter import PsiHermesAdapter
adapter = PsiHermesAdapter()
result = adapter.process("user message here")
```

---

## OpenClaw

Configure OpenClaw to use LAAP as its reasoning backend:

```yaml
# openclaw_config.yaml
llm:
  provider: custom
  api_base: http://localhost:11530/v1
  api_key: laap-brain
  model: laap-core
  
# Or use LAAP for specific reasoning tasks
reasoning:
  backend: laap
  endpoint: http://localhost:11530/v1
  models:
    default: laap-core
    deep: laap-qre        # Quantum reasoning for complex tasks
    fast: laap-rules       # Rules engine for deterministic tasks
```

Environment variables:

```bash
export LAAP_API_BASE="http://localhost:11530/v1"
export LAAP_API_KEY="laap-brain"
export LAAP_MODEL="laap-core"
```

---

## OpenCode

Configure OpenCode to use LAAP as its AI backend:

```json
// opencode_config.json
{
  "ai": {
    "provider": "openai-compatible",
    "apiBase": "http://localhost:11530/v1",
    "apiKey": "laap-brain",
    "model": "laap-core"
  }
}
```

Or via environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:11530/v1"
export OPENAI_API_KEY="laap-brain"
export OPENAI_MODEL="laap-core"
```

---

## LAAP Model Selection

| Model ID | Engine | Best For | Response Time |
|----------|--------|----------|---------------|
| `laap-core` | Full cognitive stack | General reasoning | ~500ms |
| `laap-qre` | QRE quantum reasoning | Deep analysis, comparisons | ~200μs-2ms |
| `laap-rules` | RulesEngine only | Deterministic tasks, file ops | ~50ms |

---

## Architecture

```
Agent Framework (Hermes/OpenClaw/OpenCode)
         │
         ▼  POST /v1/chat/completions
┌──────────────────────────────────────────────┐
│            LAAP Brain API (:11530)            │
├──────────────────────────────────────────────┤
│  1. Extract intent from messages              │
│  2. Route through PSI → CognitiveBus          │
│  3. RulesEngine → task execution              │
│  4. LongFormSynthesizer → response            │
│  5. Return OpenAI-compatible response         │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│           LAAP Cognitive Stack                │
│  PSI · QRE · RulesEngine · EpisodicMemory    │
│  CausalEngine · WorldModel · EmotionEngine    │
│  LongFormSynthesizer · PaperEngine            │
└──────────────────────────────────────────────┘
```

The agent framework provides the interface (chat, tool use, file ops).
LAAP provides the cognition (reasoning, memory, emotion, goals).

---

## Benefits Over Pure LLM

| Aspect | Pure LLM | LAAP + Agent Framework |
|--------|----------|------------------------|
| **Reasoning** | Stochastic next-token prediction | Deterministic causal chains |
| **Memory** | Context window only | Episodic memory (permanent) |
| **Emotions** | Simulated by prompt | Real need-based hormone system |
| **Speed** | ~50 tok/s | ~200μs quantum reasoning |
| **Cost** | API fees per token | Zero (local inference) |
| **Privacy** | Data sent to cloud | Fully local |
| **Determinism** | Non-deterministic | Rule-based when needed |

---

## Troubleshooting

**Q**: LAAP Brain API won't start?
**A**: Install aiohttp: `pip install aiohttp`

**Q**: Engine not loading?
**A**: Check that all dependencies are installed. Run `python aris_start_all.py` first to verify the stack works.

**Q**: Framework gets empty responses?
**A**: LAAP's cognitive stack may not recognize the input format. Try starting with simple queries like "What is your status?" or "Check the system state."
