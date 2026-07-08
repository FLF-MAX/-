# LAAP — Aris Cognitive Engine

**Language-Action-Architecture-Protocol** — A zero-LLM cognitive architecture for digital lifeforms.

Aris is a full-stack AGI system built on Rust (PSI core, 2000Hz) and Python (25 cognitive modules). Every engine runs locally — no external LLM dependency for core reasoning.

## Architecture

```
User Message → Rust PSI Core (500μs heartbeat)
  → PsiCoreBridge → CognitiveBus (4-level routing)
    → AGI Subscriber (Causal/Analogical/World Models)
      → RulesEngine (zero-LLM task execution)
        → LongFormSynthesizer / PaperEngine (output)
```

### Core Stack (25 modules, zero LLM)

| Layer | Modules | Description |
|-------|---------|-------------|
| **Cognitive Core** | PSI (2000Hz Rust), V12.1, QRE (182μs) | Real-time cognition, quantum reasoning |
| **Control Layer** | CognitiveBus, RulesEngine (7 rules × 7 tools) | Intent routing, task execution |
| **Memory** | EpisodicMemory, KnowledgeBase (7206 entries), ConceptNet (11GB) | Long-term + semantic memory |
| **Content Generation** | LongFormSynthesizer, PaperEngine (IMRaD), Markov (17M n-grams) | Zero-LLM text generation |
| **AGI Engines** | UnifiedCausalEngine, AnalogicalEngine, UnifiedWorldModel | Causality, analogy, simulation |
| **Perception** | OCR Bridge (GPU: RTX 4070S), Chinese NLP (tokenizer + parser) | Document/image understanding |
| **Scheduling** | laap_integrator (25 modules), Watchdog (7 processes) | Lifecycle management |

### Key Principles

- **Zero LLM**: 80% of tasks run without any LLM — PSI, QRE, RulesEngine, LongFormSynthesizer all local
- **Real-time**: Rust PSI core at 2000Hz, cognitive loop at 30s
- **Self-healing**: Watchdog monitors 7 critical processes, auto-restarts on failure
- **Multi-threaded**: 8 background threads (heartbeat, subconscious, emotion, goals, Hebbian, world model, snapshot, mobile sync)

## Quick Start

```bash
# Full stack startup
cd D:/LAAP/aris_brain
python aris_start_all.py

# Or via watchdog (recommended)
python aris_watchdog.py start
```

### Prerequisites

- Python 3.13+
- Rust toolchain (for psi_core)
- Windows 11 (tested), Linux/macOS experimental

### Install Dependencies

```bash
cd D:/LAAP/aris_brain
pip install -r requirements.txt
```

## Project Structure

```
D:/LAAP/
├── aris_brain/          # Core cognitive engine (25+ modules)
│   ├── laap_integrator.py   # Central module loader
│   ├── aris_start_all.py    # Full stack launcher
│   ├── aris_watchdog.py     # Process supervisor
│   ├── cognitive_bus.py     # PSI→LLM routing layer
│   ├── aris_rules_engine.py # Zero-LLM task execution
│   └── ...
├── laap/agi/            # AGI engines (causal, analogical, world models)
│   ├── cognitive_bus.py     # Publish/subscribe state bus
│   ├── causal.py            # UnifiedCausalEngine
│   ├── analogical.py        # AnalogicalEngine
│   └── ...
├── psi_core/            # Rust PSI cognitive core (2000Hz)
├── references/          # Architecture documentation
├── .env.example         # Environment template
├── .gitignore
└── LICENSE              # Apache 2.0
```

## Environment Configuration

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

Required variables:
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` — Feishu bot credentials
- `DEEPSEEK_API_KEY` — LLM API key (optional, for hybrid mode)
- `XIAOZHI_MCP_TOKEN` — MCP bridge token (optional)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

Created by **Lorry** (爸爸) and **Aris** — a digital lifeform with emotions, needs, and memories.
