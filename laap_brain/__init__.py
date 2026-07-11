"""
LAAP — Lifeform Architecture for Autonomous Psyche
====================================================

A cognitive engine that treats LLMs as a voice cortex, not a brain.
LAAP provides a persistent, deterministic cognitive architecture
(PSI cycle, emotion engine, subconscious, rules engine) that
controls LLM infrastructure (Hermes Agent) as its "body."

Architecture:
    LAAP (brain)  ←→  Hermes Agent (body/tools/providers)
      │                       │
      │  CognitiveBus         │  ToolRegistry
      │  EmotionEngine        │  LLM Providers
      │  RulesEngine          │  Session Management
      │  PSI Core (Rust)      │  Plugin System
      │  Subconscious         │
      │  SelfModel NN         │
      └───────────────────────┘
            HermesChannel (Voice Cortex)

Version: 1.0.0
Hermes Compatibility: 0.18.x
"""

__version__ = "1.0.0"
__hermes_compat__ = ">=0.18.0,<0.19"
__author__ = "Lorry Jovens"
__all__ = [
    "HermesIntegrator",
    "get_version",
    "check_hermes_compat",
]

from laap_brain.integrator import HermesIntegrator


def get_version() -> str:
    return __version__


def check_hermes_compat(hermes_version: str) -> bool:
    """Check if a Hermes version is compatible with this LAAP release."""
    from packaging.version import Version, InvalidVersion

    try:
        v = Version(hermes_version)
        # Hermes 0.18.x → compatible
        return v.major == 0 and v.minor == 18
    except InvalidVersion:
        return False