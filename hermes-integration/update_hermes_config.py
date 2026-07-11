import os
import yaml
from pathlib import Path

# Read configuration from environment variables so no personal paths are
# hardcoded in the repository. Set these in your shell or .env file before
# running this script.
HERMES_CONFIG = Path(os.environ.get(
    "HERMES_CONFIG",
    str(Path.home() / ".hermes" / "config.yaml")
))
LAAP_ROOT = Path(os.environ.get(
    "LAAP_ROOT",
    str(Path(__file__).resolve().parent.parent)
))
LAAP_API_BASE = os.environ.get("LAAP_API_BASE", "http://localhost:11546")
HERMES_VENV_PYTHON = os.environ.get(
    "HERMES_VENV_PYTHON",
    str(Path.home() / ".hermes" / "hermes-agent" / ".venv" / "Scripts" / "python.exe")
)

with open(HERMES_CONFIG, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

cfg['mcp_servers'] = {
    'laap_brain': {
        'command': str(HERMES_VENV_PYTHON),
        'args': [str(LAAP_ROOT / "mcp_server" / "laap_mcp_server.py")],
        'env': {'LAAP_API_BASE': LAAP_API_BASE},
        'timeout': 30,
        'connect_timeout': 10,
        'keepalive_interval': 60,
    }
}

skills = cfg.setdefault('skills', {})
preload = skills.setdefault('preload', [])
if 'laap-bridge' not in preload:
    preload.append('laap-bridge')

with open(HERMES_CONFIG, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=120)

print(f'Hermes config updated: {HERMES_CONFIG}')
print(f'  LAAP root: {LAAP_ROOT}')
print(f'  LAAP API base: {LAAP_API_BASE}')
print(f'  Hermes venv python: {HERMES_VENV_PYTHON}')
