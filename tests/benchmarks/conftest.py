import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "aris_brain"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))