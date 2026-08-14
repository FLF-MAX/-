"""LAAP 一键启动器：加载 .env -> 设置 PYTHONPATH -> 启动 Brain API"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(ROOT)
sys.path.insert(0, str(ROOT))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# 确保关键变量
os.environ.setdefault("LAAP_PORT", "11546")

# 启动主服务
sys.argv = ["laap_brain_api.py", "--port", os.environ.get("LAAP_PORT", "11546")]
from aris_brain.laap_brain_api import main
main()
