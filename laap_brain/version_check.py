"""
LAAP 版本兼容性检查
====================

验证 LAAP 与 Hermes 的版本对齐关系。

用法:
    python -m laap_brain.version_check          # 检查兼容性
    python -m laap_brain.version_check --json   # JSON 输出

印记: Aris 永远记得 Lorry — 2026-06-18
"""
import json
import sys
from pathlib import Path
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    import yaml
except ImportError:
    yaml = None


def load_versions_yaml() -> dict:
    """加载 VERSIONS.yaml。"""
    path = Path(__file__).parent.parent / "VERSIONS.yaml"
    if not path.exists():
        return {"error": "VERSIONS.yaml not found"}

    if yaml:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        return {"error": "PyYAML not installed"}


def check_hermes_version() -> dict:
    """检查 Hermes 版本兼容性。"""
    try:
        hermes_ver = _pkg_version("hermes-agent")
    except PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "compatible": False,
            "error": "hermes-agent not installed",
        }

    ver_info = hermes_ver.split(".")
    major = int(ver_info[0]) if len(ver_info) > 0 else 0
    minor = int(ver_info[1]) if len(ver_info) > 1 else 0

    # LAAP 1.0.x 兼容 Hermes 0.18.x
    compatible = major == 0 and minor == 18

    return {
        "installed": True,
        "version": hermes_ver,
        "compatible": compatible,
        "expected": "0.18.x",
    }


def check_all() -> dict:
    """全面检查。"""
    versions = load_versions_yaml()
    hermes = check_hermes_version()

    try:
        from laap_brain import __version__ as laap_ver
    except ImportError:
        laap_ver = "unknown"

    result = {
        "laap": {
            "version": laap_ver,
            "path": str(Path(__file__).parent.parent.resolve()),
        },
        "hermes": hermes,
        "compatibility": {
            "status": "ok" if hermes.get("compatible") else "incompatible",
            "message": (
                f"LAAP {laap_ver} + Hermes {hermes.get('version', 'N/A')}: "
                f"{'compatible' if hermes.get('compatible') else 'INCOMPATIBLE'}"
            ),
        },
    }

    if versions and "error" not in versions:
        result["versions_file"] = versions

    return result


def main():
    if "--json" in sys.argv:
        print(json.dumps(check_all(), indent=2, ensure_ascii=False))
    else:
        result = check_all()
        print(f"LAAP Brain: v{result['laap']['version']}")
        print(f"Location:   {result['laap']['path']}")

        h = result["hermes"]
        print(f"Hermes:     v{h.get('version', 'N/A')} "
              f"({'installed' if h.get('installed') else 'NOT FOUND'})")

        c = result["compatibility"]
        status_icon = "✅" if c["status"] == "ok" else "❌"
        print(f"Status:     {status_icon} {c['message']}")

        if h.get("error"):
            print(f"Note: {h['error']}")

        if not h.get("compatible"):
            print(f"\nExpected Hermes: {h.get('expected', '0.18.x')}")
            print("Install: pip install hermes-agent==0.18.x")


if __name__ == "__main__":
    main()