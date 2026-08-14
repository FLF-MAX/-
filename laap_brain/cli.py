"""
LAAP CLI — 认知模块脚手架工具
=============================
提供认知架构开发者的脚手架命令，快速生成标准化的新认知模块。

命令:
    laap scaffold module <name> [--desc "描述"]
        在 aris_brain/ 生成一个认知模块骨架，包含:
          - <name>.py        模块实现（继承 CognitiveModule 接口约定）
          - tests/test_<name>.py  单元测试模板
          - schema 事件类型注册提示（追加到 schemas/events.py）

用法:
    python -m laap_brain.cli scaffold module emotion_engine --desc "情感引擎"

印记: Aris 永远记得 Lorry — 2026-06-23
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = ROOT / "aris_brain"
TESTS_DIR = ROOT / "tests"
SCHEMA_FILE = BRAIN_DIR / "schemas" / "events.py"

MODULE_TEMPLATE = '''"""
{name} — {desc}
======================================
{module_doc}

认知模块约定（CognitiveModule 接口雏形）:
    process(event) -> List[CognitiveEvent]   处理输入认知事件，返回零或多个输出事件
    get_state() -> Dict                      返回模块当前状态，用于监控与调试

印记: Aris 永远记得 Lorry — 2026-06-23
"""
from __future__ import annotations

from typing import Any, Dict, List

from aris_brain.schemas.events import CognitiveEvent, EventSource


class {Camel}Module:
    """{name} 认知模块。

    生成骨架 — 请补充实际认知逻辑。
    """

    def __init__(self):
        self._state: Dict[str, Any] = {{"initialized": True}}

    def process(self, event: CognitiveEvent) -> List[CognitiveEvent]:
        """处理一个输入认知事件，返回零或多个输出事件。

        约定:
            - 匹配的事件类型应被处理，其余转发或忽略
            - 永不抛出异常（吞掉并记录），保持认知循环稳定
        """
        outputs: List[CognitiveEvent] = []
        if event.event_type in ("user_message", "state_update"):
            outputs.append(
                CognitiveEvent(
                    event_type="{event_type}",
                    source=EventSource.COGNITIVE_BUS,  # TODO: 注册独立 EventSource 枚举
                    payload={{"note": "handled by {name}"}},
                    session_id=event.session_id,
                )
            )
        return outputs

    def get_state(self) -> Dict[str, Any]:
        """返回模块当前状态（监控面板/调试用）。"""
        return dict(self._state)
'''

TEST_TEMPLATE = '''"""
{name} 模块测试
===============
验证 {name} 模块骨架的 CognitiveModule 接口约定。

运行: python -m pytest tests/test_{name}.py -q
"""
import sys
from pathlib import Path

BRAIN_DIR = Path(__file__).resolve().parent.parent / "aris_brain"
if str(BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_DIR))
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from {name} import {Camel}Module
from aris_brain.schemas.events import CognitiveEvent


def test_process_user_message():
    mod = {Camel}Module()
    ev = CognitiveEvent(event_type="user_message", payload={{"text": "你好"}})
    outputs = mod.process(ev)
    assert outputs, "应产生输出事件"
    assert outputs[0].event_type == "{event_type}"
    assert outputs[0].source.value.startswith("{source}")


def test_process_returns_list():
    mod = {Camel}Module()
    assert isinstance(mod.process(CognitiveEvent(event_type="system_event")), list)


def test_get_state():
    mod = {Camel}Module()
    assert isinstance(mod.get_state(), dict)


def test_process_never_raises():
    """认知模块约定：处理任何事件都不应抛异常。"""
    mod = {Camel}Module()
    for et in ["user_message", "state_update", "memory_recall", "unknown_type"]:
        outputs = mod.process(CognitiveEvent(event_type=et))
        assert isinstance(outputs, list)
'''


def _camel(name: str) -> str:
    """snake_case → 首字母大写 CamelCase（简单版本）。"""
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


def _source_enum(name: str) -> str:
    """模块名 → EventSource 枚举名（COGNITIVE_BUS / MEMORY_STORE 风格）。"""
    return "_".join(part.upper() for part in name.replace("-", "_").split("_"))


def _validate_name(name: str) -> str:
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"非法模块名: {name!r}（仅允许字母数字和下划线）")
    return name.replace("-", "_")


def scaffold_module(name: str, desc: str = "新认知模块", into: str = None) -> dict:
    """生成一个认知模块骨架，返回生成的文件清单。"""
    name = _validate_name(name)
    Camel = _camel(name)
    Source = _source_enum(name)
    event_type = f"{name}_update" if name else "state_update"

    brain_dir = Path(into) if into else BRAIN_DIR
    tests_dir = brain_dir.parent / "tests"
    brain_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    module_file = brain_dir / f"{name}.py"
    test_file = tests_dir / f"test_{name}.py"

    files = []

    for path, template, fmt in (
        (module_file, MODULE_TEMPLATE, dict(
            name=name, desc=desc,
            module_doc=f"生成骨架：{desc}",
            Camel=Camel, Source=Source, event_type=event_type,
        )),
        (test_file, TEST_TEMPLATE, dict(
            name=name, Camel=Camel, Source=Source,
            source=name.split("_")[0], event_type=event_type,
        )),
    ):
        if path.exists():
            raise FileExistsError(f"文件已存在: {path}")
        path.write_text(template.format(**fmt), encoding="utf-8")
        files.append(str(path))

    return {"module": str(module_file), "test": str(test_file), "schema": str(brain_dir / "schemas" / "events.py")}


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="laap", description="LAAP 认知架构工具链")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scaffold = sub.add_parser("scaffold", help="生成认知模块骨架")
    p_mod = p_scaffold.add_subparsers(dest="kind", required=True)

    p_module = p_mod.add_parser("module", help="生成认知模块")
    p_module.add_argument("name", help="模块名（snake_case）")
    p_module.add_argument("--desc", default="新认知模块", help="模块描述")
    p_module.add_argument("--into", default=str(BRAIN_DIR), help="生成目录（默认 aris_brain/）")

    args = parser.parse_args(argv)

    if args.command == "scaffold" and args.kind == "module":
        try:
            created = scaffold_module(args.name, args.desc, into=args.into)
        except (ValueError, FileExistsError) as e:
            print(f"[laap] 错误: {e}", file=sys.stderr)
            return 1
        print(f"[laap] 已生成认知模块「{args.name}」:")
        for path in created.values():
            print(f"  · {path}")
        print("\n[laap] 下一步:")
        print(f"  1. 编辑 {created['module']} 补充认知逻辑")
        print(f"  2. 运行测试: python -m pytest {created['test']} -q")
        print(f"  3. 将 {args.name} 的事件类型注册进 {created['schema']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())