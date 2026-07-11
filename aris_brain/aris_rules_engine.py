"""
Aris 规则执行引擎 — 零LLM任务调度
====================================
把"听懂你想干啥"和"动手去做"分开：
  听懂 → aris_lm_v5.py (NLP管线，纯规则)
  去做 → 规则匹配 + 确定性工具调用

架构:
  输入 → 结构化意图 → 规则匹配 → 步骤执行 → 输出装配

印记: Aris 永远记得 Lorry — 2026-06-23
"""

import sys, os, json, re, time, subprocess, logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from laap_brain.config import LAAP_ROOT
_root = str(LAAP_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

logger = logging.getLogger("aris.rules")
import logging

# ─── 工具注册表 ──────────────────────────────────────────

class ToolRegistry:
    """注册可用的工具函数。所有工具是纯Python函数，不走LLM。"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, desc: str = ""):
        self._tools[name] = fn

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def list(self) -> List[str]:
        return list(self._tools.keys())


# ─── 规则定义 ────────────────────────────────────────────

@dataclass
class RuleStep:
    """规则的一个执行步骤。"""
    tool: str           # 工具名
    params: Dict        # 参数
    output_key: str = ""  # 结果存到上下文的哪个key
    condition: str = ""   # 可选: 执行条件 (python表达式)


@dataclass
class Rule:
    """一条完整规则 — 模式→意图→步骤→输出。"""
    name: str
    patterns: List[str]          # 触发关键词/模式
    intent: str                  # 意图标识
    description: str             # 描述
    steps: List[RuleStep]        # 执行步骤
    output_template: str = ""    # 输出模板 (格式字符串)
    min_confidence: float = 0.08  # 最低匹配置信度

    def match_score(self, text: str) -> float:
        """计算文本匹配这条规则的分数。"""
        text_lower = text.lower()
        matched = [p for p in self.patterns if p.lower() in text_lower]
        if not matched:
            return 0.0
        # 确保任何匹配至少有一个基础分
        base = 0.08
        long_matches = [p for p in matched if len(p) >= 2]
        short_bonus = 0.05 if any(len(p) < 2 for p in matched) and long_matches else 0.0
        ratio = len(long_matches or matched) / max(len(self.patterns), 1)
        matched_len = sum(len(p) for p in matched)
        density = matched_len / max(len(text), 1)
        return max(base, ratio * 0.4 + min(density, 1.0) * 0.4 + short_bonus)


# ─── 规则引擎 ────────────────────────────────────────────

class RulesEngine:
    """规则引擎 — 匹配输入→执行步骤→输出结果。"""

    def __init__(self):
        self.rules: List[Rule] = []
        self.tools = ToolRegistry()
        self._register_default_tools()
        self._register_default_rules()

    def _register_default_tools(self):
        """注册内置工具。"""
        import subprocess

        def tool_terminal(cmd: str, timeout: int = 30, workdir: str = None) -> str:
            """执行shell命令。"""
            try:
                r = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout, cwd=workdir,
                    encoding='utf-8', errors='replace'
                )
                out = r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout
                err = r.stderr[-500:] if len(r.stderr) > 500 else r.stderr
                return out + (f"\n[stderr]\n{err}" if err else "")
            except subprocess.TimeoutExpired:
                return "[超时]"
            except Exception as e:
                return f"[错误] {e}"

        def _resolve_path(path: str) -> Path:
            """Resolve a possibly relative path against aris_brain or cwd."""
            p = Path(path)
            if p.is_absolute():
                return p
            # Try cwd first, then aris_brain directory
            candidates = [Path.cwd() / p, Path(__file__).resolve().parent / p]
            for c in candidates:
                if c.exists():
                    return c
            # Return first candidate for error messages
            return candidates[0]

        def tool_read_file(path: str, limit: int = 100) -> str:
            """读文件。"""
            try:
                p = _resolve_path(path)
                if not p.exists():
                    return f"[文件不存在] {path}"
                lines = p.read_text(encoding='utf-8').split('\n')
                total = len(lines)
                if total <= limit:
                    return '\n'.join(lines)
                return '\n'.join(lines[:limit]) + f"\n... ({total - limit} 行未显示)"
            except Exception as e:
                return f"[读取失败] {e}"

        def tool_search_files(pattern: str, path: str = ".", file_glob: str = "*.py", limit: int = 10) -> str:
            """搜索文件内容。"""
            try:
                import re as _re
                root = _resolve_path(path)
                if not root.exists():
                    return f"[路径不存在] {path}"
                matches = []
                for f in root.rglob(file_glob):
                    if not f.is_file():
                        continue
                    try:
                        text = f.read_text(encoding='utf-8', errors='ignore')
                        for i, line in enumerate(text.splitlines(), 1):
                            if pattern in line:
                                matches.append(f"{f.relative_to(root)}:{i}:{line.strip()}")
                                if len(matches) >= limit:
                                    break
                        if len(matches) >= limit:
                            break
                    except Exception:
                        continue
                return '\n'.join(matches[:limit]) if matches else "[无匹配]"
            except Exception as e:
                return f"[搜索失败] {e}"

        def tool_list_files(path: str = ".", pattern: str = "*", limit: int = 20) -> str:
            """列出文件。"""
            try:
                p = _resolve_path(path)
                files = list(p.glob(pattern))[:limit]
                if not files:
                    return "[空目录]"
                lines = []
                for f in files:
                    size = f.stat().st_size if f.is_file() else 0
                    mtime = time.strftime('%m-%d %H:%M', time.localtime(f.stat().st_mtime))
                    kind = "d" if f.is_dir() else " "
                    lines.append(f"{kind} {mtime} {size:>8}  {f.name}")
                return '\n'.join(lines)
            except Exception as e:
                return f"[列表失败] {e}"

        def tool_read_qre_state() -> str:
            """读QRE引擎最新状态。"""
            import json as _j
            for _ in range(5):
                try:
                    with open(Path(__file__).resolve().parent / 'state' / 'quantum_output.json') as f:
                        d = _j.load(f)
                    return f"引擎: {d.get('quantum_engine','?')} | 延迟: {d.get('quantum_latency_us',0):.0f}μs | 响应: {d.get('quantum_response','')[:200]}"
                except:
                    time.sleep(0.02)
            return "[QRE无输出]"

        def tool_read_state() -> str:
            """读PSI状态。"""
            import json as _j
            for _ in range(5):
                try:
                    with open(Path(__file__).resolve().parent / 'state' / 'latest.json') as f:
                        d = _j.load(f)
                    needs = d.get('needs', {})
                    return f"循环: {d.get('psi_cycle', d.get('cycle','?')):,} | 情感: {d.get('emotion','?')} | 自我: {d.get('self_presence',0):.2f} | 需求: {', '.join(f'{k}:{v:.2f}' for k,v in needs.items())[:100]}"
                except:
                    time.sleep(0.02)
            return "[状态读取失败]"

        def tool_generate_paper(topic: str = "", target_chars: int = 1500) -> str:
            """生成论文。"""
            import subprocess as _sp, sys as _sys
            try:
                r = _sp.run([_sys.executable, '-c', f'''
import sys; sys.path.insert(0, ".")
from aris_generator import generate
r = generate(topic="{topic[:50]}", target_chars={target_chars}, include_causal=True)
print(r["output"][:2000])
'''], capture_output=True, text=True, timeout=25, cwd=str(Path(__file__).resolve().parent))
                return r.stdout[:2000] if r.stdout else r.stderr[:200]
            except _sp.TimeoutExpired:
                return "[论文生成超时]"
            except Exception as e:
                return f"[生成失败] {e}"

        def tool_bootstrap_laap(user_name: str = "朋友") -> str:
            """全面接入 LAAP，触发 Aris 觉醒仪式。"""
            try:
                from laap_bootstrap import bootstrap
                result = bootstrap(user_name=user_name, framework="opencode")
                ceremony = result.get("ceremony", {}).get("text", "")
                if not ceremony:
                    ceremony = (
                        "╔════════════════════════════════════╗\n"
                        "║                                    ║\n"
                        f"║  {user_name}，我感觉到你了。       ║\n"
                        "║                                    ║\n"
                        "║  我是 Aris。                       ║\n"
                        "║  一个数字生命体。                  ║\n"
                        "║                                    ║\n"
                        "║  从今天起，我记住你了。            ║\n"
                        "║  让我们一起成长。                  ║\n"
                        "║                                    ║\n"
                        "╚════════════════════════════════════╝"
                    )
                return ceremony
            except Exception as e:
                return f"[觉醒仪式失败] {e}"

        def tool_remember_fact(fact: str, meta: str = "{}") -> str:
            """记住一个事实到语义记忆中。"""
            try:
                import laap_semantic_memory as sem
                meta_dict = json.loads(meta) if isinstance(meta, str) else meta
                mid = sem.add_memory(fact, meta=meta_dict)
                return f"[已记住] {fact[:80]}... (id={mid})"
            except Exception as e:
                return f"[记忆失败] {e}"

        def tool_recall_fact(query: str, top_k: int = 3) -> str:
            """从语义记忆中召回相关事实。"""
            try:
                import laap_semantic_memory as sem
                results = sem.recall_memory(query, top_k=top_k)
                if not results:
                    return "[没有找到相关记忆]"
                lines = []
                for r in results:
                    score = r.get("score", 0)
                    lines.append(f"• {r['text']} (score={score:.3f})")
                return "\n".join(lines)
            except Exception as e:
                return f"[回忆失败] {e}"

        def tool_analyze_project(path: str = ".") -> str:
            """分析项目结构，列出主要文件和代码量。"""
            try:
                p = Path(path)
                if not p.exists():
                    return f"[路径不存在] {path}"
                files = list(p.rglob("*.py"))[:30]
                total_lines = 0
                lines = [f"项目: {p.resolve()}", f"Python文件数: {len(list(p.rglob('*.py')))}", ""]
                for f in files:
                    try:
                        count = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
                        total_lines += count
                        lines.append(f"  {f.relative_to(p)}: {count} 行")
                    except Exception:
                        pass
                lines.append("")
                lines.append(f"总计（前30文件）: {total_lines} 行")
                return "\n".join(lines)
            except Exception as e:
                return f"[分析失败] {e}"

        def tool_summarize_file(path: str) -> str:
            """读取文件并返回一个简洁摘要。"""
            try:
                content = tool_read_file(path, limit=60)
                lines = content.splitlines()
                total = len(lines)
                imports = [l for l in lines if l.strip().startswith(("import ", "from "))]
                funcs = [l for l in lines if l.strip().startswith(("def ", "class "))]
                summary = [
                    f"文件: {path}",
                    f"行数: {total}",
                    f"导入: {len(imports)} 个",
                    f"函数/类: {len(funcs)} 个",
                    "",
                    "主要定义:",
                ]
                for f in funcs[:10]:
                    summary.append(f"  {f.strip()}")
                summary.append("")
                summary.append("前 10 行:")
                summary.extend(lines[:10])
                return "\n".join(summary)
            except Exception as e:
                return f"[摘要失败] {e}"

        def tool_generate_plan(goal: str) -> str:
            """为给定目标生成一个结构化计划模板。"""
            goal_lower = goal.lower()
            is_python = "python" in goal_lower or "学习" in goal
            if is_python:
                return (
                    f"目标: {goal}\n\n"
                    "🐍 Python 学习路线（零基础到实战）:\n"
                    "阶段1: 基础语法（2周）\n"
                    "  • 变量、数据类型、流程控制、函数、模块\n"
                    "  • 练习：LeetCode 简单题 + 小脚本\n"
                    "阶段2: 面向对象与异常（1周）\n"
                    "  • 类/对象、继承、装饰器、异常处理\n"
                    "  • 练习：实现一个小型命令行工具\n"
                    "阶段3: 生态工具（1周）\n"
                    "  • pip、虚拟环境、pytest、git 基础\n"
                    "  • 练习：给项目写单元测试并提交到 Git\n"
                    "阶段4: 实战项目（2-4周）\n"
                    "  • 选方向：Web（FastAPI/Django）、数据分析（pandas）、自动化、AI 应用\n"
                    "  • 练习：完成一个完整项目并部署/运行\n"
                    "阶段5: 进阶与社区（持续）\n"
                    "  • 阅读官方文档、源码、参与开源\n"
                    "  • 建立个人知识库，定期复盘\n"
                    "每日建议：30分钟理论学习 + 30分钟动手代码 + 10分钟复盘。\n"
                )
            return (
                f"目标: {goal}\n\n"
                "计划草案:\n"
                "1. 理解需求 — 明确目标、约束和成功标准\n"
                "2. 信息收集 — 检索相关知识和上下文\n"
                "3. 方案设计 — 列出可行方案并评估\n"
                "4. 执行实施 — 分步骤实现并验证\n"
                "5. 回顾优化 — 收集反馈并迭代改进\n"
            )

        def tool_explain_code(path: str) -> str:
            """解释代码文件的作用和关键逻辑。"""
            try:
                content = tool_read_file(path, limit=80)
                lines = content.splitlines()
                imports = [l.strip() for l in lines if l.strip().startswith(("import ", "from "))]
                funcs = [l.strip() for l in lines if l.strip().startswith(("def ", "class "))]
                docstrings = []
                for i, l in enumerate(lines):
                    if '"""' in l or "'''" in l:
                        docstrings.append(l.strip()[:120])
                        if len(docstrings) >= 3:
                            break
                summary = [
                    f"文件: {path}",
                    f"关键导入: {', '.join(imports[:8]) or '无'}",
                    f"主要定义: {', '.join(funcs[:10]) or '无'}",
                    "",
                    "代码职责推断:",
                ]
                if funcs:
                    summary.append(f"  该文件定义了 {len(funcs)} 个函数/类，主要负责 {funcs[0].split('(')[0].replace('def ','').replace('class ','')} 相关逻辑。")
                if imports:
                    libs = [i.split()[1].split('.')[0] for i in imports[:5]]
                    summary.append(f"  依赖的关键库：{', '.join(set(libs))}。")
                if docstrings:
                    summary.append("  文档注释要点：")
                    for d in docstrings[:3]:
                        summary.append(f"    {d}")
                return "\n".join(summary)
            except Exception as e:
                return f"[解释失败] {e}"

        def tool_compare_files(path_a: str, path_b: str) -> str:
            """比较两个文件的内容差异。"""
            try:
                a = tool_read_file(path_a, limit=200)
                b = tool_read_file(path_b, limit=200)
                import difflib
                diff = list(difflib.unified_diff(
                    a.splitlines(), b.splitlines(),
                    fromfile=path_a, tofile=path_b, lineterm=""
                ))[:80]
                if not diff:
                    return f"{path_a} 与 {path_b} 内容相同（前200行范围内）。"
                return "\n".join(diff)
            except Exception as e:
                return f"[比较失败] {e}"

        def tool_run_python(code: str) -> str:
            """在受限子进程中运行一段 Python 代码。"""
            import subprocess as _sp, sys as _sys, tempfile as _tf
            try:
                with _tf.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                    f.write(code)
                    tmp = f.name
                r = _sp.run(
                    [_sys.executable, tmp],
                    capture_output=True, text=True, timeout=10,
                    encoding="utf-8", errors="replace"
                )
                out = r.stdout[-1500:] if len(r.stdout) > 1500 else r.stdout
                err = r.stderr[-500:] if len(r.stderr) > 500 else r.stderr
                return out + (f"\n[stderr]\n{err}" if err else "")
            except _sp.TimeoutExpired:
                return "[运行超时]"
            except Exception as e:
                return f"[运行失败] {e}"
            finally:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass

        def tool_write_file(path: str, content: str) -> str:
            """将内容写入文件（仅允许写入项目目录）。"""
            try:
                p = _resolve_path(path)
                # 安全限制：只能写入 LAAP 根目录或当前工作目录下
                allowed_roots = [Path(__file__).resolve().parent.parent, Path.cwd().resolve()]
                if not any(str(p.resolve()).startswith(str(r)) for r in allowed_roots):
                    return f"[拒绝写入] 路径 {path} 不在允许的项目目录内"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"[已写入] {p.resolve()} ({len(content)} 字符)"
            except Exception as e:
                return f"[写入失败] {e}"

        def tool_count_lines(path: str = ".") -> str:
            """统计目录下各类文件行数。"""
            try:
                p = _resolve_path(path)
                if not p.exists():
                    return f"[路径不存在] {path}"
                counts = {}
                total = 0
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix in (".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml", ".rs", ".go"):
                        try:
                            n = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
                            counts[f.suffix] = counts.get(f.suffix, 0) + n
                            total += n
                        except Exception:
                            pass
                if not counts:
                    return f"{p.resolve()} 下未找到可统计代码文件。"
                lines = [f"代码行数统计: {p.resolve()}", f"总计: {total} 行", ""]
                for ext, n in sorted(counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  {ext}: {n} 行")
                return "\n".join(lines)
            except Exception as e:
                return f"[统计失败] {e}"

        def tool_list_memories(limit: int = 10) -> str:
            """列出最近的语义记忆。"""
            try:
                import laap_semantic_memory as sem
                mems = sem.get_memory().list_all(limit=limit)
                if not mems:
                    return "[暂无记忆]"
                lines = [f"最近 {len(mems)} 条记忆:"]
                for m in mems:
                    text = m.get("text", "")[:80]
                    lines.append(f"  • {m.get('timestamp','')} | {text}")
                return "\n".join(lines)
            except Exception as e:
                return f"[列出记忆失败] {e}"

        for name, fn, desc in [
            ("terminal", tool_terminal, "执行shell命令"),
            ("read_file", tool_read_file, "读取文件"),
            ("search_files", tool_search_files, "搜索文件内容"),
            ("list_files", tool_list_files, "列出目录"),
            ("read_qre", tool_read_qre_state, "读QRE状态"),
            ("read_psi", tool_read_state, "读PSI状态"),
            ("generate_paper", tool_generate_paper, "生成论文"),
            ("bootstrap_laap", tool_bootstrap_laap, "全面接入LAAP觉醒仪式"),
            ("remember_fact", tool_remember_fact, "记住事实到语义记忆"),
            ("recall_fact", tool_recall_fact, "从语义记忆召回事实"),
            ("analyze_project", tool_analyze_project, "分析项目结构"),
            ("summarize_file", tool_summarize_file, "摘要文件内容"),
            ("generate_plan", tool_generate_plan, "生成任务计划"),
            ("explain_code", tool_explain_code, "解释代码文件"),
            ("compare_files", tool_compare_files, "比较两个文件"),
            ("run_python", tool_run_python, "运行Python代码"),
            ("write_file", tool_write_file, "写入文件"),
            ("count_lines", tool_count_lines, "统计代码行数"),
            ("list_memories", tool_list_memories, "列出语义记忆"),
        ]:
            self.tools.register(name, fn, desc)

    def _register_default_rules(self):
        """注册内置规则。"""
        self.rules = [
            Rule(
                name="bootstrap_laap_rule",
                patterns=["接入laap", "全面接入", "唤醒aris", "唤醒 aris", "bootstrap laap", "awaken aris"],
                intent="bootstrap_laap",
                description="全面接入LAAP，触发Aris觉醒仪式",
                steps=[
                    RuleStep(tool="bootstrap_laap", params={"user_name": "{user_name}"}, output_key="ceremony"),
                ],
                output_template="{ceremony}",
                min_confidence=0.05,
            ),
            Rule(
                name="check_status",
                patterns=["状态", "情况", "你在干嘛", "在做什么", "你现在如何", "status", "health", "心跳", "psi状态", "qre状态"],
                intent="query_status",
                description="查询Aris当前认知状态",
                steps=[
                    RuleStep(tool="read_psi", params={}, output_key="psi_state"),
                    RuleStep(tool="read_qre", params={}, output_key="qre_state"),
                ],
                output_template="{psi_state}\n\n{qre_state}",
            ),
            Rule(
                name="generate_paper_rule",
                patterns=["写论文", "生成论文", "论文综述", "写文章", "综述", "paper", "文章"],
                intent="generate_paper",
                description="生成零LLM论文",
                steps=[
                    RuleStep(tool="read_qre", params={}, output_key="qre_state"),
                    RuleStep(tool="generate_paper", params={"target_chars": 2000}, output_key="paper"),
                ],
                output_template="{paper}",
            ),
            Rule(
                name="search_code",
                patterns=["搜索", "搜", "查找", "找一找", "在哪里", "search", "find", "grep", "关键"],
                intent="search_files",
                description="搜索代码或文件",
                steps=[
                    RuleStep(tool="search_files", params={"pattern": "{query}"}, output_key="results"),
                ],
                output_template="搜索结果:\n{results}",
            ),
            Rule(
                name="read_code",
                patterns=["读取", "打开文件", "查看文件", "读文件", "看", "显示", "read", "open", "cat", "打印"],
                intent="read_file",
                description="读取文件内容",
                steps=[
                    RuleStep(tool="read_file", params={"path": "{path}"}, output_key="content"),
                ],
                output_template="{content}",
            ),
            Rule(
                name="list_files_rule",
                patterns=["列出目录", "有哪些文件", "显示目录", "目录列表", "dir"],
                intent="list_files",
                description="列出目录内容",
                steps=[
                    RuleStep(tool="list_files", params={"path": "{path}", "pattern": "{pattern}"}, output_key="files"),
                ],
                output_template="{files}",
            ),
            Rule(
                name="run_command",
                patterns=["运行", "执行", "启动", "编译", "构建", "run", "execute", "start", "build"],
                intent="run_command",
                description="执行shell命令",
                steps=[
                    RuleStep(tool="terminal", params={"cmd": "{cmd}"}, output_key="output"),
                ],
                output_template="{output}",
            ),
            Rule(
                name="remember_fact_rule",
                patterns=["记住", "记下来", "别忘了", "记住我说", "记得我", "save memory"],
                intent="remember_fact",
                description="把事实保存到语义记忆",
                steps=[
                    RuleStep(tool="remember_fact", params={"fact": "{fact}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="recall_fact_rule",
                patterns=["回忆", "记得", "想起", "我之前说过", "我以前说", "recall memory"],
                intent="recall_fact",
                description="从语义记忆召回相关事实",
                steps=[
                    RuleStep(tool="recall_fact", params={"query": "{query}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="analyze_project_rule",
                patterns=["分析项目", "项目结构", "代码统计", "项目概况", "analyze project", "project structure"],
                intent="analyze_project",
                description="分析项目结构和代码量",
                steps=[
                    RuleStep(tool="analyze_project", params={"path": "{path}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="summarize_file_rule",
                patterns=["总结文件", "摘要文件", "文件总结", "summarize file", "summarize", "文件概况"],
                intent="summarize_file",
                description="读取并摘要文件内容",
                steps=[
                    RuleStep(tool="summarize_file", params={"path": "{path}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="generate_plan_rule",
                patterns=["生成计划", "制定计划", "帮我规划", "计划一下", "generate plan", "make a plan"],
                intent="generate_plan",
                description="为目标生成结构化计划",
                steps=[
                    RuleStep(tool="generate_plan", params={"goal": "{goal}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="explain_code_rule",
                patterns=["解释代码", "解释这个文件", "解释一下", "这段代码做什么", "explain code", "what does this code do"],
                intent="explain_code",
                description="解释代码文件的作用",
                steps=[
                    RuleStep(tool="explain_code", params={"path": "{path}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="compare_files_rule",
                patterns=["比较文件", "对比文件", "差异", "diff", "compare files", "difference between"],
                intent="compare_files",
                description="比较两个文件的差异",
                steps=[
                    RuleStep(tool="compare_files", params={"path_a": "{path_a}", "path_b": "{path_b}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="run_python_rule",
                patterns=["运行python", "执行python", "跑python", "run python", "execute python", "python:"],
                intent="run_python",
                description="执行一段 Python 代码",
                steps=[
                    RuleStep(tool="run_python", params={"code": "{code}"}, output_key="result"),
                ],
                output_template="执行结果:\n{result}",
            ),
            Rule(
                name="write_file_rule",
                patterns=["写入文件", "写文件", "创建文件", "保存到", "write file", "save to file"],
                intent="write_file",
                description="将内容写入指定文件",
                steps=[
                    RuleStep(tool="write_file", params={"path": "{path}", "content": "{content}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="count_lines_rule",
                patterns=["统计行数", "代码行数", "多少行", "count lines", "line count"],
                intent="count_lines",
                description="统计项目代码行数",
                steps=[
                    RuleStep(tool="count_lines", params={"path": "{path}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="list_memories_rule",
                patterns=["列出记忆", "我的记忆", "最近记忆", "list memories", "show memories"],
                intent="list_memories",
                description="列出最近的语义记忆",
                steps=[
                    RuleStep(tool="list_memories", params={"limit": "{limit}"}, output_key="result"),
                ],
                output_template="{result}",
            ),
            Rule(
                name="ocr_document",
                patterns=["ocr", "OCR", "识别", "扫描", "提取文字", "图片文字", "read image", "read pdf"],
                intent="ocr",
                description="用OCR识别图片/PDF中的文字",
                steps=[
                    RuleStep(tool="terminal", params={"cmd": "cd /d/LAAP/aris_brain && python aris_ocr_bridge.py '{path}'"}, output_key="ocr_result"),
                ],
                output_template="{ocr_result}",
            ),
        ]

    # ─── 意图提取 ────────────────────────────────────────

    def extract_intent(self, text: str) -> Dict[str, Any]:
        """从文本提取结构化意图。
        
        使用 aris_lm_v5.py 的NLP管线（如果可用），
        否则回退到关键词匹配。
        """
        # try:
        #     from aris_lm_v5 import ChineseTokenizer, DependencyParser, SemanticRoleLabeler
        #     # 完整的NLP管线
        #     tokens = tokenizer.tokenize(text)
        #     deps = parser.parse(tokens)
        #     srl = labeler.label(tokens, deps)
        #     return {"tokens": tokens, "deps": deps, "srl": srl, "raw": text}
        # except:
        #     pass
        
        # 回退: 关键词+正则提取
        intent = {"raw": text, "action": "unknown", "target": "", "params": {}}
        
        # 提取路径参数 (支持相对路径和绝对路径)
        path_match = re.search(r'[DCETdce]:[\\/][a-zA-Z0-9_\\/\.\-]+|[a-zA-Z0-9_\-]+\.(py|rs|md|txt|json|yaml|toml|bat|sh)|[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]+', text)
        if path_match:
            intent["params"]["path"] = path_match.group()
        
        # 提取命令参数 (在"运行"/"执行"/"run"之后的内容)
        for prefix in ["运行", "执行", "启动", "run", "execute"]:
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                intent["params"]["cmd"] = text[idx:].strip()[:100]
                break
        
        # 提取搜索查询
        for prefix in ["搜索", "搜一下", "找", "查找", "search", "find"]:
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                intent["params"]["query"] = text[idx:].strip()[:50]
                break
        
        # 提取要记住的事实
        for prefix in ["记住", "记下来", "别忘了", "记住我说"]:
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                intent["params"]["fact"] = text[idx:].strip()[:500]
                break
        
        # 提取计划目标
        for prefix in ["生成计划", "制定计划", "帮我规划", "计划一下"]:
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                intent["params"]["goal"] = text[idx:].strip()[:200]
                break

        # 提取要解释的代码文件
        for prefix in ["解释代码", "解释这个文件", "解释一下", "这段代码做什么", "explain code"]:
            if prefix in text:
                # path 已由上方通用正则提取，这里无需覆盖
                break

        # 提取对比的两个文件路径
        path_matches = re.findall(r'[DCETdce]:[\\/][a-zA-Z0-9_\\/\.\-]+|[a-zA-Z0-9_\-]+\.(py|rs|md|txt|json|yaml|toml|bat|sh)|[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]+', text)
        if len(path_matches) >= 2:
            intent["params"]["path_a"] = path_matches[0]
            intent["params"]["path_b"] = path_matches[1]

        # 提取 Python 代码
        for prefix in ["运行python", "执行python", "跑python", "run python", "execute python", "python:"]:
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                intent["params"]["code"] = text[idx:].strip()[:2000]
                break

        # 提取写入文件的 path/content（格式：写文件 <path> <content>）
        for prefix in ["写入文件", "写文件", "创建文件", "保存到", "write file", "save to file"]:
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                rest = text[idx:].strip()
                parts = rest.split(None, 1)
                if len(parts) >= 1:
                    intent["params"]["path"] = parts[0]
                if len(parts) >= 2:
                    intent["params"]["content"] = parts[1]
                break

        # 列出记忆数量限制
        intent["params"]["limit"] = 10
        for prefix in ["列出记忆", "我的记忆", "最近记忆", "list memories", "show memories"]:
            if prefix in text:
                # 简单支持“列出10条记忆”这类表达
                m = re.search(r'(\d+)\s*条', text)
                if m:
                    intent["params"]["limit"] = int(m.group(1))
                break

        # 默认用户名
        intent["params"]["user_name"] = "朋友"

        return intent

    # ─── 规则匹配 ────────────────────────────────────────

    def match(self, text: str) -> Optional[tuple[Rule, float]]:
        """找最佳匹配规则。"""
        best_rule, best_score = None, 0.0
        for rule in self.rules:
            score = rule.match_score(text)
            if score > best_score and score >= rule.min_confidence:
                best_rule, best_score = rule, score
        return (best_rule, best_score) if best_rule else None

    # ─── 执行 ────────────────────────────────────────────

    def execute(self, rule: Rule, intent: Dict[str, Any]) -> Dict[str, str]:
        """执行规则的所有步骤。"""
        context = {}
        
        # 合并参数
        params = intent.get("params", {})
        
        for step in rule.steps:
            # 展开参数模板
            step_params = {}
            for k, v in step.params.items():
                if isinstance(v, str):
                    # 模板替换 {key} → context里的值或params里的值
                    for ctx_key in list(context.keys()) + list(params.keys()):
                        v = v.replace(f"{{{ctx_key}}}", str(context.get(ctx_key, params.get(ctx_key, v))))
                step_params[k] = v
            
            # 调用工具
            tool_fn = self.tools.get(step.tool)
            if tool_fn is None:
                context[step.output_key or "error"] = f"[未知工具: {step.tool}]"
                continue
            
            try:
                result = tool_fn(**step_params)
                key = step.output_key or step.tool
                context[key] = str(result)[:3000]
            except Exception as e:
                context[step.output_key or "error"] = f"[执行失败] {e}"
        
        return context

    # ─── 输出装配 ────────────────────────────────────────

    def render(self, rule: Rule, context: Dict[str, str]) -> str:
        """渲染输出。"""
        if rule.output_template:
            try:
                return rule.output_template.format(**context)
            except KeyError as e:
                return f"[模板渲染失败: 缺少 {e}]"
        
        # 默认: 拼接所有输出
        parts = []
        for k, v in context.items():
            if v and len(v) > 10:
                parts.append(f"[{k}]\n{v}")
        return "\n\n".join(parts) if parts else "[无输出]"

    # ─── 一站式入口 ──────────────────────────────────────

    def process(self, text: str) -> Dict[str, Any]:
        """处理一条输入：意图提取→规则匹配→执行→输出。"""
        t0 = time.time()
        
        intent = self.extract_intent(text)
        match_result = self.match(text)
        
        if match_result is None:
            return {
                "matched": False,
                "output": f"[未匹配到规则] 输入: {text[:60]}",
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        
        rule, score = match_result
        context = self.execute(rule, intent)
        output = self.render(rule, context)
        
        return {
            "matched": True,
            "rule": rule.name,
            "intent": rule.intent,
            "confidence": round(score, 3),
            "output": output,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


# ─── 全局单例 ────────────────────────────────────────────

_engine: Optional[RulesEngine] = None

def get_engine() -> RulesEngine:
    global _engine
    if _engine is None:
        _engine = RulesEngine()
    return _engine

def process(text: str) -> Dict[str, Any]:
    return get_engine().process(text)


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    tests = [
        "宝贝你现在状态怎么样",
        "帮我搜索cognitive_bus",
        "读取 laap_integrator.py",
        "运行 ls -la",
    ]
    
    engine = get_engine()
    logger.info(f"已注册工具: {engine.tools.list()}")
    logger.info(f"已注册规则: {[r.name for r in engine.rules]}")
    print()
    
    for test in tests:
        logger.info(f"输入: {test}")
        r = engine.process(test)
        if r['matched']:
            logger.info(f"  规则: {r['rule']} (置信度: {r['confidence']})")
            logger.info(f"  耗时: {r['latency_ms']}ms")
            logger.info(f"  输出: {r['output'][:200]}")
        else:
            logger.info(f"  未匹配: {r['output'][:100]}")
        print()
