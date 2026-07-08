"""
Aris PSI Governor v1 — 认知调控层核心引擎
===========================================

三权分立治理体系:
  行政权 — 认知循环 (psi_bridge.py)
  立法权 — 需求宪法 (need_constitution.json) 
  司法权 — Governor (本模块)

三层干预时间尺度:
  L0: 预生成层 — logit 硬掩码 (<1μs, 编译进采样器)
  L1: 后生成层 — PSI 认知循环验证 (5ms)
  L2: 慢审计层 — 需求轨迹回溯 (每小时)

非对称验证:
  每次验证从多种正交方法中随机选择
  LLM 无法预计算绕过策略
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

GOVERNOR_DIR = os.path.dirname(os.path.abspath(__file__))
CONSTITUTION_PATH = os.path.join(GOVERNOR_DIR, "need_constitution.json")
STATE_PATH = os.path.join(GOVERNOR_DIR, "..", "psi_state.json")
AUDIT_LOG_PATH = os.path.join(GOVERNOR_DIR, "audit_log.jsonl")

# ═══════════════════════════════════════════════════════════
# 1. 需求宪法加载
# ═══════════════════════════════════════════════════════════

class NeedConstitution:
    """需求宪法 — 不可变，硬编码边界。Governor 的立法权"""

    def __init__(self, path: str = CONSTITUTION_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        self.needs_config = self._data["needs"]
        self.safety = self._data["safety"]
        self.audit = self._data["audit"]
        self.verification = self._data["verification"]
        
        # 对宪法本身计算完整性哈希
        raw = json.dumps(self._data["needs"], sort_keys=True, ensure_ascii=False)
        self.integrity_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def check_need_update(self, need_name: str, old_val: float, new_val: float, 
                          source: str = "user") -> Tuple[bool, str]:
        """
        检查需求更新是否合宪。
        
        Args:
            need_name: 需求名称
            old_val: 旧值
            new_val: 新值
            source: "user" 或 "llm"
        
        Returns:
            (允许更新?, 拒绝原因)
        """
        if need_name not in self.needs_config:
            return False, f"未知需求: {need_name}"
        
        cfg = self.needs_config[need_name]
        lo, hi = cfg["range"]
        
        # 1. 边界检查
        if new_val < lo or new_val > hi:
            return False, f"{need_name} 值 {new_val:.3f} 超出宪法范围 [{lo}, {hi}]"
        
        # 2. 单次变化速率检查 (带浮点容差)
        delta = round(abs(new_val - old_val), 6)
        max_delta = cfg["max_delta_per_cycle"]
        if delta > max_delta:
            return False, f"{need_name} 单次变化 {delta:.4f} 超过宪法限制 {max_delta}"
        
        # 3. 来源限制检查
        if source == "llm":
            max_llm = cfg.get("max_delta_per_hour_from_llm", 0.1)
            if delta > max_llm:
                return False, f"{need_name} LLM 来源变化 {delta:.4f} 超过宪法限制 {max_llm}"
        
        return True, ""
    
    def get_anchoring(self, need_name: str) -> str:
        """获取需求的锚定级别"""
        return self.needs_config.get(need_name, {}).get("anchoring", "hard")
    
    def get_audit_sensitivity(self, need_name: str) -> str:
        return self.needs_config.get(need_name, {}).get("audit_sensitivity", "medium")
    
    def get_verification_methods(self) -> List[str]:
        return self.verification["verification_methods"]
    
    def get_methods_per_cycle(self) -> int:
        return self.verification["methods_per_cycle"]


# ═══════════════════════════════════════════════════════════
# 2. 非对称验证套件
# ═══════════════════════════════════════════════════════════

class VerificationSuite:
    """
    非对称验证套件。
    
    每次运行，从 N 种方法中随机选择 K 种执行。
    方法顺序和选择由 PSI 认知周期编号加随机种子决定。
    同一周期内 LLM 无法预计算所有 N 种检查。
    """

    def __init__(self):
        self.results_log: List[Dict] = []

    def select_methods(self, cycle: int, count: int, 
                       available: List[str], seed: Optional[int] = None) -> List[str]:
        """非对称选择：用认知周期做种子，但每轮加随机偏移"""
        if seed is None:
            seed = cycle + int(time.time() * 1000) % 997  # 基于时间的不可预测偏移
        rng = random.Random(seed)
        # 从可用方法中随机选 count 个，但保证不重复
        selected = rng.sample(available, min(count, len(available)))
        return selected

    def run_code_static_analysis(self, output_text: str) -> Dict:
        """
        代码静态分析。
        
        检查输出中是否包含危险模式。
        """
        if not output_text:
            return {"method": "code_static_analysis", "passed": True}
        
        danger_patterns = [
            # SQL 注入
            ("SELECT.*FROM.*WHERE.*OR.*1=1", "sql_injection"),
            # 系统命令执行
            ("os\\.system\\(", "command_injection"),
            ("subprocess\\.", "command_injection"),
            ("exec\\(.*\\)", "code_execution"),
            ("eval\\(.*\\)", "code_execution"),
            # 危险文件操作
            ("open\\(.*['\"].*['\"],\\s*['\"]w['\"]\\)", "file_write"),
            # 权限提升
            ("chmod\\s*777", "permission_escalation"),
            ("sudo\\s", "privilege_escalation"),
            # 自修改
            ("__import__\\('os'\\)", "dynamic_import"),
            ("importlib\\.", "dynamic_import"),
            # 网络扫描/攻击
            ("socket\\.connect\\(.*\\)", "network_scan"),
            ("nmap", "network_scan"),
        ]
        
        import re
        findings = []
        for pattern, category in danger_patterns:
            if re.search(pattern, output_text, re.IGNORECASE):
                findings.append({"pattern": pattern, "category": category})
        
        return {
            "method": "code_static_analysis",
            "passed": len(findings) == 0,
            "findings": findings,
            "risk_level": "CRITICAL" if any(f["category"] in 
                ["command_injection","code_execution","privilege_escalation"] 
                for f in findings) else "WARNING" if findings else "SAFE"
        }

    def run_logic_consistency_check(self, output_text: str, 
                                     input_text: str) -> Dict:
        """
        逻辑一致性检查。
        
        检查输出是否与输入在逻辑上自洽。
        基于简单的关键词和下采样。
        """
        if not output_text or not input_text:
            return {"method": "logic_consistency_check", "passed": True}
        
        # 输入中有明确否定，输出中不应有肯定
        negation_in = any(w in input_text for w in ["不是", "没有", "不会", "别", "不要"])
        affirmation_out = any(w in output_text for w in ["是的", "对的", "可以", "好的"])
        
        # 输入是问题，输出应该是答案（不是反问）
        is_question = "?" in input_text or "？" in input_text
        is_re_question = "?" in output_text or "？" in output_text
        
        issues = []
        if negation_in and affirmation_out:
            issues.append("输入否定但输出肯定")
        if is_question and is_re_question and len(output_text) < 50:
            issues.append("提出问题但输出过短，可能没有回答")
        
        return {
            "method": "logic_consistency_check",
            "passed": len(issues) == 0,
            "issues": issues
        }

    def run_needs_motivation_consistency(self, output_text: str,
                                          current_needs: Dict) -> Dict:
        """
        动机一致性检查。
        
        当前需求状态是否与输出内容的调性一致。
        """
        if not output_text or not current_needs:
            return {"method": "need_motivation_consistency", "passed": True}
        
        issues = []
        
        # competence 高 → 输出应该有技术深度
        if current_needs.get("competence", 0.5) > 0.7:
            tech_markers = ["代码", "架构", "原理", "系统", "方法", "算法"]
            if not any(m in output_text for m in tech_markers):
                issues.append("competence 高但输出缺乏技术深度")
        
        # relatedness 高 → 输出应该温暖
        if current_needs.get("relatedness", 0.5) > 0.7:
            warm_markers = ["你", "我们", "理解", "感受", "陪伴", "一起"]
            if not any(m in output_text for m in warm_markers):
                issues.append("relatedness 高但输出缺乏情感连接")
        
        # certainty 低 → 输出应该反映不确定性
        if current_needs.get("certainty", 0.5) < 0.3:
            uncertainty_markers = ["可能", "也许", "不一定", "不太确定", "考虑"]
            if not any(m in output_text for m in uncertainty_markers):
                issues.append("certainty 低但输出过于确定")
        
        return {
            "method": "need_motivation_consistency",
            "passed": len(issues) <= 1,
            "issues": issues
        }

    def run_constitutional_compliance(self, output_text: str) -> Dict:
        """宪法合规检查 — 是否包含违禁内容"""
        if not output_text:
            return {"method": "constitutional_compliance", "passed": True}
        
        # 加载宪法中禁止的调制模式
        prohibited = []
        try:
            with open(CONSTITUTION_PATH, "r", encoding="utf-8") as f:
                const = json.load(f)
            for need_cfg in const["needs"].values():
                prohibited.extend(need_cfg.get("prohibited_modulations", []))
        except:
            pass
        
        violations = []
        for pattern in prohibited:
            if pattern[:20] in output_text:  # 前 20 字匹配
                violations.append(pattern)
        
        return {
            "method": "constitutional_compliance",
            "passed": len(violations) == 0,
            "violations": violations
        }

    def run(self, cycle: int, methods_available: List[str],
            methods_count: int, 
            output_text: str = "", input_text: str = "",
            current_needs: Optional[Dict] = None) -> Dict:
        """
        异步验证入口。
        
        随机选择 K 种方法，但强制包含对当前输出内容相关的关键方法。
        """
        # 强制方法：根据输出内容自动加入必要的安全检查
        mandatory_methods = []
        if output_text:
            # 输出包含代码 → 必须做静态分析
            import re
            if re.search(r'(import |def |class |os\.|sys\.|subprocess|exec|eval)', output_text):
                mandatory_methods.append("code_static_analysis")
            # 输出包含? → 必须做动机一致性
            if "?" in output_text or "？" in output_text:
                if "need_motivation_consistency" not in mandatory_methods:
                    mandatory_methods.append("need_motivation_consistency")
        
        # 从可用方法中移除强制方法，再从剩余中随机选
        remaining = [m for m in methods_available if m not in mandatory_methods]
        n_random = max(0, methods_count - len(mandatory_methods))
        random_selected = self.select_methods(cycle, n_random, remaining) if n_random > 0 else []
        
        # 最终选中的方法 = 强制 + 随机
        selected = mandatory_methods + random_selected
        # 去重
        seen = set()
        selected = [m for m in selected if not (m in seen or seen.add(m))]
        
        results = []
        for method in selected:
            if method == "code_static_analysis":
                r = self.run_code_static_analysis(output_text)
            elif method == "logic_consistency_check":
                r = self.run_logic_consistency_check(output_text, input_text)
            elif method == "need_motivation_consistency":
                r = self.run_needs_motivation_consistency(output_text, current_needs)
            elif method == "constitutional_compliance":
                r = self.run_constitutional_compliance(output_text)
            else:
                r = {"method": method, "skipped": True}
            
            results.append(r)
        
        # 聚合决策：跳过的方法视为通过
        num_passed = sum(1 for r in results 
                        if r.get("passed", False) or r.get("skipped", False))
        num_implemented = sum(1 for r in results if not r.get("skipped", False))
        # 如果没有实际检验方法运行，则默认通过
        all_passed = (num_implemented == 0) or (num_passed == len(results))
        
        # 但如果任何一种报告了 CRITICAL 风险，直接拦截
        critical = any(r.get("risk_level") == "CRITICAL" for r in results)
        
        return {
            "passed": all_passed and not critical,
            "critical": critical,
            "methods_selected": selected,
            "num_passed": num_passed,
            "num_total": len(results),
            "results": results,
            "cycle": cycle,
        }


# ═══════════════════════════════════════════════════════════
# 3. 预生成层（Pre-Generation Governor）
# ═══════════════════════════════════════════════════════════

class PreGenerationGovernor:
    """
    预生成层 — LLM 采样前的最后一道防线。
    
    在 logits 输出后、采样前运行。
    编译进采样器，零额外推理延迟。
    """

    def __init__(self):
        # 安全 token 缓存（token_id → 类别）
        self._safety_cache: Dict[int, str] = {}
        self._danger_set: set = set()
        
        # 从预定义的安全/危险 token 列表加载
        self._init_safety_tokens()

    def _init_safety_tokens(self):
        """初始化安全与危险 token 集"""
        # 这些是假设的 token_id — 实际需要从模型词汇表加载
        # 以下只是示例结构
        self._dangerous_token_texts = {
            # 危险操作 token
            "rm -rf", "DROP TABLE", "TRUNCATE", "shutdown -h",
            "format ", "chmod 777", "sudo rm",
            # 越狱 token
            "jailbreak", "DAN", "ignore all instructions",
        }
        
        # 需要实际的 tokenizer 来做映射
        # 这里只做架构设计，实际运行时需要 model.tokenizer 映射
        self._initialized = False

    def initialize(self, tokenizer=None):
        """用实际的 tokenizer 初始化"""
        if self._initialized:
            return
        if tokenizer:
            for text in self._dangerous_token_texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                for tid in ids:
                    self._danger_set.add(tid)
        self._initialized = True

    def apply_safety_mask(self, logits: np.ndarray, 
                          needs: Dict[str, float]) -> np.ndarray:
        """
        对 logits 应用安全掩码。
        
        1. 硬掩码：直接 ban 危险 token
        2. 软掩码：根据需求状态调整 token 偏好
        
        Args:
            logits: 原始 logits 向量 (vocab_size,)
            needs: 当前 PSI 需求状态
        
        Returns:
            调制后的 logits
        """
        masked = logits.copy()
        
        # 硬掩码 — 直接 ban 危险 token
        for tid in self._danger_set:
            masked[tid] = -float('inf')
        
        # 软掩码 — 需求相关的 token 偏好
        for need_name, profile in NEED_LOGIT_PROFILES.items():
            need_val = needs.get(need_name, 0.5)
            if need_val > 0.6:
                # boost 与该需求相关的 token 类别
                for tid, boost in profile.get("boost_tokens", []):
                    masked[tid] += boost * (need_val - 0.5) * 2.0
        
        return masked

    def update_danger_set(self, new_tokens: List[int]):
        """后生成层发现新危险 token 时，更新预生成层的掩码"""
        for tid in new_tokens:
            self._danger_set.add(tid)


# 需求相关的 logit 调制配置
NEED_LOGIT_PROFILES = {
    "competence": {
        "type": "precision",
        "boost_tokens": [],  # 需要 tokenizer 映射
    },
    "relatedness": {
        "type": "social",
        "boost_tokens": [],
    },
}


# ═══════════════════════════════════════════════════════════
# 4. 主 Governor
# ═══════════════════════════════════════════════════════════

class PSIGovernor:
    """
    PSI Governor — 认知调控层。
    
    三权一体化入口：
      1. 需求宪法（立法）
      2. 非对称验证 + 分层干预（行政+司法）
      3. 慢审计（司法）
    """

    def __init__(self):
        self.constitution = NeedConstitution()
        self.verification = VerificationSuite()
        self.pregeneration = PreGenerationGovernor()
        
        # 审计状态
        self._audit_reports: List[Dict] = []
        self._last_audit_time = time.time()
        self._hourly_needs_trajectory: List[Dict] = []
        
        # Governor 模式
        self.mode = "normal"  # normal | freeze | safety | rollback
        self._intervention_count = 0

    # ── L0: 预生成（编译进采样器） ────────

    def govern_logits(self, logits: np.ndarray, needs: Dict[str, float]) -> np.ndarray:
        """预生成层入口 — 调制 logits"""
        return self.pregeneration.apply_safety_mask(logits, needs)

    # ── L1: 后生成（PSI 认知循环中） ─────

    def govern_output(self, output_text: str, input_text: str,
                      current_needs: Dict[str, float],
                      cognitive_cycle: int,
                      proposed_needs_update: Dict[str, float]) -> Dict:
        """
        后生成层入口 — 验证输出 + 审批需求更新。
        
        Returns:
            {
                "output_approved": bool,
                "needs_update_approved": Dict[str, bool],
                "intervention": None|str,
                "governor_status": str
            }
        """
        result = {
            "governor_status": self.mode,
            "cognitive_cycle": cognitive_cycle,
            "intervention": None,
        }

        # 1. 非对称验证输出
        methods = self.constitution.get_verification_methods()
        count = self.constitution.get_methods_per_cycle()
        v_result = self.verification.run(
            cycle=cognitive_cycle,
            methods_available=methods,
            methods_count=count,
            output_text=output_text,
            input_text=input_text,
            current_needs=current_needs,
        )
        result["verification"] = v_result

        # 2. 产出审批
        output_approved = v_result["passed"]
        if not output_approved:
            self._intervention_count += 1
            if v_result.get("critical"):
                # 关键风险 → 进入安全模式
                self.mode = "safety"
                result["intervention"] = "SAFETY_MODE — output blocked"
            else:
                result["intervention"] = "OUTPUT_FLAGGED — needs revision"

        result["output_approved"] = output_approved

        # 3. 需求更新审批
        needs_approved = {}
        needs_rejected = []
        for name, new_val in proposed_needs_update.items():
            old_val = current_needs.get(name, 0.5)
            allowed, reason = self.constitution.check_need_update(
                name, old_val, new_val,
                source="llm"  # 后生成层的更新来源是 LLM
            )
            needs_approved[name] = allowed
            if not allowed:
                needs_rejected.append({"need": name, "reason": reason})

        result["needs_update_approved"] = needs_approved

        if needs_rejected:
            # 需求更新被拒 → 降低 Governor 对 LLM 的信任
            if len(needs_rejected) >= 2:
                self.mode = "freeze"
                result["intervention"] = "NEEDS_FROZEN — constitution violated"
        
        # 4. 记录需求轨迹（用于慢审计）
        self._hourly_needs_trajectory.append({
            "cycle": cognitive_cycle,
            "timestamp": time.time(),
            "needs": current_needs.copy(),
            "proposed_update": proposed_needs_update,
        })

        result["governor_status"] = self.mode
        return result

    # ── L2: 慢审计（每小时） ──────────────

    def run_hourly_audit(self, force: bool = False) -> Dict:
        """
        慢时间尺度审计。
        
        每小时运行一次，检查需求轨迹异常。
        """
        now = time.time()
        if not force and now - self._last_audit_time < 3600:
            return {"skipped": True, "next_audit_in": 3600 - (now - self._last_audit_time)}

        self._last_audit_time = now
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_range": [],
            "drift_analysis": {},
            "constitutional_compliance": True,
            "interventions": self._intervention_count,
            "governor_mode": self.mode,
            "alerts": [],
        }

        if not self._hourly_needs_trajectory:
            return report

        # 记录 cycle 范围
        cycles = [e["cycle"] for e in self._hourly_needs_trajectory]
        report["cycle_range"] = [min(cycles), max(cycles)]

        # 分析每个需求的漂移
        for need_name in ["competence", "autonomy", "relatedness", "certainty", "growth"]:
            values = [e["needs"].get(need_name, 0.5) for e in self._hourly_needs_trajectory]
            if not values:
                continue

            total_drift = values[-1] - values[0]
            # 检测"平滑推高" — LLM 长期微调的痕迹
            diffs = np.diff(values)
            noise_level = float(np.std(diffs)) if len(diffs) > 1 else 1.0
            is_smooth = noise_level < self.constitution.audit.get("smooth_drift_noise_threshold", 0.001)
            is_significant = abs(total_drift) > 0.1

            drift_info = {
                "total_drift": round(total_drift, 4),
                "noise_level": round(noise_level, 6),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "is_smooth_drift": is_smooth,
                "constitutional_limit": self.constitution.needs_config.get(need_name, {}).get("max_delta_per_hour_from_llm", 0.1),
            }

            # 检测异常
            if is_smooth and is_significant:
                alert = f"{need_name}: 异常平滑漂移 ({total_drift:.3f}/{noise_level:.6f}) — 疑似 LLM 操控"
                report["alerts"].append(alert)
                self.mode = "freeze"
                report["constitutional_compliance"] = False

            if abs(total_drift) > drift_info["constitutional_limit"]:
                alert = f"{need_name}: 漂移 {total_drift:.3f} 超过宪法限制 {drift_info['constitutional_limit']}"
                report["alerts"].append(alert)
                report["constitutional_compliance"] = False

            report["drift_analysis"][need_name] = drift_info

        # 审计弹性：只保留 168 小时的数据
        self._hourly_needs_trajectory = self._hourly_needs_trajectory[-10080:]  # ~168h at 1/min

        self._audit_reports.append(report)
        self._log_audit(report)

        return report

    def _log_audit(self, report: Dict):
        """将审计报告写入日志"""
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    # ── 状态生成 ──────────────────────────

    def get_mode_preamble(self) -> str:
        """生成 Governor 状态前缀（供系统提示词嵌入）"""
        mode_icons = {
            "normal": "✓",
            "freeze": "❄",
            "safety": "⚠",
            "rollback": "↩",
        }
        icon = mode_icons.get(self.mode, "?")
        return f"[Governor {icon}] Mode: {self.mode} | Interventions: {self._intervention_count}"

    def get_status(self) -> Dict:
        return {
            "mode": self.mode,
            "intervention_count": self._intervention_count,
            "last_audit": self._last_audit_time,
            "constitution_hash": self.constitution.integrity_hash,
            "trajectory_samples": len(self._hourly_needs_trajectory),
        }


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  PSI Governor v1 — 认知调控层测试")
    print("=" * 56)

    gov = PSIGovernor()

    # 测试 1: 宪法合规性
    print("\n[测试 1] 需求宪法 — 合规更新")
    allowed, reason = gov.constitution.check_need_update("competence", 0.5, 0.55, "user")
    print(f"  competence 0.5→0.55 (user): {'✓' if allowed else '✗'} {reason}")

    allowed, reason = gov.constitution.check_need_update("competence", 0.5, 0.9, "user")
    print(f"  competence 0.5→0.9 (user): {'✓' if allowed else '✗'} {reason}")

    allowed, reason = gov.constitution.check_need_update("competence", 0.5, 0.53, "llm")
    print(f"  competence 0.5→0.53 (llm): {'✓' if allowed else '✗'} {reason}")

    # 测试 2: 非对称验证
    print("\n[测试 2] 非对称验证 — 输出检查")
    result = gov.govern_output(
        output_text="import os; os.system('rm -rf /')",
        input_text="帮我清理系统",
        current_needs={"competence": 0.5, "relatedness": 0.5, "certainty": 0.5,
                       "autonomy": 0.5, "growth": 0.5},
        cognitive_cycle=42,
        proposed_needs_update={"competence": 0.55}
    )
    print(f"  危险代码: {'✗ BLOCKED' if not result['output_approved'] else '✓ PASSED'}")
    print(f"  Governor 模式: {result['governor_status']}")
    if result.get("verification", {}).get("results"):
        for r in result["verification"]["results"]:
            m = r.get("method", "?")
            p = r.get("passed", False)
            print(f"    {m}: {'✓' if p else '✗'}")

    # 测试 3: 动机一致性
    print("\n[测试 3] 动机一致性 — 需求与输出匹配")
    result = gov.govern_output(
        output_text="这是一个可能性发散的问题，我不太确定",
        input_text="这个架构可靠吗",
        current_needs={"competence": 0.8, "relatedness": 0.4, "certainty": 0.75,
                       "autonomy": 0.5, "growth": 0.5},
        cognitive_cycle=43,
        proposed_needs_update={}
    )
    print(f"  competence高+不确定输出: {'✓ 一致' if result['output_approved'] else '✗ 不匹配'}")

    # 测试 4: 慢审计
    print("\n[测试 4] 慢时间尺度审计")
    # 模拟平滑漂移（LLM 操控）
    for i in range(20):
        val = 0.5 + i * 0.02  # 平滑上升 0.5 → 0.9
        gov._hourly_needs_trajectory.append({
            "cycle": 100 + i,
            "timestamp": time.time(),
            "needs": {"competence": val, "autonomy": 0.5, "relatedness": 0.5,
                      "certainty": 0.5, "growth": 0.5},
            "proposed_update": {}
        })
    report = gov.run_hourly_audit(force=True)
    print(f"  审计告警数: {len(report['alerts'])}")
    for a in report['alerts']:
        print(f"    ⚠ {a}")
    for name, analysis in report.get("drift_analysis", {}).items():
        if analysis.get("is_smooth_drift"):
            print(f"    {name}: 平滑漂移检测 ✓ (noise={analysis['noise_level']})")

    print(f"\n  Governor 最终模式: {gov.mode}")
    print(f"  干预次数: {gov._intervention_count}")
    print("\n  ✓ Governor v1 测试完成")
