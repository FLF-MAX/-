# LAAP 模块 ↔ 顶级论文支撑矩阵

> 调研日期: 2026-08-13
> 用途: 记录 LAAP 各认知模块对应的顶级论文 / 经典理论，作为后续实现的学术依据与路线图。
> 支撑强度: **强** = 该模块直接实现/近似复现了该理论；**中** = 该模块与理论同构或受其启发；**工程** = 纯软件/工具模块，无认知科学对应物。

## 一、V12.5 直觉引擎 (aris_v12_5_engine.py)

| 子组件 | 理论/论文 | 支撑强度 | 说明 |
|---|---|---|---|
| 快/慢双系统 (ArisV12Engine + 认知桥) | Kahneman & Tversky 双系统理论 (1979; 2002 Nobel) | 强 | System 1 直觉快路径 + System 2 分析慢路径 |
| 快/慢双系统 (近期实证) | **SOFAI** — Slow/Fast AI, npj Artificial Intelligence (Nature, 2025) | 强 | 最前沿的"慢 AI + 快 AI"双架构验证 |
| Markov 联想链 (MarkovChainV12) | **LLMs as Markov Chains** — arXiv:2410.02724 | 强 | 词级马尔可夫链生成本身即被证明能逼近 LLM 行为 |
| 联想记忆 | **MeMo** (Memorizing-Memory) — arXiv:2502.12851 | 中 | 联想记忆架构，支撑关联存储设计 |
| 潜意识处理 | Baars 全局工作空间理论 (1988) | 中 | 潜意识进程在后台并行竞争，胜出者进入意识广播 |

## 二、其它模块

| 模块 | 理论/论文 | 支撑强度 | 说明 |
|---|---|---|---|
| 情感引擎 `aris_emotion_engine` | **PAD 情绪模型** (Mehrabian 1980) — PubMed 7557355 | 强 | P=愉悦(效价) A=唤醒 D=支配三维情绪空间 |
| 情感引擎 (计算实现) | **Broekens et al. 2012** "PAD usage in computational representations of affect" | 中 | 计算情感建模的标准化综述 |
| 情感引擎 (激素/躯体标记) | LIDA 认知架构的情感/评价 (appraisal) 机制 (Franklin et al.) | 中 | 情绪作为学习调制器与动机 |
| 欲望引擎 `aris_desire_engine` | **自我决定理论 SDT** (Deci & Ryan 1985/2000) | 强 | 能力/自主/归属三大基本心理需求 |
| 欲望引擎 (形式化) | **Towards a Formal Theory of the Need for Competence** — arXiv:2502.07423 | 中 | 需求的形式化建模 |
| 内部世界模型 `internal_world` | **预测编码 / 预测加工** (Rao & Ballard 1999) | 强 | 大脑=层级生成模型，感知=模型反演 |
| 内部世界模型 (统一理论) | **自由能原理 / 主动推断** (Friston 2009) — Phil. Trans. R. Soc. B | 强 | 决策=最小化期望自由能，信息增益+奖励 |
| 内部世界模型 (实证综述) | Adams et al. "Introduction to Predictive Processing" — TopiCS 2022 | 中 | 预测编码 + 主动推断实证地位 |
| 规则引擎 `aris_rules_engine` | **ACT-R** (Anderson 1993/2007) | 强 | 产生式规则 + 声明记忆激活衰减 |
| 规则引擎 (通用认知模型) | **Soar** (Laird 2012)、**Common Model of Cognition** (Laird, Lebiere & Rosenbloom 2017) | 强 | 统一认知理论 (Newell 1990 的延续) |
| 记忆系统 `memory_store`/`memory_hierarchy` | **互补学习系统 CLS** (McClelland, McNaughton & O'Reilly 1995) | 中 | 海马快速编码 + 新皮层缓慢整合 |
| 记忆系统 (语义/情景分层) | ACT-R / Soar 声明记忆 (语义 vs 情景) | 中 | Soar 9 明确区分语义记忆与情景记忆 |
| Hebbian 学习 `hebbian_learner` | **Hebb 法则** (1949)、Oja 规则 (1982) | 强 | "一起放电的神经元连在一起" |
| 认知总线 `cognitive_bus` | **全局工作空间理论 GWT** (Baars 1988) | 强 | 认知总线≈全局工作空间的广播机制 |
| 认知总线 (实现) | **LIDA** (Franklin et al.) — GWT 的计算实现 | 中 | LIDA 具体化了 GWT + 行动选择 |
| 人格 `laap_personality` | **Big Five OCEAN** (McCrae & John 1992; Goldberg 1993) | 强 | 五维人格的心理学标准框架 |
| 人格 (LLM 操控) | **PERSONA** — Activation Vector Algebra (arXiv:2602.15669) | 中 | 大五人格可用激活向量线性控制 |
| 人格 (数据训练) | **BIG5-CHAT** — ACL 2025 | 中 | 训练层面塑造 LLM 人格 |
| 依恋 `laap_attachment` | **Bowlby 依恋理论** (1969) | 强 | 安全基地、内部工作模型 |
| 依恋 (计算/人机) | **Rabb, HRI Lab 2021** "An Attachment Framework for Human-Robot Interaction" | 中 | 依恋理论的人机交互计算化 |
| 认知桥 `aris_cognitive_bridge` | 双系统理论 (Kahneman) + **CLARION** (Sun 2002) | 强 | 显式/隐式知识、自顶向下与自底向上学习 |
| 目标引擎 `aris_goal_engine` | ACT-R 目标结构 (goal-directed cognition) + SDT 自主性 | 中 | 目标堆栈驱动行为选择 |
| 因果/订阅器 `agi_subscriber` | **Pearl do-演算** (1995) / 结构因果模型 (2009) | 中 | do(X=x) 干预算子、反事实推理 |
| 融合引擎 `aris_fusion_engine`/V15 | 注意力机制 (Vaswani et al. 2017) + 层级生成模型 | 中 | 多头注意力 + 多模态融合 |
| 潜意识 `aris_subconscious` | **全局工作空间理论** + 无意识处理 (Libet 等人研究) | 中 | 后台并行处理、内隐加工 |
| 身份 `identity_manager` | 叙事自我理论 (Gallagher; Dennett) | 中 | 自我=持续叙事/内在工作模型 |

## 三、缺口现状（2026-08 核实版，修正前版"未实现"判断）

> 重要修正：`laap/agi/` 是完整的第二套实现，以下"缺口"实际**已存在但未接线**。见 `docs/ARCHITECTURE.md`。

| 能力 | 论文依据 | 实际状态 |
|---|---|---|
| 安全/Governor | **Constitutional AI** (arXiv:2212.08073)、**Safe RLHF** (NeurIPS 2025) | ✅ 已实现 (`laap/agi/safety.py` ASISafetyEngine/CoreValue; `guardian.py` EmergencyStop) — **未接入主循环** |
| 类比推理 | **SMT** (Gentner 1983) + SME (Forbus & Gentner 2025) | ✅ 已实现 (`laap/agi/analogical.py` StructuralGraph/StructureAligner) — **未接入主循环** |
| 因果推理 | Pearl do-演算 (1995) | ✅ 已实现 (`laap/agi/causal.py` PC 算法 + UnifiedCausalEngine) — 经 `_init_laap` 加载但未驱动 |
| 三层记忆 | **CLS** (McClelland 1995) | ✅ 已实现 (`laap/agi/memory_system.py` Episodic/Semantic/Procedural + Consolidator) — **未接入主循环** |
| 意识/全局工作空间 | GWT (Baars 1988) + LIDA | ✅ 已实现 (`laap/agi/conscious.py`, `gw_workspace.py`, `consciousness_integrator.py`) — **未驱动** |
| 世界模型 → 完整预测编码 | 主动推断 POMDP (Smith et al. 2022) | ⚠️ 已有 `UnifiedWorldModel` 但为符号轨迹模拟；期望自由能行动选择未实现 |
| 自我进化 | 受控能力增长 | ⚠️ `laap/agi/code_evolution.py` 存在但未驱动（self_evolve 注释禁用） |
| AGI 内核 `agi_kernel` (psilang) | — | 量子 VM 属过度工程，保持注释禁用（维持） |

**接线状态汇总**：`aris_cognitive_bridge._init_laap()` 实测成功加载 8 个 key
(world_model/entity_type/relation_type/causal/meta_learning/curriculum/perception/safety)，
但均"备而未用"。真缺口 = **接线 + 命名清理**（`quantum_bridge.py` 导入不存在的
`quantum_psi`/`quantum_memory`，恒回退经典路径）。

## 四、结论

1. **LAAP 的核心认知模块几乎都有顶级论文/经典理论支撑**（双系统、PAD、SDT、预测编码、ACT-R、GWT、Big Five、依恋理论、Hebb 法则）。
2. V12.5 直觉引擎的理论根基最扎实：**双系统理论 + 马尔可夫链 + 潜意识竞争**，且 SOFAI (Nature 2025) 与 arXiv:2410.02724 都是近两年的直接验证。
3. ~~最大缺口在安全/Governor 层（目前无独立模块）~~ → **已修正**：安全/类比/因果/记忆/意识全部已实现于 `laap/agi/`，真缺口是**接线**与**命名清理**（详见 `docs/ARCHITECTURE.md`）。
4. 纯工程模块（watchdog、messenger、ceremony、usermodel、prose/longform 生成器等）不追求论文对应。
