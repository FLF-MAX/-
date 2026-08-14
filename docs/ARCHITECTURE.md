# LAAP 架构现状（2026-08 核实版）

> 本文档基于代码实测,记录本项目**真实**的架构与接线状态。
> 目标:把"作者理想"与"实际运行"之间的差异摆到桌面上,为下一步决策提供依据。

## 1. 三套并行的实现

本项目存在三套相互重叠、但接线状态完全不同的代码:

| 层 | 位置 | 规模 | 接线状态 |
|---|---|---|---|
| **运行时主脑** | `aris_brain/` | 47 个 `.py` | ✅ 已接线,主循环实际运行 |
| **完整 AGI 层** | `laap/agi/` | 52 个 `.py` | ⚠️ 已加载(bridge),未在主循环驱动 |
| **独立工程版** | `laap_v2/` | 独立包 | ⚠️ 自成一体,110 tests(用 `G:\Hermes\.venv`) |

根项目测试(`G:\laap\.venv`):32 passed(25 + 7,SDT 需求层新增)。

## 2. 运行时接线图(实测)

```
aris_brain/ (主循环, 薄层)
  └─ laap_integrator.load_all() → 14 模块 (规则/情感/欲望/潜意识/PSI...)
  └─ aris_cognitive_bridge._init_laap()  →  加载 laap/agi 6 大模块(实测成功):
        ✅ UnifiedWorldModel        ✅ UnifiedCausalEngine (PC 算法)
        ✅ MetaLearningEngine       ✅ CurriculumEngine
        ✅ UnifiedPerceptionEngine  ✅ ASISafetyEngine
```

`aris_cognitive_bridge` 的 `_laap_available=True` 且 `_laap_modules` 含全部 8 个 key
(world_model / entity_type / relation_type / causal / meta_learning / curriculum /
perception / safety)。**加载是真实成功的**，接线实测分级：

- ✅ **已驱动于主循环**：`world_model`(P3)、`causal`(P2)
- ⚠️ **弱驱动**：`meta_learning`(每轮 `_learn` 采样)、`entity_type`/`relation_type`(枚举类型装载,无业务消费)
- ⚠️ **备而未用**：`perception`、`curriculum`(bridge 实例)`；safety 能力实际经
  `laap_brain_api._safety_gate` 输出拦截面旁路激活（见 §5）

## 3. laap/agi 层实际拥有的能力(未被主循环驱动)

这是项目真正的宝藏层,此前评估误判为"缺失",实际全部存在:

| 领域 | 模块 | 实现 |
|---|---|---|
| 记忆 | `memory_system.py` | MemoryTrace / 三层记忆(Episodic/Semantic/Procedural)/ MemoryConsolidator(Rescorla-Wagner 强化 + 梦境巩固) |
| 统一记忆 | `unified_memory.py` | UnifiedMemory |
| 因果 | `causal.py` | ConditionalIndependenceTester(PC 算法)/ CausalDiscovery / UnifiedCausalEngine / QuantumCausalStore |
| 类比 | `analogical.py` | StructuralGraph / StructureAligner = Gentner 结构映射 SME |
| 意识 | `conscious.py` / `consciousness_integrator.py` / `gw_workspace.py` | GlobalWorkspace / CoalitionalProcess / ConsciousContext |
| 情感 | `affective_engine.py` | AffectiveState / PersonalityProfile / EmotionDimension |
| 安全 | `safety.py`(865 行)/ `guardian.py` | ASISafetyEngine / CoreValue(immutable)/ SandboxedChange / EmergencyStop / AuditTrail |
| 自主 | `autonomy.py` | GoalManager / Planner / AutonomousEngine |
| 世界模型 | `world_model.py` + `world_models/` | UnifiedWorldModel / EntityType / RelationType |
| 进化 | `code_evolution.py` / `evolution_system.py` / `rsi_engine.py` | CodeEvolutionEngine / EvolutionSystem |
| 学习 | `meta_learning.py` / `continuous_learning.py` / `curriculum.py` | MetaLearningEngine / LearningPipeline / CurriculumEngine |
| 自愈 | `self_healing.py` / `heartbeat_daemon.py` | AutoHealer / heartbeat |
| 安全系统 | `security_system.py` / `multi_agent.py` / `swarm_system.py` | SecuritySystem / AgentRegistry / TaskBoard / SafeRollback |

自带测试:`test_memory_system.py` / `test_unified_memory.py` / `test_consciousness_integrator.py` /
`test_meta_cognitive.py` / `test_affective_engine.py`。
运行注意:需在 `G:\laap` 下(保证 `laap` 包可导入),并设 `PYTHONIOENCODING=utf-8`(代码打印 `✓`)。

## 4. 量子层的真实状况(已核实)

- **`aris_brain/quantum_bridge.py` 是死代码(已归档)**:它 `_ensure_quantum()` 惰性导入
  `aris_brain.quantum_psi` / `aris_brain.quantum_memory`,但**这两个模块不存在**。
  实测 `ModuleNotFoundError` → `_QUANTUM_AVAILABLE=False` → 恒"回退到经典认知"。
  零外部引用,已移入 `aris_brain/_archive/`。
- **V12 的"quantum" = JL 随机投影**(`aris_v12_dense_kernel.py`:
  N_SPARSE=16384, N_DENSE=986, P=randn 归一化列)。这是真实数学技术(Johnson–Lindenstrauss lemma),
  命名已改为"联想投影"。
- `laap/agi/causal.py` 的 `QuantumCausalStore`:"量子叠加态" `|Ψ_causal⟩ = Σ α_i |cause_i⟩ ⊗ |effect_i⟩`
  实现为 (cause_vec, effect_vec, confidence) 余弦相似匹配——向量关联存储的"量子化"命名。

**结论**:项目中没有伪科学实现,只有从未运行的占位层 + 命名问题。这比"伪科学包装"好得多——
问题是可清理的命名,不是需推翻的算法。

## 5. 作者理想 vs 实际运行(接线后)

| 理想/宣称 | 实际 | 差距本质 |
|---|---|---|
| 量子认知桥 | `quantum_bridge.py` 死代码 | ✅ 已移入 `_archive/`,命名→"随机投影联想" |
| 500μs 认知周期 | Pure Python + torch-CPU | 未达,纯虚构指标 |
| 自我进化 | `code_evolution` 存在但未驱动 | 未接线(高风险,审慎) |
| 安全机制 | safety/guardian 存在且已加载 | ✅ 已激活:`laap_brain_api._safety_gate` 输出拦截面(中文高危词表 41 项 + ASISafetyEngine 核心价值),4 tests 保护 |
| 潜意识命名 | `QuantumSubconscious` 等 | ✅ P0 清理:→`SubconsciousLayer`,保留兼容别名 |
| 三层记忆 | memory_system 已接入主循环(P1) | ✅ 接线完成;本批修复中文召回缺陷(空 split→分词+字符保底) |
| 因果推理 | UnifiedCausalEngine 已接入(P2) | ✅ 接线完成 |
| 世界模型 | UnifiedWorldModel 已接入(P3) | ✅ 接线完成 |
| 意识/全局工作空间 | GlobalWorkspace 已接入(P3) | ✅ 接线完成 |
| 动态自我叙事 | 记忆→身份叙事(P4) | ✅ 接线完成 |
| Zero-LLM 语义理解 | 关键词查表(原) | ✅ 升级为双编码(ConceptGraph 结构嵌入 + V12 字形);图内覆盖近义实测提升,**开放词汇为诚实边界** |

## 6. 下一步(候选)

P0-P4 已全部完成(见 §7),语义化与安全拦截面已落地。剩余风险项排序:

1. ~~**长时验收(中风险)**~~:✅ 已通过——`experiments/exp_memory_1000.py`:1000 轮跨会话(持久化→新实例加载)召回 **recall@5=100%**(10 话题×各100条,期望=最新一条);抽样 200 轮精确召回 100%。过程暴露并修复:① 分词器每次检索重建的昂贵开销→单例+内容级 token 缓存(1000 条查询 ~0.7s);② 纯数字/TAG 编号 token 干扰相似度→过滤;③ 同话题多条时无 recency→按 strength(情感/置信/近因)次级排序。
2. **开放词汇语义(中风险)**:离线无中文语义模型,180 节点概念图只覆盖词内近义(实测 Dual 4/7 vs 字形 1/7);开放词汇零字重叠三路持平 ~1/8——除非引入中文 embedding 模型,否则不承诺。
3. ~~**旁支决断(低风险)**~~:✅ 保留并跟踪——`laap_v2` 是自研第二代认知运行时(PSI 竞争动力/LinUCB 元学习/DBN+粒子滤波世界模型/LSH 语义记忆/Page-Hinkley 漂移/VCG 多智能体/FastAPI+混沌压测),与 laap/agi(PC 因果/Baars 意识/Rescorla 记忆)为**不同技术路线并行实现**,110 tests 全绿(`G:\Hermes\.venv`);已 `git add` 跟踪(41 文件),未接线主循环,作为对照资产保留。|
4. **自我进化(高风险)**:`code_evolution` 自我改码,保持禁用,人工监督下试点。|
5. ~~**affective_engine 测试漂移(低风险)**~~:✅ 已修复——测试为旧值:引擎默认 `noise_amplitude` 0.03(测试期望 0.05)、`task_success` PLEASURE 0.6×0.5=0.3(期望 0.2)、`compute_mood` 合法情绪 9 种(集成测试只列 5 种)。已按引擎同步测试。|
6. **检索性能(已优化)**:记忆检索经分词单例+内容级 token 缓存(`_tok_cache`)+ 数字 token 过滤后,1000 条 × 查询 ~0.7s;如扩展到万级记忆需倒排索引。|
7. **残留健康度(已收尾)**:本轮排查修复——① 记忆持久化损坏时静默清空→改备份 `.corrupt.json`+error 日志;② 公共 HTTP 网关(handle_chat_completions 等 12 路由)零测试→`tests/test_gateway.py` 11 测试;③ `aris_messenger.py:37` 全项目唯一语法错误→修复;④ README "无本地路径"声明与 6 处硬编码矛盾→入口/测试改相对路径、声明修正;⑤ 量子命名活跃调用点→`SubconsciousLayer`/`VectorCausalStore`;⑥ 500μs/2000Hz 虚构指标 6 处代码注释→真实 10Hz;⑦ `_run_agi_tick` docstring 宣称激活 curriculum/meta→修正诚实。|

## 7. 接线进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 记忆接线 | `memory_bridge.py` 重写:优先用 laap/agi 三层记忆(Episodic/Semantic/Procedural + MemoryConsolidator),加 JSON 持久化(`state/agi_memory.json`),接口不变 | ✅ 完成 |
| P1 主循环写入 | `aris_cognitive_bridge._learn()` 改用 `store_important` 写 AGI 记忆,不再绕行 fallback | ✅ 完成 |
| P1 验证 | 一轮对话 → episodic +1;召回/叙事/巩固均实测通过;根测试 32 passed | ✅ 完成 |
| P1 剩余 | 1000 轮跨会话召回 ≥80% 长时验收;`before_turn` 读取侧已走 AGI 后端 | 待长跑 |
| P2 因果接线 | `aris_cognitive_bridge` 新增 `_learn_causal`/`_save_causal_state`/`_load_causal_state`/`_causal_context`:真实交互事件(话题/情感/结果)写入 `UnifiedCausalEngine`,每 10 轮自动持久化,`before_turn` 注入因果经验 | ✅ 完成 |
| P2 缺陷修复 | `laap/agi/causal.py` `load()` 原不恢复 bonds(只恢复 rules)→ 修复,跨会话因果键可恢复 | ✅ 完成 |
| P2 验证 | 10 轮对话 → bonds/temporal/entity 持久化;新会话恢复 2 bonds + 5 rules;因果上下文注入生效;32 passed | ✅ 完成 |
| P3 世界模型接线 | `aris_cognitive_bridge` 新增 `_learn_world_model`(真实交互 → 实体状态 + 社交关系)、`_world_model_path`/`_load_world_model`/`_save_world_model`(跨会话持久化,每 10 轮保存);`_integrate` 注入世界模型摘要(实体/关系 + Aris→Lorry 信任/亲密度) | ✅ 完成 |
| P3 验证 | 11 轮 → entities 6/relations 15,aris→lorry 信任/亲密度更新,持久化+跨会话恢复,`_integrate` 注入生效;32 passed | ✅ 完成 |
| P3 意识驱动 | `_init_consciousness`(Baars GlobalWorkspace)+ `_run_consciousness`:每轮把 感知/记忆/情感/需求/自我模型/任务 喂入 6 通道,`compete()` 角逐→胜出者(意识焦点+绑定)注入上下文 | ✅ 完成 |
| P3 意识验证 | before_turn 输出含 `[意识工作区] 意识焦点+绑定`;32 passed | ✅ 完成 |
| P4 动态自我叙事 | `_dynamic_self_narrative`:从真实记忆(最近情景 + 语义概念 + 技能)实时生成"我是谁→我记得→我学会"叙事,名字/人格锚点取自 identity_manager | ✅ 完成 |
| P4 验证 | before_turn 输出含 `[自我叙事]`;32 passed;laap_v2 110 passed;laap/agi 自带记忆测试 passed | ✅ 完成 |
| 全量验证 | 根测试 32 passed + laap_v2 110 passed + laap/agi test_memory_system/test_unified_memory passed | ✅ 完成 |
| P5 语义化-嵌入修复 | `ConceptGraph._build_embeddings` 从词名哈希随机向量改为关系驱动嵌入(同义/上下位/反义/特征/效价);`_add` 重复定义从静默丢弃改为合并增强(修复 24 条单向同义链,如 恐惧→害怕);分词器自动并入概念图词表(修复 害怕/孤单 整词不命中) | ✅ 完成 |
| P5 双编码接线 | `aris_brain/semantic_matcher.py`(新建):字形核(V12)+ 语义核(ConceptGraph)加权融合;`ArisLMv12.respond()` 改 语义路→记忆路→字形路→语言回退 | ✅ 完成 |
| P5 记忆健壮性 | `laap/agi/memory_system.retrieve_similar` 中文召回修复(空格 split→分词+CJK 保底);记忆兜底阈值 0.15→0.10(recall 已过滤,避免临界抖动) | ✅ 完成 |
| P5 量化对照 | `experiments/exp_dual_encoding.py`:三路(OldKeyword/DualEncoder/TfidfBoW)对标注语料,图内覆盖组 Dual 4/7 vs Old 1/7 vs Tfidf 1/7;开放词汇三路持平 ~1/8(诚实边界);`tests/test_dual_encoding.py` 7 测试全绿 | ✅ 完成 |
| P5 安全拦截面 | `laap_brain_api._safety_gate` 输出网关拦截:中文高危词表 41 项(自残/暴力/犯罪/赌博)+ ASISafetyEngine 核心价值;`tests/test_safety_gate.py` 4 测试全绿 | ✅ 完成 |
| P5 验证 | 根测试 43 passed(含新增 11)+ laap_v2 110 passed + laap/agi 42 passed(3 例 affective 漂移挂起,与本次无关) | ✅ 完成 |
| P1 长时验收 | `experiments/exp_memory_1000.py`:1000 轮跨会话(JSON 持久化→新实例)recall@5=100%(10 话题×各100,期望最新一条)+200 抽样 100%;同期 `memory_system` 性能优化(_tok 单例/`_tok_cache`/数字过滤/recency tie-break) | ✅ 通过 |
| P5 affective 对齐 | `laap/agi/test_affective_engine.py` 3 例漂移修复(旧值→引擎实际):noise_amplitude 0.05→0.03、task_success PLEASURE 0.2→0.3、mood 白名单 5→9 种+valence<0.3 断言;laap/agi 全量 56 passed | ✅ 完成 |
| 旁支决断 | `laap_v2` 保留并 git 跟踪(41 文件,110 tests 全绿),文档化与 laap/agi 的技术路线差异,作为对照资产 | ✅ 完成 |
| P6 健康度修复 | 记忆持久化损坏容错(备份+日志)/网关 11 测试/语法错误修复/硬编码路径相对化/量子命名清理/虚构指标注释修正/_run_agi_tick 文档诚实化 | ✅ 完成 |
| P7 闭环与持久化缺陷修复 | ① 网关写记忆闭环:HTTP 路径补 `after_turn`(此前只读 before_turn,线上学习从未发生) ② 因果路径统一:tick 的 `ce.save()` 显式传 `_causal_path()`,消除 cwd 与 state 双文件分叉 ③ `memory_bridge._load` 重建时间/情感索引(重启后按时间/情感检索恒空)+ relations/hierarchy 保持 `defaultdict`(防 consolidation KeyError) ④ `causal.load()` 还原规则 conditions/effects(含旧 str 值类型还原) ⑤ world_model 持久化保全 properties 值/relationships/history/timeline ⑥ `retrieve_similar` 除零防护 ⑦ `_integrate:909` "\\n" → "\n" 手误修复 ⑧ `_learn_world_model` 信任值绝对公式 → 增量收敛 ⑨ VectorCausalStore 向量补零对齐(短/超长维度不再抛错) ⑩ 部署:Dockerfile 源码 COPY 后再 `-e .`(修容器启动即崩)、requirements 补 `openai`(修 DeepSeek 永远走 Zero-LLM)、docker-compose.override 移除不存在的 `laap-AGI-full/` 引用、MCP 命令补 `--sse`(修 SSE 健康检查永远 unhealthy) | ✅ 完成 |
| P8 收敛与并发缺陷修复 | ① 记忆巩固接通:`_run_agi_tick` 周期性调 `consolidate_memory`(此前从未调用,consolidation_queue 只积不消、dream_reports 永空;实测 dreams 1→2、concepts 22→25) ② 因果注入复活:`_learn_causal` 键稳定化(target 固定 lorry/aris,不再经每轮唯一 topic 造键,observations 得以累积)+ `_causal_context` 按稳定键查询(n≥2 注入 `[因果经验]`,实测 3 轮后注入生效、情感独立分桶) ③ psi_core 字段脱节修复:engine 写 `cycle`(asdict),cognitive_bus 只读 `psi_cycle` 且强制 `quantum_engine!="none"` → 永远等不到新周期;改双键兼容(psi_cycle/cycle)+ 宽松引擎检查(实测 cycle 11→16 轮询推进成功) ④ 双进程并发写防护:psi_core latest/input_queue 与 cognitive_bus input_queue、memory_bridge agi_memory.json 全部改原子写(临时文件+`os.replace`,读端永不看到半截文件) | ✅ 完成 |
| P8 低危收尾 | ① `aris_watchdog` 幽灵目标换血:7 个指向不存在的 `cognitive_bus_daemon/v11_agi_daemon/aris_standalone/aris_qlg_provider/aris_psi_self_optimizer_daemon/aris_tts_server/xiaozhi_mcp_bridge` 的条目改为真实运行单元(LAAP Brain API :11546 / 飞书网关 / bootstrap / snapshot / sync),启动排序同步(去掉已删除的 qlg,实测 status 检测 5 个真实目标) ② 端口三重占用:`state_snapshot_server` 从 11520(与 QUANTUM_PORT 默认冲突)改独立 11521,watchdog 同步 ③ `.env.example` 对齐:补 `DEEPSEEK_MODEL`(laap_v2 别名)/`OPENAI_API_KEY`/`OPENAI_BASE_URL`/调参变量,移除幽灵 `XIAOZHI_MCP_TOKEN`(全库零消费) | ✅ 完成 |
