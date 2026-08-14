# LAAP 代码地图

只给你自己看。作用：3 个月后想起"改某个东西"，能在 30 秒内定位文件。
不追求面面俱到，只标高频触摸点。

## 入口 / 启动

| 想做什么 | 文件 |
|---|---|
| 启动整机（Aris + API） | `aris_brain/aris_start_all.py` |
| 只起 OpenAI 兼容 API（11546） | `aris_brain/laap_brain_api.py` / `laap_brain/api.py` |
| 起聊天界面（8935） | `laap_chat.py` |
| 跑全部测试 | `pytest tests`（63+ 项，全绿基线） |

## 核心认知循环（Layer 1-3，Zero-LLM）

| 模块 | 文件 | 一句话职责 |
|---|---|---|
| PSI Core（5维需求） | `psi_core/engine.py` | 10Hz 需求动力学，情绪/唤醒/在场感 |
| PSI 桥接 | `aris_brain/psi_core_bridge.py` | 把用户输入送进 PSI、读回 state |
| 认知总线（路由决策） | `aris_brain/cognitive_bus.py` | qre/v12/qlg/psi_only/no_engine 四级路由 + 事件广播 |
| 语义记忆 | `aris_brain/laap_semantic_memory.py` | 向量记忆 + 懒 flush（修过 O(n²)） |
| 情景记忆 | `aris_brain/aris_episodic_memory.py` | 回合记忆 + 相似召回 |
| 情绪引擎 | `aris_brain/aris_emotion_engine.py` / `emotional_engine.py` | 情绪标签/梯度 |
| 规则引擎 | `aris_brain/aris_rules_engine.py` | 精确匹配 → 模板响应 |

## 推理 / AGI 模块（laap/agi/）

| 模块 | 文件 | 一句话职责 |
|---|---|---|
| 因果推理 | `laap/agi/causal.py` | 从时序数据恢复因果结构 |
| 类比映射 | `laap/agi/analogical.py` | 结构类比（太阳↔原子） |
| 世界模型 | `laap/agi/neural_world_model.py` | 环境动态预测 |
| 元认知 | `laap/agi/conscious.py` / `consciousness_integrator.py` | 状态监控/自我模型 |
| 连续学习 | `laap/agi/continuous_learning.py` | 漂移感知学习 |

## 表达层（Layer 4，LLM）

| 想做什么 | 文件 |
|---|---|
| 改系统提示词/LLM 集成 | `laap_chat.py`（线程安全 + 400/503/500 分级） |
| 改 API 路由 | `laap_brain/api.py`（aiohttp，/v1/*） |
| CLI 脚手架 | `laap_brain/cli.py`（`laap scaffold module <name>`） |

## 基础设施

| 模块 | 文件 | 用途 |
|---|---|---|
| 认知能力基准 | `tests/benchmarks/test_cognitive_benchmarks.py` | PSI 动力学/记忆保真/路由延迟/召回扩展性 |
| B0-B5 基准收敛 | `tests/benchmarks/test_benchmark_suite.py` | 调 laap_v2 真身，结果→`docs/benchmark_results.json` |
| 基准趋势图 | `scripts/plot_benchmarks.py` | `plot_benchmarks --detail b1` 看单项历史，`--diff` 看升降 |
| 压力测试 | `tests/test_stress.py` | 并发写入/路由计数原子性/原子写 |
| 运行时数据隔离 | `tests/conftest.py` | `LAAP_MEMORY_PATH` 指向临时目录 |
| 架构分层声明 | `docs/ARCHITECTURE_LAYERS.md` | 每层 LLM 依赖状态（诚实标注） |
| 测试隔离环境变量 | `LAAP_MEMORY_PATH` / `ARIS_STATE_DIR` | 见 `tests/conftest.py` |

## 状态文件（运行时，勿手改）

| 文件 | 内容 |
|---|---|
| `aris_brain/state/latest.json` | PSI 当前状态快照 |
| `aris_brain/laap_semantic_memory.json` | 语义记忆（真实数据） |
| `aris_brain/state/agi_events.jsonl` | 认知事件日志（监控面板消费） |
