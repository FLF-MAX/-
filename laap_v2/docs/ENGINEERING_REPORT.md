# laap_v2 工程报告 (Engineering Report)

## 1. 概述

`laap_v2` 是 laap-AGI 项目的第二代认知运行时，用一个 `process_input()` 单入口把 8 个认知模块串成一个完整认知环。设计目标：

- **认知闭环**：感知(PSI) → 元学习(策略) → 世界模型(预测) → 记忆(存储/检索) → 输出；
- **工业级韧性**：每个模块都有异常边界，任一模组崩溃 → 自动降级 → 在线自愈；
- **可观测**：结构化日志、指标、健康探针、混沌注入全具备；
- **零硬依赖**：除 numpy 外全部标准库；LLM 桥可选（有 DeepSeek key 时启用）。

## 2. 模块清单与职责

| 文件 | 职责 | 核心算法 |
|---|---|---|
| `psi_core_v2.py` | 情绪/需要/注意力动力学 | 竞争型需要动力系统 + PAD 情绪向量 + 预测误差驱动 |
| `meta_learning_engine.py` | 认知策略选择 | LinUCB（bandit） + REINFORCE 梯度策略 |
| `probabilistic_world_model.py` | 世界概率模型 | 动态贝叶斯网络(DBN) + 粒子滤波 + BIC 结构学习 |
| `neural_world_model.py` | 世界神经网络模型 | MLP + 异方差 NLL + Adam + 课程式训练 |
| `hierarchical_memory.py` | 分层记忆 | LSH 近邻索引 + 时间衰减 + 语义召回 |
| `drift_aware_meta_learning.py` | 分布漂移检测 | Page-Hinkley 检验 + 自适应滑动窗口 |
| `deep_analogical_engine.py` | 结构类比推理 | Gentner 结构映射 + 匈牙利最优对齐 |
| `multi_agent_coordination.py` | 多智能体协调 | VCG 拍卖 + 联盟形成 + Shapley 值 |
| `production_infra.py` | 生产基础设施 | 结构化日志/配置/校验/熔断/指标/健康 |
| `laap_integration.py` | 端到端编排 | PSI 心跳自愈 + 降级模式 + 线程安全 + 状态持久化 |
| `api_server.py` | HTTP 服务 | FastAPI + 令牌桶限流 + 请求链路追踪 |
| `benchmark_suite.py` | 标准化基准 | 6 项指标化评测 |
| `stress_tests.py` | 稳定性测试 | 压测/模糊/混沌/记忆/长跑 |

## 3. 关键设计决策

### 3.1 竞争型需要动力系统（PSI）
五个基本需要（competence/relatedness/growth/certainty/autonomy）构成一个**竞争动力系统**：
```
d(need_i)/dt = 需求增长 - 满足衰减 - ∑ 其他需要对其的抑制
```
注意力自动指向"最紧迫"的需要（组合需要的收益与随时间增长的紧迫度）。预测误差(PE)注入后，能驱动需要变化，从而影响注意与情绪——这正是"学习动机来自惊讶"的机制。`heartbeat_ok()`/`recover()` 支持心跳检测与自愈。

### 3.2 元学习双通道
- **LinUCB**：对每个策略维护独立特征权重，按置信上界选择，适合在线冷启动；
- **REINFORCE**：随机策略梯度，负责对历史选择给出全局回报归因；
- 两者在 `update()` 中同时更新，`best_strategy()` 给出当前最优策略。

### 3.3 世界模型三态
- DBN + 粒子滤波：显式因果结构（`learn_structure` 用 BIC 评分做结构搜索），可解释；
- MLP + 异方差 NLL：近似灵活动力学，输出均值+不确定度；
- 双模型按数据量自动切换（`num_observations() >= 2` 用 DBN，否则用 MLP）。

### 3.4 记忆的确定性哈希
记忆嵌入使用 **FNV-1a 稳定哈希**（而非 Python 内建 `hash()`）。原因：`hash()` 受 `PYTHONHASHSEED` 影响，跨进程随机化会导致 LSH 分桶不一致、召回排名不稳定——压测时实测抓到该缺陷，已修复。

### 3.5 容错与自愈（降级模式）
- 每个认知阶段单独 try/except；
- PSI 心跳丢失 → `psi_recover()` 自动恢复；
- 混沌注入（`degrade_module`）→ 下一次 `process_input` 必须仍然有响应；
- 顶层兜底：`degraded_mode=True` 时返回降级响应而非抛异常。

### 3.6 状态持久化（重启不丢记忆）
`LaapCognitiveSystem.save_state/load_state` 把完整认知状态序列化为 JSON：
- **PSI**：需要/速度/情绪/PAD 向量；
- **元学习**：LinUCB 权重矩阵、各策略成败计数、EMA；
- **漂移**：EMA、滑动窗口、Page-Hinkley 内部状态；
- **记忆**：完整条目 + **精确向量**（而非重新嵌入），重启后 LSH 召回质量完全一致；
- 人格档案与对话计数。

这套设计让"认知体"可以停机升级、重启后带着旧记忆与旧偏好继续工作——是生产级运行时区别于一次性脚本的关键能力。

## 4. 评测结果

### 4.1 基准（`python benchmark_suite.py`）

| 基准 | 得分 | 基线 | 说明 |
|---|---|---|---|
| psi_prediction_error_injection | 1.000 | 0.5 | PE 注入后成长需要确实上升 |
| causal_structure_recovery | 1.000 | 0.33 | 恢复 T→P、T→H 两条因果边 |
| analogical_structure_mapping | 1.000 | 0.25 | 太阳系↔原子 结构映射一致性 1.0 |
| drift_aware_continuous_learning | 1.000 | 0.5 | 40步后检测漂移并 100% 选最优 |
| nonlinear_world_model_mae | 0.993 | 0.0 | Z=X*Y MAE 0.032 vs 基线 4.35 |
| linucb_best_strategy_identification | 0.933 | 0.5 | 60轮后 93% 概率选最优策略 |
| **平均** | **0.988** | — | — |

### 4.2 稳定性（`python stress_tests.py`，9/9 通过）
- 压测：300 条突发消息，约 560 条/秒，全程无异常；
- 模糊：30 组畸形/超长/非 UTF8 输入，零崩溃；
- 混沌：PSI 心跳损坏→自动恢复；记忆清空→仍正常响应；元学习置空→策略回退；
- 记忆：2000 插入容量锁在 300 上限；语义召回命中目标；
- 匈牙利：200 组随机形状矩阵全部合法分配；
- 漂移：分布切换被 Page-Hinkley 检出。

## 5. 复现基准时修复的缺陷

1. `multi_agent_coordination.py` 缺少 `import math`（`shapley()` 用 `math.factorial`）→ 补上；
2. `hierarchical_memory.py` 用 `hash()` 导致跨进程召回不稳定 → 改为 FNV-1a 稳定哈希；
3. `deep_analogical_engine.py` 的 `hungarian` 曾对矩形矩阵/空图越界 → 改为经典的 e-maxx 势能版（1-based，`size+1` 数组），并保持"空图返回空对齐"语义；
4. `production_infra.py` 的 `SystemConfig.to_dict` 引用未定义变量 → 返回参数副本；
5. `benchmark_suite.py` 早期 stdout 与文件写入相互污染（环境问题，非代码缺陷）——已统一用唯一时间戳文件隔离。

## 6. 部署形态

- **本地/开发**：`python demo.py`（交互式）、`python api_server.py`（HTTP 服务）；
- **容器**：Dockerfile + docker-compose（多副本 + 健康检查 + 自动重启）；
- **生产**：Kubernetes（Deployment + HPA + Service + ConfigMap）、systemd 单元、GitHub Actions CI。
- 配置：环境变量 `LAAP_*`（见 `.env.example`）；LLM 可选 `DEEPSEEK_API_KEY`。

## 7. 结论

laap_v2 以 8 个可独立评测的认知模块 + 工业级编排层交付，6 项基准平均 0.988、9 项稳定性测试全通过。系统在任何单点模块失败时保持在线并自愈，满足"可作为生产级认知运行时"的交付标准。
