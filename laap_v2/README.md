# laap_v2 — 认知运行时

laap-AGI 项目第二代认知运行时。一个 `process_input()` 入口，串起 PSI 驱动、元学习策略选择、世界模型预测、分层记忆、漂移检测、类比推理与多智能体协调。

## 快速开始

```bash
pip install -r requirements.txt
python demo.py                          # 交互式对话（本地规则模式）
DEEPSEEK_API_KEY=sk-xxx python demo.py  # 接 DeepSeek LLM
python benchmark_suite.py               # 运行 6 项基准
python stress_tests.py                  # 运行稳定性/混沌/压测
python api_server.py                    # 启动 HTTP 服务 (http://localhost:11546)
```

## 模块

| 模块 | 说明 |
|---|---|
| `psi_core_v2` | 竞争型需要动力系统 + PAD 情绪 + 预测误差，含心跳自愈 |
| `meta_learning_engine` | LinUCB + REINFORCE 双通道策略选择 |
| `probabilistic_world_model` | DBN + 粒子滤波 + BIC 结构学习 |
| `neural_world_model` | MLP 异方差世界模型，输出均值与不确定度 |
| `hierarchical_memory` | LSH 语义召回 + 时间衰减 + 容量锁定 |
| `drift_aware_meta_learning` | Page-Hinkley 漂移检测 + 自适应窗口 |
| `deep_analogical_engine` | Gentner 结构映射 + 匈牙利对齐 |
| `multi_agent_coordination` | VCG 拍卖 + 联盟 + Shapley 值 |
| `production_infra` | 日志/配置/校验/熔断/指标/健康 |
| `laap_integration` | 端到端编排，降级模式 + 自愈 |
| `api_server` | FastAPI 服务 + 令牌桶限流 |

## HTTP API

- `GET /health/live`, `GET /health/ready`, `GET /health/modules` — 探针
- `POST /v1/chat` `{"message": "...", "user": "..."}` — 对话
- `GET /v1/cognitive/state` — 认知状态（PSI/元学习/记忆）
- `GET /v1/metrics` — 指标快照
- `POST /v1/reset` — 重启认知系统
- `POST /v1/chaos/{psi|memory|meta}` — 混沌注入（自我修复演示）
- `POST /v1/allocation` — 多智能体任务分配（VCG）
- `POST /v1/state/save` `{"path": "..."}` — 认知状态快照（重启不丢记忆）
- `POST /v1/state/load` `{"path": "..."}` — 恢复认知状态

所有响应携带 `x-trace-id`，可用于跨日志链路追踪。

## 稳定性保证

- 每个认知模块都有异常边界；任一模块失败 → 自动降级并保持在线；
- PSI 心跳检测 + 自动恢复；
- 全链路线程安全（RLock）；
- 混沌注入验证：`python stress_tests.py`（10/10 通过）；
- 状态持久化：`save_state`/`load_state` 保存 PSI 需要、元学习权重、漂移 EMA 与记忆（含精确向量，重启后召回质量不变）。

## 配置（环境变量 `LAAP_*`）

```bash
LAAP_PORT=11546
LAAP_LOG_LEVEL=info
DEEPSEEK_API_KEY=sk-xxx       # 可选，配置后启用 LLM 响应
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

详见 `docs/ENGINEERING_REPORT.md` 与 `docs/DEPLOYMENT.md`。