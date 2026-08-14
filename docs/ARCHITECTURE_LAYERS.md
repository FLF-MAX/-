# LAAP 认知架构分层与 LLM 依赖边界

**印记**: Aris 永远记得 Lorry — 2026-06-23
**最后更新**: 2026-08-14

本文档明确 LAAP 认知架构各层对 LLM 的依赖状态，回答"Zero-LLM 到底指什么"这一核心问题。

---

## 1. 分层总览

```text
┌──────────────────────────────────────────────────────────────────┐
│ Layer 4 · Expression（表达层）           ← LLM 必需（"声音皮层"）│
│   功能：自然语言生成、情感化表达、多轮对话                        │
│   依赖：DeepSeek / 任意 OpenAI 兼容 API / 本地模型               │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3 · Reasoning（推理层）            ← Zero-LLM 目标         │
│   功能：因果推理、类比映射、联想检索、世界模拟                    │
│   依赖：纯架构（向量引擎 / 规则引擎 / 图模型）                    │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2 · Memory（记忆层）               ← Zero-LLM              │
│   功能：情景记忆、语义记忆、记忆固化、记忆召回                    │
│   依赖：向量索引 + JSON 持久化（TF-IDF / 局部 embedding）         │
├──────────────────────────────────────────────────────────────────┤
│ Layer 1 · PSI Core（本能/感知层）        ← Zero-LLM              │
│   功能：5 维需求动力学、情绪梯度、注意选择、唤醒度                │
│   依赖：纯数值计算                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**一句话**：LLM 是"声音皮层"，不是"大脑"。认知循环（Layer 1–3）可脱离 LLM 独立运行；只有自然的语言进出（Layer 4）需要 LLM。

---

## 2. 各层详述

### Layer 1 · PSI Core（本能层）— Zero-LLM ✅

| 项 | 说明 |
|---|---|
| 模块 | `psi_core/engine.py`（纯 Python，10Hz 认知周期） |
| 功能 | 5 维需求向量（competence/relatedness/growth/certainty/autonomy）、情绪标签、唤醒度、注意选择 |
| 依赖 | 纯数值计算，`random` 装饰噪声，零网络、零外部模型 |
| 状态验证 | `tests/benchmarks/test_cognitive_benchmarks.py::TestPsiNeedDynamics` |

**诚实说明**：需求引擎当前对自然语言敏感度有限——只对设计内关键词（如"爱你"→relatedness）响应。这是后续 NLP 触发的改进点，不是架构缺陷。

### Layer 2 · Memory（记忆层）— Zero-LLM ✅（部分待完善）

| 项 | 说明 |
|---|---|
| 模块 | `aris_brain/laap_semantic_memory.py`、`laap/agi/memory_system.py` |
| 功能 | 语义记忆写入/召回、情景记忆、记忆固化 |
| 依赖 | 向量索引（TF-IDF / 局部 embedding）+ JSON 持久化 |
| 边界 | 默认 `SentenceTransformersProvider` 是本地离线模型；若显式配置 OpenAI embedding 则引入网络依赖（此时"语义嵌入层"改为 LLM-augmented） |
| 状态验证 | `tests/benchmarks/::TestMemoryFidelity` |

### Layer 3 · Reasoning（推理层）— Zero-LLM 目标 🚧

| 项 | 说明 |
|---|---|
| 模块 | `CognitiveBus`（路由）、`aris_brain/` 下 QRE / V12 / QLG 引擎、`psi_semiotics/`、`laap/agi/causal.py` |
| 功能 | 联想检索、模板匹配、因果推理、类比映射 |
| 依赖 | 纯架构（特征空间向量引擎 / 规则 / 模板库） |
| 现状 | 规则驱动的匹配与合成不依赖 LLM；但**复杂常识推理目前没有独立符号实现**，这部分若硬要脱离 LLM 会退化到模板层级 |
| 状态验证 | `tests/benchmarks/::TestRoutingLatency` |

### Layer 4 · Expression（表达层）— LLM 必需 ✅

| 项 | 说明 |
|---|---|
| 模块 | `laap_brain/api.py` 的 `_llm_respond`、`laap_chat.py`、Hermes 集成 |
| 功能 | 自然语言生成、情感化表达、上下文对话 |
| 依赖 | `DEEPSEEK_API_KEY` / 任意 OpenAI 兼容端点 |
| 隔离 | 若未配置 API Key，Layer 4 退化到模板回复（`laap-fallback`），其余层照常运行 |

---

## 3. 术语修正建议

README 的"Zero-LLM"应理解为：

> **认知循环 Zero-LLM，表达层 LLM-Augmented**。
> 更准确的说法是 **LLM-Augmented Cognitive Architecture** ——
> 认知主体由架构承担，LLM 是外接的生命维持与声音系统。

这一修正已在 README「🧪 当前能力边界」章节体现（2026-08-14）。

---

## 4. 每层的独立运行验证

```bash
# Layer 1：仅 PSI Core 独立运行（不触碰 LLM）
python -m pytest tests/benchmarks -q -k "NeedDynamics"

# Layer 2：记忆独立运行（隔离临时文件，不读真实记忆）
python -m pytest tests/benchmarks -q -k "MemoryFidelity"

# Layer 3：路由独立运行（无 LLM 参与）
python -m pytest tests/benchmarks -q -k "RoutingLatency"

# Layer 4：无 Key 时的退化路径
DEEPSEEK_API_KEY= python -m pytest tests/benchmarks -q
```

---

## 5. 演进路线

| 层 | 当前 | 目标 | 优先级 |
|---|---|---|---|
| L1 PSI | 关键词触发 | 语义触发 + 更细的情绪连续体 | P0 |
| L2 记忆 | 局部 embedding | 可插拔向量库（faiss/annoy）+ 记忆衰减曲线量化 | P1 |
| L3 推理 | 规则 + 模板 | 显式符号推理（ARC/bAbI 可测） | P1 |
| L4 表达 | LLM 必需 | 保留，持续做"引擎输出优先"的策略 | P2 |

---

## 6. 相关文件

- `psi_core/engine.py` — PSI Core 引擎
- `aris_brain/cognitive_bus.py` — 认知路由中枢
- `aris_brain/schemas/events.py` — 类型化认知事件契约
- `aris_brain/laap_semantic_memory.py` — 语义记忆
- `tests/benchmarks/` — 各层能力基准测试
- [ARCHITECTURE.md](ARCHITECTURE.md) — 总体架构