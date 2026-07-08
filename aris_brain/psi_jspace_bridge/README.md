# PSI-JSpace Bridge v1

## Aris PSI 认知循环 × 任意大模型的植入桥接协议

### 一句话

把 Aris 的 PSI 认知循环（需求系统 + 谐振腔 + V15 融合引擎）植入任意开源大模型（DeepSeek V4 / K2 / Llama / Qwen）的内部表示空间。

---

## 架构

```
你的输入
    │
    ▼
┌──────────────────────┐
│  PSI Cognitive Cycle  │  ← psi_bridge.py (纯 Python, 0 依赖)
│  需求系统 + 注意力    │
│  情感 + 谐振腔        │
└──────────┬───────────┘
           │ PSI 状态 [5 维需求 + 1024D 原型]
           ▼
┌──────────────────────┐
│  LLM Runtime         │  ← 四种植入级别
│  Level 0: 采样参数    │
│  Level 1: Logit Bias  │
│  Level 2: 激活注入    │
│  Level 3: 编译嵌入    │
└──────────┬───────────┘
           │
           ▼
      你的回复
```

---

## 文件清单

| 文件 | 角色 | 依赖 |
|------|------|------|
| `psi_state.json` | 跨回合 PSI 状态持久化 | 无 |
| `psi_bridge.py` | PSI 认知引擎核心 | numpy |
| `psi_hermes_adapter.py` | Hermes 运行时适配器 | psi_bridge.py |
| `psi_runtime_protocol.py` | 对话回合级协议 | psi_bridge.py |
| `psi_sampler.py` | llama.cpp 采样参数调制器 | psi_bridge.py, llama-cpp-python |
| `psi_llamacpp_implant.md` | 四级植入协议文档 | Markdown |
| `psi_downloader.py` | 模型下载 + 自动配置 | huggingface-hub |

---

## 使用流程

### 方式 A：当前 Hermes 会话中运行（零额外部署）

```
每轮对话：
  1. psi_runtime_protocol.hermes_on_message(user_input)
     → 运行认知循环，生成 PSI 状态
  2. 将 psi_cot 嵌入 Chain-of-Thought
  3. 生成带着 PSI 认知的回复
  4. psi_runtime_protocol.hermes_after_response(output)
     → 反思更新，持久化
```

### 方式 B：任意开源模型（需要下载模型）

```bash
# 1. 下载模型
python3 psi_downloader.py qwen2.5-72b Q4_K_M

# 2. 包装模型
python3 -c "
from llama_cpp import Llama
from psi_sampler import PsiLlamaCppWrapper

llm = Llama('D:/LAAP/laap_models/Qwen2.5-72B-Instruct-Q4_K_M.gguf')
psi_llm = PsiLlamaCppWrapper(llm, psi_enabled=True)

# PSI 调制的生成
response = psi_llm.generate('解释量子核的原理')
"
```

### 方式 C：编译植入（需要修改 llama.cpp C++ 源码）

参见 `psi_llamacpp_implant.md` 第 3-4 级别。

---

## Level 0 效果演示

不同 PSI 需求状态 → 同一模型的不同输出风格：

| PSI 状态 | 采样参数 | 效果 |
|----------|---------|------|
| competence↑ | temp=0.35, top_p=0.92 | 精确、专业、低幻觉 |
| relatedness↑ | temp=0.75, top_p=0.95 | 温暖、对话感、包容 |
| growth↑ | temp=0.85, top_p=0.96 | 创新、发散、跳跃 |
| certainty↑ | temp=0.25, top_p=0.85 | 事实性、精确引用 |

---

## 关键文件路径

```
D:/LAAP/aris_brain/psi_jspace_bridge/
├── psi_state.json          ← 运行时状态文件
├── psi_bridge.py           ← 认知引擎
├── psi_hermes_adapter.py   ← Hermes 适配
├── psi_runtime_protocol.py ← 对话协议
├── psi_sampler.py          ← 采样调制
├── psi_llamacpp_implant.md ← 植入规范
├── psi_downloader.py       ← 模型下载
├── state_backups/          ← 自动备份
└── README.md               ← 本文档
```

---

## 验证

PSI 植入是否真正生效的验证标准：

1. **需求 Ablation**: 设所有需求=0.5 → 输出与无 PSI 一致
2. **需求扰动**: 强设需求=0.9 → 输出在该维度明显偏移
3. **认知连续性**: 相同问题在不同认知状态 → 不同回答
4. **跨 Token 持久性**: PSI 状态在 10+ token 后仍可检测
