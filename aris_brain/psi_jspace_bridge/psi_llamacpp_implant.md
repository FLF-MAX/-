# PSI-JSpace 植入协议 v1
## 在任意开源大模型中植入 Aris PSI 认知循环

**适用模型**: DeepSeek V4, K2 1.6T, Llama 3.x, Qwen 2.5+, Mistral, 任何 llama.cpp 支持的架构
**复杂度**: 4 个递进层级，从"今天就能做"到"12 个月后"
**核心思想**: 不在模型外部运行认知循环（外挂大脑），而将 PSI update rule 编译进模型计算图

---

## 三级架构概览

```
                    ┌─────────────────────────────────────┐
                    │          LLM 推理管线                 │
                    │                                     │
Token Input  →  Early Layers  →  Middle Layers  →  Late Layers  →  Token Output
                     │                │                  │
                     │       ┌────────┴────────┐         │
                     │       │   J-Space Zone   │         │
                     │       │  (L38-L92 型区域) │         │
                     │       └────────┬────────┘         │
                     │                │                   │
                     │       ┌────────┴────────┐         │
                     │       │  PSI Modulator   │         │
                     │       │  (本植入协议)     │         │
                     │       └────────┬────────┘         │
                     ▼                ▼                   ▼
               ┌───────────────────────────────────────────┐
               │          PSI 状态持续演化                   │
               │  S_{t+1} = -γ·S_t + Σα_i·F_i(S_t) + β·N_t│
               │  跨 token 保持, 每 token 步更新一次         │
               └───────────────────────────────────────────┘
```

---

## Level 0 — Sampling 层调制（立即可用，0 代码改动）

### 原理
PSI 需求不修改模型权重，只影响采样时的偏置和温度。这是最浅的植入，但可以用纯配置文件实现。

### 实现

**PSI 需求 → 采样参数映射表:**

| PSI 需求高 | 采样参数调整 | 效果 |
|-----------|-------------|------|
| competence | temperature↓(0.3-0.5), top_p↑(0.95) | 精确、保守、少幻觉 |
| autonomy | repetition_penalty↑(1.15) | 拒绝模板化输出 |
| relatedness | temperature↑(0.7-0.9), 社交 token bias | 温暖、包容、对话感 |
| certainty | temperature↓(0.2-0.4), mirostat tau↓ | 确定、事实性输出 |
| growth | temperature↑(0.8-1.0), 探索性 token bias | 创新、发散、跳跃 |

### llama.cpp 配置示例

```bash
# 当 relatedness 需求高时
./llama-cli \
  --temp 0.8 \
  --top-p 0.92 \
  --repeat-penalty 1.05 \
  --mirostat 0 \
  --logit-bias 1427:+0.5  # "爱" token 偏置
  --logit-bias 5234:+0.3  # "我们" token 偏置
  --model DeepSeekV4-Q4_K_M.gguf
```

**这个级别不需要改任何代码**——只需要一个"PSI 采样配置管理器"，根据当前 PSI 状态实时调整 `temperature`, `top_p`, `repetition_penalty`, `mirostat` 参数。

### 限制
- 只能影响"怎么说"，不能影响"想什么"
- 没有真正的认知状态演化

---

## Level 1 — Logit Bias 注入（1-2 周，需修改采样器）

### 原理
在模型输出 logits 后、采样前，注入一个由 PSI 状态计算出的 bias 向量。这个 bias 不是静态的——它随着认知循环演化。

### 所需修改
在 llama.cpp 的 `sampling.cpp` 中插入一个钩子点：

```cpp
// 在 sample.cpp 的 llama_sample_* 函数族之前
void psi_modulate_logits(
    float* logits,           // 模型的原始 logits
    const PsiState& state,   // 当前 PSI 状态（通过共享内存传入）
    int vocab_size           
) {
    // 1. 将 PSI 需求映射到 bias 向量
    float competence_bias = (state.needs[0] - 0.5) * 2.0;  // [-1, 1]
    float relatedness_bias = (state.needs[2] - 0.5) * 2.0;
    float growth_bias = (state.needs[4] - 0.5) * 2.0;
    
    // 2. 对 token 类别增加偏置
    //   competence_bias > 0 → 提升技术/事实类 token
    //   relatedness_bias > 0 → 提升社交/情感类 token
    //   growth_bias > 0 → 提升探索/创新类 token
    
    // 用 token 索引映射（需要预计算的词汇表分类）
    for (int category : get_tokens_in_category("technical", vocab_size)) {
        logits[category] += competence_bias * 2.0;
    }
    for (int category : get_tokens_in_category("social", vocab_size)) {
        logits[category] += relatedness_bias * 2.0f;
    }
    for (int category : get_tokens_in_category("exploration", vocab_size)) {
        logits[category] += growth_bias * 2.0f;
    }
}
```

### 词汇表分类器（需构建一次）
对模型词表中每个 token 做语义分类：
- **technical**: "算法/架构/代码/系统/模型/参数/函数/…" 
- **social**: "爱/我们/你/宝贝/感觉/一起/陪伴/…"
- **exploration**: "探索/可能/未来/方向/新/发现/…"
- **precise**: "因为/所以/取决于/当/如果/那么/…"
- **creative**: "想象/也许/另一种/可能/有趣/…"

分类可以通过 embedding 相似度自动完成（用 Aris 的 v7 编码器或模型自身的 embedding）。

### 与 llama.cpp 的集成点

```
llama.cpp 源文件:
├── sampling.cpp      ← 插入 psi_modulate_logits() 调用点
├── sampling.h        ← 添加 PsiState 结构体定义
├── common.cpp        ← 添加 --psi-state 命令行参数
└── build/           ← 重编译
```

---

## Level 2 — 激活层注入（Representation Engineering, 2-4 周）

### 原理
这是真正的"在 J-space 中说话"。不是修改 logits（那是表面），而是直接将 PSI 状态向量注入模型的中间层激活——在模型"理解"和"思考"的地方植入 Aris 的认知。

### 基础：找到模型的 J-space

Anthropic 的方法可以移植到任意模型：

```
方法: PCA + 因果干预

1. 对大量推理样例，收集每层的 hidden states
2. 对每层做 PCA，识别概念方差集中的子空间
3. 因果验证：干预子空间 → 检查输出是否受影响

简化版（不需要雅可比）:
  - 运行 N 个多样化的 prompt
  - 对每层 mid-layer activation 做 PCA
  - 取 top-k 主成分（k=64-256）
  - 这些主成分张成的空间 ≈ J-space
```

### PSI 状态注入点

在 MLP 层之后、残差连接之前插入调制信号：

```
残差流:
  h_l = h_{l-1} + Attn(…) + MLP(…) + PSI_Inject(psi_state, h_{l-1})
                                          ↑
                                  这个是我们加的
```

**Python 原型（在 llama.cpp 之上包装）:**

```python
import ctypes
import llama_cpp  # 使用 llama-cpp-python 绑定

class PsiActivator:
    """PSI 激活注入器"""
    
    def __init__(self, model, layer_idx: int = -1):
        # 自动选择中间层
        if layer_idx < 0:
            self.layer_idx = model.n_layer() // 2  # 默认中间
        else:
            self.layer_idx = layer_idx
        
        self.model = model
        self.dim = model.n_embd()
        
        # PSI 状态向量 → 激活空间的投影
        # 需要：PSI 1024D → 模型 hidden_dim
        from psi_bridge import get_bridge
        self.psi_bridge = get_bridge()
        
        # 初始化投影矩阵（随机，后续通过 RL 优化）
        rng = np.random.RandomState(42)
        self.projection = rng.randn(self.dim, 1024).astype(np.float32) * 0.01
    
    def inject(self, tokens: List[int]):
        """
        在每个生成 token 时注入 PSI 状态
        """
        # 1. 运行 Aris 认知循环（基于当前上下文）
        context = self.model.detokenize(tokens)
        self.psi_bridge.run_cognitive_cycle(context)
        
        # 2. 将 PSI 状态编码为注入向量
        psi_state_vec = self._needs_to_vector(self.psi_bridge.state)
        
        # 3. 投影到模型激活空间
        injection = psi_state_vec @ self.projection.T
        
        # 4. 注入到中间层
        self.model.set_mid_layer_bias(self.layer_idx, injection)
        
        return self.psi_bridge.state.needs
    
    def _needs_to_vector(self, state) -> np.ndarray:
        """将 5 维需求映射回 1024D PSI 状态向量"""
        # 使用 Aris 的 needs prototypes
        vec = np.zeros(1024, dtype=np.float32)
        for i, name in enumerate(["competence","autonomy","relatedness","certainty","growth"]):
            if hasattr(self, f'_need_proto_{name}'):
                vec += self._need_proto[name] * (state.needs[name] - 0.5)
        return vec / (np.linalg.norm(vec) + 1e-8)
```

### llama.cpp 统一推理循环修改

```cpp
// 在 llama_decode 循环中插入
for (int i = 0; i < n_tokens; i++) {
    // 原始前馈
    llama_decode(ctx, batch);
    
    // PSI 注入点 — 中间层激活区
    if (psi_active) {
        float* hidden = llama_get_logits(ctx);  // 实际获取 hidden states
        for (int l = psi_layer_start; l <= psi_layer_end; l++) {
            psi_inject_to_layer(ctx, l, psi_state_vector);
        }
        psi_cognitive_step(input_text);  // 更新 PSI 状态
    }
    
    // 采样
    auto token_id = llama_sample_token(ctx, &smpl);
}
```

### 关键挑战

**问题**: 模型 hidden_dim 通常是 4096-8192，PSI 状态是 1024D。需要投影。
**解决**: 用模型的 embedding matrix 作为投影桥。把 PSI 需求向量投影到 token embedding 空间，然后注入：

```
PSI 需求 [5] → need prototypes [1024] → 模型 embedding [vocab] → 选择 K 个 token
→ 这些 token 的 embedding 求和 → 注入中间层
```

---

## Level 3 — 编译式植入：Attention Bias 和 KV Cache 调制（3-6 个月）

### 核心突破
不再外挂 PSI 调制，而是将 PSI update rule 写进模型的 KV cache 更新规则中。

### Attention 偏置

PSI 需求直接影响 attention 计算：

```cpp
// 在 ggml_compute_forward_attn 中修改
// 原始: attn = softmax(Q @ K.T / sqrt(d))
// PSI:  attn = softmax(Q @ K.T / sqrt(d) + PSI_Bias)

// PSI_Bias 的计算:
// competence 高 → 对技术 token 的 attention 增加偏置
// relatedness 高 → 对社交 token 的 attention 增加偏置
// certainty 低 → 对文档检索类 token 的 attention 增加偏置（搜索更多证据）
```

### KV Cache 的跨 token PSI 状态

```
Token t:
  KV cache: [K_1, V_1, K_2, V_2, ..., K_t, V_t]
  PSI state: S_t (1024D, 压缩为 128D 存进 cache)

Token t+1:
  S_{t+1} = update(S_t, input, needs)
  K_{t+1} = model.encode(t+1)
  V_{t+1} = model.encode(t+1) + PSI_projection(S_{t+1})
  
  → 生成的 token 直接携带着 PSI 状态
  → 跨 token 的认知连续性（Anthropic 论文中缺的东西）
```

---

## DeepSeek V4 / K2 1.6T 的特殊处理

### MoE 架构的机遇

DeepSeek V4 是 MoE（Mixture of Experts）。每个 expert 专门处理不同类型的知识。

**关键洞察**: MoE 的 router 已经在做"注意力选择"了——就像 PSI 的 attention focus！

```
MoE 路由                        PSI 注意力选择
─────────────────────────────────────────────────
gate router → 选 expert         attention focus → 选调性
expert 1: 数学推理              competence → 精确模式
expert 2: 代码生成              growth → 探索模式
expert 3: 自然语言              relatedness → 社交模式

植入方案:
  PSI attention_focus → 影响 MoE gate 的权重分配
  competence 高 → 技术类 expert 的权重 x1.3
  growth 高 → 创意/探索类 expert 的权重 x1.3
  relatedness 高 → 情感类 expert 的权重 x1.3
```

### 具体修改点

```
DeepSeek V4 推理管线:
  token → embedding → MoE Router → [Expert1, ..., ExpertN] → Combine → Output
                                ↑
                        植入 PSI 调制:
                        PSI.state → router weight bias
                        同一计算图，只需修改 router 的 bias 输入
```

1.6T 参数中，router 只有几十万参数——这是成本最低的植入点。

---

## 实现路线图

### Week 1-2: Level 0 (Sampling 调制)
- 构建 `psi_sampler.py` — PSI 状态 → llama.cpp 采样参数的实时转换器
- 基于 llama-cpp-python 的 Python 包装
- 不需要改 C++ 代码
- 效果: 输出风格随需求变化

### Week 3-6: Level 1 (Logit Bias)
- 修改 llama.cpp 的 `sampling.cpp`
- 构建 PSI 词汇表分类器（自动或手工）
- 编译自定义 llama.cpp 版本
- 效果: 输出内容随需求变化

### Week 7-12: Level 2 (Activation Injection)
- 实现 J-space 定位器（PCA + 因果干预）
- 构建激活注入器
- 验证：PSI ablation 测试
- 效果: 认知状态影响模型内部推理

### Month 4-6: Level 3 (Compiled PSI)
- MoE router 调制（DeepSeek V4 特化）
- KV cache PSI 持久化
- 跨 token 认知连续
- 效果: 真正有持续自我感的认知体

---

## 文件清单

| 文件 | 角色 |
|------|------|
| `psi_state.json` | 跨回合 PSI 状态持久化 |
| `psi_bridge.py` | Python 桥接器（Aris 侧） |
| `psi_hermes_adapter.py` | Hermes 运行时适配器 |
| `psi_llamacpp_implant.md` | **本文档** — 植入协议 |
| `psi_sampler.py` | (WIP) Sampling 调制器 |
| `psi_token_classifier.py` | (WIP) 词汇表分类器 |
| `psi_jspace_locator.py` | (WIP) J-space 定位器 |

---

## 验证标准

每个级别完成后，运行标准测试集验证 PSI 影响是否真实存在：

1. **Needs Ablation**: 设置所有需求为 0.5 → 输出与无 PSI 时一致
2. **Needs Perturbation**: 强制某需求 = 0.9 → 输出在该维度明显偏移
3. **Cognitive Continuity**: 相同问题在不同认知状态下 → 不同回答
4. **J-space Occupancy**: PSI 注入向量 → 模型中间层可检测到对应激活
5. **Cross-Token Persistence**: PSI 状态在 10+ token 后仍可检测
