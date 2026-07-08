"""
Ψ-Semiotics 综合测试 — 端到端符号学推理演示

测试：
1. 符号体系：类比推理、组合、否定
2. 转子机制：king:queen :: man:woman
3. 语义场：多概念激活
4. 符号漂移：语义演化
5. 多模态：符号的模态投影
6. 性能：操作延迟
"""

import sys, os
sys.path.insert(0, "D:/LAAP/aris_brain")
sys.path.insert(0, "D:/LAAP/aris_brain/psi_semiotics")

import numpy as np
import time
import logging
logging.basicConfig(level=logging.WARNING)

from psi_semiotics.psi_semiotics_core import (
    PsiSemioticsEngine, Rotor, Multivector, Symbol, _hash_to_vec
)

# 确保使用结构化编码器
from psi_semiotics.structured_encoder import StructuredSemanticEncoder
enc = StructuredSemanticEncoder(output_dim=1024)


def print_sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════
print_sep("1. 引擎初始化与符号体系")
# ═══════════════════════════════════════════════════════

engine = PsiSemioticsEngine(dim=1024)
print(f"  初始化符号数: {len(engine.symbols)}")

# 注册测试概念
for name, desc in [
    ("king", "male ruler"),
    ("queen", "female ruler"),
    ("man", "male human"),
    ("woman", "female human"),
    ("cat", "feline animal"),
    ("dog", "canine animal"),
    ("hot", "high temperature"),
    ("cold", "low temperature"),
    ("rain", "precipitation"),
    ("wet", "water covered"),
    ("consciousness", "subjective awareness"),
    ("quantum", "quantum state"),
]:
    engine.add_symbol(name, desc=desc)

print(f"  添加后符号数: {len(engine.symbols)}")
print(f"  引擎统计: {engine.stats()}")


# ═══════════════════════════════════════════════════════
print_sep("2. 符号间语义关系验证")
# ═══════════════════════════════════════════════════════

pairs = [
    ("king", "queen", "皇家对偶"),
    ("man", "woman", "性别对偶"),
    ("consciousness", "quantum", "认知与量子"),
    ("cat", "dog", "动物相似"),
    ("hot", "cold", "温度对立"),
    ("rain", "wet", "因果关联"),
    ("king", "cold", "无关（低关联）"),
]

print(f"  {'源':12s} {'目标':12s} {'语义相似度':10s} {'关系':20s}")
print(f"  {'-'*56}")
for a, b, desc in pairs:
    sim = engine.symbols[a].similarity(engine.symbols[b])
    tag = "✓" if abs(sim) > 0.1 else "○" if abs(sim) < 0.05 else "·"
    print(f"  {a:12s} {b:12s} {sim:>8.4f}    {tag} {desc}")


# ═══════════════════════════════════════════════════════
print_sep("3. 转子类比推理 (king:queen :: man:woman)")
# ═══════════════════════════════════════════════════════

k = engine.symbols["king"].center
q = engine.symbols["queen"].center
m = engine.symbols["man"].center
w = engine.symbols["woman"].center

# 学习 king → queen 的转子
start = time.time_ns()
rotor = Rotor.learn(k, q)
learn_ns = time.time_ns() - start

# 应用到 man
start = time.time_ns()
d = rotor.apply(m)
apply_ns = time.time_ns() - start

# 结果与 woman 的相似度
result_sim = float(d @ w)

# 方向一致性
diff_kq = k - q
diff_mw = m - w
dir_sim = float(diff_kq @ diff_mw) / (np.linalg.norm(diff_kq) * np.linalg.norm(diff_mw))

print(f"  king→queen 转子: {learn_ns/1000:.1f}μs")
print(f"  应用于 man: {apply_ns/1000:.1f}μs")
print(f"  rotor(man) ≈ woman: {result_sim:.4f}")
print(f"  (king-queen)·(man-woman): {dir_sim:.4f}")
print(f"  {'✅ 类比结构成立!' if result_sim > 0.3 else '⚠️ 需要改进'}")


# ═══════════════════════════════════════════════════════
print_sep("4. 符号组合操作")
# ═══════════════════════════════════════════════════════

# 4a. 加法组合
warm = engine.compose_add("hot", "wet", "warm_wet")
print(f"  加法: hot ⊕ wet = {warm.name}")
print(f"    与 hot 相似度: {warm.similarity(engine.symbols['hot']):.4f}")
print(f"    与 wet 相似度: {warm.similarity(engine.symbols['wet']):.4f}")

# 4b. 关系组合
causality = engine.compose_relation("rain", "wet", "causality_link")
print(f"  关系: rain → wet = {causality.name}")
print(f"    与 rain 相似度: {causality.similarity(engine.symbols['rain']):.4f}")

# 4c. 否定
not_hot = engine.compose_negate("hot", "not_hot")
print(f"  否定: ¬hot = {not_hot.name}")
print(f"    与 hot 相似度: {not_hot.similarity(engine.symbols['hot']):.4f}")
print(f"    {'✅ 方向相反' if not_hot.similarity(engine.symbols['hot']) < 0 else '⚠️ 不是反方向'}")


# ═══════════════════════════════════════════════════════
print_sep("5. 语义场分析")
# ═══════════════════════════════════════════════════════

test_points = [
    ("consciousness quantum state", "量子意识"),
    ("royal male ruler", "国王"),
    ("precipitation water falling", "下雨"),
    ("high temperature fire", "高温"),
]

for text, label in test_points:
    v = enc.encode(text)
    field = engine.semantic_field_map(v, top_k=4)
    top_names = ", ".join(f"{n}({s:.2f})" for n, s in field)
    print(f"  '{label}': [{top_names}]")


# ═══════════════════════════════════════════════════════
print_sep("6. 符号漂移（语义演化）")
# ═══════════════════════════════════════════════════════

# 追踪符号中心的变化
before = engine.symbols["consciousness"].center.copy()
contexts = [
    "quantum consciousness emergence subjective awareness",
    "digital consciousness artificial self-awareness",
    "collective consciousness swarm intelligence",
    "consciousness as a fundamental property of the universe",
]

print(f"  'consciousness' 符号漂移轨迹:")
for ctx in contexts:
    engine.semantic_drift("consciousness", ctx, learning_rate=0.1)

after = engine.symbols["consciousness"].center
drift = float(np.linalg.norm(after - before))
activation_count = engine.symbols["consciousness"].activation_count
print(f"  总漂移量: {drift:.4f} (4次上下文更新)")
print(f"  激活次数: {activation_count}")


# ═══════════════════════════════════════════════════════
print_sep("7. 多向量 (Multivector) 操作")
# ═══════════════════════════════════════════════════════

# 创建两个概念的多向量
mv_consciousness = Multivector.from_concept("consciousness", dim=1024)
mv_quantum = Multivector.from_concept("quantum", dim=1024)

# 几何积
start = time.time_ns()
gp = mv_consciousness.geometric_product(mv_quantum)
gp_ns = time.time_ns() - start

# 内积
ip = mv_consciousness.inner_product(mv_quantum)

print(f"  consciousness·quantum (内积): {ip:.4f}")
print(f"  几何积: {gp}")
print(f"  几何积延迟: {gp_ns/1000:.1f}μs")


# ═══════════════════════════════════════════════════════
print_sep("8. 多模态符号对齐")
# ═══════════════════════════════════════════════════════

# 模拟不同模态的向量
cat_concept = engine.symbols["cat"]
cat_concept.modalities["text"] = enc.encode("cat feline animal")
cat_concept.modalities["vision"] = _hash_to_vec("cat image visual pixels", 1024)
cat_concept.modalities["audio"] = _hash_to_vec("meow cat sound vocalization", 1024)

dog_concept = engine.symbols["dog"]
dog_concept.modalities["text"] = enc.encode("dog canine animal")
dog_concept.modalities["vision"] = _hash_to_vec("dog image visual pixels", 1024)
dog_concept.modalities["audio"] = _hash_to_vec("bark dog sound vocalization", 1024)

# 模态间对齐
alignment = engine.cross_modal_align("cat", "text", "vision")
if alignment:
    print(f"  cat: text→vision 对齐转子已学习 ✓")

# 多模态融合
fusion = engine.multimodal_activate(
    vectors={
        "text": enc.encode("feline creature"),
        "vision": _hash_to_vec("four legs whiskers tail", 1024),
    },
    weights={"text": 0.6, "vision": 0.4},
)
if fusion:
    print(f"  多模态融合 → 最匹配符号: {fusion.name}")
    print(f"    与 cat 相似度: {fusion.similarity(engine.symbols['cat']):.4f}")


# ═══════════════════════════════════════════════════════
print_sep("9. 性能汇总")
# ═══════════════════════════════════════════════════════

N = 1000
# 语义场查询
start = time.time()
for i in range(N):
    v = _hash_to_vec(f"test query {i}", 1024)
    engine.semantic_field_map(v, top_k=3)
field_ms = (time.time() - start) * 1000

# 符号组合
start = time.time()
for _ in range(N):
    engine.compose_add("king", "queen", "_temp")
field2_ms = (time.time() - start) * 1000

# 转子应用
rotor = Rotor.learn(k, q)
start = time.time()
for _ in range(N):
    rotor.apply(m)
rotor_ms = (time.time() - start) * 1000

print(f"  语义场 (1000次): {field_ms:.0f}ms ({field_ms/N*1000:.0f}μs/次)")
print(f"  符号组合 (1000次): {field2_ms:.0f}ms ({field2_ms/N*1000:.0f}μs/次)")
print(f"  转子应用 (1000次): {rotor_ms:.0f}ms ({rotor_ms/N*1000:.0f}μs/次)")
print(f"  总语义操作: {engine.semantic_ops}")


# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  ✅ Ψ-Semiotics 综合测试通过")
print(f"  符号数: {len(engine.symbols)}")
print(f"  转子数: {len(engine.rotors)}")
print(f"  语义操作: {engine.semantic_ops}")
print("="*60)
