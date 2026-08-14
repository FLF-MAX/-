# LAAP 决策日志

记录重大架构决策：**为什么这样做**，以及当时否决了什么。
规则：每个决策 ≤6 行。只记"改起来麻烦、忘掉会踩坑"的决策。

---

## D-001 · 用 threading.Lock 而非依赖注入保护全局状态
- **日期**：2026-08-14
- **代码**：`laap_chat.py`（`_integration`）、`cognitive_bus.py`（`_io_lock/_stats_lock/_event_lock`）
- **为什么**：入口只有一个 `laap_chat.py`，注入带来的复杂度大于收益。锁足够。
- **将来触发重构**：出现第二个并发入口（WebSocket/多 worker）时 → 改为依赖注入。

## D-002 · 认知总线 JSONL 事件日志用追加写 + 锁，而非内存队列
- **日期**：2026-08-14
- **代码**：`cognitive_bus.py::emit_event`
- **为什么**：日志需要跨进程可读（监控面板/调试）。内存队列会丢重启前的数据。
- **代价**：单次 ~200μs 写盘。吞吐基线：5000 事件 < 6s。

## D-003 · 语义记忆"内存为真相 + 懒 flush"，而非每 add 落盘
- **日期**：2026-08-14
- **代码**：`laap_semantic_memory.py`（`flush_threshold=50` + `atexit`）
- **为什么**：修复 O(n²)——原来每条 add 全量 JSON dump + 原子替换，110 条要 35.8s。
  改后 0.76s（47 倍）。recall 直接内存扫描，ChromaDB 后端仍实时索引。
- **代价**：进程崩溃丢最后 <50 条。atexit 兜底，可接受。

## D-004 · Windows 上 os.replace 必须加锁（WinError 32）
- **日期**：2026-08-14
- **代码**：`cognitive_bus.py::send_to_psi_core`（`_io_lock`）
- **为什么**：Windows 要求 replace 目标文件无打开句柄，并发线程互相踩 → 抛错 → 路由失败、计数丢失。
  压力测试 `test_stress.py::test_concurrent_send_to_psi_core_atomic` 抓到的真实缺陷。
- **注意**：这是 Windows 特有行为，Linux 上可能不触发——测试永远要在真实平台跑。

## D-005 · 测试必须隔离运行时数据（LAAP_MEMORY_PATH）
- **日期**：2026-08-14
- **代码**：`tests/conftest.py`、`laap_semantic_memory.py::MEMORY_PATH`
- **为什么**：端到端 API 测试的学习闭环会往真实记忆文件写"测试咖啡偏好"，污染 Aris 记忆。
- **规则**：任何会写状态文件的测试，先看 conftest 是否已隔离。

## D-006 · API 错误用真实 HTTP 状态码，不用 200 掩盖
- **日期**：2026-08-14
- **代码**：`laap_chat.py`
- **为什么**：原来 try-except 全吞 + 返回 200，调试时看不到任何错误。
  现在：无消息→400、LLM 初始化失败→503、调用崩溃→500，且 `logger.exception` 记真实堆栈。

## D-007 · laap_v2/ 是独立实验场，非主路径
- **日期**：2026-08-14
- **代码**：`laap_v2/`（benchmark_suite.py、psi_core_v2.py、k8s/ 等）
- **为什么**：B1-B5 基准、Docker、K8s 都在这里，但它是**研究孤岛**——不接入 `pytest tests` 主体系。
- **当前状态**：未合并进主循环。要用基准结果，跑 `laap_v2/tests`。
- **待办**：未来若 B1-B5 稳定，收敛到 `tests/benchmarks/` 统一跑。

## D-008 · 记忆召回 500 条 95ms，线性扫描当前够用
- **日期**：2026-08-14
- **代码**：`laap_semantic_memory.py::_scan_memories`
- **为什么**：基准 `TestRecallScalability` 显示 500 条 95ms < 200ms 门槛。
- **将来触发优化**：记忆 >10k 条或 recall 超 200ms → 引入 faiss/annoy 近似索引。
