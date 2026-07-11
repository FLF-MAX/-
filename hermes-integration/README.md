# LAAP + Hermes Agent 整合

本目录包含将 LAAP Brain 接入 Hermes Agent 的完整方案，包含两种集成方式：

1. **MCP Server + Skill 集成**（推荐，非侵入式）
2. **源码级 System Prompt 注入**（侵入式，更深）

## 已完成的改动

### LAAP 侧

- 扩展 `aris_brain/laap_brain_api.py`：
  - `POST /v1/cognitive_state` — 获取 PSI 认知状态
  - `POST /v1/recall_memory` — 召回 LAAP 记忆
  - `POST /v1/reflect` — 反思并更新状态
- 新建 `mcp_server/laap_mcp_server.py`：LAAP 的 MCP 服务器（stdio/SSE 双模式）

### Hermes 侧

- 新建 `skills/laap-bridge/SKILL.md`：教 Hermes 何时/如何调用 LAAP 工具
- 修改 `agent/system_prompt.py`：在 volatile system prompt 中自动注入 LAAP 认知状态
- 备份：`agent/system_prompt.py.laap-backup`

## 快速启动

### 1. 启动 LAAP API

```powershell
cd D:\laap-AGI\aris_brain
python laap_brain_api.py --port 11546
```

### 2. 配置 Hermes MCP

把 `hermes-integration/hermes-config-laap-example.yaml` 中的 `mcp_servers` 块复制到：

```
C:\Users\<you>\.hermes\config.yaml
```

注意：把路径改成你自己的 Python 路径和 LAAP 路径。

### 3. 启动 Hermes

```powershell
hermes chat --skills laap-bridge
```

## 一键启动脚本

```powershell
D:\laap-AGI\hermes-integration\start_laap_hermes.bat 11546
```

## 源码级集成说明

修改后的 `agent/system_prompt.py` 会在每次构建 system prompt 时：

1. 读取环境变量 `LAAP_API_BASE`（默认 `http://localhost:11546`）
2. 调用 `POST /v1/cognitive_state` 获取 PSI 状态
3. 把 preamble 注入到 volatile system prompt tier

这不会破坏 prompt caching，因为只影响 volatile tier。

## 回滚

如果源码级集成导致问题，恢复备份：

```powershell
$HermesHome = "$env:LOCALAPPDATA\hermes\hermes-agent"
Copy-Item -Path "$HermesHome\agent\system_prompt.py.laap-backup" -Destination "$HermesHome\agent\system_prompt.py" -Force
```
