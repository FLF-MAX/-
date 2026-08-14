# laap_v2 部署指南 (Deployment Guide)

## 环境要求

- Python ≥ 3.10
- 依赖：`numpy>=1.24`（核心）；`fastapi`, `uvicorn`, `requests`（API/LLM）
- 可选：`DEEPSEEK_API_KEY` 启用 LLM 响应；无 key 时使用本地规则合成器

## 方式一：裸机 / systemd

```bash
pip install -r requirements.txt
cat > /etc/systemd/system/laap-v2.service <<'EOF'
[Unit]
Description=laap_v2 cognitive runtime
After=network.target

[Service]
Type=simple
User=laap
WorkingDirectory=/srv/laap_v2
Environment=DEEPSEEK_API_KEY=sk-xxx
ExecStart=/usr/bin/python /srv/laap_v2/api_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now laap-v2
```

## 方式二：Docker

```bash
docker build -t laap-v2 .
docker run -d --name laap-v2 -p 11546:11546 \
  -e DEEPSEEK_API_KEY=sk-xxx \
  --restart unless-stopped laap-v2
```

或 docker-compose（多副本 + 健康检查）：

```bash
docker compose up --scale api=3 -d
```

## 方式三：Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

- `deployment.yaml` 含就绪/存活探针（`/health/ready`, `/health/live`）；
- `hpa.yaml` 按 CPU 自动扩缩（min 2，max 10）；
- 所有配置通过 `ConfigMap` 注入（不把密钥写进镜像）。

## 健康检查语义

| 探针 | 路径 | 语义 |
|---|---|---|
| 存活 | `/health/live` | 进程响应即存活 |
| 就绪 | `/health/ready` | PSI 心跳 + 记忆 + 元学习全部正常才就绪 |

## 混沌自愈演练

```bash
curl -X POST localhost:11546/v1/chaos/psi   # 制造 PSI 心跳故障
curl -X POST localhost:11546/v1/chat -d '{"message":"hello"}'  # 自动恢复
```

## 限流与指标

- 每 IP 令牌桶限流（默认 60 次/分钟，`LAAP_SERVER_RATE_LIMIT` 可调）；
- `GET /v1/metrics` 返回 uptime/吞吐/延迟 p50-p99/错误率。

## 观测

结构化日志默认输出到 stdout；设置 `LAAP_LOG_DIR=/var/log/laap` 开启滚动文件。