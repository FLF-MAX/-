"""LAAP 健康仪表盘 API 服务器
提供实时健康数据供 _dashboard.html 读取。
端口: 11521 (独立；11520 为 QUANTUM_PORT 默认，避免冲突)
启动: python state_snapshot_server.py
"""
import json, time, logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

BRAIN = Path(__file__).parent.resolve()
STATE = BRAIN / "state"
PORT = 11521

# 延迟导入（避免循环依赖）
_snapshot = None
def _get_snapshot():
    global _snapshot
    if _snapshot is None:
        import state_snapshot as _snapshot
    return _snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [VIZ] %(message)s")
logger = logging.getLogger("aris.viz")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        snap = _get_snapshot()
        
        if self.path == "/health":
            health = snap.compute_health_score()
            timeline = snap.get_health_timeline(hours=72)
            snaps = snap.list_snapshots()
            versions = snap.version_list()
            
            data = {
                "health": {
                    "total": health["total"],
                    "details": {k: round(v["score"], 3) for k, v in health.get("details", {}).items()},
                },
                "timeline": timeline,
                "snapshots": snaps[:20],
                "versions": versions,
                "timestamp": time.time(),
            }
            self._json(data)

        elif self.path == "/" or self.path == "/dashboard":
            self._file("snapshots/_dashboard.html", "text/html")

        elif self.path.startswith("/snap/"):
            snap_name = self.path[6:]
            result = snap.restore_incremental(snap_name, dry_run=True)
            self._json(result)

        elif self.path == "/snapshot":
            meta = snap.snapshot_on_event("manual", "仪表盘手动触发")
            self._json(meta or {"error": "snapshot failed"})

        elif self.path == "/versions":
            self._json(snap.version_list())

        elif self.path == "/health/now":
            # 立即执行一次完整健康检查
            health = snap.compute_health_score()
            snap._append_health_timeline(health["total"])
            self._json(health)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "not found"}')

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _file(self, rel_path, mime):
        fp = BRAIN / rel_path
        if fp.exists():
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.end_headers()
            self.wfile.write(fp.read_bytes())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        logger.info(f"{self.client_address[0]} {fmt % args}")


def start_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    logger.info(f"📊 LAAP 仪表盘: http://localhost:{PORT}/dashboard")
    logger.info(f"📡 健康API: http://localhost:{PORT}/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        logger.info("仪表盘服务器已停止")


if __name__ == "__main__":
    start_server()
