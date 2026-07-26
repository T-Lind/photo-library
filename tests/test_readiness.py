from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from photolib.readiness import run_when_ready, wait_for_server


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        status = 200 if self.path == "/api/v1/health" else 404
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_wait_for_server_requires_a_real_health_response():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert wait_for_server(url, attempts=2, interval=0.001)
    finally:
        server.shutdown()
        server.server_close()


def test_run_when_ready_invokes_callback_once():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    calls = []
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        probe = run_when_ready(url, lambda: calls.append(1))
        probe.join(timeout=2)
        assert calls == [1]
    finally:
        server.shutdown()
        server.server_close()
