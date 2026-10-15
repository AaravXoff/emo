"""Flask web remote controller only; local facial animation stays in Pygame."""
from __future__ import annotations
import hmac
import logging
import threading

log = logging.getLogger(__name__)


class RemoteController:
    def __init__(self, status, command_handler, host="0.0.0.0", port=8766, token="") -> None:
        self.status, self.command_handler = status, command_handler
        self.host, self.port = host, int(port)
        self.token = (token or "").strip()
        self.app = self.server = self.thread = None

    def start(self) -> tuple[bool, str]:
        if not self.token:
            return False, "Set REMOTE_CONTROL_TOKEN in .env to enable remote control"
        try:
            from flask import Flask, jsonify, render_template, request
            from werkzeug.serving import make_server
        except ImportError:
            return False, "Flask is missing; install requirements.txt"
        app = Flask("emo_remote", template_folder="templates")

        def authorized():
            return hmac.compare_digest(self.token, request.headers.get("X-EMO-Token", ""))

        @app.get("/")
        def index():
            return render_template("remote.html")

        @app.get("/health")
        def health():
            return jsonify({"ok": True})

        @app.get("/api/status")
        def api_status():
            if not authorized():
                return jsonify({"error": "Unauthorized"}), 401
            return jsonify(self.status.snapshot())

        @app.post("/api/command")
        def api_command():
            if not authorized():
                return jsonify({"error": "Unauthorized"}), 401
            payload = request.get_json(silent=True) or {}
            command = str(payload.get("command", "")).strip().casefold()
            argument = str(payload.get("argument", "")).strip()[:300]
            try:
                result = self.command_handler(command, argument)
            except Exception as exc:
                log.exception("Remote command failed")
                return jsonify({"error": str(exc)[:180]}), 400
            if not result:
                return jsonify({"error": "Unsupported command or motion"}), 400
            return jsonify({"ok": True, "result": str(result)[:180]})

        self.app = app
        try:
            self.server = make_server(self.host, self.port, app, threaded=True)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="EMO-Remote")
            self.thread.start()
        except Exception as exc:
            self.server = None
            return False, f"Could not start remote controller: {exc}"
        return True, f"Remote controller listening at http://{self.host}:{self.port}"

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        self.thread = None
