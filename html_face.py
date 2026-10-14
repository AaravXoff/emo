"""Small Flask adapter for the HTML/CSS face renderer."""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from flask import Flask, jsonify, send_file
from werkzeug.serving import make_server


class HtmlFace:
    def __init__(self, status=None, host="127.0.0.1", port=8765):
        self.status = status
        self.host = host
        self.port = port
        self.html_path = Path(__file__).with_name("face.html")
        self.lock = threading.RLock()
        self.state = "boot"
        self.state_started = time.monotonic()
        self.state_until = None
        self.talking = False
        self.app = Flask("emo_face")

        @self.app.get("/")
        def index():
            return send_file(self.html_path, max_age=0)

        @self.app.get("/api/state")
        def api_state():
            return jsonify(self.snapshot())

        @self.app.get("/api/boot")
        def api_boot():
            data = self.snapshot()
            return jsonify({
                "boot_complete": data["boot_complete"],
                "current_stage": data["current_stage"],
                "current_message": data["current_message"],
                "boot_steps": data["boot_steps"],
            })

        @self.app.get("/health")
        def health():
            return jsonify({"ok": True})

        self.server = None
        self.server_thread = None
        self._server_lock = threading.RLock()

    def start_server(self):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        with self._server_lock:
            if self.server_thread and self.server_thread.is_alive():
                return
            self.server = make_server(self.host, self.port, self.app, threaded=True)
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True,
                name="EMO-HTML-Server",
            )
            self.server_thread.start()
            print(f"EMO HTML face: http://{self.host}:{self.port}/")

    def set_state(self, state, duration=None):
        with self.lock:
            self.state = str(state)
            self.state_started = time.monotonic()
            self.state_until = (
                self.state_started + max(0, float(duration))
                if duration is not None
                else None
            )

    def get_state(self):
        return self.snapshot()["state"]

    def set_talking(self, active):
        with self.lock:
            self.talking = bool(active)

    def update(self):
        # Compatibility with the previous face interface.
        pass

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if self.state_until is not None and now >= self.state_until:
                self.state = "idle"
                self.state_until = None
            state = self.state
            talking = self.talking

        runtime = {}
        if self.status is not None:
            try:
                runtime = self.status.snapshot()
            except Exception:
                runtime = {}
        temperature = float(runtime.get("temperature", 0.0) or 0.0)
        warning = bool(runtime.get("thermal_warning", False))
        lockout = bool(runtime.get("thermal_lockout", False))
        if warning or lockout:
            state = "overheat"

        return {
            "state": state,
            "talking": talking,
            "temperature": temperature,
            "thermal_warning": warning,
            "thermal_lockout": lockout,
            "camera_found": bool(runtime.get("camera_found", False)),
            "last_gesture": str(runtime.get("last_gesture", "")),
            "boot_complete": bool(runtime.get("boot_complete", True)),
            "current_stage": str(runtime.get("current_stage", "")),
            "current_message": str(runtime.get("current_message", "")),
            "boot_steps": runtime.get("boot_steps", {}),
            "last_action": str(runtime.get("last_action", "")),
            "last_error": str(runtime.get("last_error", "")),
        }

    def stop_server(self):
        # Keep start/stop serialized so a quick restart cannot race a still-exiting thread.
        with self._server_lock:
            server = self.server
            thread = self.server_thread
            if server is not None:
                server.shutdown()
                server.server_close()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)
            self.server = None
            self.server_thread = thread if thread is not None and thread.is_alive() else None
