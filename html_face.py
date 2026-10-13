import threading
import time
import logging
from pathlib import Path
from flask import Flask, jsonify, send_file
from werkzeug.serving import make_server


class HtmlFace:
    """HTML/CSS face renderer with a small JSON state endpoint."""

    def __init__(self, status=None, host="127.0.0.1", port=8765):
        self.status = status
        self.host = host
        self.port = port
        self.html_path = Path(__file__).with_name("face.html")
        self.lock = threading.Lock()
        self.state = "boot"
        self.state_started = time.monotonic()
        self.state_until = self.state_started + 2.5
        self.talking = False
        self.app = Flask("emo_face")

        @self.app.get("/")
        def index():
            return send_file(self.html_path, max_age=0)

        @self.app.get("/api/state")
        def api_state():
            return jsonify(self.snapshot())

        @self.app.get("/health")
        def health():
            return jsonify({"ok": True})

        self.server = None
        self.server_thread = None

    def start_server(self):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        if self.server_thread and self.server_thread.is_alive():
            return
        self.server = make_server(self.host, self.port, self.app, threaded=True)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
            name="EMO-HTML-Server",
        )
        self.server_thread.start()
        print(f"HTML face: http://{self.host}:{self.port}/")

    def set_state(self, state, duration=None):
        with self.lock:
            self.state = str(state)
            self.state_started = time.monotonic()
            self.state_until = (
                self.state_started + max(0, duration)
                if duration is not None
                else None
            )

    def get_state(self):
        return self.snapshot()["state"]

    def set_talking(self, active):
        with self.lock:
            self.talking = bool(active)

    def update(self):
        # Kept for compatibility with the old face controller.
        pass

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if self.state_until is not None and now >= self.state_until:
                self.state = "idle"
                self.state_until = None
            state = self.state
            talking = self.talking

        temperature = 0.0
        warning = False
        lockout = False
        camera_found = False
        last_gesture = ""

        if self.status is not None:
            with self.status.lock:
                temperature = float(getattr(self.status, "temperature", 0.0))
                warning = bool(getattr(self.status, "thermal_warning", False))
                lockout = bool(getattr(self.status, "thermal_lockout", False))
                camera_found = bool(getattr(self.status, "camera_found", False))
                last_gesture = str(getattr(self.status, "last_gesture", ""))

        if warning or lockout:
            state = "overheat"

        return {
            "state": state,
            "talking": talking,
            "temperature": temperature,
            "thermal_warning": warning,
            "thermal_lockout": lockout,
            "camera_found": camera_found,
            "last_gesture": last_gesture,
        }

    def stop_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
