"""Optional background camera loop connecting requested vision packages to EMO."""
from __future__ import annotations
import logging
import threading
import time

log = logging.getLogger(__name__)


class VisionRuntime:
    def __init__(self, config, status, on_gesture=None):
        self.config, self.status, self.on_gesture = config, status, on_gesture
        self._stop = threading.Event()
        self._thread = None
        self._gesture = self._faces = self._classifier = None

    def start(self):
        if not self.config.camera_enabled:
            return True, "Camera pipeline disabled; enable tasks individually for lower CPU/RAM"
        if not (self.config.gesture_enabled or self.config.face_recognition_enabled or self.config.object_classification_enabled):
            return False, "Camera enabled but no vision task is enabled"
        self._thread = threading.Thread(target=self._run, daemon=True, name="EMO-Vision")
        self._thread.start()
        return True, "Optional camera worker starting"

    def _load_models(self):
        if self.config.gesture_enabled:
            from vision.hand_gestures import HandGestureRecognizer
            self._gesture = HandGestureRecognizer(complexity=0)
        if self.config.face_recognition_enabled:
            from vision.face_identity import FaceIdentityRecognizer
            self._faces = FaceIdentityRecognizer(self.config.known_faces_dir)
        if self.config.object_classification_enabled:
            from vision.item_classification import ItemClassifier
            self._classifier = ItemClassifier(self.config.vision_model, image_size=320)

    def _run(self):
        capture = None
        try:
            import cv2
            self._load_models()
            capture = cv2.VideoCapture(self.config.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera index {self.config.camera_index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            last_gesture, last_action_at, frame_index = "", 0.0, 0
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._stop.wait(0.25)
                    continue
                frame_index += 1
                if self._gesture is not None:
                    result = self._gesture.recognize(frame)
                    gesture = result["gesture"] if result else ""
                    self.status.set_camera(True, gesture)
                    if gesture and gesture != last_gesture and time.monotonic() - last_action_at > 1.5:
                        last_gesture, last_action_at = gesture, time.monotonic()
                        if self.on_gesture:
                            self.on_gesture(gesture)
                if self._faces is not None and frame_index % 6 == 0:
                    log.debug("Detected local identities: %s", [x["name"] for x in self._faces.recognize(frame)])
                if self._classifier is not None and frame_index % 10 == 0:
                    log.debug("Detected objects: %s", [x["label"] for x in self._classifier.classify(frame)[:6]])
                self._stop.wait(0.08)
        except Exception as exc:
            log.warning("Vision worker disabled: %s", exc)
            self.status.set_boot_step("Vision", "warning", str(exc))
        finally:
            if capture is not None:
                capture.release()
            for model in (self._gesture, self._faces):
                if model is not None:
                    try: model.close()
                    except Exception: pass

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None
