"""MediaPipe landmarks plus upstream keypoint_classifier.tflite when available."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hand-gesture-recognition-using-mediapipe"


class HandGestureRecognizer:
    """Use the supplied upstream TFLite model; fall back to a light landmark heuristic."""

    def __init__(self, complexity: int = 0, model_root: Path = VENDOR_ROOT) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install the optional vision requirements to enable hand gestures.") from exc
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=complexity,
            min_detection_confidence=0.55, min_tracking_confidence=0.55,
        )
        self._interpreter = None
        self._input_index = None
        self._output_index = None
        self._labels: list[str] = []
        model_path = model_root / "model" / "keypoint_classifier" / "keypoint_classifier.tflite"
        labels_path = model_root / "model" / "keypoint_classifier" / "keypoint_classifier_label.csv"
        if model_path.is_file() and labels_path.is_file():
            try:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    import tensorflow as tf
                    Interpreter = tf.lite.Interpreter
                self._interpreter = Interpreter(model_path=str(model_path), num_threads=1)
                self._interpreter.allocate_tensors()
                self._input_index = self._interpreter.get_input_details()[0]["index"]
                self._output_index = self._interpreter.get_output_details()[0]["index"]
                with labels_path.open(encoding="utf-8-sig", newline="") as handle:
                    self._labels = [row[0].strip() for row in csv.reader(handle) if row]
                if not self._labels:
                    raise ValueError("The upstream classifier label CSV is empty")
                log.info("Loaded upstream hand gesture model (%d labels)", len(self._labels))
            except Exception as exc:
                log.warning("Upstream TFLite gesture model unavailable; using landmark fallback: %s", exc)
                self._interpreter = None
                self._labels = []

    @staticmethod
    def _finger_up(points, tip: int, pip: int) -> bool:
        return points[tip].y < points[pip].y

    @staticmethod
    def _preprocess_keypoints(points, width: int, height: int) -> list[float]:
        # This matches the coordinate normalization used by upstream app.py.
        coordinates = [[int(point.x * width), int(point.y * height)] for point in points]
        base_x, base_y = coordinates[0]
        for xy in coordinates:
            xy[0] -= base_x
            xy[1] -= base_y
        flattened = [value for xy in coordinates for value in xy]
        scale = max((abs(value) for value in flattened), default=0)
        return [float(value) / scale for value in flattened] if scale else [0.0] * len(flattened)

    def _classify_upstream(self, points, width: int, height: int):
        import numpy as np
        keypoints = self._preprocess_keypoints(points, width, height)
        self._interpreter.set_tensor(self._input_index, np.asarray([keypoints], dtype=np.float32))
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)
        class_id = int(np.argmax(np.squeeze(output)))
        if class_id < 0 or class_id >= len(self._labels):
            return None
        label = self._labels[class_id]
        gesture = {"open": "wave", "close": "stop", "pointer": "point"}.get(label.casefold(), label.casefold())
        return {"gesture": gesture, "label": label}

    def recognize(self, frame):
        """Return a label from an OpenCV BGR camera frame."""
        import cv2
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        points = result.multi_hand_landmarks[0].landmark
        if self._interpreter is not None:
            try:
                classified = self._classify_upstream(points, width, height)
                if classified:
                    return classified
            except Exception:
                log.debug("Upstream TFLite classification failed; using heuristic", exc_info=True)
        raised = (
            points[4].x < points[3].x,
            self._finger_up(points, 8, 6),
            self._finger_up(points, 12, 10),
            self._finger_up(points, 16, 14),
            self._finger_up(points, 20, 18),
        )
        count = sum(raised)
        label = {0: "stop", 1: "point", 2: "two", 3: "three", 4: "four", 5: "wave"}[count]
        return {"gesture": label, "label": label, "finger_count": count}

    def close(self) -> None:
        self._hands.close()
