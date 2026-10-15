"""Gesture recognition adapter using the requested upstream TFLite classifier when present."""
from __future__ import annotations

import csv
import importlib.util
import logging
from pathlib import Path

log = logging.getLogger(__name__)
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hand-gesture-recognition-using-mediapipe"


class HandGestureRecognizer:
    """MediaPipe landmarks plus upstream keypoint_classifier.tflite when available.

    The upstream model labels are Open/Close/Pointer. If the vendor model/dependency
    isn't installed, a small landmark heuristic remains available for basic fallback.
    """

    def __init__(self, complexity: int = 0, model_root: Path = VENDOR_ROOT) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install the optional vision requirements to enable hand gestures.") from exc
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=complexity,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._classifier = None
        self._labels: list[str] = []
        classifier_py = model_root / "model" / "keypoint_classifier" / "keypoint_classifier.py"
        classifier_model = model_root / "model" / "keypoint_classifier" / "keypoint_classifier.tflite"
        labels_csv = model_root / "model" / "keypoint_classifier" / "keypoint_classifier_label.csv"
        if classifier_py.is_file() and classifier_model.is_file() and labels_csv.is_file():
            try:
                spec = importlib.util.spec_from_file_location("emo_upstream_keypoint_classifier", classifier_py)
                if spec is None or spec.loader is None:
                    raise ImportError("Could not load upstream KeyPointClassifier module")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._classifier = module.KeyPointClassifier(str(classifier_model), num_threads=1)
                with labels_csv.open(encoding="utf-8-sig", newline="") as handle:
                    self._labels = [row[0].strip() for row in csv.reader(handle) if row]
                if not self._labels:
                    raise ValueError("The upstream classifier label CSV is empty")
                log.info("Loaded upstream MediaPipe keypoint classifier with %d labels", len(self._labels))
            except Exception as exc:
                log.warning("Upstream TFLite gesture model unavailable; using landmark heuristic: %s", exc)
                self._classifier = None
                self._labels = []

    @staticmethod
    def _finger_up(points, tip: int, pip: int) -> bool:
        return points[tip].y < points[pip].y

    @staticmethod
    def _preprocess_official_keypoints(points, width: int, height: int) -> list[float]:
        """Port upstream calc_landmark_list + pre_process_landmark normalization."""
        coordinates = [[int(point.x * width), int(point.y * height)] for point in points]
        base_x, base_y = coordinates[0]
        for xy in coordinates:
            xy[0] -= base_x
            xy[1] -= base_y
        flattened = [value for xy in coordinates for value in xy]
        max_value = max((abs(value) for value in flattened), default=0)
        if max_value == 0:
            return [0.0] * len(flattened)
        return [value / max_value for value in flattened]

    def recognize(self, frame):
        """Return a gesture label from an OpenCV BGR camera frame."""
        import cv2
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        points = result.multi_hand_landmarks[0].landmark
        if self._classifier is not None:
            try:
                class_id = int(self._classifier(self._preprocess_official_keypoints(points, width, height)))
                label = self._labels[class_id] if 0 <= class_id < len(self._labels) else "Unknown"
                normalized = label.casefold()
                gesture = {"open": "wave", "close": "stop", "pointer": "point"}.get(normalized, normalized)
                return {"gesture": gesture, "label": label}
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
        labels = {0: "stop", 1: "point", 2: "two", 3: "three", 4: "four", 5: "wave"}
        return {"gesture": labels[count], "label": labels[count], "finger_count": count}

    def close(self) -> None:
        self._hands.close()
