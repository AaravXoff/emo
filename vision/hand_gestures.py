"""MediaPipe hand landmarks and lightweight gesture classification."""
from __future__ import annotations


class HandGestureRecognizer:
    def __init__(self, complexity: int = 0) -> None:
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

    @staticmethod
    def _finger_up(points, tip: int, pip: int) -> bool:
        return points[tip].y < points[pip].y

    def recognize(self, frame):
        """Return a coarse action label and confidence from a BGR OpenCV frame."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        points = result.multi_hand_landmarks[0].landmark
        raised = (
            points[4].x < points[3].x,
            self._finger_up(points, 8, 6),
            self._finger_up(points, 12, 10),
            self._finger_up(points, 16, 14),
            self._finger_up(points, 20, 18),
        )
        count = sum(raised)
        labels = {0: "stop", 1: "point", 2: "two", 3: "three", 4: "four", 5: "wave"}
        return {"gesture": labels[count], "confidence": 0.70, "finger_count": count}

    def close(self):
        self._hands.close()
