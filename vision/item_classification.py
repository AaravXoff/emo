"""Optional YOLO item classification using Ultralytics."""
from __future__ import annotations


class ItemClassifier:
    def __init__(self, model_path: str = "yolo11n.pt", image_size: int = 320) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-vision.txt on a capable machine; Ultralytics/PyTorch can be too heavy on Pi 3B+."
            ) from exc
        self.model = YOLO(model_path)
        self.image_size = max(160, min(640, int(image_size)))

    def classify(self, frame):
        results = self.model.predict(frame, imgsz=self.image_size, verbose=False, device="cpu")
        output = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = [int(value) for value in box.xyxy[0].tolist()]
                output.append({
                    "label": str(result.names.get(class_id, class_id)),
                    "confidence": round(confidence, 4),
                    "box": xyxy,
                })
        return output
