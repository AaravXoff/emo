"""Optional face-recognition wrapper using ageitgey's face_recognition library."""
from __future__ import annotations

from pathlib import Path


class FaceIdentityRecognizer:
    def __init__(self, known_faces_dir: Path) -> None:
        try:
            import face_recognition
        except ImportError as exc:
            raise RuntimeError("Install requirements-vision.txt to enable face recognition.") from exc
        self._face_recognition = face_recognition
        self.names = []
        self.encodings = []
        root = Path(known_faces_dir)
        root.mkdir(parents=True, exist_ok=True)
        for image in sorted(root.glob("*")):
            if image.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                loaded = face_recognition.load_image_file(str(image))
                encodings = face_recognition.face_encodings(loaded)
                if encodings:
                    self.names.append(image.stem)
                    self.encodings.append(encodings[0])
            except Exception:
                continue

    def recognize(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = self._face_recognition.face_locations(rgb, model="hog")
        encodings = self._face_recognition.face_encodings(rgb, locations)
        found = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            matches = self._face_recognition.compare_faces(self.encodings, encoding, tolerance=0.50)
            name = "unknown"
            if any(matches):
                distances = self._face_recognition.face_distance(self.encodings, encoding)
                best = int(distances.argmin())
                if matches[best]:
                    name = self.names[best]
            found.append({"name": name, "box": [int(left), int(top), int(right), int(bottom)]})
        return found
