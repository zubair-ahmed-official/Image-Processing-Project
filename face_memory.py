import os
import json
import uuid
import numpy as np
import cv2
import face_recognition

FACES_DB_PATH = "known_faces.json"
MATCH_THRESHOLD = 0.48


class FaceMemory:
    def __init__(self, db_path=FACES_DB_PATH, match_threshold=MATCH_THRESHOLD):
        self.db_path = db_path
        self.match_threshold = match_threshold
        self.known_faces = self._load_db()

    def _load_db(self):
        if not os.path.exists(self.db_path):
            return []

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.known_faces, f, indent=2)

    def _get_encoding(self, face_img_bgr):
        if face_img_bgr is None or face_img_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if not encodings:
            return None

        return encodings[0]

    def recognize_or_register(self, face_img_bgr):
        encoding = self._get_encoding(face_img_bgr)
        if encoding is None:
            return "unknown", "none"

        if not self.known_faces:
            new_id = str(uuid.uuid4())[:8]
            self.known_faces.append({
                "id": new_id,
                "encoding": encoding.tolist()
            })
            self._save_db()
            return "new_person", new_id

        stored_encodings = [
            np.array(person["encoding"], dtype=np.float64)
            for person in self.known_faces
        ]

        distances = face_recognition.face_distance(stored_encodings, encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance <= self.match_threshold:
            return "remembered_person", self.known_faces[best_index]["id"]

        new_id = str(uuid.uuid4())[:8]
        self.known_faces.append({
            "id": new_id,
            "encoding": encoding.tolist()
        })
        self._save_db()
        return "new_person", new_id