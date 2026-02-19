# attention_mode.py
import time
import math

class AttentionMode:
    """
    Attention Mode = person is likely "staring" at camera if:
      - face bbox is near frame center
      - face bbox is not tiny
      - face stays for a continuous duration

    This avoids heavy gaze libraries and works well for webcam demos.
    """

    def __init__(
        self,
        seconds_required=1.6,
        center_radius_ratio=0.22,   # how close to center (0.22 = 22% of min(frame_w, frame_h))
        min_face_area_ratio=0.03,   # ignore tiny faces (3% of frame area)
        speak_cooldown_sec=12
    ):
        self.seconds_required = seconds_required
        self.center_radius_ratio = center_radius_ratio
        self.min_face_area_ratio = min_face_area_ratio
        self.speak_cooldown_sec = speak_cooldown_sec

        self._start_ts = None
        self._engaged = False
        self._last_spoken_ts = 0.0

    def reset(self):
        self._start_ts = None
        self._engaged = False

    def _is_centered(self, bbox, frame_shape):
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]

        face_cx = x + w / 2
        face_cy = y + h / 2
        frame_cx = fw / 2
        frame_cy = fh / 2

        dist = math.hypot(face_cx - frame_cx, face_cy - frame_cy)
        radius = self.center_radius_ratio * min(fw, fh)
        return dist <= radius

    def _is_big_enough(self, bbox, frame_shape):
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]
        face_area = w * h
        frame_area = fw * fh
        return (face_area / max(frame_area, 1)) >= self.min_face_area_ratio

    def update(self, bbox, frame_shape, stable=True):
        """
        Returns:
          engaged_now (bool): True when attention mode is active
          should_speak (bool): True only once per engage with cooldown
        """
        now = time.time()

        centered = self._is_centered(bbox, frame_shape)
        big_enough = self._is_big_enough(bbox, frame_shape)

        # If conditions not met, reset attention timer
        if not (stable and centered and big_enough):
            self.reset()
            return False, False

        # Start timing once conditions are met
        if self._start_ts is None:
            self._start_ts = now
            return False, False

        # Engage if sustained for required seconds
        if not self._engaged and (now - self._start_ts) >= self.seconds_required:
            self._engaged = True

        # Speak only if engaged and cooldown passed
        should_speak = False
        if self._engaged and (now - self._last_spoken_ts) >= self.speak_cooldown_sec:
            should_speak = True
            self._last_spoken_ts = now

        return self._engaged, should_speak
