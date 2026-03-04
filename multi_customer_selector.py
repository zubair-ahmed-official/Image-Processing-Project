import time
import cv2
import numpy as np


class MultiCustomerSelector:
    """
    Chooses ONE face when multiple faces exist:
    - Ask to raise hand
    - Detect raised hands (MediaPipe Hands)
    - Map raised hand -> nearest face
    - Lock selection for N seconds
    """

    def __init__(
        self,
        hold_seconds: float = 8.0,         # keep selected person for this long
        hand_raise_y_ratio: float = 0.25,   # hand is "raised" if hand is above (face_top + ratio*face_h)
        face_expand_ratio: float = 0.35,    # expand face box to associate a hand with a face
        min_hand_conf: float = 0.55
    ):
        self.hold_seconds = hold_seconds
        self.hand_raise_y_ratio = hand_raise_y_ratio
        self.face_expand_ratio = face_expand_ratio
        self.min_hand_conf = min_hand_conf

        self._mp = None
        self._hands = None

        self._selected_idx = None
        self._selected_until = 0.0

    def _ensure_hands(self):
        if self._hands is not None:
            return
        import mediapipe as mp
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _detect_hands(self, frame_bgr):
        """Returns list of hand dicts: {cx, cy, miny, score} in pixel coords."""
        self._ensure_hands()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)

        hands = []
        if not res.multi_hand_landmarks:
            return hands

        for i, lm in enumerate(res.multi_hand_landmarks):
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]

            cx = int(np.mean(xs) * w)
            cy = int(np.mean(ys) * h)
            miny = int(np.min(ys) * h)

            # MediaPipe doesn't directly give a numeric "score" here reliably;
            # so we treat presence as valid. We'll filter using miny/position instead.
            hands.append({"cx": cx, "cy": cy, "miny": miny})

        return hands

    def _expand_face(self, face, frame_shape):
        """Expand face bbox to help linking hands to the right customer."""
        x, y, w, h = face
        fh, fw = frame_shape[:2]
        ex = int(w * self.face_expand_ratio)
        ey = int(h * self.face_expand_ratio)
        nx = max(0, x - ex)
        ny = max(0, y - ey)
        nw = min(fw - nx, w + 2 * ex)
        nh = min(fh - ny, h + 2 * ey)
        return (nx, ny, nw, nh)

    def _is_hand_raised_for_face(self, hand, face):
        """
        Hand raised rule:
        - if the TOP of the hand (miny) is above face_top + (ratio * face_h)
        This is robust enough for demos.
        """
        x, y, w, h = face
        threshold_y = int(y + self.hand_raise_y_ratio * h)
        return hand["miny"] < threshold_y

    def _hand_belongs_to_face(self, hand, face_expanded):
        """Check if hand center is inside expanded face region."""
        hx, hy = hand["cx"], hand["cy"]
        x, y, w, h = face_expanded
        return (x <= hx <= x + w) and (y <= hy <= y + h)

    def pick_customer(self, frame_bgr, faces):
        """
        Returns selected face index or None.
        - If already locked, keep it until timeout.
        - Else choose face whose raised hand best matches.
        """
        now = time.time()

        # keep previous selection while held
        if self._selected_idx is not None and now < self._selected_until:
            if self._selected_idx < len(faces):
                return self._selected_idx
            else:
                self._selected_idx = None

        if len(faces) == 0:
            self._selected_idx = None
            return None

        # If only one face, select automatically
        if len(faces) == 1:
            self._selected_idx = 0
            self._selected_until = now + self.hold_seconds
            return 0

        # Multiple faces: need raised hand
        hands = self._detect_hands(frame_bgr)
        if not hands:
            return None

        # Score candidates
        best = None  # (score, idx)
        for idx, face in enumerate(faces):
            face_exp = self._expand_face(face, frame_bgr.shape)

            for hand in hands:
                # first: link hand to face region
                if not self._hand_belongs_to_face(hand, face_exp):
                    continue

                # second: raised check
                if not self._is_hand_raised_for_face(hand, face):
                    continue

                # scoring: prefer hand closer to face center
                x, y, w, h = face
                fc = (x + w // 2, y + h // 2)
                dist = abs(hand["cx"] - fc[0]) + abs(hand["cy"] - fc[1])
                score = 1_000_000 - dist  # higher is better

                if best is None or score > best[0]:
                    best = (score, idx)

        if best is None:
            return None

        self._selected_idx = best[1]
        self._selected_until = now + self.hold_seconds
        return self._selected_idx

    def reset(self):
        self._selected_idx = None
        self._selected_until = 0.0