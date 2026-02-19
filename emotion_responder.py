# emotion_responder.py
import time

RESPONSES = {
    "happy": "You look happy today! Glad to see you happy!",
    "sad": "You look a bit sad. I hope everything is okay. I'm here with you.",
    "angry": "You seem angry. Take a deep breath. I'm here if you need a moment.",
}

# Optional fallbacks
DEFAULT_RESPONSE = "I can see an emotion on your face."

class EmotionResponder:
    def __init__(self, min_conf=0.70, stable_frames=10, cooldown_sec=12):
        self.min_conf = min_conf
        self.stable_frames = stable_frames
        self.cooldown_sec = cooldown_sec

        self._last_emotion = None
        self._stable_count = 0
        self._last_spoken_time = 0.0
        self._last_spoken_emotion = None

    def update(self, emotion: str, confidence: float):
        """
        Returns text to speak OR None.
        Speaks only when:
          - confidence >= min_conf
          - emotion is stable for stable_frames
          - cooldown passed (and avoids repeating same emotion too often)
        """
        now = time.time()

        if not emotion or confidence < self.min_conf:
            self._stable_count = 0
            self._last_emotion = None
            return None

        # stability check
        if emotion == self._last_emotion:
            self._stable_count += 1
        else:
            self._last_emotion = emotion
            self._stable_count = 1

        if self._stable_count < self.stable_frames:
            return None

        # cooldown check
        if (now - self._last_spoken_time) < self.cooldown_sec:
            return None

        # don’t repeat the same emotion line again and again
        if self._last_spoken_emotion == emotion and (now - self._last_spoken_time) < (self.cooldown_sec * 2):
            return None

        self._last_spoken_time = now
        self._last_spoken_emotion = emotion

        return RESPONSES.get(emotion.lower(), DEFAULT_RESPONSE)
