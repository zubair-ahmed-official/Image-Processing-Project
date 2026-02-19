import cv2
import time
from face_detection import detect_faces
from emotion_predictor import predict_emotion
from person_tracker import should_greet
from voice import say_hello, start_voice_worker, say_text
from attention_mode import AttentionMode

start_voice_worker()

cap = cv2.VideoCapture(0)

hello_text_frames = 0
attention_text_frames = 0

# ----------------------------
# Emotion speech tuning
# ----------------------------
EMOTION_MIN_CONF = 0.55          # lower if your model confidence is low
EMOTION_STABLE_FRAMES = 6        # emotion must stay same for N frames
EMOTION_COOLDOWN_SEC = 10        # don't speak too often

_last_emotion = None
_stable_emotion_count = 0
_last_emotion_spoken_time = 0.0
_last_emotion_spoken_label = None

def get_emotion_sentence(emotion: str) -> str | None:
    if not emotion:
        return None
    e = emotion.lower()

    if "happy" in e:
        return "You look happy today! Glad to see you happy!"
    if "sad" in e:
        return "You look a bit sad. I hope everything is okay."
    if "angry" in e:
        return "You seem angry. Take a deep breath. I'm here if you need a moment."
    if "surprise" in e:
        return "You look surprised! Something interesting happened?"
    if "fear" in e:
        return "You look worried. It's okay. Take your time."
    if "neutral" in e:
        return None  # usually better not to speak for neutral

    return None
# ----------------------------

# Attention mode controller
attention = AttentionMode(
    seconds_required=1.6,
    center_radius_ratio=0.22,
    min_face_area_ratio=0.03,
    speak_cooldown_sec=12
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    # If no faces, reset attention and emotion stability
    if len(faces) == 0:
        attention.reset()
        _last_emotion = None
        _stable_emotion_count = 0

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # ✅ Emotion (compute early so we can speak after Hello in correct order)
        emotion, confidence = predict_emotion(face_img)

        # ✅ Greeting (new person) — Hello FIRST, then emotion sentence
        is_new = should_greet(face_img)
        if is_new:
            say_hello()
            hello_text_frames = 45

            sentence = get_emotion_sentence(emotion)
            if sentence and confidence >= EMOTION_MIN_CONF:
                say_text(sentence)

            # Reset emotion cooldown so it doesn't immediately speak again
            _last_emotion_spoken_time = time.time()
            _last_emotion_spoken_label = emotion
            _last_emotion = emotion
            _stable_emotion_count = 1

        # ✅ Attention Mode (staring / engaged)
        engaged, should_speak = attention.update((x, y, w, h), frame.shape, stable=True)
        if engaged:
            attention_text_frames = 45

        # ---- Emotion stability + cooldown speech (ONLY for non-new person) ----
        if not is_new:
            now = time.time()

            # Track stable emotion
            if emotion == _last_emotion:
                _stable_emotion_count += 1
            else:
                _last_emotion = emotion
                _stable_emotion_count = 1

            if (
                _stable_emotion_count >= EMOTION_STABLE_FRAMES
                and confidence >= EMOTION_MIN_CONF
                and (now - _last_emotion_spoken_time) >= EMOTION_COOLDOWN_SEC
            ):
                sentence = get_emotion_sentence(emotion)

                # Avoid repeating same sentence again and again
                if sentence and _last_emotion_spoken_label != emotion:
                    say_text(sentence)
                    _last_emotion_spoken_time = now
                    _last_emotion_spoken_label = emotion
        # ---------------------------------------------------------------------

        # Draw face box + emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Hello overlay
    if hello_text_frames > 0:
        cv2.putText(frame, "Hello!", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 3)
        hello_text_frames -= 1

    # Attention overlay (smaller)
    if attention_text_frames > 0:
        cv2.putText(frame, "ATTENTION MODE", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        attention_text_frames -= 1

    cv2.imshow("Matilda's Eye: Emotion + Attention Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
