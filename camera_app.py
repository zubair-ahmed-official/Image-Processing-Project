import cv2
import time
from face_detection import detect_faces
from emotion_predictor import predict_emotion
from person_tracker import should_greet
from voice import start_voice_worker, say_hello, say_text
from attention_mode import AttentionMode
from multi_customer_selector import MultiCustomerSelector

start_voice_worker()
cap = cv2.VideoCapture(0)

# --- overlays ---
hello_text_frames = 0
attention_text_frames = 0
instruction_frames = 0

# --- selection / multi-customer ---
selector = MultiCustomerSelector(hold_seconds=10.0)

# --- emotion speech tuning ---
EMOTION_MIN_CONF = 0.55
EMOTION_STABLE_FRAMES = 6
EMOTION_COOLDOWN_SEC = 10

_last_emotion = None
_stable_emotion_count = 0
_last_emotion_spoken_time = 0.0
_last_emotion_spoken_label = None

def get_emotion_sentence(emotion: str):
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
        return None
    return None

# Attention mode controller
attention = AttentionMode(
    seconds_required=1.6,
    center_radius_ratio=0.22,
    min_face_area_ratio=0.03,
    speak_cooldown_sec=12
)

# To avoid talking to the wrong person when switching selection
_selected_last_idx = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    # no faces -> reset everything
    if len(faces) == 0:
        attention.reset()
        selector.reset()
        _last_emotion = None
        _stable_emotion_count = 0
        _selected_last_idx = None
        cv2.imshow("Matilda's Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # MULTI-CUSTOMER: choose one by hand raise
    selected_idx = selector.pick_customer(frame, faces)

    if len(faces) > 1 and selected_idx is None:
        # show instruction overlay while waiting
        instruction_frames = 20

    # If selection changed, reset emotion stability so it doesn't speak instantly
    if selected_idx is not None and selected_idx != _selected_last_idx:
        _last_emotion = None
        _stable_emotion_count = 0
        _selected_last_idx = selected_idx

    # Draw all faces; process ONLY selected
    for idx, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]

        # highlight selected face
        if selected_idx == idx:
            box_color = (0, 255, 255)  # yellow
            thickness = 3
        else:
            box_color = (0, 255, 0)    # green
            thickness = 2

        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, thickness)

        # Only detect emotion + speak for selected customer
        if selected_idx is None or idx != selected_idx:
            continue

        # ✅ Greeting first (new person)
        # (This will happen only for the selected person)
        if should_greet(face_img):
            say_hello()
            hello_text_frames = 45

        # ✅ Attention mode on selected customer only
        engaged, should_speak = attention.update((x, y, w, h), frame.shape, stable=True)
        if engaged:
            attention_text_frames = 45

        # ✅ Emotion on selected customer only
        emotion, confidence = predict_emotion(face_img)

        # Draw label on selected face
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ---- emotion stability + cooldown speech ----
        now = time.time()

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
            if sentence and _last_emotion_spoken_label != emotion:
                # IMPORTANT: Hello should be first, then emotion
                # Your greeting happens above; here we only speak emotion afterward.
                say_text(sentence)
                _last_emotion_spoken_time = now
                _last_emotion_spoken_label = emotion
        # --------------------------------------------

    # Overlays
    if hello_text_frames > 0:
        cv2.putText(frame, "Hello!", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 3)
        hello_text_frames -= 1

    if attention_text_frames > 0:
        cv2.putText(frame, "ATTENTION MODE", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        attention_text_frames -= 1

    if instruction_frames > 0:
        cv2.putText(frame, "Multiple customers detected:", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Raise your hand to talk to Matilda", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        instruction_frames -= 1

    cv2.imshow("Matilda's Eye", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()