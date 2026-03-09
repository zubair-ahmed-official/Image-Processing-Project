import cv2
import time
import threading
import speech_recognition as sr

from face_detection import detect_faces
from emotion_predictor import predict_emotion
from person_tracker import should_greet
from voice import start_voice_worker, say_hello, say_text
from attention_mode import AttentionMode

start_voice_worker()
cap = cv2.VideoCapture(0)

# -----------------------------
# overlays
# -----------------------------
hello_text_frames = 0
attention_text_frames = 0
instruction_frames = 0

# -----------------------------
# conversation / selection state
# -----------------------------
selected_idx = None
selected_until = 0.0
SELECTION_HOLD_SEC = 15

conversation_active = False

# prompt cooldown
last_multi_prompt_time = 0.0
MULTI_PROMPT_COOLDOWN_SEC = 8

# -----------------------------
# wake word detection
# -----------------------------
wake_word_detected = False

# -----------------------------
# emotion speech tuning
# -----------------------------
EMOTION_MIN_CONF = 0.55
EMOTION_STABLE_FRAMES = 6
EMOTION_COOLDOWN_SEC = 10

_last_emotion = None
_stable_emotion_count = 0
_last_emotion_spoken_time = 0.0
_last_emotion_spoken_label = None

# greeting cooldown
last_hello_time = 0.0
HELLO_COOLDOWN_SEC = 8


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

    return None


# -----------------------------
# Attention Mode
# -----------------------------
attention = AttentionMode(
    seconds_required=1.6,
    center_radius_ratio=0.22,
    min_face_area_ratio=0.03,
    speak_cooldown_sec=12
)


# -----------------------------
# Wake word listener thread
# -----------------------------
def wake_listener():
    global wake_word_detected

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=3)

            text = recognizer.recognize_google(audio).lower()

            if "hi matilda" in text:
                wake_word_detected = True
                print("Wake word detected")

        except Exception:
            pass


threading.Thread(target=wake_listener, daemon=True).start()


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)
    now = time.time()

    # ---------------------------------
    # No faces detected
    # ---------------------------------
    if len(faces) == 0:

        attention.reset()
        selected_idx = None
        conversation_active = False

        _last_emotion = None
        _stable_emotion_count = 0

        cv2.imshow("Matilda's Eye", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # ---------------------------------
    # expire conversation if timeout
    # ---------------------------------
    if conversation_active and now > selected_until:
        conversation_active = False
        selected_idx = None

    # ---------------------------------
    # MULTI CUSTOMER LOGIC
    # ---------------------------------
    if len(faces) > 1:

        # prompt only if idle
        if not conversation_active:

            if now - last_multi_prompt_time > MULTI_PROMPT_COOLDOWN_SEC:
                say_text(
                    "Hello everyone. If you want to talk to me, please say Hi Matilda."
                )
                last_multi_prompt_time = now
                instruction_frames = 60

            # wake word triggered
            if wake_word_detected:

                # choose largest face (closest speaker)
                selected_idx = max(
                    range(len(faces)),
                    key=lambda i: faces[i][2] * faces[i][3]
                )

                conversation_active = True
                selected_until = now + SELECTION_HOLD_SEC
                wake_word_detected = False

                say_text("Hello! I am listening to you.")
                hello_text_frames = 45

        else:
            # if selected face disappears
            if selected_idx is None or selected_idx >= len(faces):
                conversation_active = False
                selected_idx = None

    else:
        # only one person
        selected_idx = 0
        conversation_active = True
        selected_until = now + SELECTION_HOLD_SEC

    # ---------------------------------
    # PROCESS FACES
    # ---------------------------------
    for idx, (x, y, w, h) in enumerate(faces):

        face_img = frame[y:y+h, x:x+w]

        if selected_idx == idx:
            color = (0, 255, 255)
            thickness = 3
        else:
            color = (0, 255, 0)
            thickness = 2

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        if idx != selected_idx:
            continue

        # greeting
        if should_greet(face_img) and (now - last_hello_time > HELLO_COOLDOWN_SEC):
            say_hello()
            hello_text_frames = 45
            last_hello_time = now

        # attention mode
        engaged, _ = attention.update((x, y, w, h), frame.shape, stable=True)

        if engaged:
            attention_text_frames = 45

        # emotion detection
        emotion, confidence = predict_emotion(face_img)

        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # emotion stability
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

                say_text(sentence)

                _last_emotion_spoken_time = now
                _last_emotion_spoken_label = emotion

    # ---------------------------------
    # UI overlays
    # ---------------------------------
    if hello_text_frames > 0:

        cv2.putText(
            frame,
            "Hello!",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 0, 0),
            3
        )

        hello_text_frames -= 1

    if attention_text_frames > 0:

        cv2.putText(
            frame,
            "ATTENTION MODE",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        attention_text_frames -= 1

    if instruction_frames > 0 and len(faces) > 1 and not conversation_active:

        cv2.putText(
            frame,
            "Multiple customers detected",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Say 'Hi Matilda' to talk",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        instruction_frames -= 1

    cv2.imshow("Matilda's Eye", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()