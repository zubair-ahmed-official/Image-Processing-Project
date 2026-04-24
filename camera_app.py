import cv2
import time
import threading
import asyncio
import json
import base64
import requests
from collections import deque, Counter
import speech_recognition as sr
import websockets

from face_detection import detect_faces
from emotion_predictor import predict_emotion
from person_tracker import should_greet
from voice import start_voice_worker, say_hello, say_text
from attention_mode import AttentionMode


# -----------------------------
# WebSocket config
# -----------------------------
connected_clients = set()
ws_loop = None

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS
last_frame_sent_time = 0.0


async def ws_handler(websocket):
    connected_clients.add(websocket)
    print("Browser connected")

    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print("Browser disconnected")


async def broadcast(payload):
    if not connected_clients:
        return

    message = json.dumps(payload)

    await asyncio.gather(
        *[client.send(message) for client in list(connected_clients)],
        return_exceptions=True
    )


def send_to_browser(payload):
    if ws_loop and connected_clients:
        asyncio.run_coroutine_threadsafe(broadcast(payload), ws_loop)


def encode_frame(frame):
    # Reduce size for browser stability
    frame = cv2.resize(frame, (640, 480))
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


async def ws_main():
    async with websockets.serve(ws_handler, "0.0.0.0", 8080):
        print("WebSocket server running on ws://0.0.0.0:8080")
        await asyncio.Future()


def start_ws_server():
    global ws_loop

    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(ws_main())


def send_status_ws(detection_status, emotion_status, faces_count=0):
    send_to_browser({
        "type": "camera_status",
        "detection_status": detection_status,
        "emotion_status": emotion_status,
        "faces_count": faces_count,
        "timestamp": time.time()
    })


def send_frame_to_browser(frame, faces_count, selected_idx, conversation_active, emotion, person_present):
    global last_frame_sent_time

    now = time.time()

    if now - last_frame_sent_time < FRAME_INTERVAL:
        return

    image = encode_frame(frame)

    if image is None:
        return

    send_to_browser({
        "type": "camera_frame",
        "image": image,
        "faces_count": faces_count,
        "selected_idx": selected_idx,
        "conversation_active": conversation_active,
        "emotion": emotion if emotion else "none",
        "person_present": person_present,
        "timestamp": now
    })

    last_frame_sent_time = now


# Start WebSocket once
threading.Thread(target=start_ws_server, daemon=True).start()
time.sleep(1)


# -----------------------------
# Voice and camera
# -----------------------------
start_voice_worker()
cap = cv2.VideoCapture(0)


# -----------------------------
# State variables
# -----------------------------
hello_text_frames = 0
attention_text_frames = 0
instruction_frames = 0

selected_idx = None
selected_until = 0.0
SELECTION_HOLD_SEC = 15

conversation_active = False

last_multi_prompt_time = 0.0
MULTI_PROMPT_COOLDOWN_SEC = 8

wake_word_detected = False

EMOTION_MIN_CONF = 0.60
EMOTION_STABLE_FRAMES = 10
EMOTION_COOLDOWN_SEC = 12

_last_emotion = None
_stable_emotion_count = 0
_last_emotion_spoken_time = 0.0
_last_emotion_spoken_label = None

EMOTION_HISTORY_SIZE = 12
SMOOTHED_MIN_COUNT = 6
emotion_history = deque(maxlen=EMOTION_HISTORY_SIZE)

last_hello_time = 0.0
HELLO_COOLDOWN_SEC = 8

person_present = False
last_sent_emotion = None


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


def reset_emotion_tracking():
    global _last_emotion
    global _stable_emotion_count
    global _last_emotion_spoken_label
    global emotion_history

    _last_emotion = None
    _stable_emotion_count = 0
    _last_emotion_spoken_label = None
    emotion_history.clear()


def reset_person_state():
    global person_present
    global last_sent_emotion

    person_present = False
    last_sent_emotion = None


def get_smoothed_emotion(current_emotion: str, current_confidence: float):
    global emotion_history

    if current_emotion and current_confidence >= EMOTION_MIN_CONF:
        emotion_history.append(current_emotion.lower())

    if not emotion_history:
        return current_emotion.lower() if current_emotion else current_emotion

    counts = Counter(emotion_history)
    most_common_emotion, count = counts.most_common(1)[0]

    if count >= SMOOTHED_MIN_COUNT:
        return most_common_emotion

    return current_emotion.lower() if current_emotion else current_emotion


attention = AttentionMode(
    seconds_required=1.6,
    center_radius_ratio=0.22,
    min_face_area_ratio=0.03,
    speak_cooldown_sec=12
)


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


# Enable later if needed
# threading.Thread(target=wake_listener, daemon=True).start()


# -----------------------------
# Main camera loop
# -----------------------------
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        faces = detect_faces(frame)
        now = time.time()

        # -----------------------------
        # No face
        # -----------------------------
        if len(faces) == 0:
            if person_present:
                send_status_ws("person_left", "none", 0)
                reset_person_state()
                print("No one detected")

            attention.reset()
            selected_idx = None
            conversation_active = False
            reset_emotion_tracking()

        else:
            # -----------------------------
            # Conversation timeout
            # -----------------------------
            if conversation_active and now > selected_until:
                if person_present:
                    send_status_ws("person_left", "none", 0)
                    reset_person_state()
                    print("No one detected")

                conversation_active = False
                selected_idx = None
                reset_emotion_tracking()

            # -----------------------------
            # Multi-customer logic
            # -----------------------------
            if len(faces) > 1:
                if not conversation_active:
                    if now - last_multi_prompt_time > MULTI_PROMPT_COOLDOWN_SEC:
                        say_text("Hello everyone. If you want to talk to me, please say Hi Matilda.")
                        last_multi_prompt_time = now
                        instruction_frames = 60

                    if wake_word_detected:
                        selected_idx = max(
                            range(len(faces)),
                            key=lambda i: faces[i][2] * faces[i][3]
                        )

                        conversation_active = True
                        selected_until = now + SELECTION_HOLD_SEC
                        wake_word_detected = False
                        reset_emotion_tracking()
                        reset_person_state()

                        say_text("Hello! I am listening to you.")
                        hello_text_frames = 45

                else:
                    if selected_idx is None or selected_idx >= len(faces):
                        conversation_active = False
                        selected_idx = None
                        reset_emotion_tracking()

                        if person_present:
                            send_status_ws("person_left", "none", 0)
                            reset_person_state()
                            print("No one detected")

            # -----------------------------
            # Single customer
            # -----------------------------
            else:
                if selected_idx != 0 or not conversation_active:
                    reset_emotion_tracking()
                    reset_person_state()

                selected_idx = 0
                conversation_active = True
                selected_until = now + SELECTION_HOLD_SEC

            # -----------------------------
            # Process faces
            # -----------------------------
            for idx, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y + h, x:x + w]

                if selected_idx == idx:
                    color = (0, 255, 255)
                    thickness = 3
                else:
                    color = (0, 255, 0)
                    thickness = 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                if idx != selected_idx:
                    continue

                if should_greet(face_img) and (now - last_hello_time > HELLO_COOLDOWN_SEC):
                    say_hello()
                    hello_text_frames = 45
                    last_hello_time = now

                engaged, _ = attention.update((x, y, w, h), frame.shape, stable=True)

                if engaged:
                    attention_text_frames = 45

                raw_emotion, confidence = predict_emotion(face_img)
                emotion = get_smoothed_emotion(raw_emotion, confidence)

                cv2.putText(
                    frame,
                    f"{emotion} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                if emotion == _last_emotion:
                    _stable_emotion_count += 1
                else:
                    _last_emotion = emotion
                    _stable_emotion_count = 1

                if (
                    emotion
                    and confidence >= EMOTION_MIN_CONF
                    and _stable_emotion_count >= EMOTION_STABLE_FRAMES
                ):
                    if not person_present:
                        send_status_ws("person_detected", emotion, len(faces))
                        person_present = True
                        last_sent_emotion = emotion

                    elif emotion != last_sent_emotion:
                        send_status_ws("person_alive", emotion, len(faces))
                        last_sent_emotion = emotion

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

        # -----------------------------
        # UI overlays
        # -----------------------------
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

        # -----------------------------
        # Send frame ONCE per loop, throttled
        # -----------------------------
        send_frame_to_browser(
            frame=frame,
            faces_count=len(faces),
            selected_idx=selected_idx,
            conversation_active=conversation_active,
            emotion=_last_emotion,
            person_present=person_present
        )

except KeyboardInterrupt:
    print("Camera stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()