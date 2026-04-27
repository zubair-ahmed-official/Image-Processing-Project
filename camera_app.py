import asyncio
import base64
import json
import time
from collections import deque, Counter

import cv2
import numpy as np
from aiohttp import web

from face_detection import detect_faces
from emotion_predictor import predict_emotion
from voice import start_voice_worker, say_text


device_states = {}

LOOKING_REQUIRED_SEC = 3.0
PERSON_LEFT_DELAY_SEC = 1.5
EMOTION_RESPONSE_COOLDOWN_SEC = 6.0

# -----------------------------
# Stable emotion config
# -----------------------------
EMOTION_HISTORY_SIZE = 12
EMOTION_STABLE_COUNT = 7


def get_stable_emotion(state, new_emotion):
    if not new_emotion or new_emotion == "none":
        return state.get("stable_emotion", "none")

    state["emotion_history"].append(new_emotion)

    counts = Counter(state["emotion_history"])
    most_common_emotion, count = counts.most_common(1)[0]

    if count >= EMOTION_STABLE_COUNT:
        state["stable_emotion"] = most_common_emotion

    return state.get("stable_emotion", new_emotion)


def get_emotion_response(emotion):
    if not emotion:
        return None

    emotion = emotion.lower()

    if emotion == "happy":
        return "You look happy today. That is wonderful to see."
    if emotion == "sad":
        return "You seem a little sad. I hope your day gets better."
    if emotion == "angry":
        return "You seem angry. Please take a moment, I am here to help."
    if emotion == "surprise":
        return "You look surprised. Did something interesting happen?"
    if emotion == "fear":
        return "You look worried. It is okay, take your time."
    if emotion == "neutral":
        return "Hello, how can I help you today?"
    if emotion == "disgust":
        return "Something seems uncomfortable. Let me know how I can help."

    return None


def speak_based_on_emotion(state, emotion, now):
    if not emotion or emotion == "none":
        return

    if (
        emotion != state["last_spoken_emotion"]
        and now - state["last_emotion_speak_time"] >= EMOTION_RESPONSE_COOLDOWN_SEC
    ):
        response_text = get_emotion_response(emotion)

        if response_text:
            say_text(response_text)
            state["last_spoken_emotion"] = emotion
            state["last_emotion_speak_time"] = now


def decode_browser_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def is_looking_at_camera(face_box, frame_shape):
    x, y, w, h = face_box
    frame_h, frame_w = frame_shape[:2]

    face_center_x = x + w / 2
    face_center_y = y + h / 2

    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    center_tolerance_x = frame_w * 0.22
    center_tolerance_y = frame_h * 0.28

    face_area = w * h
    frame_area = frame_w * frame_h

    is_centered = (
        abs(face_center_x - frame_center_x) < center_tolerance_x
        and abs(face_center_y - frame_center_y) < center_tolerance_y
    )

    is_close_enough = face_area > frame_area * 0.025

    return is_centered and is_close_enough


def get_closest_face_index(faces):
    return max(range(len(faces)), key=lambda i: faces[i][2] * faces[i][3])


def get_face_emotion(frame, face, state):
    x, y, w, h = face
    face_img = frame[y:y + h, x:x + w]

    emotion, confidence = predict_emotion(face_img)
    raw_emotion = emotion if emotion else "none"
    stable_emotion = get_stable_emotion(state, raw_emotion)

    return stable_emotion, confidence


async def index_handler(request):
    return web.FileResponse("index.html")


async def safe_send(ws, payload):
    if ws.closed:
        return False

    try:
        await ws.send_str(json.dumps(payload))
        return True
    except ConnectionResetError:
        return False
    except Exception as e:
        print("WebSocket send error:", e)
        return False


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
    await ws.prepare(request)

    device_id = id(ws)

    device_states[device_id] = {
        "person_present": False,
        "no_face_start_time": None,
        "selected_idx": None,
        "customer_selected": False,
        "looking_start_time": None,
        "last_prompt_time": 0.0,
        "last_spoken_emotion": None,
        "last_emotion_speak_time": 0.0,

        # stable emotion memory
        "emotion_history": deque(maxlen=EMOTION_HISTORY_SIZE),
        "stable_emotion": "none"
    }

    print(f"Device connected: {device_id}")

    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue

            data = json.loads(msg.data)

            if data.get("type") != "camera_frame":
                continue

            frame = decode_browser_image(data["image"])

            if frame is None:
                continue

            faces = detect_faces(frame)
            state = device_states[device_id]
            now = time.time()

            current_emotion = "none"

            if len(faces) == 0:
                state["selected_idx"] = None
                state["customer_selected"] = False
                state["looking_start_time"] = None
                state["last_spoken_emotion"] = None
                state["emotion_history"].clear()
                state["stable_emotion"] = "none"

                if state["person_present"]:
                    if state["no_face_start_time"] is None:
                        state["no_face_start_time"] = now

                    elif now - state["no_face_start_time"] >= PERSON_LEFT_DELAY_SEC:
                        state["person_present"] = False
                        state["no_face_start_time"] = None

                        ok = await safe_send(ws, {
                            "type": "event",
                            "event_name": "person_left",
                            "detection_status": "person_left",
                            "emotion_status": "none",
                            "faces_count": 0,
                            "timestamp": now
                        })

                        if not ok:
                            break
                else:
                    state["no_face_start_time"] = None

            elif len(faces) == 1:
                state["no_face_start_time"] = None
                state["selected_idx"] = 0
                state["customer_selected"] = True
                state["looking_start_time"] = None

                current_emotion, confidence = get_face_emotion(frame, faces[0], state)
                speak_based_on_emotion(state, current_emotion, now)

                if not state["person_present"]:
                    state["person_present"] = True
                    say_text("Hi, this is Matilda")

                    ok = await safe_send(ws, {
                        "type": "event",
                        "event_name": "person_detected",
                        "detection_status": "person_detected",
                        "emotion_status": current_emotion,
                        "confidence": confidence,
                        "faces_count": 1,
                        "timestamp": now
                    })

                    if not ok:
                        break

            else:
                state["no_face_start_time"] = None

                if not state["customer_selected"]:
                    closest_idx = get_closest_face_index(faces)
                    closest_face = faces[closest_idx]

                    if now - state["last_prompt_time"] > 6:
                        say_text("Multiple customers detected. Please look at the camera if you want to speak with me.")
                        state["last_prompt_time"] = now

                    if is_looking_at_camera(closest_face, frame.shape):
                        if state["looking_start_time"] is None:
                            state["looking_start_time"] = now

                        looking_duration = now - state["looking_start_time"]

                        x, y, w, h = closest_face
                        cv2.putText(
                            frame,
                            f"Looking: {looking_duration:.1f}s",
                            (x, y - 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        if looking_duration >= LOOKING_REQUIRED_SEC:
                            state["selected_idx"] = closest_idx
                            state["customer_selected"] = True
                            state["person_present"] = True
                            state["looking_start_time"] = None
                            state["emotion_history"].clear()
                            state["stable_emotion"] = "none"

                            current_emotion, confidence = get_face_emotion(frame, closest_face, state)
                            speak_based_on_emotion(state, current_emotion, now)

                            say_text("Hello, I am listening to you.")

                            ok = await safe_send(ws, {
                                "type": "event",
                                "event_name": "customer_selected",
                                "detection_status": "person_detected",
                                "emotion_status": current_emotion,
                                "confidence": confidence,
                                "faces_count": len(faces),
                                "selected_face_index": closest_idx,
                                "timestamp": now
                            })

                            if not ok:
                                break
                    else:
                        state["looking_start_time"] = None

                else:
                    if state["selected_idx"] is None or state["selected_idx"] >= len(faces):
                        state["selected_idx"] = None
                        state["customer_selected"] = False
                        state["looking_start_time"] = None
                        state["emotion_history"].clear()
                        state["stable_emotion"] = "none"

            for idx, (x, y, w, h) in enumerate(faces):
                is_selected = (
                    state["customer_selected"]
                    and state["selected_idx"] == idx
                )

                color = (0, 255, 255) if is_selected else (0, 255, 0)
                thickness = 3 if is_selected else 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                if len(faces) > 1 and not state["customer_selected"]:
                    cv2.putText(
                        frame,
                        "Look at camera",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    continue

                if not is_selected:
                    continue

                face_img = frame[y:y + h, x:x + w]
                emotion, confidence = predict_emotion(face_img)

                raw_emotion = emotion if emotion else "none"
                current_emotion = get_stable_emotion(state, raw_emotion)

                speak_based_on_emotion(state, current_emotion, now)

                cv2.putText(
                    frame,
                    f"{current_emotion} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                ok = await safe_send(ws, {
                    "type": "event",
                    "event_name": "emotion_detected",
                    "detection_status": "person_alive",
                    "emotion_status": current_emotion,
                    "confidence": confidence,
                    "selected_face_index": idx,
                    "timestamp": now
                })

                if not ok:
                    break

            if len(faces) > 1 and not state["customer_selected"]:
                cv2.putText(
                    frame,
                    "Multiple customers detected",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    "Please look at the camera for 3 seconds",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            encoded = encode_frame(frame)

            if encoded:
                await ws.send_str(json.dumps({
                    "type": "processed_frame",
                    "image": encoded,
                    "faces_count": len(faces),
                    "emotion": current_emotion,
                    "person_present": state["person_present"],
                    "customer_selected": state["customer_selected"],
                    "selected_face_index": state["selected_idx"]
                }))

    finally:
        device_states.pop(device_id, None)
        print(f"Device disconnected: {device_id}")

    return ws


async def main():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/index.html", index_handler)
    app.router.add_get("/ws", ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()

    print("Server running on http://127.0.0.1:8080")
    await asyncio.Future()


if __name__ == "__main__":
    start_voice_worker()
    asyncio.run(main())