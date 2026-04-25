import asyncio
import base64
import json
import time

import cv2
import numpy as np
from aiohttp import web

from face_detection import detect_faces
from emotion_predictor import predict_emotion
from voice import start_voice_worker, say_text


device_states = {}


def decode_browser_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_frame(frame):
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 65]
    )

    if not ok:
        return None

    return base64.b64encode(buffer).decode("utf-8")


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
        "no_face_start_time": None
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

            # -----------------------------
            # Send person event only on state change
            # -----------------------------
            now = time.time()

            if len(faces) > 0:
                state["no_face_start_time"] = None

                if not state["person_present"]:
                    state["person_present"] = True

                    ok = await safe_send(ws, {
                        "type": "event",
                        "event_name": "person_detected",
                        "faces_count": len(faces),
                        "timestamp": now
                    })

                    if not ok:
                        break

            else:
                if state["person_present"]:
                    if state["no_face_start_time"] is None:
                        state["no_face_start_time"] = now

                    elif now - state["no_face_start_time"] >= 1.5:
                        state["person_present"] = False
                        state["no_face_start_time"] = None

                        ok = await safe_send(ws, {
                            "type": "event",
                            "event_name": "person_left",
                            "faces_count": 0,
                            "timestamp": now
                        })

                        if not ok:
                            break

            current_emotion = "none"

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]

                emotion, confidence = predict_emotion(face_img)
                current_emotion = emotion if emotion else "none"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.putText(
                    frame,
                    f"{current_emotion}",
                    (x, y - 10),
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
                    "person_present": state["person_present"]
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


asyncio.run(main())