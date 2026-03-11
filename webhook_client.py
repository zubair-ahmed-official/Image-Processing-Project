import requests

WEBHOOK_URL = "http://127.0.0.1:5000/webhook"

def send_webhook(detection_status: str, emotion_status: str):
    payload = {
        "detection_status": detection_status,
        "emotion_status": emotion_status
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=3)
        print("Webhook sent:", payload, "Status:", response.status_code)
    except Exception as e:
        print("Webhook error:", e)