from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from emotion_predictor import predict_emotion
from llm_explainer import explain_emotion

app = FastAPI()

@app.post("/detect-emotion")
async def detect_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    emotion, confidence = predict_emotion(frame)
    explanation = explain_emotion(emotion, confidence)

    return {
        "emotion": emotion,
        "confidence": round(confidence, 2),
        "explanation": explanation
    }
