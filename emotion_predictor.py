from deepface import DeepFace

def predict_emotion(face_img):
    try:
        result = DeepFace.analyze(
            face_img,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]
        confidence = max(result[0]["emotion"].values()) / 100

        return emotion.capitalize(), confidence

    except Exception as e:
        return "Unknown", 0.0
