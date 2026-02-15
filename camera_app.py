import cv2
from face_detection import detect_faces
from emotion_predictor import predict_emotion
from person_tracker import should_greet
from voice import say_hello, start_voice_worker

start_voice_worker()

cap = cv2.VideoCapture(0)
hello_text_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # ✅ stable + new-person greeting
        if should_greet(face_img):
            say_hello()
            hello_text_frames = 45

        emotion, confidence = predict_emotion(face_img)

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

    cv2.imshow("Emotion + New Person Greeting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
