import cv2
from face_detection import detect_faces
from emotion_predictor import predict_emotion
from voice import say_hello
from person_tracker import is_new_person

cap = cv2.VideoCapture(0)

hello_text_frames = 0  # show "Hello!" for a short time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    # ✅ Detect new person using MediaPipe signature
    if len(faces) > 0 and is_new_person(frame):
        say_hello()
        hello_text_frames = 30  # show text for ~30 frames

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
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
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                3
            )

    if hello_text_frames > 0:
        hello_text_frames -= 1

    cv2.imshow("Emotion + New Person Greeting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
