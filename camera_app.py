import cv2
from face_detection import detect_faces
from emotion_predictor import predict_emotion
from voice import say_hello

cap = cv2.VideoCapture(0)

human_present = False
hello_display = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    # If new human arrives
    if len(faces) > 0 and not human_present:
        say_hello()
        human_present = True
        hello_display = True

    # If no human anymore
    if len(faces) == 0:
        human_present = False
        hello_display = False

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        emotion, confidence = predict_emotion(face_img)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show emotion
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Show Hello text
        if hello_display:
            cv2.putText(
                frame,
                "Hello!",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                3
            )

    cv2.imshow("Emotion + Voice AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
