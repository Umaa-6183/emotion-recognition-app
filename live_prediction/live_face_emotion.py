import cv2
from deepface import DeepFace


def detect_emotion_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return None

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            result = DeepFace.analyze(
                frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f'Emotion: {emotion}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            emotion = "Not detected"
            cv2.putText(frame, 'Emotion: N/A', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion
