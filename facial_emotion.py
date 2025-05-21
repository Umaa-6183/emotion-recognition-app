from fer import FER
import cv2

def detect_facial_emotion():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    emotion, score = detector.top_emotion(frame)
    cap.release()
    return emotion, score
