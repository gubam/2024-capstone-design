import cv2
import mediapipe as mp
import mp_keypoint

VIDEO_SRC = "C:/Users/gubam/Desktop/수어 데이터셋/원본영상/나_WORD1157.mp4"

#미디어파이프 인스턴스 선언 부분

keypoint = mp_keypoint.keypoint(mp.solutions.drawing_utils,
                                mp.solutions.holistic,
                                mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))

cap = cv2.VideoCapture(VIDEO_SRC)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image, output = keypoint.extract_keypoint(frame)
    
    #print(output)
    cv2.imshow('video',cv2.resize(image,dsize=(960,540)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
