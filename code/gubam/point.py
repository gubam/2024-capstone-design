import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
holistic = mp_holistic.Holistic(static_image_mode=False)

# 비디오 파일 열기
cap = cv2.VideoCapture("C:/Users/82109/Desktop/test1/0.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 미러링
    h, w, _ = frame.shape

    # 입력 전처리
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # 검은 배경 생성
    black_img = np.zeros_like(frame)

    # 그리기 도우미 함수 (굵게 수정)
    def draw_landmarks(landmarks, connections=None, color=(255, 255, 255)):
        if landmarks:
            for lm in landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(black_img, (cx, cy), 3, color, -1)  # 점 굵기 증가
            if connections:
                for con in connections:
                    start_idx, end_idx = con
                    start = landmarks.landmark[start_idx]
                    end = landmarks.landmark[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(black_img, (x1, y1), (x2, y2), color, 3)  # 선 굵기 증가

    # 랜드마크 그리기
    draw_landmarks(results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, (0, 255, 0))        # 초록: 포즈
    draw_landmarks(results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, (255, 0, 0))    # 파랑: 왼손
    draw_landmarks(results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, (0, 0, 255))   # 빨강: 오른손
    draw_landmarks(results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, (200, 200, 200)) # 회색: 얼굴

    # 출력
    cv2.imshow('Landmarks on Black Background', black_img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
