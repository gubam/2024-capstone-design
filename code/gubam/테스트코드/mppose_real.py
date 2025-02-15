import mediapipe as mp
import cv2
import numpy as np
import json

# MediaPipe의 그림 및 전체 모델 유틸리티
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 웹캠 또는 동영상 캡처 초기화
cap = cv2.VideoCapture("C:/Users/gubam/Desktop/수어 데이터셋/원본영상/나_WORD1157.mp4")

# 전체 모델 사용
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_count = 0  # JSON에 프레임별로 쉼표를 추가하기 위해 프레임 수 추적
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        temp = []
        pointDic = {}

        # 웹캠의 BGR 프레임을 MediaPipe 처리를 위해 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 오른손 랜드마크 
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                #mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                temp.append([landmark.x, landmark.y, landmark.z])
        else:
            for i in range(20):
                temp.append(-1)
        pointDic["right"] = temp
        temp = []

        # 왼손 랜드마크 처리
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                #mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                temp.append([landmark.x, landmark.y, landmark.z])
        else:
            for i in range(20):
                temp.append(-1)
        pointDic["left"] = temp
        temp = []

        # 신체 랜드마크 처리 (11-22번)
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark[11:23]):
                temp.append([landmark.x, landmark.y, landmark.z])
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                #mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        else:
            for i in range(12):
                temp.append(-1)
        pointDic["body"] = temp

        # with open(f"C:/Users/SAMSUNG/OneDrive/바탕 화면/Coding/캡스톤/데이터/{frame_count}.json", "w") as outfile:
        #     json.dump(pointDic, outfile)
                
        frame_count += 1

        # 결과 프레임 표시
        cv2.imshow('Holistic Model Detection', image)
        print(pointDic)

        # 'q'를 누르면 루프 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# JSON 배열 닫고 파일 닫기
# output_file.write("\n]")
# output_file.close()

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
