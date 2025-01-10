import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

VIDEO_PATH = "C:/Users/SAMSUNG/Downloads/KakaoTalk_20241025_181310548.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 0 : 오른손, 1 : 왼손, 2 : 상체
        temp_point = [0,0,0]
        pointDic ={}

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #오른손 추출후 그리기 만약 추출이 안된다면 리스트에 -1대입
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
            # 오른손 좌표따기
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                temp_point[0].append( [landmark.x , landmark.y, landmark.z] )
        else:
            for i in range(20):
                temp_point.append(-1)


        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                temp_point[1].append( [landmark.x , landmark.y, landmark.z] )
        else:
            for i in range(20):
                temp_point[1].append(-1)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark[11:23]):  # 11~22번만 추출
                temp_point[2].append( [landmark.x , landmark.y, landmark.z] )

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        else:
            for i in range(12):
                temp_point[2].append(-1)

        #추출한 좌표 리스트 "right" value값 넣기
        pointDic["right"] = temp_point[0]
        pointDic["left"] = temp_point[1]
        pointDic["body"] = temp_point[2]

        cv2.imshow('image', image)
        print(pointDic)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
