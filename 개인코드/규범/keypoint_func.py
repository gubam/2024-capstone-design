'''
keypoint 추출을 위한 함수

해당 함수의 입력은 cv2의 capture frame
출력은 오른손 왼손 몸의 nparray

오른손, 왼손 각각 20point
상체는 12point(11 ~ 22)
numpy array로 출력하기
'''
import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def keypoint(frame):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

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

            # Draw only landmarks 11 to 22
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
        
        result = pointDic.items()
        data = list(result)
        nparray = np.array(data)

        return nparray


