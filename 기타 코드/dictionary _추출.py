import mediapipe as mp
import cv2
import numpy as np

# Drawing and holistic utilities from MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize webcam capture
cap = cv2.VideoCapture("C:/Users/SAMSUNG/Downloads/KakaoTalk_20241025_181310548.mp4")

# Use the holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        temp = []
        pointDic ={}

        # Convert the BGR frame from webcam to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw right hand landmarks if detected and print their coordinates
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
            # 출력: 오른손 랜드마크 좌표
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                temp.append( [landmark.x , landmark.y, landmark.z] )
        else:
            for i in range(20):
                temp.append(-1)
        pointDic["right"] = temp
        temp = []

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            # 출력: 왼손 랜드마크 좌표
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                temp.append( [landmark.x , landmark.y, landmark.z] )
        else:
            for i in range(20):
                temp.append(-1)

        pointDic["left"] = temp
        temp = []

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark[11:23]):  # 11~22번만 추출
                temp.append( [landmark.x , landmark.y, landmark.z] )

            # Draw only landmarks 11 to 22
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        else:
            for i in range(12):
                temp.append(-1)
        pointDic["body"] = temp

        # Display the resulting frame
        cv2.imshow('Holistic Model Detection', image)
        print(pointDic)
        # Exit loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
