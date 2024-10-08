import mediapipe as mp
import cv2
import numpy as np

# Drawing and holistic utilities from MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Use the holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame from webcam to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Convert the image color back to BGR for rendering in OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks if detected and print their coordinates
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )
            # 출력: 얼굴 랜드마크 좌표
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                print(f"Face Landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

        # Draw right hand landmarks if detected and print their coordinates
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
            # 출력: 오른손 랜드마크 좌표
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                print(f"Right Hand Landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

        # Draw left hand landmarks if detected and print their coordinates
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            # 출력: 왼손 랜드마크 좌표
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                print(f"Left Hand Landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

        # Draw pose landmarks if detected and print their coordinates
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            # 출력: 신체 랜드마크 좌표
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                print(f"Pose Landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

        # Display the resulting frame
        cv2.imshow('Holistic Model Detection', image)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
