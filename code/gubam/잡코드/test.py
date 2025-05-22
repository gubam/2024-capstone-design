import mediapipe as mp
import cv2
import json

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = holistic.process(image_rgb)

def extract_coords(landmarks, indices):
    coords = {}
    for idx in indices:
        lm = landmarks.landmark[idx]
        coords[str(idx)] = [lm.x, lm.y, lm.z]
    return coords

if results.pose_landmarks and results.right_hand_landmarks:
    # 상체: 11~16번 (어깨, 팔꿈치, 손목 등)
    pose_coords = extract_coords(results.pose_landmarks, range(11, 17))
    # 오른손 손가락: 0~20
    hand_coords = extract_coords(results.right_hand_landmarks, range(21))
    
    all_coords = {
        "pose": pose_coords,
        "hand": hand_coords
    }

    with open("joint_coords.json", "w") as f:
        json.dump(all_coords, f, indent=2)

cap.release()
