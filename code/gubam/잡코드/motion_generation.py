import os
import json
import cv2
import numpy as np
from glob import glob

def parse_keypoints(flat_list, keep_indices=None):
    kps = [(int(flat_list[i]), int(flat_list[i+1]), flat_list[i+2]) for i in range(0, len(flat_list), 3)]
    if keep_indices is not None:
        return [kps[i] if i < len(kps) else (0, 0, 0) for i in keep_indices]
    return kps


def draw_stick_figure(canvas, kps, pairs, color, radius=5, thickness=3):
    for x, y, conf in kps:
        if conf > 0.2:
            cv2.circle(canvas, (x, y), radius, color, -1)
    for a, b in pairs:
        if a < len(kps) and b < len(kps):
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if ca > 0.2 and cb > 0.2:
                cv2.line(canvas, (xa, ya), (xb, yb), color, thickness)

def render_openpose_folder(folder_path, canvas_size=1500, display_size=800):
    json_files = sorted(glob(os.path.join(folder_path, '*_keypoints.json')))
    if not json_files:
        print(f"❌ No JSON files in {folder_path}")
        return False  # skip

    print(f"▶️ Playing folder: {os.path.basename(folder_path)} ({len(json_files)} frames)")

    POSE_PAIRS = [
        (1, 2), (2, 3), (3, 4),         # 오른팔: Neck → RShoulder → RElbow → RWrist
        (1, 5), (5, 6), (6, 7),         # 왼팔: Neck → LShoulder → LElbow → LWrist

    ]


    HAND_PAIRS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20)
    ]

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        person = data.get("people", {})
        if not person:
            continue

        pose_kps = parse_keypoints(person.get("pose_keypoints_2d", []))
        hand_l_kps = parse_keypoints(person.get("hand_left_keypoints_2d", []))
        hand_r_kps = parse_keypoints(person.get("hand_right_keypoints_2d", []))
        face_kps = parse_keypoints(person.get("face_keypoints_2d", []))

        draw_stick_figure(canvas, pose_kps, POSE_PAIRS, color=(0, 0, 0), thickness=6)
        draw_stick_figure(canvas, hand_l_kps, HAND_PAIRS, color=(0, 0, 200), thickness=3)
        draw_stick_figure(canvas, hand_r_kps, HAND_PAIRS, color=(0, 120, 0), thickness=3)

        for x, y, conf in face_kps:
            if conf > 0.3:
                cv2.circle(canvas, (x, y), 2, (80, 80, 80), -1)

        filename = os.path.basename(file_path)
        cv2.putText(canvas, filename, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)

        display = cv2.resize(canvas, (display_size, display_size))
        cv2.imshow("OpenPose Viewer (q to quit)", display)

        key = cv2.waitKey(30)
        if key == ord('q'):
            return True  # 전체 종료

    return False  # 정상 완료

def render_multiple_folders(folder_list):
    for folder in folder_list:
        should_quit = render_openpose_folder(folder)
        if should_quit:
            print("🛑 User requested quit.")
            break
        print(f"✅ Finished: {folder}\n")

    cv2.destroyAllWindows()
    print("🎉 All folders completed.")

# 사용 예:
folders = [
    "C:/Users/82109/Desktop/NIA_SL_WORD0001_SYN01_F",
    "C:/Users/82109/Desktop/NIA_SL_WORD0001_SYN01_F",
    "C:/Users/82109/Desktop/NIA_SL_WORD0001_SYN01_F"
]

render_multiple_folders(folders)
