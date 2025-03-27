import cv2
import numpy as np
import os
import glob

root_dir = r"C:/Users/SAMSUNG/OneDrive/바탕 화면/Coding/캡스톤/영상변환/영상"

# 🔥 폴더 자동 탐색
word_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

scales = {
    "_s": (0.8, 1.0),
    "_w": (1.2, 1.0),
    "_0.75": (1.0, 0.75),
    "_1.25": (1.0, 1.25),
    "_s_0.75": (0.8, 0.75),
    "_s_1.25": (0.8, 1.25),
    "_w_0.75": (1.2, 0.75),
    "_w_1.25": (1.2, 1.25),
}

for word in word_folders:
    word_path = os.path.join(root_dir, word)
    
    # 내부 폴더 순회
    for i in range(14):
        folder_name = f"{word}{i}"
        input_video_path = os.path.join(word_path, folder_name, f"{folder_name}.mp4")
        output_dir = os.path.join(word_path, folder_name)

        if not os.path.exists(input_video_path):
            print(f"❌ {input_video_path} 없음! 스킵")
            continue

        # 기존 파일 삭제
        old_files = glob.glob(os.path.join(output_dir, f"{folder_name}_*.mp4"))
        for f in old_files:
            os.remove(f)
            print(f"🗑️ {f} 삭제 완료")

        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        print(f"✅ {folder_name} - {len(frames)} 프레임 로드 완료")

        # 변환
        for suffix, (scale_x, speed_ratio) in scales.items():
            out_path = os.path.join(output_dir, f"{folder_name}{suffix}.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            time_acc = 0.0
            for idx in range(len(frames)):
                frame = frames[idx]

                resized = cv2.resize(frame, None, fx=scale_x, fy=1.0)
                if scale_x < 1.0:
                    diff = w - resized.shape[1]
                    pad_left = diff // 2
                    pad_right = diff - pad_left
                    resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
                elif scale_x > 1.0:
                    crop = (resized.shape[1] - w) // 2
                    resized = resized[:, crop:crop+w]

                time_acc += speed_ratio
                while time_acc >= 1.0:
                    out.write(resized)
                    time_acc -= 1.0

            out.release()
            print(f"➡ {out_path} 저장 완료")

print("🎉 자동 폴더 탐색 + 전체 변환 완료!")
