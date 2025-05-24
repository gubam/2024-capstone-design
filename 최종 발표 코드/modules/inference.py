# modules/infer_video.py
import torch
import cv2
import os
from .predictor import SignLanguagePredictor
from .kp_extractor import keypoint # your own module


def run_sign_inference(video_path):
    output = []
    keypoint_inst = keypoint(kf_sw=True, draw_graph_sw=False, z_kill=True)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoint_inst.extract_keypoint(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    angle_tensor = torch.tensor(keypoint_inst.angle_list, dtype=torch.float32)

###여기서 불러옴
    predictor = SignLanguagePredictor()
    results = predictor.predict_with_stride(
        angle_tensor, window_size=100, stride=10, min_confidence=0.80
    )

    for start, end, label, conf in results:
        print(f"🟦 {start}-{end} 프레임: {label} ({conf*100:.1f}%)")
        output.append(label)

    return output
