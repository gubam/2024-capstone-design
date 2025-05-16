#################
# 학습 모델을 불러와서 
# 특정 Confidence 이상이면 리스트 더해서 출력하기 위한 모듈
#################
import torch
import torch.nn.functional as F
from .model import LSTMModel

label_map = {
    0: "치료하다", 1: "머리", 2: "왼쪽", 3: "나",
    4: "간호사", 5: "오른쪽", 6: "아프다", 7: "너",
    8: "손ㅋ", 9: "건강하다"
}

class SignLanguagePredictor:
    def __init__(self, model_path, input_size=102, hidden_size=128, num_layers=2, num_classes=10, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✅ 모델 로드 완료")

    def predict_with_stride(self, angle_seq, window_size=100, stride=10, min_confidence=0.7):
        self.model.eval()
        T = angle_seq.shape[0]
        results = []

        if T < window_size:
            pad_len = window_size - T
            pad = torch.zeros((pad_len, angle_seq.shape[1]))
            angle_seq = torch.cat([angle_seq, pad], dim=0)
            T = angle_seq.shape[0]

        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = angle_seq[start:end]
            input_tensor = window.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()

            if confidence >= min_confidence:
                label = label_map[pred_idx]
                results.append((start, end, label, confidence))

        return results
