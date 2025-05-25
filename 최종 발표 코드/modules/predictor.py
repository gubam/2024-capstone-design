#################
# 학습 모델을 불러와서 
# 특정 Confidence 이상이면 리스트 더해서 출력하기 위한 모듈
# 아래에다가 모델 정의해두기
# 모델과 라벨맵 그리고 pt경로 설정하기 위한 모듈
#################
import torch
import torch.nn.functional as F
import os
import torch
import torch.nn as nn

label_map = {
    0: "검사하다", 1: "손", 2: "머리", 3: "안녕하세요",
    4: "오른쪽", 5: "아프다", 6: "감사합니다", 7: "상처"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
##########모델 .pt path설정 파트
##########최종발표코드 -> file 경로에 pt파일 넣고 이름만 바꾸기 -> 지금은 4_words.pt로 이걸 파일명으로 바꿔
MODEL_PATH = os.path.join(BASE_DIR, "..", "file", "1dcnn.pt")

class SignLanguagePredictor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##학습 모델정의부분 
        self.model = CNNBackbone().to(self.device)
        ##여기만 가져온모델로 바꾸세요

        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
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

##################
#모델정의
##################
class LSTMModel(nn.Module):
    def __init__(self, input_size=102, hidden_size=128, num_layers=2, num_classes=10, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        if self.lstm.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            out = hn[-1]
        return self.fc(out)
    
    
class CNNBackbone(nn.Module):
    def __init__(self, input_size=102, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, T, D)
        x = x.transpose(1, 2)  # → (B, D, T)
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)