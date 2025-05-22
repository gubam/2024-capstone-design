import torch
import torch.nn as nn
import torch.nn.functional as F
import mp_keypoint
import cv2

label_map = {
    0: "cure", 1: "head", 2: "left", 3: "me",
    4: "nurse", 5: "right", 6: "sick", 7: "you"
    , 8: "hand", 9: "health"
}


VIDEO_SRC = "C:/Users/82109/Desktop/test1/0.mp4"

#인스턴스 선언 부분
keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True)
cap = cv2.VideoCapture(VIDEO_SRC)
# sampling = training.ScoreSampling(50 ,skip_sample=0)

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
        # x: (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)  # hn: (num_layers * num_directions, batch, hidden_size)
        # 마지막 layer의 forward와 backward hidden state 연결
        if self.lstm.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # (batch, hidden_size * 2)
        else:
            out = hn[-1]
        return self.fc(out)  # (batch, num_classes)

class SignLanguagePredictor:
    def __init__(self, model_path, input_size=102, hidden_size=128, num_layers=2, num_classes=10, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✅ 모델 로드 완료")

    def predict(self, angle_list):
        """
        angle_list: List[List[float]] or Tensor of shape [T, 50]
        returns: (label: str, confidence: float)
        """
        if isinstance(angle_list, list):
            angle_tensor = torch.tensor(angle_list, dtype=torch.float32)
        else:
            angle_tensor = angle_list

        if angle_tensor.ndim != 2 or angle_tensor.shape[1] != 50:
            raise ValueError("입력은 [T, 50] 형태의 텐서 또는 리스트여야 합니다.")

        input_tensor = angle_tensor.unsqueeze(0).to(self.device)  # [1, T, 50]

        with torch.no_grad():
            output = self.model(input_tensor)  # [1, num_classes]
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        return label_map[pred_idx], confidence
    
def predict_sliding_window(model, angle_seq, label_map, device, window_size=100, stride=1):
    """
    angle_seq: Tensor [T, 50]
    returns: list of (start, end, label, confidence)
    """
    model.eval()
    results = []

    for start in range(0, angle_seq.shape[0] - window_size + 1, stride):
        window = angle_seq[start:start+window_size]
        input_tensor = window.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
            results.append((start, start+window_size, label_map[pred_idx], confidence))

    return results

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    keypoint.extract_keypoint(frame)
    image = keypoint.frame
    output = keypoint.angle
    
    cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
score = keypoint.score_list
frame = keypoint.frame_list
angle = keypoint.angle_list

# 추론기 생성
predictor = SignLanguagePredictor("C:/Users/gubam/Documents/GitHub/2024-capstone-design/code/gubam/pt_file/feature.pt")
# Convert list to tensor
angle_tensor = torch.tensor(angle, dtype=torch.float32)

def predict_with_stride_and_confidence(
    model, angle_seq, label_map, device,
    window_size=100, stride=10, min_confidence=0.7
):
    model.eval()
    T = angle_seq.shape[0]
    results = []

    if T < window_size:
        # 🔻 부족한 경우: padding 적용
        pad_len = window_size - T
        pad = torch.zeros((pad_len, angle_seq.shape[1]))
        padded_seq = torch.cat([angle_seq, pad], dim=0)
        input_tensor = padded_seq.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        if confidence >= min_confidence:
            label = label_map[pred_idx]
            results.append((0, T, label, confidence))
        return results

    # 🔁 일반적인 경우
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window = angle_seq[start:end]
        input_tensor = window.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        if confidence >= min_confidence:
            label = label_map[pred_idx]
            results.append((start, end, label, confidence))

    return results

# 각도 시퀀스 준비
angle_tensor = torch.tensor(angle, dtype=torch.float32)

# 추론 실행
results = predict_with_stride_and_confidence(
    predictor.model,
    angle_tensor,
    label_map,
    predictor.device,
    window_size=100,
    stride=10,
    min_confidence=0.5  # 원하는 기준에 따라 조정
)

# 결과 출력
for start, end, label, conf in results:
    print(f"🟦 {start}-{end} 프레임: {label} ({conf*100:.1f}%)")
