import torch
import torch.nn as nn
import mp_keypoint

label_map = {
    0: "cure",
    1: "head",
    2: "left",
    3: "me",
    4: "nurse",
    5: "right",
    6: "sick",
    7: "you"
}



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # ✅ 2층 LSTM
        self.fc = nn.Linear(hidden_size, num_classes)  # 마지막 출력층

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM 처리
        last_out = lstm_out[:, -1, :]  # 마지막 시퀀스 출력 사용
        out = self.fc(last_out)  # FC Layer
        return out
input_size = 50  # 특징 개수 (예제에서 keypoints의 개수)
hidden_size = 100  # LSTM의 hidden 크기
num_layers = 2  # ✅ LSTM 층 개수 (2층 LSTM)
num_classes = 8  # 출력 클래스 개수 (cure, head, me, right, sick)


# ✅ 모델 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
 
# ✅ 저장된 모델 불러오기
model.load_state_dict(torch.load("C:/Users/82109/Desktop/2024-capstone-design/code/gubam/pt_file/model_8class.pt",map_location=torch.device('cpu')))
print("모델이 성공적으로 불러와졌습니다!")


# ✅ 모델을 평가 모드로 설정
model.eval()
import cv2
import mp_keypoint
import training

VIDEO_SRC = "C:/Users/82109/Downloads/KakaoTalk_20250312_141834806.mp4"
FOLDER_NAME = "오른쪽"

#인스턴스 선언 부분
keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True)
cap = cv2.VideoCapture(VIDEO_SRC)
sampling = training.ScoreSampling(50)
counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    keypoint.extract_keypoint(frame)
    image = keypoint.frame
    output = keypoint.angle
    #print(len(keypoint.angle_list))
    
    cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    counter += 1



frame_list, angle_list = sampling.sampling(keypoint.score_list,keypoint.frame_list,keypoint.angle_list)
print(len(frame_list))
for frame in frame_list:
    cv2.imshow("Video Playback",cv2.resize(frame,dsize=(960,540)) )  # 이미지 표시
    if cv2.waitKey(int(1/30 * 1000)) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()

input = torch.tensor(angle_list)
print(input.shape)
output = model(input.unsqueeze(0))
print(model(input.unsqueeze(0)))

max_index = torch.argmax(output)
matched_label = label_map[int(max_index)]

print(matched_label) 