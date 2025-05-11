import torch
import torch.nn as nn

label_map = {
    0: "ambulance", 1: "cold", 2: "come", 3: "constipation", 4: "counsel",
    5: "cure", 6: "diarrhea", 7: "discharge", 8: "doctor", 9: "examine",
    10: "fracture", 11: "hand", 12: "head", 13: "health", 14: "hospitalization",
    15: "left", 16: "me", 17: "neck", 18: "nurse", 19: "patient",
    20: "recover", 21: "right", 22: "runnynose", 23: "sick", 24: "standby",
    25: "start", 26: "temperature", 27: "weight", 28: "wound", 29: "you"
}


#메인에서 해당클래스 객체 선언 예정
class ModelLoader:
    def __init__(self, model_name ,input_size = 50, hidden_size = 50, num_layers = 2 , num_classes = 30 ):
        self.input_size = input_size
        self.hidden_size =hidden_size  
        self.num_layers =  num_layers 
        self.num_classes = num_classes
        self.model =  self._create_model(model_name)
        #파라미터 로드
        self.model.load_state_dict(torch.load("C:/Users/gubam/Documents/GitHub/2024-capstone-design/code/gubam/pt_file/30class.pt",map_location=torch.device('cpu')))
        self.model.eval()
        print("모델이 성공적으로 불러와졌습니다!")

    def inference_output(self, input):        
        input = torch.tensor(input)
        output = self.model(input.unsqueeze(0))

        print(output)
        #인덱스로 결과값 찾기
        if torch.max(output) < 2:
            return 'None' 
        max_index = torch.argmax(output)

        matched_label = label_map[int(max_index)]

        return matched_label
    

    def _create_model(self, model_name):
        model_map = {
            'lstm': LSTMModel,
            # 'cnn': CNNModel,  # 추가 모델 생기면 여기에 추가
            # 'transformer': TransformerModel
        }

        model_class = model_map.get(model_name.lower())

        if model_class is None:
            raise ValueError(f"Unknown model name: {model_name}")

        return model_class(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.num_classes
        )

#모델 정의
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