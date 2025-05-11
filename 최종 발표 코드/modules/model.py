#########
#LSTM 모델
#########
import torch
import torch.nn as nn

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