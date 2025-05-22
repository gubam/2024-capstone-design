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
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_size=102, model_dim=128, num_heads=4, num_layers=2, num_classes=10, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.input_proj = nn.Linear(input_size, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)  # (batch, seq_len, model_dim)
        x = self.transformer_encoder(x)  # (batch, seq_len, model_dim)
        x = x.mean(dim=1)  # 평균 pooling (or x[:, 0, :] if using [CLS] token idea)
        return self.classifier(x)  # (batch, num_classes)