import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_LSTM_Model, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # grayscale input
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 → 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 14 → 7
        )

        # feature size = 32 * 7 * 7
        self.lstm = nn.LSTM(
            input_size=32 * 7 * 7,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1, 28, 28)
        b, seq_len, c, h, w = x.size()

        cnn_out = []
        for t in range(seq_len):
            out = self.cnn(x[:, t])   # (b, 32, 7, 7)
            out = out.view(b, -1)     # flatten
            cnn_out.append(out)

        cnn_out = torch.stack(cnn_out, dim=1)  # (b, seq_len, features)

        lstm_out, _ = self.lstm(cnn_out)

        out = lstm_out[:, -1, :]  # last timestep
        out = self.fc(out)

        return out