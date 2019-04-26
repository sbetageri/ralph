import torch
import torch.nn as nn

class ListenNet(nn.Module):
    def __init__(self, device):
        super(ListenNet, self).__init__()
        self.lstm = nn.LSTM(input_size=13, hidden_size=256, num_layers=3)
        self.lstm = self.lstm.to(device)

    def forward(self, x):
        lstm_out, (hidden, carry)= self.lstm(x)
        return lstm_out, hidden[:, -1]

    def get_parameters(self):
        return self.lstm.parameters()