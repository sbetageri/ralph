import torch
import torch.nn as nn

class ListenNet:
    def __init__(self):
        self.lstm = nn.LSTM(input_size=13, hidden_size=256, num_layers=3)

    def forward(self, x):
        lstm_out, (hidden, carry)= self.lstm(x)
        return lstm_out, hidden

    def get_parameters(self):
        return self.lstm.parameters()