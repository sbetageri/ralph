import torch
import torch.nn as nn

from attention import Attention


class SpellNet(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size, device):
        super(SpellNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedded = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.attentionVideo = Attention.AttentionNet(hidden_size, hidden_size)
        self.attentionAudio = Attention.AttentionNet(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        ## Move to Device
        self.embedded = self.embedded.to(device)
        self.lstm = self.lstm.to(device)
        self.attentionVideo = self.attentionVideo.to(device)
        self.attentionAudio = self.attentionAudio.to(device)
        self.mlp = self.mlp.to(device)

    def forward(self, input, hidden_state, cell_state, watch_outputs, listen_outputs):
        input = self.embedded(input)
        context = torch.zeros_like(input)
        concatenated = torch.cat([input, context], dim=1)
        output, (hidden_state, cell_state) = self.lstm(concatenated, (hidden_state, cell_state))
        video_context = self.attentionVideo(hidden_state[-1], watch_outputs)
        audio_context = self.attentionVideo(hidden_state[-1], listen_outputs)
        combined_context = torch.cat([output, video_context, audio_context], dim=1)
        output = self.mlp(combined_context).unsqueeze(1)

        return output, hidden_state, cell_state, context
