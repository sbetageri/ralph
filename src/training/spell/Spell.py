import torch
import torch.nn as nn

from attention import Attention
from attention import AttentionRNN


class SpellNet(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size, device):
        super(SpellNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedded = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)

        self.attention_video_rnn = AttentionRNN.AttnDecoderRNN(256, 40)
        self.attention_audio_rnn = AttentionRNN.AttnDecoderRNN(256, 40)

        self.attentionVideo = Attention.AttentionNet(hidden_size, hidden_size)
        self.attentionAudio = Attention.AttentionNet(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        self.inp_mlp = nn.Sequential(
            nn.Linear(100, 40),
            nn.ReLU()
        )

        ## Move to Device
        self.embedded = self.embedded.to(device)
        self.lstm = self.lstm.to(device)
        self.attentionVideo = self.attentionVideo.to(device)
        self.attentionAudio = self.attentionAudio.to(device)
        self.mlp = self.mlp.to(device)

    def forward(self, input, hidden_state, cell_state, watch_outputs, listen_outputs):
        emb_input = self.embedded(input)
        context = torch.zeros_like(input)
        concatenated = torch.cat([input, context], dim=1)
        encoder_output, (hidden_state, cell_state) = self.lstm(emb_input, (hidden_state, cell_state))
        # target_length = input.size(1)

        # vid_hidden = hidden_state

        # for b in range(input.size(0)):
        #     print(b)
        #     inp = emb_input[b].view(1, *emb_input[b].size())
        #     hidden_state = hidden_state[:, b]
        #     hidden_state = hidden_state.view(3, 1, -1)
        #     cell_state = cell_state[:, b]
        #     cell_state = cell_state.view(3, 1, -1)
        #     print('Hidden : ', hidden_state.size())
        #     print('inp : ', inp.size())
        #     print('cell : ', cell_state.size())
        #     encoder_output, (hidden_state, cell_state) = self.lstm(inp, (hidden_state, cell_state))
        #     target_length = input.size(1)
        #     for i in range(target_length):
        #         txt = input[b, i]
        #         vid_attn, vid_hidden, decoder_attn = self.attention_video_rnn(txt, vid_hidden, encoder_output)
        #         print(vid_attn.size())
        #         assert False

        video_context = self.attentionVideo(hidden_state[-1], watch_outputs)
        audio_context = self.attentionAudio(hidden_state[-1], listen_outputs)
        # print(video_context.size())
        # print(audio_context.size())
        combined_context = torch.cat([video_context, audio_context], dim=1)
        # print('Combined')
        # print(combined_context.size())
        output = self.mlp(combined_context.view(2, -1))

        # return combined_context
        return output, hidden_state, cell_state, context
