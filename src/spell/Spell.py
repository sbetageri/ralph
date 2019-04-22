import torch
import torch.nn as nn

class SpellNet:
	def __init__(self, num_layers, hidden_size, output_size):
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers

		self.embedded = nn.Embedding(self.output_size, self.hidden_size)
		self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, self.num_layers, batch_first=True)
		self.attentionVideo = AttentionNet(hidden_size, hidden_size)
		self.mlp = nn.Sequential(
			nn.Linear(hidden_size * 2, hidden_size),
			nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Linear(256, output_size)
		)

	def forward(self, input, hidden_state, cell_state, watch_outputs, context):
		input = self.embedded(input)
		concatenated = torch.cat([input, context], dim=2)
		output, (hidden_state, cell_state) = self.lstm(concatenated, (hidden_state, cell_state))
		context = self.attentionVideo(hidden_state[-1], watch_outputs)
		output = self.mlp(torch.cat([output, context], dim=2).squeeze(1)).unsqueeze(1)

		return output, hidden_state, cell_state, context