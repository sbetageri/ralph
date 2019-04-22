import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionNet: #(nn.Module)
	def __init__(self, hidden_size, annotation_size):
		self.dense = nn.Sequential(
			nn.Linear(hidden_size + annotation_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1)
		)

	def forward(self, prev_hidden_state, annotations):
		batch_size, sequence_length, _ = annotations.size()

		prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)

		concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
		attn_energies = self.dense(concatenated).squeeze(2)
		alpha = F.softmax(attn_energies).unsqueeze(1)
		context = alpha.bmm(annotations)

		return context