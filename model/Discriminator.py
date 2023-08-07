import torch
import torch.nn as nn
import math
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding


# MusicTransformer
class Discriminator(nn.Module):
	def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
				 dropout=0.1, max_sequence=2048, rpr=False):
		super(Discriminator, self).__init__()
		self.nlayers = n_layers
		self.nhead = num_heads
		self.d_model = d_model
		self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
		self.d_ff = dim_feedforward
		self.dropout = dropout
		self.max_seq = max_sequence
		self.rpr = rpr

		self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
		self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=num_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
		)
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer,
			num_layers=n_layers,
		)
		self.fc = nn.Linear(d_model, d_model)
		self.classifier = nn.Linear(d_model, 1)
		self.activation = nn.Sigmoid()

	def forward(self, x):
		x = self.embedding(x) * math.sqrt(self.d_model)
		x = self.positional_encoding(x)
		x = self.transformer_encoder(x)
		x = x.mean(dim=1)
		x = self.fc(x)
		return self.activation(self.classifier(x))