import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


# MusicTransformer
class Generator(nn.Module):
    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False):
        super(Generator, self).__init__()

        self.dummy      = DummyDecoder()
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr
        self.name = 'gumbelgan'
        self.temperature = 1.0  # init value is 1.0
        self.theta = None
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Base transformer
        if(not self.rpr):
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)


    def step(self, x, mask=True):

        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        x = self.embedding(x)
        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1,0,2)
        x = self.positional_encoding(x)
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)
        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)
        y = self.Wout(x_out)
        del mask
        y = self.softmax(y)[..., :TOKEN_END]
        y = y[:, -1,:]
        gumbel_t = self.add_gumbel(y)
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        pred = F.softmax(gumbel_t * self.temperature, dim=-1)
        next_token_onehot = None

        # They are trained to predict the next note in sequence (we don't need the last one)
        return pred, next_token, next_token_onehot

    # generate
    def sample(self, primer=None, target_seq_length=256, one_hot = False):

        #                     2 when not generating / 1 when generating
        gen_seq = torch.full((1, target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        #not generating:
        #num_primer = primer[0].size(dim=0)
        #generating:
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        cur_i = num_primer

        all_preds = torch.zeros(2, target_seq_length//2, 388).to(get_device()) #batch_size x seq_length x vocab_size

        while(cur_i < target_seq_length):
            pred_y, next_token, _ = self.step(gen_seq[..., :cur_i])
            if (one_hot):
                all_preds[:, cur_i-128] = pred_y

            gen_seq[:, cur_i] = next_token
            if (not one_hot):
                if (next_token == TOKEN_END):
                    break

            cur_i += 1

        if (one_hot): return all_preds
        return gen_seq[:, :cur_i]

    @staticmethod
    def add_gumbel(theta, eps=1e-10):
        u = torch.zeros(theta.size())
        u = u.to(get_device())
        u.uniform_(0, 1)
        gumbel_t = torch.log(theta + eps) - torch.log(-torch.log(u + eps) + eps)
        return gumbel_t

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory

