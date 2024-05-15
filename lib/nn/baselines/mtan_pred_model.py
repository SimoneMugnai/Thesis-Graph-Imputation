import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tsl.nn.blocks import RNN, MLPDecoder


class MultiTimeAttention(nn.Module):

    def __init__(self, input_size: int,
                 hidden_size: int = 16,
                 emb_size: int = 16,
                 num_heads: int = 1,
                 dropout: float = 0.):
        super(MultiTimeAttention, self).__init__()
        assert emb_size % num_heads == 0, \
            "Embedding size must be divisible by the number of heads."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.emb_size_k = emb_size // num_heads
        self._scaling_factor = 1 / math.sqrt(self.emb_size_k)
        self.num_heads = num_heads
        self.lin_q = nn.Linear(emb_size, emb_size)
        self.lin_k = nn.Linear(emb_size, emb_size, bias=False)
        self.lin_v = nn.Linear(input_size * num_heads, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query: Tensor, key: Tensor, value: Tensor,
                  mask: Optional[Tensor] = None):
        """Compute Scaled Dot Product Attention"""
        # query: [b h t_q e]
        # key: [b h t_k e]
        # value: [b 1 t_k f]
        scores = torch.einsum('bhqe,bhke->bhqk', query, key)
        scores = scores * self._scaling_factor
        scores = repeat(scores, 'b h q k -> b h q k f', f=self.input_size)
        if mask is not None:
            mask = mask.unsqueeze(-3)  # mask: [b 1 t_k f] -> [b 1 1 t_k f]
            scores = scores.masked_fill(~mask, -1e9)  # mask == 0
        alpha = F.softmax(scores, dim=-2)
        alpha = self.dropout(alpha)
        out = torch.einsum('bhqkf,bhkf->bhqf', alpha, value)
        return out, alpha  # out: [b h t_q f], alpha: [b h t_q t_k]

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[Tensor] = None):
        # query: [b t_q e]
        # key: [b t_k e]
        # value, mask: [b t_k f]

        # Project query and key and split heads
        query = rearrange(self.lin_q(query), "b q (h e) -> b h q e",
                          h=self.num_heads)
        # .view(x.size(0), -1, self.num_heads, self.emb_size_k).transpose(1, 2)
        key = rearrange(self.lin_k(key), "b k (h e) -> b h k e",
                        h=self.num_heads)

        # Add dummy head dimension in value and mask
        value = rearrange(value, "b k f -> b 1 k f")
        if mask is not None:
            mask = rearrange(mask, "b k f -> b 1 k f")

        out, _ = self.attention(query, key, value, mask)
        out = rearrange(out, "b h q f -> b q (h f)")
        return self.lin_v(out)  # out: [b t d]


class MTANPredictionModel(nn.Module):

    def __init__(self,
                 input_size: int,
                 n_nodes: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 16,
                 emb_size: int = 16,
                 num_heads: int = 1,
                 rnn_layers: int = 1,
                 ff_layers: int = 1,
                 learn_emb: bool = True,
                 freq: float = 10.,
                 dropout: float = 0.):
        super(MTANPredictionModel, self).__init__()
        assert emb_size % num_heads == 0, \
            "Embedding size must be divisible by the number of heads."
        self.input_size = input_size
        self.n_nodes = n_nodes
        self.horizon = horizon

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.learn_emb = learn_emb
        self.freq = freq

        if learn_emb:
            self.time_emb = nn.Linear(1, emb_size)
        else:
            self.register_parameter('time_emb', None)

        self.mtan = MultiTimeAttention(input_size=2 * input_size * n_nodes,
                                       hidden_size=hidden_size * n_nodes,
                                       emb_size=emb_size,
                                       num_heads=num_heads)
        self.rnn = RNN(input_size=hidden_size, hidden_size=hidden_size,
                       exog_size=exog_size,
                       n_layers=rnn_layers,
                       return_only_last_state=True,
                       dropout=dropout)
        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=input_size,
                                  horizon=horizon,
                                  n_layers=ff_layers,
                                  dropout=dropout)

    def learn_time_embedding(self, time_steps):
        time_steps = time_steps.unsqueeze(-1)  # [b t] -> [b t 1]
        time_enc = self.time_emb(time_steps)
        time_enc[:, :, 1:] = torch.sin(time_enc[:, :, 1:])
        return time_enc

    def time_embedding(self, time_steps):
        time_enc = torch.zeros(time_steps.shape[0], time_steps.shape[1],
                               self.emb_size, requires_grad=False,
                               device=time_steps.device)
        position = 48. * time_steps.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.emb_size, 2) *
                             -(np.log(self.freq) / self.emb_size))
        time_enc[:, :, 0::2] = torch.sin(position * div_term)
        time_enc[:, :, 1::2] = torch.cos(position * div_term)
        return time_enc

    def forward(self, x, u, input_mask, time_steps):
        # x: [batch, time, nodes, features]
        # time_steps: [batch, time]

        if self.learn_emb:
            query = key = self.learn_time_embedding(time_steps)
        else:
            query = key = self.time_embedding(time_steps)
        # Concatenate mask to input
        x = torch.cat((x, input_mask), -1)  # [b t n 2f]
        input_mask = torch.cat((input_mask, input_mask), -1)  # [b t n 2f]

        x = rearrange(x, "b t n f -> b t (n f)")
        input_mask = rearrange(input_mask, "b t n f -> b t (n f)")

        out = self.mtan(query, key, x, input_mask)
        out = rearrange(out, "b t (n f) -> b t n f", n=self.n_nodes)

        h = self.rnn(out, u)
        return self.readout(h)


if __name__ == '__main__':
    model = MTANPredictionModel(input_size=1,
                                n_nodes=207,
                                horizon=12,
                                exog_size=4,
                                hidden_size=32,  # 128
                                emb_size=128,
                                num_heads=1,
                                learn_emb=True,
                                freq=10.,
                                dropout=0.)
    observed_data = torch.rand(32, 24, 207, 1)
    observed_mask = torch.rand(32, 24, 207, 1) > 0.5
    time_steps = torch.arange(32)[:, None] + torch.arange(24)[None]
    time_steps = time_steps.to(torch.float32)
    u = torch.rand(32, 24, 4)
    out = model(observed_data, u, observed_mask, time_steps)
