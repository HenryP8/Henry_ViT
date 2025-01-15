import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Attention(nn.Module):
    def __init__(self, d_embed, n_heads, device):
        super(Attention, self).__init__()
        self.w_q = nn.Linear(d_embed, d_embed)
        self.w_k = nn.Linear(d_embed, d_embed)
        self.w_v = nn.Linear(d_embed, d_embed)
        self.w_o = nn.Linear(d_embed, d_embed)

        self.norm = nn.LayerNorm(d_embed)
        self.drop = nn.Dropout(0.2)

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_k = d_embed / n_heads

        self.device = device

    def split(self, Q, K, V):
        Q_i = torch.split(Q, int(self.d_k), 2)
        K_i = torch.split(K, int(self.d_k), 2)
        V_i = torch.split(V, int(self.d_k), 2)

        return Q_i, K_i, V_i

    def scaled_dot_product(self, Q, K, V):
        QK = torch.matmul(Q, torch.transpose(K, 1, 2))
        masked_QK = self.mask(QK)
        scaled_QK = masked_QK / math.sqrt(self.d_k)
        scaled_QK = torch.softmax(scaled_QK, dim=2)
        QKV = torch.matmul(scaled_QK, V)

        return QKV

    def mask(self,A):
        mask = torch.tril(torch.ones(A.shape)).to(self.device)
        masked = A.masked_fill(mask==0, float('-inf'))

        return masked

    def concat(self, res_i):
        ret = torch.cat(res_i, 2)
        
        return ret

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q_i, K_i, V_i = self.split(Q, K, V)

        QKV_i = [self.scaled_dot_product(Q_i[i], K_i[i], V_i[i]) for i in range(len(Q_i))]

        QKV = self.w_o(self.concat(QKV_i))
        QKV = self.drop(QKV)

        add_norm = x + self.norm(QKV)

        return add_norm


class FeedForward(nn.Module):
    def __init__(self, d_embed, hidden):
        super(FeedForward, self).__init__()

        self.norm = nn.LayerNorm(d_embed)
        self.l1 = nn.Linear(d_embed, hidden)
        self.l2 = nn.Linear(hidden, d_embed)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        ret = self.l1(x)
        ret = self.gelu(ret)
        ret = self.l2(ret)
        ret = self.drop(ret)

        ret = x + self.norm(ret)

        return ret


class Transformer(nn.Module):
    def __init__(self, n_blocks, d_embed, n_heads, hidden, dict_size, device):
        super(Transformer, self).__init__()

        self.d_embed = d_embed
        self.device = device

        self.n_blocks = n_blocks
        self.block = nn.Sequential(
            Attention(d_embed, n_heads, device), 
            FeedForward(d_embed, hidden)
        )
        self.tok_embedder = nn.Embedding(dict_size, d_embed)

        self.linear = nn.Linear(d_embed, dict_size)
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = self.tok_embedder(x)
        x = self.position_encoding(x, self.d_embed, self.device)

        for _ in range(self.n_blocks):
            x = self.block(x)

        x = self.norm(x)
        x = self.linear(x)

        return x
    
    def position_encoding(self, x, d_embed, device):
        context_window = len(x[0])

        pe = torch.zeros(context_window, d_embed).to(device)

        pos = torch.arange(context_window).to(device)
        pos = pos.view(context_window, -1)
        pos = pos.expand(context_window, int(d_embed / 2))

        odd_denom = torch.arange(1, d_embed, 2).to(device)
        odd_denom = torch.sub(odd_denom, 1)
        odd_denom = torch.div(odd_denom, d_embed)
        odd_denom = torch.pow(10000, odd_denom)

        even_denom = torch.arange(0, d_embed, 2).to(device)
        even_denom = torch.div(even_denom, d_embed)
        even_denom = torch.pow(10000, even_denom)

        pe[:, 0::2] = torch.sin(torch.div(pos, even_denom))
        pe[:, 1::2] = torch.cos(torch.div(pos, odd_denom))

        return x + pe
