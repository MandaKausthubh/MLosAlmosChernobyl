import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):

    def __init__(self, N) -> None:
        super().__init__()
        self.d = N 
        self.V = nn.Linear(N, N)
        self.K = nn.Linear(N, N)
        self.Q = nn.Linear(N, N)
        self.SM = nn.Softmax(-1)

    def forward(self, x):
        val = self.V(x)
        key = self.K(x)
        query = self.Q(x)

        # calculating the Attention values
        attention = self.SM(
            (torch.matmul(query, key.transpose(-2, -1)))/
                torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        )
        return attention @ val


class Encoder(nn.Module):

    def __init__(self, N, M) -> None:
        super().__init__()
        self.N = N
        self.M = M
        self.attention = AttentionBlock(N)
        self.linear = nn.Linear(N, M)
        self.norm = nn.LayerNorm(N)

    def forward(self, x):
        x = self.norm(self.attention(x) + x)
        return self.norm(self.linear(x) + x)

